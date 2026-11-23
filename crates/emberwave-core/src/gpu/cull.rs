//! Which inverse-transform tiles a frame has to run.
//!
//! The encoder repeats nonzero blocks even when their bytes did not change, so
//! a tile is skipped only after its payload matches the previous frame. The
//! first frame, a concealed frame and a frame that changes most of its blocks
//! dispatch the whole level.

use crate::bitstream::consts::{BLOCK_SIZE, DECOMPOSITION_LEVELS};
use crate::bitstream::{Band, Component, FrameLayout};

/// Output tile of the inverse transform, in samples.
pub(crate) const TILE_W: u32 = 64;
pub(crate) const TILE_H: u32 = 32;
/// Samples of extension the lifting reads past the tile.
const APRON: i32 = 4;

const DETAIL: [Band; 3] = [Band::Hl, Band::Lh, Band::Hh];

/// Blocks whose payload differs from the frame already on the GPU.
pub(crate) struct ChangeSet {
    pub bits: Vec<u64>,
    /// Blocks whose payload differed. The dispatch uses [`Self::force_full`]
    /// and [`Self::bits`]; tests read the count.
    #[cfg_attr(not(test), allow(dead_code))]
    pub count: u32,
    /// Dispatch every tile. The bitset is then unused.
    pub force_full: bool,
}

/// Output tiles of one component that this frame must recompute.
#[derive(Clone)]
pub(crate) struct CompCoverage {
    pub tiles_x: u32,
    pub tiles_y: u32,
    pub bits: Vec<u64>,
}

/// Tiles to dispatch, and the output coverage the next finer level reads.
pub(crate) struct Selected {
    pub ids: Vec<u32>,
    pub coverage: Vec<CompCoverage>,
}

/// Where one level's tile ids live in the GPU buffer.
pub(crate) struct TileRegion {
    pub offset: u64,
    pub size: u64,
}

/// Full tile lists for every level, coarse to fine, packed for the GPU.
pub(crate) struct TilePlan {
    pub regions: Vec<TileRegion>,
    pub counts: Vec<u32>,
    pub bytes: Vec<u8>,
}

/// Previous block payloads, so a repeated block does not redraw its tiles.
#[derive(Debug, Default)]
pub(crate) struct PyramidCull {
    start: Vec<u32>,
    len: Vec<u32>,
    arena: Vec<u32>,
    next_start: Vec<u32>,
    next_len: Vec<u32>,
    next_arena: Vec<u32>,
    have_image: bool,
}

impl PyramidCull {
    /// The next frame redraws every tile. Used after the decoder is cleared and
    /// after a concealed frame, whose picture is not a function of the
    /// payloads alone.
    pub(crate) fn forget(&mut self) {
        self.have_image = false;
    }

    /// Compares each block with the previous frame and keeps the new payloads.
    /// `block` returns the block's words, or an empty slice when it is missing.
    pub(crate) fn observe<'payload>(
        &mut self,
        block_count: u32,
        mut block: impl FnMut(u32) -> &'payload [u32],
    ) -> ChangeSet {
        let n = block_count as usize;
        let mut bits = vec![0u64; n.div_ceil(64)];
        let mut count = 0u32;
        self.next_start.clear();
        self.next_len.clear();
        self.next_arena.clear();
        self.next_start.reserve(n);
        self.next_len.reserve(n);

        for index in 0..n {
            let words = block(index as u32);
            if self.have_image && self.previous(index) != words {
                bits[index / 64] |= 1u64 << (index % 64);
                count += 1;
            }
            let start = u32::try_from(self.next_arena.len()).unwrap_or(u32::MAX);
            self.next_start.push(start);
            self.next_len
                .push(u32::try_from(words.len()).unwrap_or(u32::MAX));
            self.next_arena.extend_from_slice(words);
        }

        std::mem::swap(&mut self.start, &mut self.next_start);
        std::mem::swap(&mut self.len, &mut self.next_len);
        std::mem::swap(&mut self.arena, &mut self.next_arena);

        let force_full = !self.have_image || count.saturating_mul(2) > block_count;
        self.have_image = true;
        ChangeSet {
            bits,
            count,
            force_full,
        }
    }

    fn previous(&self, index: usize) -> &[u32] {
        let start = self.start[index] as usize;
        let len = self.len[index] as usize;
        &self.arena[start..start + len]
    }
}

/// `x | (y << 16) | (component << 28)`, in tile units. Grids of a legal frame
/// fit: the widest tile row is `16384 / 64` and the tallest is `16384 / 32`.
pub(crate) fn pack_tile(x: u32, y: u32, component: u32) -> u32 {
    x | (y << 16) | (component << 28)
}

/// Every tile of `level`, coarse levels included. `skip` is left out of the
/// list (luma at level 1 when that lowpass is built inside the finest level).
pub(crate) fn full_ids(layout: &FrameLayout, level: usize, skip: Option<Component>) -> Vec<u32> {
    let (tiles_x, tiles_y) = tile_grid(layout, level);
    let mut ids = Vec::new();
    for component in Component::ALL {
        if level < layout.finest_level(component) || skip == Some(component) {
            continue;
        }
        let z = component.index() as u32;
        for y in 0..tiles_y {
            for x in 0..tiles_x {
                ids.push(pack_tile(x, y, z));
            }
        }
    }
    ids
}

/// Tile lists for the fast inverse transform, one region per level from coarse
/// to fine. Regions start on `align` so each can be bound on its own.
pub(crate) fn plan_tiles(layout: &FrameLayout, align: u64, fuse_y_l1: bool) -> TilePlan {
    let align = align.max(4);
    let mut offset = 0u64;
    let mut bytes = Vec::new();
    let mut regions = Vec::with_capacity(DECOMPOSITION_LEVELS);
    let mut counts = Vec::with_capacity(DECOMPOSITION_LEVELS);
    for level in (0..DECOMPOSITION_LEVELS).rev() {
        let skip = (fuse_y_l1 && level == 1).then_some(Component::Y);
        let ids = full_ids(layout, level, skip);
        if bytes.len() as u64 != offset {
            bytes.resize(offset as usize, 0);
        }
        for id in &ids {
            bytes.extend_from_slice(&id.to_le_bytes());
        }
        // A binding size of zero is not a buffer binding. An empty level still
        // occupies one word and dispatches nothing.
        let size = (ids.len() as u64 * 4).max(4);
        regions.push(TileRegion { offset, size });
        counts.push(ids.len() as u32);
        offset = align_up(offset + size, align);
    }
    if bytes.len() as u64 != offset {
        bytes.resize(offset as usize, 0);
    }
    TilePlan {
        regions,
        counts,
        bytes,
    }
}

/// Tiles of `level` whose inputs differ in `changed`. `parent` is the coarser
/// level's output coverage; the coarsest level has none. `skip` is computed
/// into `coverage` and left out of `ids`.
pub(crate) fn select_tiles(
    layout: &FrameLayout,
    level: usize,
    changed: &[u64],
    parent: Option<&[CompCoverage]>,
    skip: Option<Component>,
) -> Selected {
    let (band_w, band_h) = layout.level_size(level);
    let (tiles_x, tiles_y) = tile_grid(layout, level);
    let bands: &[Band] = if level + 1 == DECOMPOSITION_LEVELS {
        Band::ALL.as_slice()
    } else {
        &DETAIL
    };
    let mut ids = Vec::new();
    let mut coverage = Vec::with_capacity(Component::ALL.len());
    for component in Component::ALL {
        if level < layout.finest_level(component) {
            coverage.push(CompCoverage {
                tiles_x: 0,
                tiles_y: 0,
                bits: Vec::new(),
            });
            continue;
        }
        let mut bits = vec![0u64; (tiles_x as usize * tiles_y as usize).div_ceil(64)];
        let info: Vec<_> = bands
            .iter()
            .filter_map(|band| layout.band(component, level, *band))
            .collect();
        let parent_comp = parent.and_then(|levels| levels.get(component.index()));
        for ty in 0..tiles_y {
            for tx in 0..tiles_x {
                let (bx0, by0, bx1, by1) =
                    covered_band(tx, ty, band_w, band_h, 2 * band_w, 2 * band_h);
                let detail = info.iter().any(|band| {
                    rect_hits(
                        changed,
                        band.first_block,
                        band.blocks_x,
                        band.blocks_y,
                        bx0,
                        by0,
                        bx1,
                        by1,
                    )
                });
                let from_parent =
                    parent_comp.is_some_and(|parent| parent_hits(parent, bx0, by0, bx1, by1));
                if !detail && !from_parent {
                    continue;
                }
                let index = ty * tiles_x + tx;
                bits[index as usize / 64] |= 1u64 << (index % 64);
                if skip != Some(component) {
                    ids.push(pack_tile(tx, ty, component.index() as u32));
                }
            }
        }
        coverage.push(CompCoverage {
            tiles_x,
            tiles_y,
            bits,
        });
    }
    Selected { ids, coverage }
}

fn tile_grid(layout: &FrameLayout, level: usize) -> (u32, u32) {
    let (band_w, band_h) = layout.level_size(level);
    ((2 * band_w).div_ceil(TILE_W), (2 * band_h).div_ceil(TILE_H))
}

/// Inclusive band-coordinate span of the tile's neighbourhood, after mirroring
/// into the image. The lowpass the coarser level wrote uses the same span.
fn covered_band(
    tx: u32,
    ty: u32,
    band_w: u32,
    band_h: u32,
    image_w: u32,
    image_h: u32,
) -> (u32, u32, u32, u32) {
    let tile_x = tx as i32 * TILE_W as i32;
    let tile_y = ty as i32 * TILE_H as i32;
    let mut min_x = u32::MAX;
    let mut max_x = 0u32;
    let mut min_y = u32::MAX;
    let mut max_y = 0u32;
    let x_end = tile_x + TILE_W as i32 + APRON;
    let y_end = tile_y + TILE_H as i32 + APRON;
    for x in (tile_x - APRON)..x_end {
        let bx = (mirror(x, image_w as i32) as u32) >> 1;
        min_x = min_x.min(bx);
        max_x = max_x.max(bx);
    }
    for y in (tile_y - APRON)..y_end {
        let by = (mirror(y, image_h as i32) as u32) >> 1;
        min_y = min_y.min(by);
        max_y = max_y.max(by);
    }
    // A neighbourhood that the image cannot contain still has to name a block.
    let max_bx = band_w.saturating_sub(1);
    let max_by = band_h.saturating_sub(1);
    (
        min_x.min(max_bx),
        min_y.min(max_by),
        max_x.min(max_bx),
        max_y.min(max_by),
    )
}

fn rect_hits(
    changed: &[u64],
    first_block: u32,
    blocks_x: u32,
    blocks_y: u32,
    bx0: u32,
    by0: u32,
    bx1: u32,
    by1: u32,
) -> bool {
    if blocks_x == 0 || blocks_y == 0 {
        return false;
    }
    let x0 = (bx0 / BLOCK_SIZE).min(blocks_x - 1);
    let x1 = (bx1 / BLOCK_SIZE).min(blocks_x - 1);
    let y0 = (by0 / BLOCK_SIZE).min(blocks_y - 1);
    let y1 = (by1 / BLOCK_SIZE).min(blocks_y - 1);
    for by in y0..=y1 {
        for bx in x0..=x1 {
            let index = first_block + by * blocks_x + bx;
            if bit(changed, index) {
                return true;
            }
        }
    }
    false
}

fn parent_hits(parent: &CompCoverage, bx0: u32, by0: u32, bx1: u32, by1: u32) -> bool {
    if parent.tiles_x == 0 || parent.tiles_y == 0 {
        return false;
    }
    let x0 = (bx0 / TILE_W).min(parent.tiles_x - 1);
    let x1 = (bx1 / TILE_W).min(parent.tiles_x - 1);
    let y0 = (by0 / TILE_H).min(parent.tiles_y - 1);
    let y1 = (by1 / TILE_H).min(parent.tiles_y - 1);
    for py in y0..=y1 {
        for px in x0..=x1 {
            let index = py * parent.tiles_x + px;
            if bit(&parent.bits, index) {
                return true;
            }
        }
    }
    false
}

fn bit(bits: &[u64], index: u32) -> bool {
    let index = index as usize;
    bits.get(index / 64)
        .is_some_and(|word| word & (1u64 << (index % 64)) != 0)
}

fn mirror(index: i32, n: i32) -> i32 {
    if n <= 1 {
        return 0;
    }
    let mut m = index;
    if m < 0 {
        m = -m;
    }
    if m > n - 1 {
        m = 2 * (n - 1) - m;
    }
    m.clamp(0, n - 1)
}

fn align_up(value: u64, align: u64) -> u64 {
    value.div_ceil(align) * align
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitstream::ChromaSubsampling;

    fn layout_512() -> FrameLayout {
        FrameLayout::new(512, 256, ChromaSubsampling::Yuv420).unwrap()
    }

    fn bitset(block_count: u32, indexes: &[u32]) -> Vec<u64> {
        let mut bits = vec![0u64; (block_count as usize).div_ceil(64)];
        for index in indexes {
            bits[*index as usize / 64] |= 1u64 << (*index % 64);
        }
        bits
    }

    #[test]
    fn repeated_payloads_select_nothing() {
        let mut cull = PyramidCull::default();
        let first = cull.observe(4, |_| &[]);
        assert!(first.force_full);
        let second = cull.observe(4, |_| &[]);
        assert!(!second.force_full);
        assert_eq!(second.count, 0);

        let mut blocks = [Vec::<u32>::new(), vec![7, 8], Vec::new(), Vec::new()];
        let changed = cull.observe(4, |index| blocks[index as usize].as_slice());
        assert!(!changed.force_full);
        assert_eq!(changed.count, 1);
        assert!(bit(&changed.bits, 1));
        assert!(!bit(&changed.bits, 0));

        let same = cull.observe(4, |index| blocks[index as usize].as_slice());
        assert_eq!(same.count, 0);
        blocks[1] = vec![7, 9];
        let edited = cull.observe(4, |index| blocks[index as usize].as_slice());
        assert_eq!(edited.count, 1);
        assert!(bit(&edited.bits, 1));

        // More than half the blocks is a full dispatch: building the tile list
        // costs more than running the level.
        let many = cull.observe(4, |index| if index < 3 { &[1u32][..] } else { &[] });
        assert!(many.force_full);

        cull.forget();
        let after_clear = cull.observe(4, |_| &[]);
        assert!(after_clear.force_full);
    }

    #[test]
    fn one_detail_block_marks_nearby_tiles_only() {
        let layout = layout_512();
        let hl = layout.band(Component::Y, 0, Band::Hl).unwrap();
        // Block (0, 0) of luma's finest horizontal detail.
        let block = hl.first_block;
        let changed = bitset(layout.block_count(), &[block]);
        let selected = select_tiles(&layout, 0, &changed, None, None);
        let (tiles_x, tiles_y) = tile_grid(&layout, 0);

        assert!(selected.ids.contains(&pack_tile(0, 0, 0)));
        assert!(selected.ids.contains(&pack_tile(1, 1, 0)));
        // Tile (6, 5) sits inside the picture and its neighbourhood does not
        // reach block (0, 0).
        assert!(!selected.ids.contains(&pack_tile(6, 5, 0)));
        assert!(selected.ids.len() < (tiles_x * tiles_y) as usize);
        assert!(selected.coverage[1].bits.is_empty());
    }

    #[test]
    fn a_coarse_change_reaches_the_finest_level() {
        let layout = FrameLayout::new(3840, 2160, ChromaSubsampling::Yuv420).unwrap();
        let ll = layout.band(Component::Y, 4, Band::Ll).unwrap();
        let changed = bitset(layout.block_count(), &[ll.first_block]);
        let mut parent = None;
        let mut finest = None;
        for level in (0..DECOMPOSITION_LEVELS).rev() {
            let selected = select_tiles(&layout, level, &changed, parent.as_deref(), None);
            parent = Some(selected.coverage);
            if level == 0 {
                finest = Some(selected.ids);
            }
        }
        let ids = finest.expect("finest level");
        let (tiles_x, tiles_y) = tile_grid(&layout, 0);
        assert!(ids.contains(&pack_tile(0, 0, 0)));
        assert!(ids.len() < (tiles_x * tiles_y) as usize);
    }

    #[test]
    fn skipped_luma_stays_in_the_parent_coverage() {
        let layout = FrameLayout::new(3840, 2160, ChromaSubsampling::Yuv420).unwrap();
        let hl = layout.band(Component::Y, 1, Band::Hl).unwrap();
        let changed = bitset(layout.block_count(), &[hl.first_block]);
        let level1 = select_tiles(&layout, 1, &changed, None, Some(Component::Y));
        assert!(level1.ids.iter().all(|id| id >> 28 != 0));
        assert!(bit(&level1.coverage[0].bits, 0));

        let level0 = select_tiles(&layout, 0, &changed, Some(&level1.coverage), None);
        assert!(level0.ids.contains(&pack_tile(0, 0, 0)));
        assert!(level0.ids.iter().all(|id| id >> 28 == 0));
    }

    #[test]
    fn full_lists_leave_fused_luma_out_of_level_one() {
        let layout = FrameLayout::new(3840, 2160, ChromaSubsampling::Yuv420).unwrap();
        let plan = plan_tiles(&layout, 256, true);
        assert_eq!(plan.regions.len(), DECOMPOSITION_LEVELS);
        assert_eq!(plan.counts.len(), DECOMPOSITION_LEVELS);
        // Coarse to fine: index 3 is level 1, index 4 is level 0.
        let level1 = &plan.counts[3];
        let (tiles_x, tiles_y) = tile_grid(&layout, 1);
        assert_eq!(*level1, tiles_x * tiles_y * 2);
        let level0 = plan.counts[4];
        let (x0, y0) = tile_grid(&layout, 0);
        assert_eq!(level0, x0 * y0);
        for region in &plan.regions {
            assert_eq!(region.offset % 256, 0);
        }
        assert_eq!(plan.bytes.len() as u64 % 4, 0);
    }

    #[test]
    fn mirror_matches_the_shader() {
        assert_eq!(mirror(-1, 10), 1);
        assert_eq!(mirror(-4, 10), 4);
        assert_eq!(mirror(10, 10), 8);
        assert_eq!(mirror(3, 10), 3);
    }
}
