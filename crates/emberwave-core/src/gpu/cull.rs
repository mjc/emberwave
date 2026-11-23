//! Which inverse-transform tiles a frame has to run.
//!
//! The encoder repeats nonzero blocks even when their coefficients did not
//! change. A tile is skipped only after those words match the previous frame.
//! The three sequence bits in each block header change every frame and the
//! inverse transform never reads them, so they are left out of the compare.
//! The first frame, a concealed frame and a frame that changes most of its
//! blocks dispatch the whole level.

use crate::bitstream::consts::{BLOCK_SIZE, DECOMPOSITION_LEVELS};
use crate::bitstream::{Band, Component, FrameLayout, MISSING_BLOCK};

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

/// Sequence field of a block header (bits 28–30). The inverse transform does
/// not read it.
const SEQUENCE_BITS: u32 = 0x7000_0000;

/// Previous block payloads, so a repeated block does not redraw its tiles.
#[derive(Debug, Default)]
pub(crate) struct PyramidCull {
    offsets: Vec<u32>,
    payload: Vec<u32>,
    scratch: Vec<u64>,
    have_image: bool,
}

impl PyramidCull {
    /// The next frame redraws every tile. Used after the decoder is cleared and
    /// after a concealed frame, whose picture is not a function of the
    /// payloads alone.
    pub(crate) fn forget(&mut self) {
        self.have_image = false;
    }

    /// Compares each block's coefficients with the previous frame and keeps the
    /// new payloads when any differ. `offsets` are word offsets into `payload`,
    /// or [`MISSING_BLOCK`] where a block was not received. A frame that
    /// changes more than half its blocks stops there: the bitset is unused
    /// once every tile is dispatched.
    pub(crate) fn observe_frame(
        &mut self,
        block_count: u32,
        offsets: &[u32],
        payload: &[u32],
    ) -> ChangeSet {
        let n = block_count as usize;
        if !self.have_image || self.offsets.len() != n {
            self.remember(n, offsets, payload);
            self.have_image = true;
            return ChangeSet {
                bits: Vec::new(),
                count: 0,
                force_full: true,
            };
        }

        // A handful of blocks is enough to notice a frame that changes most
        // of them. The full walk then only runs for a sparse change, which is
        // the one where skipping tiles pays for the compare.
        if self.sample_is_dense(n, offsets, payload) {
            self.remember(n, offsets, payload);
            return ChangeSet {
                bits: Vec::new(),
                count: block_count,
                force_full: true,
            };
        }

        self.scratch.resize(n.div_ceil(64), 0);
        self.scratch.fill(0);
        let mut count = 0u32;
        let mut force_full = false;
        for index in 0..n {
            let old = block_words(&self.offsets, &self.payload, index);
            let new = block_words(offsets, payload, index);
            if same_coefficients(old, new) {
                continue;
            }
            self.scratch[index / 64] |= 1u64 << (index % 64);
            count += 1;
            if broad_change(block_count, count) {
                force_full = true;
                break;
            }
        }

        if force_full || count != 0 {
            self.remember(n, offsets, payload);
        }
        let bits = if force_full || count == 0 {
            Vec::new()
        } else {
            self.scratch.clone()
        };
        ChangeSet {
            bits,
            count,
            force_full,
        }
    }

    /// More than half of 32 evenly spaced blocks differ. Redrawing the frame
    /// is then cheaper than listing the tiles, and a single changed block
    /// almost never lands on the sample.
    fn sample_is_dense(&self, n: usize, offsets: &[u32], payload: &[u32]) -> bool {
        let samples = 32.min(n);
        if samples == 0 {
            return false;
        }
        let mut changed = 0u32;
        for step in 0..samples {
            let index = step * n / samples;
            let old = block_words(&self.offsets, &self.payload, index);
            let new = block_words(offsets, payload, index);
            if !same_coefficients(old, new) {
                changed += 1;
            }
        }
        changed.saturating_mul(2) > samples as u32
    }

    fn remember(&mut self, n: usize, offsets: &[u32], payload: &[u32]) {
        self.offsets.clear();
        let have = n.min(offsets.len());
        self.offsets.extend_from_slice(&offsets[..have]);
        self.offsets.resize(n, MISSING_BLOCK);
        self.payload.clear();
        self.payload.extend_from_slice(payload);
    }
}

/// Words of one block, header included, or empty when it is missing or short.
fn block_words<'a>(offsets: &'a [u32], payload: &'a [u32], index: usize) -> &'a [u32] {
    let Some(&offset) = offsets.get(index) else {
        return &[];
    };
    if offset == MISSING_BLOCK {
        return &[];
    }
    let offset = offset as usize;
    let Some(&header) = payload.get(offset) else {
        return &[];
    };
    let words = ((header >> 16) & 0xfff) as usize;
    match offset.checked_add(words) {
        Some(end) if words > 0 && end <= payload.len() => &payload[offset..end],
        _ => &[],
    }
}

/// More than half the blocks, or more than one in sixteen on a large frame.
///
/// A 4K scene that changes about a thousand blocks still marks every finest
/// tile: the apron and the coarse levels fill the picture, and building that
/// tile list costs more than drawing it.
fn broad_change(block_count: u32, count: u32) -> bool {
    count.saturating_mul(2) > block_count
        || (block_count > 1024 && count.saturating_mul(16) > block_count)
}

/// Coefficient words, ignoring the frame counter in the first header word.
fn same_coefficients(old: &[u32], new: &[u32]) -> bool {
    if old.len() != new.len() {
        return false;
    }
    if old.is_empty() {
        return true;
    }
    if (old[0] ^ new[0]) & !SEQUENCE_BITS != 0 {
        return false;
    }
    old[1..] == new[1..]
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
    let x0 = tile_x - APRON;
    let x1 = tile_x + TILE_W as i32 + APRON;
    let y0 = tile_y - APRON;
    let y1 = tile_y + TILE_H as i32 + APRON;
    let interior = x0 >= 0 && x1 <= image_w as i32 && y0 >= 0 && y1 <= image_h as i32;
    let (min_x, max_x, min_y, max_y) = if interior {
        (
            (x0 as u32) >> 1,
            ((x1 - 1) as u32) >> 1,
            (y0 as u32) >> 1,
            ((y1 - 1) as u32) >> 1,
        )
    } else {
        covered_band_edges(x0, x1, y0, y1, image_w, image_h)
    };
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

/// Band span of a tile whose apron crosses the picture edge, where samples mirror.
fn covered_band_edges(
    x0: i32,
    x1: i32,
    y0: i32,
    y1: i32,
    image_w: u32,
    image_h: u32,
) -> (u32, u32, u32, u32) {
    let mut min_x = u32::MAX;
    let mut max_x = 0u32;
    let mut min_y = u32::MAX;
    let mut max_y = 0u32;
    for x in x0..x1 {
        let bx = (mirror(x, image_w as i32) as u32) >> 1;
        min_x = min_x.min(bx);
        max_x = max_x.max(bx);
    }
    for y in y0..y1 {
        let by = (mirror(y, image_h as i32) as u32) >> 1;
        min_y = min_y.min(by);
        max_y = max_y.max(by);
    }
    (min_x, max_x, min_y, max_y)
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

    /// `blocks[i] == None` is a block that was not received. The words are the
    /// coefficient payload after the two header words.
    fn frame(blocks: &[Option<&[u32]>], sequence: u8) -> (Vec<u32>, Vec<u32>) {
        let mut offsets = vec![MISSING_BLOCK; blocks.len()];
        let mut payload = Vec::new();
        for (index, block) in blocks.iter().enumerate() {
            let Some(coeffs) = block else {
                continue;
            };
            offsets[index] = payload.len() as u32;
            let words = (2 + coeffs.len()) as u32;
            payload.push((words << 16) | (u32::from(sequence) << 28));
            payload.push((index as u32) << 8);
            payload.extend_from_slice(coeffs);
        }
        (offsets, payload)
    }

    fn watch(cull: &mut PyramidCull, blocks: &[Option<&[u32]>], sequence: u8) -> ChangeSet {
        let (offsets, payload) = frame(blocks, sequence);
        cull.observe_frame(blocks.len() as u32, &offsets, &payload)
    }

    #[test]
    fn repeated_payloads_select_nothing() {
        let mut cull = PyramidCull::default();
        let none = [None, None, None, None];
        let first = watch(&mut cull, &none, 0);
        assert!(first.force_full);
        let second = watch(&mut cull, &none, 1);
        assert!(!second.force_full);
        assert_eq!(second.count, 0);

        let mut blocks = [None, Some(&[7, 8][..]), None, None];
        let changed = watch(&mut cull, &blocks, 2);
        assert!(!changed.force_full);
        assert_eq!(changed.count, 1);
        assert!(bit(&changed.bits, 1));
        assert!(!bit(&changed.bits, 0));

        // The frame counter moved and the coefficients did not.
        let same = watch(&mut cull, &blocks, 3);
        assert_eq!(same.count, 0);
        assert!(!same.force_full);
        blocks[1] = Some(&[7, 9][..]);
        let edited = watch(&mut cull, &blocks, 4);
        assert_eq!(edited.count, 1);
        assert!(bit(&edited.bits, 1));

        // More than half the blocks is a full dispatch: building the tile list
        // costs more than running the level.
        let many = watch(
            &mut cull,
            &[Some(&[1][..]), Some(&[1][..]), Some(&[1][..]), None],
            5,
        );
        assert!(many.force_full);

        cull.forget();
        let after_clear = watch(&mut cull, &none, 6);
        assert!(after_clear.force_full);
    }

    #[test]
    fn one_block_off_the_sample_stays_sparse() {
        let mut cull = PyramidCull::default();
        let blank = vec![None; 64];
        assert!(watch(&mut cull, &blank, 0).force_full);
        let mut blocks = blank.clone();
        // Samples for 64 blocks are the even indexes.
        blocks[1] = Some(&[4, 5][..]);
        let changed = watch(&mut cull, &blocks, 1);
        assert!(!changed.force_full);
        assert_eq!(changed.count, 1);
        assert!(bit(&changed.bits, 1));

        let mut dense = vec![None; 64];
        for block in dense.iter_mut().step_by(2) {
            *block = Some(&[9][..]);
        }
        assert!(watch(&mut cull, &dense, 2).force_full);
    }

    #[test]
    fn a_wide_change_on_a_large_frame_redraws_everything() {
        let mut cull = PyramidCull::default();
        let blank = vec![None; 2048];
        assert!(watch(&mut cull, &blank, 0).force_full);

        let mut few = blank.clone();
        for block in few.iter_mut().take(10) {
            *block = Some(&[1][..]);
        }
        let sparse = watch(&mut cull, &few, 1);
        assert!(!sparse.force_full);
        assert_eq!(sparse.count, 10);

        let mut many = blank;
        for block in many.iter_mut().take(200) {
            *block = Some(&[2][..]);
        }
        assert!(watch(&mut cull, &many, 2).force_full);
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
    fn interior_tiles_cover_the_same_band_as_the_mirror_walk() {
        for (width, height) in [(3840, 2160), (1920, 1080), (64, 64), (40, 48)] {
            let layout = FrameLayout::new(width, height, ChromaSubsampling::Yuv420).unwrap();
            for level in 0..DECOMPOSITION_LEVELS {
                let (band_w, band_h) = layout.level_size(level);
                let (tiles_x, tiles_y) = tile_grid(&layout, level);
                for ty in 0..tiles_y {
                    for tx in 0..tiles_x {
                        let fast = covered_band(tx, ty, band_w, band_h, 2 * band_w, 2 * band_h);
                        let tile_x = tx as i32 * TILE_W as i32;
                        let tile_y = ty as i32 * TILE_H as i32;
                        let slow = covered_band_edges(
                            tile_x - APRON,
                            tile_x + TILE_W as i32 + APRON,
                            tile_y - APRON,
                            tile_y + TILE_H as i32 + APRON,
                            2 * band_w,
                            2 * band_h,
                        );
                        let max_bx = band_w.saturating_sub(1);
                        let max_by = band_h.saturating_sub(1);
                        let slow = (
                            slow.0.min(max_bx),
                            slow.2.min(max_by),
                            slow.1.min(max_bx),
                            slow.3.min(max_by),
                        );
                        assert_eq!(fast, slow, "{width}x{height} L{level} tile {tx},{ty}");
                    }
                }
            }
        }
    }

    #[test]
    fn mirror_matches_the_shader() {
        assert_eq!(mirror(-1, 10), 1);
        assert_eq!(mirror(-4, 10), 4);
        assert_eq!(mirror(10, 10), 8);
        assert_eq!(mirror(3, 10), 3);
    }
}
