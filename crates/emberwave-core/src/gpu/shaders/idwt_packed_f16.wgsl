// Stage: the store of the inverse transform's last level, into a buffer (64x32 tile).

@group(0) @binding(1) var<storage, read_write> packed: array<u32>;

// The last level's 64x32 tile, quantised by `to_unorm8` (0..=255).
var<workgroup> levels: array<f32, 2048>;

fn store_vertical_8(
    _comp: CompParams,
    _component: u32,
    _x: u32,
    _y: u32,
    at: u32,
    s0: f32,
    s1: f32,
    s2: f32,
    s3: f32,
    s4: f32,
    s5: f32,
    s6: f32,
    s7: f32,
) {
    levels[at + 0u * TILE_W] = to_unorm8(s0 + DC_SHIFT);
    levels[at + 1u * TILE_W] = to_unorm8(s1 + DC_SHIFT);
    levels[at + 2u * TILE_W] = to_unorm8(s2 + DC_SHIFT);
    levels[at + 3u * TILE_W] = to_unorm8(s3 + DC_SHIFT);
    levels[at + 4u * TILE_W] = to_unorm8(s4 + DC_SHIFT);
    levels[at + 5u * TILE_W] = to_unorm8(s5 + DC_SHIFT);
    levels[at + 6u * TILE_W] = to_unorm8(s6 + DC_SHIFT);
    levels[at + 7u * TILE_W] = to_unorm8(s7 + DC_SHIFT);
}

fn level_at(at: u32, x: u32, width: u32) -> u32 {
    return select(0u, u32(levels[at]), x < width);
}

// Every invocation writes two words of the 64x32 tile (512 words total across
// 256 invocations). Rows of a packed plane are padded to whole words, and the
// padding samples are zero.
fn store_plane(comp: CompParams, _component: u32, tile_x: u32, tile_y: u32, tid: u32) {
    let width = comp.plane_width;
    for (var w = 0u; w < 2u; w = w + 1u) {
        let word_idx = tid + w * 256u;
        let row = word_idx / 16u;
        let word = word_idx % 16u;
        let y = tile_y + row;
        let word_x = tile_x / 4u + word;
        let x = 4u * word_x;
        if (y < comp.plane_height && x < width) {
            let at = row * TILE_W + 4u * word;
            packed[comp.image_offset + y * comp.image_stride + word_x] = level_at(at, x, width) |
                (level_at(at + 1u, x + 1u, width) << 8u) |
                (level_at(at + 2u, x + 2u, width) << 16u) |
                (level_at(at + 3u, x + 3u, width) << 24u);
        }
    }
}

fn finish_plane(comp: CompParams, component: u32, tile_x: u32, tile_y: u32, tid: u32, is_final: bool) {
    workgroupBarrier();
    if (is_final) {
        store_plane(comp, component, tile_x, tile_y, tid);
    }
}
