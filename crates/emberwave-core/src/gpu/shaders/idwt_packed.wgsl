// Stage: the store of the inverse transform's last level, into a buffer.
//
// Behind idwt.wgsl. Upstream's last level stores the 8-bit samples into UNORM
// storage images (shaders/idwt.comp); 8-bit storage textures are outside what
// WebGPU guarantees, so this packs them four to a word into a buffer instead,
// the first sample in the low byte. The CPU reads the planes from there, the
// colour conversion takes them, and copies take them into textures (decision
// D41).

@group(0) @binding(1) var<storage, read_write> packed: array<u32>;

// The last level's 64x32 tile packed into 512 words (2048 bytes, staying under
// the 16 KiB WebGPU workgroup storage cap alongside `horizontal` 10240 bytes).
var<workgroup> levels: array<atomic<u32>, 512>;

fn init_levels(tid: u32) {
    atomicStore(&levels[tid], 0u);
    atomicStore(&levels[tid + 256u], 0u);
}

fn store_vertical_8(
    comp: CompParams,
    _component: u32,
    x: u32,
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
    if (x >= comp.plane_width) {
        return;
    }
    let col = at % TILE;
    let base_row = at / TILE;
    let word_col = col >> 2u;
    let shift = (col & 3u) * 8u;
    let w_base = base_row * 16u + word_col;
    atomicOr(&levels[w_base + 0u * 16u], u32(to_unorm8(s0 + DC_SHIFT)) << shift);
    atomicOr(&levels[w_base + 1u * 16u], u32(to_unorm8(s1 + DC_SHIFT)) << shift);
    atomicOr(&levels[w_base + 2u * 16u], u32(to_unorm8(s2 + DC_SHIFT)) << shift);
    atomicOr(&levels[w_base + 3u * 16u], u32(to_unorm8(s3 + DC_SHIFT)) << shift);
    atomicOr(&levels[w_base + 4u * 16u], u32(to_unorm8(s4 + DC_SHIFT)) << shift);
    atomicOr(&levels[w_base + 5u * 16u], u32(to_unorm8(s5 + DC_SHIFT)) << shift);
    atomicOr(&levels[w_base + 6u * 16u], u32(to_unorm8(s6 + DC_SHIFT)) << shift);
    atomicOr(&levels[w_base + 7u * 16u], u32(to_unorm8(s7 + DC_SHIFT)) << shift);
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
            packed[comp.image_offset + y * comp.image_stride + word_x] = atomicLoad(&levels[word_idx]);
        }
    }
}

// The barrier is outside the branch on purpose: every level waits, though only
// the last needs it. Inside `if (is_final)`, uniform as that is, it crashed
// WARP, Direct3D's software rasteriser.
fn finish_plane(comp: CompParams, component: u32, tile_x: u32, tile_y: u32, tid: u32, is_final: bool) {
    workgroupBarrier();
    if (is_final) {
        store_plane(comp, component, tile_x, tile_y, tid);
    }
}
