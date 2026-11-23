enable f16;

// Stage: inverse CDF 9/7 wavelet transform with capability-gated f16
// intermediate storage (horizontal workgroup staging and L3..L0 LL bands)
// using a 64x32 output tile @workgroup_size(256).
// Lifting arithmetic runs in f32 so 10-bit output accuracy is preserved.

struct CompParams {
    band_offset: vec4<u32>,
    image_offset: u32,
    image_stride: u32,
    plane_width: u32,
    plane_height: u32,
    is_plane: u32,
    // For shader_f16:
    // pad0: offset in `ll_f16` for input LL band, or 0xFFFFFFFFu to read LL from `pyramid`.
    // pad1: offset in `ll_f16` for output LL band when `is_plane == 0`.
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct LevelParams {
    band_width: u32,
    band_height: u32,
    image_width: u32,
    image_height: u32,
    comps: array<CompParams, 3>,
}

@group(0) @binding(0) var<storage, read_write> pyramid: array<f32>;
@group(0) @binding(2) var<uniform> params: LevelParams;
@group(0) @binding(6) var<storage, read_write> ll_f16: array<f16>;

const ALPHA: f32 = -1.586134342059924;
const BETA: f32 = -0.052980118572961;
const GAMMA: f32 = 0.882911075530934;
const DELTA: f32 = 0.443506852043971;
const K: f32 = 1.230174104914001;
const INV_K: f32 = 0.8128930661159609;

const TILE_W: u32 = 64u;
const TILE_H: u32 = 32u;
const APRON: i32 = 4;
const DC_SHIFT: f32 = 0.5;
const NO_F16_LL: u32 = 0xFFFFFFFFu;

// 40 * 64 = 2560 halfs = 5120 bytes of workgroup memory.
var<workgroup> horizontal: array<f16, 2560>;

fn mirror(index: i32, n: i32) -> i32 {
    var m = index;
    if (m < 0) {
        m = -m;
    }
    if (m > n - 1) {
        m = 2 * (n - 1) - m;
    }
    return clamp(m, 0, n - 1);
}

fn lift(v: ptr<function, array<f32, 16>>) {
    (*v)[0] = (*v)[0] * K;
    (*v)[1] = (*v)[1] * INV_K;
    (*v)[2] = (*v)[2] * K;
    (*v)[3] = (*v)[3] * INV_K;
    (*v)[4] = (*v)[4] * K;
    (*v)[5] = (*v)[5] * INV_K;
    (*v)[6] = (*v)[6] * K;
    (*v)[7] = (*v)[7] * INV_K;
    (*v)[8] = (*v)[8] * K;
    (*v)[9] = (*v)[9] * INV_K;
    (*v)[10] = (*v)[10] * K;
    (*v)[11] = (*v)[11] * INV_K;
    (*v)[12] = (*v)[12] * K;
    (*v)[13] = (*v)[13] * INV_K;
    (*v)[14] = (*v)[14] * K;
    (*v)[15] = (*v)[15] * INV_K;
    (*v)[2] = (*v)[2] - DELTA * ((*v)[1] + (*v)[3]);
    (*v)[4] = (*v)[4] - DELTA * ((*v)[3] + (*v)[5]);
    (*v)[6] = (*v)[6] - DELTA * ((*v)[5] + (*v)[7]);
    (*v)[8] = (*v)[8] - DELTA * ((*v)[7] + (*v)[9]);
    (*v)[10] = (*v)[10] - DELTA * ((*v)[9] + (*v)[11]);
    (*v)[12] = (*v)[12] - DELTA * ((*v)[11] + (*v)[13]);
    (*v)[14] = (*v)[14] - DELTA * ((*v)[13] + (*v)[15]);
    (*v)[3] = (*v)[3] - GAMMA * ((*v)[2] + (*v)[4]);
    (*v)[5] = (*v)[5] - GAMMA * ((*v)[4] + (*v)[6]);
    (*v)[7] = (*v)[7] - GAMMA * ((*v)[6] + (*v)[8]);
    (*v)[9] = (*v)[9] - GAMMA * ((*v)[8] + (*v)[10]);
    (*v)[11] = (*v)[11] - GAMMA * ((*v)[10] + (*v)[12]);
    (*v)[13] = (*v)[13] - GAMMA * ((*v)[12] + (*v)[14]);
    (*v)[4] = (*v)[4] - BETA * ((*v)[3] + (*v)[5]);
    (*v)[6] = (*v)[6] - BETA * ((*v)[5] + (*v)[7]);
    (*v)[8] = (*v)[8] - BETA * ((*v)[7] + (*v)[9]);
    (*v)[10] = (*v)[10] - BETA * ((*v)[9] + (*v)[11]);
    (*v)[12] = (*v)[12] - BETA * ((*v)[11] + (*v)[13]);
    (*v)[5] = (*v)[5] - ALPHA * ((*v)[4] + (*v)[6]);
    (*v)[7] = (*v)[7] - ALPHA * ((*v)[6] + (*v)[8]);
    (*v)[9] = (*v)[9] - ALPHA * ((*v)[8] + (*v)[10]);
    (*v)[11] = (*v)[11] - ALPHA * ((*v)[10] + (*v)[12]);
}

fn store_ll_f16(comp: CompParams, x: i32, y: i32, value: f32) {
    if (x < i32(params.image_width) && y < i32(params.image_height)) {
        ll_f16[comp.pad1 + u32(y) * comp.image_stride + u32(x)] = f16(value);
    }
}

fn to_unorm8(v: f32) -> f32 {
    return round(clamp(v, 0.0, 1.0) * 255.0);
}

fn horiz_segment(comp: CompParams, tile_x: i32, tile_y: i32, out_width: i32, out_height: i32, row: u32, segment: u32) {
    let raw_y = tile_y + i32(row) - APRON;
    let y = select(mirror(raw_y, out_height), raw_y, raw_y >= 0 && raw_y < out_height);
    let row_bands = u32(y & 1) << 1u;
    let band_row = u32(y >> 1u) * params.band_width;
    let odd = comp.band_offset[row_bands | 1u] + band_row;
    let x0 = tile_x + i32(8u * segment) - APRON;
    var v: array<f32, 16>;

    if (row_bands == 0u && comp.pad0 != NO_F16_LL) {
        let even = comp.pad0 + band_row;
        if (x0 >= 0 && x0 + 15 < out_width) {
            let bx = u32(x0) >> 1u;
            v[0] = f32(ll_f16[even + bx + 0u]);
            v[1] = pyramid[odd + bx + 0u];
            v[2] = f32(ll_f16[even + bx + 1u]);
            v[3] = pyramid[odd + bx + 1u];
            v[4] = f32(ll_f16[even + bx + 2u]);
            v[5] = pyramid[odd + bx + 2u];
            v[6] = f32(ll_f16[even + bx + 3u]);
            v[7] = pyramid[odd + bx + 3u];
            v[8] = f32(ll_f16[even + bx + 4u]);
            v[9] = pyramid[odd + bx + 4u];
            v[10] = f32(ll_f16[even + bx + 5u]);
            v[11] = pyramid[odd + bx + 5u];
            v[12] = f32(ll_f16[even + bx + 6u]);
            v[13] = pyramid[odd + bx + 6u];
            v[14] = f32(ll_f16[even + bx + 7u]);
            v[15] = pyramid[odd + bx + 7u];
        } else {
            v[0] = f32(ll_f16[even + u32(mirror(x0 + 0, out_width) >> 1u)]);
            v[1] = pyramid[odd + u32(mirror(x0 + 1, out_width) >> 1u)];
            v[2] = f32(ll_f16[even + u32(mirror(x0 + 2, out_width) >> 1u)]);
            v[3] = pyramid[odd + u32(mirror(x0 + 3, out_width) >> 1u)];
            v[4] = f32(ll_f16[even + u32(mirror(x0 + 4, out_width) >> 1u)]);
            v[5] = pyramid[odd + u32(mirror(x0 + 5, out_width) >> 1u)];
            v[6] = f32(ll_f16[even + u32(mirror(x0 + 6, out_width) >> 1u)]);
            v[7] = pyramid[odd + u32(mirror(x0 + 7, out_width) >> 1u)];
            v[8] = f32(ll_f16[even + u32(mirror(x0 + 8, out_width) >> 1u)]);
            v[9] = pyramid[odd + u32(mirror(x0 + 9, out_width) >> 1u)];
            v[10] = f32(ll_f16[even + u32(mirror(x0 + 10, out_width) >> 1u)]);
            v[11] = pyramid[odd + u32(mirror(x0 + 11, out_width) >> 1u)];
            v[12] = f32(ll_f16[even + u32(mirror(x0 + 12, out_width) >> 1u)]);
            v[13] = pyramid[odd + u32(mirror(x0 + 13, out_width) >> 1u)];
            v[14] = f32(ll_f16[even + u32(mirror(x0 + 14, out_width) >> 1u)]);
            v[15] = pyramid[odd + u32(mirror(x0 + 15, out_width) >> 1u)];
        }
    } else {
        let even = comp.band_offset[row_bands] + band_row;
        if (x0 >= 0 && x0 + 15 < out_width) {
            let bx = u32(x0) >> 1u;
            v[0] = pyramid[even + bx + 0u];
            v[1] = pyramid[odd + bx + 0u];
            v[2] = pyramid[even + bx + 1u];
            v[3] = pyramid[odd + bx + 1u];
            v[4] = pyramid[even + bx + 2u];
            v[5] = pyramid[odd + bx + 2u];
            v[6] = pyramid[even + bx + 3u];
            v[7] = pyramid[odd + bx + 3u];
            v[8] = pyramid[even + bx + 4u];
            v[9] = pyramid[odd + bx + 4u];
            v[10] = pyramid[even + bx + 5u];
            v[11] = pyramid[odd + bx + 5u];
            v[12] = pyramid[even + bx + 6u];
            v[13] = pyramid[odd + bx + 6u];
            v[14] = pyramid[even + bx + 7u];
            v[15] = pyramid[odd + bx + 7u];
        } else {
            v[0] = pyramid[even + u32(mirror(x0 + 0, out_width) >> 1u)];
            v[1] = pyramid[odd + u32(mirror(x0 + 1, out_width) >> 1u)];
            v[2] = pyramid[even + u32(mirror(x0 + 2, out_width) >> 1u)];
            v[3] = pyramid[odd + u32(mirror(x0 + 3, out_width) >> 1u)];
            v[4] = pyramid[even + u32(mirror(x0 + 4, out_width) >> 1u)];
            v[5] = pyramid[odd + u32(mirror(x0 + 5, out_width) >> 1u)];
            v[6] = pyramid[even + u32(mirror(x0 + 6, out_width) >> 1u)];
            v[7] = pyramid[odd + u32(mirror(x0 + 7, out_width) >> 1u)];
            v[8] = pyramid[even + u32(mirror(x0 + 8, out_width) >> 1u)];
            v[9] = pyramid[odd + u32(mirror(x0 + 9, out_width) >> 1u)];
            v[10] = pyramid[even + u32(mirror(x0 + 10, out_width) >> 1u)];
            v[11] = pyramid[odd + u32(mirror(x0 + 11, out_width) >> 1u)];
            v[12] = pyramid[even + u32(mirror(x0 + 12, out_width) >> 1u)];
            v[13] = pyramid[odd + u32(mirror(x0 + 13, out_width) >> 1u)];
            v[14] = pyramid[even + u32(mirror(x0 + 14, out_width) >> 1u)];
            v[15] = pyramid[odd + u32(mirror(x0 + 15, out_width) >> 1u)];
        }
    }
    lift(&v);
    let out = row * TILE_W + 8u * segment;
    horizontal[out + 0u] = f16(v[4]);
    horizontal[out + 1u] = f16(v[5]);
    horizontal[out + 2u] = f16(v[6]);
    horizontal[out + 3u] = f16(v[7]);
    horizontal[out + 4u] = f16(v[8]);
    horizontal[out + 5u] = f16(v[9]);
    horizontal[out + 6u] = f16(v[10]);
    horizontal[out + 7u] = f16(v[11]);
}

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) tid: u32, @builtin(workgroup_id) wg: vec3<u32>) {
    let comp = params.comps[wg.z];
    let out_width = i32(params.image_width);
    let out_height = i32(params.image_height);
    let tile_x = i32(wg.x * TILE_W);
    let tile_y = i32(wg.y * TILE_H);

    // Horizontal pass: 40 rows x 8 segments = 320 segments across 256 threads.
    let h_seg = tid % 8u;
    let row0 = 2u * ((tid % 128u) / 8u) + (tid / 128u);
    horiz_segment(comp, tile_x, tile_y, out_width, out_height, row0, h_seg);
    if (tid < 64u) {
        let row1 = 32u + 2u * ((tid % 32u) / 8u) + (tid / 32u);
        horiz_segment(comp, tile_x, tile_y, out_width, out_height, row1, h_seg);
    }
    workgroupBarrier();

    // Vertical pass: 64 columns x 4 vertical segments = 256 invocations (100% active).
    let is_final = comp.is_plane != 0u;
    let column = tid % TILE_W;
    let segment = tid / TILE_W;
    var v: array<f32, 16>;
    let base = 8u * segment * TILE_W + column;
    v[0] = f32(horizontal[base + 0u * TILE_W]);
    v[1] = f32(horizontal[base + 1u * TILE_W]);
    v[2] = f32(horizontal[base + 2u * TILE_W]);
    v[3] = f32(horizontal[base + 3u * TILE_W]);
    v[4] = f32(horizontal[base + 4u * TILE_W]);
    v[5] = f32(horizontal[base + 5u * TILE_W]);
    v[6] = f32(horizontal[base + 6u * TILE_W]);
    v[7] = f32(horizontal[base + 7u * TILE_W]);
    v[8] = f32(horizontal[base + 8u * TILE_W]);
    v[9] = f32(horizontal[base + 9u * TILE_W]);
    v[10] = f32(horizontal[base + 10u * TILE_W]);
    v[11] = f32(horizontal[base + 11u * TILE_W]);
    v[12] = f32(horizontal[base + 12u * TILE_W]);
    v[13] = f32(horizontal[base + 13u * TILE_W]);
    v[14] = f32(horizontal[base + 14u * TILE_W]);
    v[15] = f32(horizontal[base + 15u * TILE_W]);
    lift(&v);

    let x = tile_x + i32(column);
    let y = tile_y + i32(8u * segment);
    if (is_final) {
        store_vertical_8(
            comp,
            wg.z,
            u32(x),
            u32(y),
            base,
            v[4],
            v[5],
            v[6],
            v[7],
            v[8],
            v[9],
            v[10],
            v[11],
        );
    } else {
        if (x < out_width) {
            if (y + 7 < out_height) {
                let row_base = comp.pad1 + u32(y) * comp.image_stride + u32(x);
                let stride = comp.image_stride;
                ll_f16[row_base + 0u * stride] = f16(v[4]);
                ll_f16[row_base + 1u * stride] = f16(v[5]);
                ll_f16[row_base + 2u * stride] = f16(v[6]);
                ll_f16[row_base + 3u * stride] = f16(v[7]);
                ll_f16[row_base + 4u * stride] = f16(v[8]);
                ll_f16[row_base + 5u * stride] = f16(v[9]);
                ll_f16[row_base + 6u * stride] = f16(v[10]);
                ll_f16[row_base + 7u * stride] = f16(v[11]);
            } else {
                store_ll_f16(comp, x, y + 0, v[4]);
                store_ll_f16(comp, x, y + 1, v[5]);
                store_ll_f16(comp, x, y + 2, v[6]);
                store_ll_f16(comp, x, y + 3, v[7]);
                store_ll_f16(comp, x, y + 4, v[8]);
                store_ll_f16(comp, x, y + 5, v[9]);
                store_ll_f16(comp, x, y + 6, v[10]);
                store_ll_f16(comp, x, y + 7, v[11]);
            }
        }
    }

    finish_plane(comp, wg.z, u32(tile_x), u32(tile_y), tid, is_final);
}
