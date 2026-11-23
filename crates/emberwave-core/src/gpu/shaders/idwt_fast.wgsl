// Inverse CDF 9/7 for 16-bit plane stores.
//
// Behind this file, idwt_planes_r16.wgsl writes the final planes. The level
// that feeds a plane packs that lowpass two rows to a word with pack2x16float.
// Coarser lowpass bands stay in the f32 pyramid: a second f16 hop costs more
// accuracy than the bandwidth it saves.
//
// Workgroups are not a grid. tile_ids[wg.x] is x | (y << 16) | (component << 28),
// in tile units. Dispatch only the tiles whose coefficients changed.
//
// Workgroup memory is not zeroed (decision D25). `horizontal` is filled before
// the vertical lift. A barrier between the two passes is enough: staging each
// band row in shared memory and waiting twice per row was slower on Tegra.

struct CompParams {
    band_offset: vec4<u32>,
    image_offset: u32,
    image_stride: u32,
    plane_width: u32,
    plane_height: u32,
    is_plane: u32,
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
@group(0) @binding(6) var<storage, read_write> ll_packed: array<u32>;
@group(0) @binding(9) var<storage, read> tile_ids: array<u32>;

const ALPHA: f32 = -1.586134342059924;
const BETA: f32 = -0.052980118572961;
const GAMMA: f32 = 0.882911075530934;
const DELTA: f32 = 0.443506852043971;
const K: f32 = 1.230174104914001;
const INV_K: f32 = 0.8128930661159609;

const TILE: u32 = 64u;
const TILE_H: u32 = 32u;
const APRON: i32 = 4;
const DC_SHIFT: f32 = 0.5;
const ABSENT: u32 = 0xFFFFFFFFu;

var<workgroup> horizontal: array<f32, 2560>;

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

// mode 0: both samples from the pyramid. mode 1: even rows' lowpass from the
// packed buffer. `band_x` and `y` are already inside the image.
fn sample_at(comp: CompParams, band_width: u32, y: i32, band_x: u32, mode: u32) -> vec2<f32> {
    let row_bands = u32(y & 1) << 1u;
    let by = u32(y >> 1u);
    let band_row = by * band_width;
    let odd = pyramid[comp.band_offset[row_bands | 1u] + band_row + band_x];
    var even: f32;
    if (row_bands == 0u && mode == 1u) {
        let word = ll_packed[(comp.pad0 >> 1u) + (by >> 1u) * band_width + band_x];
        let pair = unpack2x16float(word);
        even = select(pair.y, pair.x, (by & 1u) == 0u);
    } else {
        even = pyramid[comp.band_offset[row_bands] + band_row + band_x];
    }
    return vec2<f32>(even, odd);
}

// One interleaved sample. The band follows the output x parity; mirroring
// keeps that parity, so the reflected index still selects the same band.
fn coeff_at(
    comp: CompParams,
    band_width: u32,
    y: i32,
    out_x: i32,
    out_width: i32,
    mode: u32,
) -> f32 {
    let mx = mirror(out_x, out_width);
    let pair = sample_at(comp, band_width, y, u32(mx >> 1u), mode);
    return select(pair.x, pair.y, (mx & 1) == 1);
}

fn store_horizontal(row: u32, segment: u32, v: ptr<function, array<f32, 16>>) {
    let out = row * TILE + 8u * segment;
    horizontal[out + 0u] = (*v)[4];
    horizontal[out + 1u] = (*v)[5];
    horizontal[out + 2u] = (*v)[6];
    horizontal[out + 3u] = (*v)[7];
    horizontal[out + 4u] = (*v)[8];
    horizontal[out + 5u] = (*v)[9];
    horizontal[out + 6u] = (*v)[10];
    horizontal[out + 7u] = (*v)[11];
}

fn horiz_segment(
    comp: CompParams,
    band_width: u32,
    tile_x: i32,
    tile_y: i32,
    out_width: i32,
    out_height: i32,
    row: u32,
    segment: u32,
    mode: u32,
) {
    let raw_y = tile_y + i32(row) - APRON;
    let y = select(mirror(raw_y, out_height), raw_y, raw_y >= 0 && raw_y < out_height);
    let x0 = tile_x + i32(8u * segment) - APRON;
    var v: array<f32, 16>;
    if (x0 >= 0 && x0 + 15 < out_width) {
        let bx = u32(x0) >> 1u;
        let p0 = sample_at(comp, band_width, y, bx + 0u, mode);
        let p1 = sample_at(comp, band_width, y, bx + 1u, mode);
        let p2 = sample_at(comp, band_width, y, bx + 2u, mode);
        let p3 = sample_at(comp, band_width, y, bx + 3u, mode);
        let p4 = sample_at(comp, band_width, y, bx + 4u, mode);
        let p5 = sample_at(comp, band_width, y, bx + 5u, mode);
        let p6 = sample_at(comp, band_width, y, bx + 6u, mode);
        let p7 = sample_at(comp, band_width, y, bx + 7u, mode);
        v[0] = p0.x;
        v[1] = p0.y;
        v[2] = p1.x;
        v[3] = p1.y;
        v[4] = p2.x;
        v[5] = p2.y;
        v[6] = p3.x;
        v[7] = p3.y;
        v[8] = p4.x;
        v[9] = p4.y;
        v[10] = p5.x;
        v[11] = p5.y;
        v[12] = p6.x;
        v[13] = p6.y;
        v[14] = p7.x;
        v[15] = p7.y;
    } else {
        v[0] = coeff_at(comp, band_width, y, x0 + 0, out_width, mode);
        v[1] = coeff_at(comp, band_width, y, x0 + 1, out_width, mode);
        v[2] = coeff_at(comp, band_width, y, x0 + 2, out_width, mode);
        v[3] = coeff_at(comp, band_width, y, x0 + 3, out_width, mode);
        v[4] = coeff_at(comp, band_width, y, x0 + 4, out_width, mode);
        v[5] = coeff_at(comp, band_width, y, x0 + 5, out_width, mode);
        v[6] = coeff_at(comp, band_width, y, x0 + 6, out_width, mode);
        v[7] = coeff_at(comp, band_width, y, x0 + 7, out_width, mode);
        v[8] = coeff_at(comp, band_width, y, x0 + 8, out_width, mode);
        v[9] = coeff_at(comp, band_width, y, x0 + 9, out_width, mode);
        v[10] = coeff_at(comp, band_width, y, x0 + 10, out_width, mode);
        v[11] = coeff_at(comp, band_width, y, x0 + 11, out_width, mode);
        v[12] = coeff_at(comp, band_width, y, x0 + 12, out_width, mode);
        v[13] = coeff_at(comp, band_width, y, x0 + 13, out_width, mode);
        v[14] = coeff_at(comp, band_width, y, x0 + 14, out_width, mode);
        v[15] = coeff_at(comp, band_width, y, x0 + 15, out_width, mode);
    }
    lift(&v);
    store_horizontal(row, segment, &v);
}

fn store_packed_8(
    comp: CompParams,
    x: i32,
    y: i32,
    s0: f32,
    s1: f32,
    s2: f32,
    s3: f32,
    s4: f32,
    s5: f32,
    s6: f32,
    s7: f32,
) {
    if (x < 0 || y < 0 || x >= i32(params.image_width) || comp.pad1 == ABSENT) {
        return;
    }
    let base = comp.pad1 >> 1u;
    let stride = params.image_width;
    let xu = u32(x);
    let h = i32(params.image_height);
    if (y + 1 < h) {
        ll_packed[base + (u32(y) >> 1u) * stride + xu] = pack2x16float(vec2<f32>(s0, s1));
    }
    if (y + 3 < h) {
        ll_packed[base + (u32(y + 2) >> 1u) * stride + xu] = pack2x16float(vec2<f32>(s2, s3));
    }
    if (y + 5 < h) {
        ll_packed[base + (u32(y + 4) >> 1u) * stride + xu] = pack2x16float(vec2<f32>(s4, s5));
    }
    if (y + 7 < h) {
        ll_packed[base + (u32(y + 6) >> 1u) * stride + xu] = pack2x16float(vec2<f32>(s6, s7));
    }
}

fn run_scalar_rows(
    comp: CompParams,
    band_width: u32,
    tile_x: i32,
    tile_y: i32,
    out_width: i32,
    out_height: i32,
    tid: u32,
    mode: u32,
) {
    let h_seg = tid % 8u;
    let row0 = 2u * ((tid % 128u) / 8u) + (tid / 128u);
    horiz_segment(comp, band_width, tile_x, tile_y, out_width, out_height, row0, h_seg, mode);
    if (tid < 64u) {
        let row1 = 32u + 2u * ((tid % 32u) / 8u) + (tid / 32u);
        horiz_segment(comp, band_width, tile_x, tile_y, out_width, out_height, row1, h_seg, mode);
    }
}

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) tid: u32, @builtin(workgroup_id) wg: vec3<u32>) {
    let packed_id = tile_ids[wg.x];
    let tile_ix = packed_id & 0xFFFFu;
    let tile_iy = (packed_id >> 16u) & 0xFFFu;
    let z = packed_id >> 28u;
    let comp = params.comps[z];
    let tile_x = i32(tile_ix * TILE);
    let tile_y = i32(tile_iy * TILE_H);
    let out_width = i32(params.image_width);
    let out_height = i32(params.image_height);
    init_levels(tid);

    // The plane reads the lowpass the previous level packed. Coarser levels
    // keep the f32 pyramid, so only one hop is rounded to f16.
    var mode: u32 = 0u;
    if (comp.is_plane != 0u && comp.pad0 != ABSENT) {
        mode = 1u;
    }

    run_scalar_rows(comp, params.band_width, tile_x, tile_y, out_width, out_height, tid, mode);
    workgroupBarrier();

    let is_final = comp.is_plane != 0u;
    let column = tid % TILE;
    let segment = tid / TILE;
    var v: array<f32, 16>;
    let base = 8u * segment * TILE + column;
    v[0] = horizontal[base + 0u * TILE];
    v[1] = horizontal[base + 1u * TILE];
    v[2] = horizontal[base + 2u * TILE];
    v[3] = horizontal[base + 3u * TILE];
    v[4] = horizontal[base + 4u * TILE];
    v[5] = horizontal[base + 5u * TILE];
    v[6] = horizontal[base + 6u * TILE];
    v[7] = horizontal[base + 7u * TILE];
    v[8] = horizontal[base + 8u * TILE];
    v[9] = horizontal[base + 9u * TILE];
    v[10] = horizontal[base + 10u * TILE];
    v[11] = horizontal[base + 11u * TILE];
    v[12] = horizontal[base + 12u * TILE];
    v[13] = horizontal[base + 13u * TILE];
    v[14] = horizontal[base + 14u * TILE];
    v[15] = horizontal[base + 15u * TILE];
    lift(&v);

    let x = tile_x + i32(column);
    let y = tile_y + i32(8u * segment);
    if (is_final) {
        store_vertical_8(comp, comp.pad2, u32(x), u32(y), base, v[4], v[5], v[6], v[7], v[8], v[9], v[10], v[11]);
    } else if (comp.pad1 != ABSENT) {
        store_packed_8(comp, x, y, v[4], v[5], v[6], v[7], v[8], v[9], v[10], v[11]);
    } else if (x >= 0 && x < out_width) {
        let stride = comp.image_stride;
        let row_base = comp.image_offset + u32(y) * stride + u32(x);
        if (y >= 0 && y + 7 < out_height) {
            pyramid[row_base + 0u * stride] = v[4];
            pyramid[row_base + 1u * stride] = v[5];
            pyramid[row_base + 2u * stride] = v[6];
            pyramid[row_base + 3u * stride] = v[7];
            pyramid[row_base + 4u * stride] = v[8];
            pyramid[row_base + 5u * stride] = v[9];
            pyramid[row_base + 6u * stride] = v[10];
            pyramid[row_base + 7u * stride] = v[11];
        } else {
            if (y + 0 >= 0 && y + 0 < out_height) { pyramid[comp.image_offset + u32(y + 0) * stride + u32(x)] = v[4]; }
            if (y + 1 >= 0 && y + 1 < out_height) { pyramid[comp.image_offset + u32(y + 1) * stride + u32(x)] = v[5]; }
            if (y + 2 >= 0 && y + 2 < out_height) { pyramid[comp.image_offset + u32(y + 2) * stride + u32(x)] = v[6]; }
            if (y + 3 >= 0 && y + 3 < out_height) { pyramid[comp.image_offset + u32(y + 3) * stride + u32(x)] = v[7]; }
            if (y + 4 >= 0 && y + 4 < out_height) { pyramid[comp.image_offset + u32(y + 4) * stride + u32(x)] = v[8]; }
            if (y + 5 >= 0 && y + 5 < out_height) { pyramid[comp.image_offset + u32(y + 5) * stride + u32(x)] = v[9]; }
            if (y + 6 >= 0 && y + 6 < out_height) { pyramid[comp.image_offset + u32(y + 6) * stride + u32(x)] = v[10]; }
            if (y + 7 >= 0 && y + 7 < out_height) { pyramid[comp.image_offset + u32(y + 7) * stride + u32(x)] = v[11]; }
        }
    }
    finish_plane(comp, comp.pad2, u32(tile_x), u32(tile_y), tid, is_final);
}
