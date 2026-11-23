// Stage: inverse CDF 9/7 for the fast 16-bit path.
//
// Bands are 2x2 quads in storage buffers. A texel (tx, ty) holds samples
// (2*tx, 2*ty), (2*tx+1, 2*ty), (2*tx, 2*ty+1), (2*tx+1, 2*ty+1). fp16 quads
// are two u32 words, even row then odd row, each packed with pack2x16float.
// f32 quads are four floats in that channel order. The coarsest level, and
// any level with an odd side, stays scalar in the pyramid.
//
// Arithmetic is f32. pad0 is the input format of all four bands: 0 scalar,
// 1 fp16 quads, 2 f32 quads. is_plane is the output: 1 an r16 plane, 2 an
// fp16 quad image, 3 an f32 quad image, 0 a scalar pyramid image. pad1 is
// the quad-buffer offset of a quad image. pad2 is the component.
//
// Tile ids are x | (y << 16) | (mask << 25) | (component << 28) | (mode << 30).
// Mode 0 loads every band. Mode 1 loads only LL. Mode 2 loads LL plus the
// high bands named by the mask: bit 0 HL, bit 1 LH, bit 2 HH.
//
// Workgroup memory is not zeroed (decision D25). `horizontal` is filled
// before the vertical lift. The neighbour exchange lives in the entry point,
// which supplies the odd column's samples to `publish`.

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
@group(0) @binding(6) var<storage, read_write> quads_f16: array<u32>;
@group(0) @binding(7) var<storage, read_write> quads_f32: array<f32>;
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

fn band_needed(band: u32, tile_mode: u32, mask: u32) -> bool {
    if (band == 0u || tile_mode == 0u) {
        return true;
    }
    if (tile_mode == 1u) {
        return false;
    }
    return ((mask >> (band - 1u)) & 1u) != 0u;
}

// Four band samples at an even bx. Quad formats pack two of them per texel.
fn band4(comp: CompParams, band: u32, by: u32, bx: u32) -> vec4<f32> {
    if (comp.pad0 == 0u) {
        let base = comp.band_offset[band] + by * params.band_width + bx;
        return vec4<f32>(
            pyramid[base],
            pyramid[base + 1u],
            pyramid[base + 2u],
            pyramid[base + 3u],
        );
    }
    let qstride = params.band_width >> 1u;
    let qx = bx >> 1u;
    let qy = by >> 1u;
    if (comp.pad0 == 1u) {
        let word = comp.band_offset[band] + (qy * qstride + qx) * 2u + (by & 1u);
        let w0 = unpack2x16float(quads_f16[word]);
        let w1 = unpack2x16float(quads_f16[word + 2u]);
        return vec4<f32>(w0.x, w0.y, w1.x, w1.y);
    }
    let base = comp.band_offset[band] + (qy * qstride + qx) * 4u + ((by & 1u) << 1u);
    return vec4<f32>(
        quads_f32[base],
        quads_f32[base + 1u],
        quads_f32[base + 4u],
        quads_f32[base + 5u],
    );
}

fn sample_one(comp: CompParams, band: u32, by: u32, bx: u32) -> f32 {
    if (comp.pad0 == 0u) {
        return pyramid[comp.band_offset[band] + by * params.band_width + bx];
    }
    let qstride = params.band_width >> 1u;
    let qx = bx >> 1u;
    let qy = by >> 1u;
    if (comp.pad0 == 1u) {
        let word = quads_f16[comp.band_offset[band] + (qy * qstride + qx) * 2u + (by & 1u)];
        let pair = unpack2x16float(word);
        return select(pair.y, pair.x, (bx & 1u) == 0u);
    }
    let base = comp.band_offset[band] + (qy * qstride + qx) * 4u;
    let channel = ((by & 1u) << 1u) | (bx & 1u);
    if (channel == 0u) {
        return quads_f32[base];
    }
    if (channel == 1u) {
        return quads_f32[base + 1u];
    }
    if (channel == 2u) {
        return quads_f32[base + 2u];
    }
    return quads_f32[base + 3u];
}

fn coeff_at(
    comp: CompParams,
    y: i32,
    out_x: i32,
    out_width: i32,
    tile_mode: u32,
    mask: u32,
) -> f32 {
    let mx = mirror(out_x, out_width);
    let band = (u32(y & 1) << 1u) | u32(mx & 1);
    if (!band_needed(band, tile_mode, mask)) {
        return 0.0;
    }
    return sample_one(comp, band, u32(y >> 1), u32(mx >> 1));
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

fn write_even8(v: ptr<function, array<f32, 16>>, a: vec4<f32>, b: vec4<f32>) {
    (*v)[0] = a.x;
    (*v)[2] = a.y;
    (*v)[4] = a.z;
    (*v)[6] = a.w;
    (*v)[8] = b.x;
    (*v)[10] = b.y;
    (*v)[12] = b.z;
    (*v)[14] = b.w;
}

fn write_odd8(v: ptr<function, array<f32, 16>>, a: vec4<f32>, b: vec4<f32>) {
    (*v)[1] = a.x;
    (*v)[3] = a.y;
    (*v)[5] = a.z;
    (*v)[7] = a.w;
    (*v)[9] = b.x;
    (*v)[11] = b.y;
    (*v)[13] = b.z;
    (*v)[15] = b.w;
}

fn zero_even8(v: ptr<function, array<f32, 16>>) {
    (*v)[0] = 0.0;
    (*v)[2] = 0.0;
    (*v)[4] = 0.0;
    (*v)[6] = 0.0;
    (*v)[8] = 0.0;
    (*v)[10] = 0.0;
    (*v)[12] = 0.0;
    (*v)[14] = 0.0;
}

fn zero_odd8(v: ptr<function, array<f32, 16>>) {
    (*v)[1] = 0.0;
    (*v)[3] = 0.0;
    (*v)[5] = 0.0;
    (*v)[7] = 0.0;
    (*v)[9] = 0.0;
    (*v)[11] = 0.0;
    (*v)[13] = 0.0;
    (*v)[15] = 0.0;
}

fn fill_tail8(
    comp: CompParams,
    y: i32,
    x: i32,
    out_width: i32,
    tile_mode: u32,
    mask: u32,
    v: ptr<function, array<f32, 16>>,
) {
    let row_bands = u32(y & 1) << 1u;
    let by = u32(y >> 1u);
    if (x >= 0 && x + 7 < out_width) {
        let bx = u32(x) >> 1u;
        if (band_needed(row_bands, tile_mode, mask)) {
            let a = band4(comp, row_bands, by, bx);
            (*v)[8] = a.x;
            (*v)[10] = a.y;
            (*v)[12] = a.z;
            (*v)[14] = a.w;
        } else {
            (*v)[8] = 0.0;
            (*v)[10] = 0.0;
            (*v)[12] = 0.0;
            (*v)[14] = 0.0;
        }
        if (band_needed(row_bands | 1u, tile_mode, mask)) {
            let a = band4(comp, row_bands | 1u, by, bx);
            (*v)[9] = a.x;
            (*v)[11] = a.y;
            (*v)[13] = a.z;
            (*v)[15] = a.w;
        } else {
            (*v)[9] = 0.0;
            (*v)[11] = 0.0;
            (*v)[13] = 0.0;
            (*v)[15] = 0.0;
        }
    } else {
        (*v)[8] = coeff_at(comp, y, x + 0, out_width, tile_mode, mask);
        (*v)[9] = coeff_at(comp, y, x + 1, out_width, tile_mode, mask);
        (*v)[10] = coeff_at(comp, y, x + 2, out_width, tile_mode, mask);
        (*v)[11] = coeff_at(comp, y, x + 3, out_width, tile_mode, mask);
        (*v)[12] = coeff_at(comp, y, x + 4, out_width, tile_mode, mask);
        (*v)[13] = coeff_at(comp, y, x + 5, out_width, tile_mode, mask);
        (*v)[14] = coeff_at(comp, y, x + 6, out_width, tile_mode, mask);
        (*v)[15] = coeff_at(comp, y, x + 7, out_width, tile_mode, mask);
    }
}

fn fill_16(
    comp: CompParams,
    y: i32,
    x0: i32,
    out_width: i32,
    tile_mode: u32,
    mask: u32,
    v: ptr<function, array<f32, 16>>,
) {
    let row_bands = u32(y & 1) << 1u;
    let by = u32(y >> 1u);
    if (x0 >= 0 && x0 + 15 < out_width) {
        let bx = u32(x0) >> 1u;
        if (band_needed(row_bands, tile_mode, mask)) {
            write_even8(v, band4(comp, row_bands, by, bx), band4(comp, row_bands, by, bx + 4u));
        } else {
            zero_even8(v);
        }
        if (band_needed(row_bands | 1u, tile_mode, mask)) {
            write_odd8(v, band4(comp, row_bands | 1u, by, bx), band4(comp, row_bands | 1u, by, bx + 4u));
        } else {
            zero_odd8(v);
        }
    } else {
        (*v)[0] = coeff_at(comp, y, x0 + 0, out_width, tile_mode, mask);
        (*v)[1] = coeff_at(comp, y, x0 + 1, out_width, tile_mode, mask);
        (*v)[2] = coeff_at(comp, y, x0 + 2, out_width, tile_mode, mask);
        (*v)[3] = coeff_at(comp, y, x0 + 3, out_width, tile_mode, mask);
        (*v)[4] = coeff_at(comp, y, x0 + 4, out_width, tile_mode, mask);
        (*v)[5] = coeff_at(comp, y, x0 + 5, out_width, tile_mode, mask);
        (*v)[6] = coeff_at(comp, y, x0 + 6, out_width, tile_mode, mask);
        (*v)[7] = coeff_at(comp, y, x0 + 7, out_width, tile_mode, mask);
        (*v)[8] = coeff_at(comp, y, x0 + 8, out_width, tile_mode, mask);
        (*v)[9] = coeff_at(comp, y, x0 + 9, out_width, tile_mode, mask);
        (*v)[10] = coeff_at(comp, y, x0 + 10, out_width, tile_mode, mask);
        (*v)[11] = coeff_at(comp, y, x0 + 11, out_width, tile_mode, mask);
        (*v)[12] = coeff_at(comp, y, x0 + 12, out_width, tile_mode, mask);
        (*v)[13] = coeff_at(comp, y, x0 + 13, out_width, tile_mode, mask);
        (*v)[14] = coeff_at(comp, y, x0 + 14, out_width, tile_mode, mask);
        (*v)[15] = coeff_at(comp, y, x0 + 15, out_width, tile_mode, mask);
    }
}

fn horiz_pair(
    comp: CompParams,
    tile_x: i32,
    tile_y: i32,
    out_width: i32,
    out_height: i32,
    row: u32,
    pair: u32,
    tile_mode: u32,
    mask: u32,
) {
    let raw_y = tile_y + i32(row) - APRON;
    let y = select(mirror(raw_y, out_height), raw_y, raw_y >= 0 && raw_y < out_height);
    let x0 = tile_x + i32(16u * pair) - APRON;
    var v: array<f32, 16>;
    fill_16(comp, y, x0, out_width, tile_mode, mask, &v);
    let s0 = v[8];
    let s1 = v[9];
    let s2 = v[10];
    let s3 = v[11];
    let s4 = v[12];
    let s5 = v[13];
    let s6 = v[14];
    let s7 = v[15];
    lift(&v);
    store_horizontal(row, 2u * pair, &v);
    v[0] = s0;
    v[1] = s1;
    v[2] = s2;
    v[3] = s3;
    v[4] = s4;
    v[5] = s5;
    v[6] = s6;
    v[7] = s7;
    fill_tail8(comp, y, x0 + 16, out_width, tile_mode, mask, &v);
    lift(&v);
    store_horizontal(row, 2u * pair + 1u, &v);
}

fn run_rows(
    comp: CompParams,
    tile_x: i32,
    tile_y: i32,
    out_width: i32,
    out_height: i32,
    tid: u32,
    tile_mode: u32,
    mask: u32,
) {
    if (tid < 160u) {
        horiz_pair(
            comp,
            tile_x,
            tile_y,
            out_width,
            out_height,
            tid / 4u,
            tid % 4u,
            tile_mode,
            mask,
        );
    }
}

fn lift_column(column: u32, segment: u32, v: ptr<function, array<f32, 16>>) {
    let base = 8u * segment * TILE + column;
    (*v)[0] = horizontal[base + 0u * TILE];
    (*v)[1] = horizontal[base + 1u * TILE];
    (*v)[2] = horizontal[base + 2u * TILE];
    (*v)[3] = horizontal[base + 3u * TILE];
    (*v)[4] = horizontal[base + 4u * TILE];
    (*v)[5] = horizontal[base + 5u * TILE];
    (*v)[6] = horizontal[base + 6u * TILE];
    (*v)[7] = horizontal[base + 7u * TILE];
    (*v)[8] = horizontal[base + 8u * TILE];
    (*v)[9] = horizontal[base + 9u * TILE];
    (*v)[10] = horizontal[base + 10u * TILE];
    (*v)[11] = horizontal[base + 11u * TILE];
    (*v)[12] = horizontal[base + 12u * TILE];
    (*v)[13] = horizontal[base + 13u * TILE];
    (*v)[14] = horizontal[base + 14u * TILE];
    (*v)[15] = horizontal[base + 15u * TILE];
    lift(v);
}

fn store_quad_pair(
    comp: CompParams,
    tx: u32,
    ty: u32,
    qstride: u32,
    s0: f32,
    n0: f32,
    s1: f32,
    n1: f32,
) {
    if (comp.is_plane == 2u) {
        let word = comp.pad1 + (ty * qstride + tx) * 2u;
        quads_f16[word] = pack2x16float(vec2<f32>(s0, n0));
        quads_f16[word + 1u] = pack2x16float(vec2<f32>(s1, n1));
        return;
    }
    let base = comp.pad1 + (ty * qstride + tx) * 4u;
    quads_f32[base] = s0;
    quads_f32[base + 1u] = n0;
    quads_f32[base + 2u] = s1;
    quads_f32[base + 3u] = n1;
}

fn store_quads_8(
    comp: CompParams,
    x: i32,
    y: i32,
    v: ptr<function, array<f32, 16>>,
    nb: ptr<function, array<f32, 8>>,
) {
    if (x < 0 || y < 0 || u32(x) >= params.image_width) {
        return;
    }
    let tx = u32(x) >> 1u;
    let qstride = params.image_width >> 1u;
    let h = i32(params.image_height);
    let ty = u32(y) >> 1u;
    if (y + 1 < h) {
        store_quad_pair(comp, tx, ty, qstride, (*v)[4], (*nb)[0], (*v)[5], (*nb)[1]);
    }
    if (y + 3 < h) {
        store_quad_pair(comp, tx, ty + 1u, qstride, (*v)[6], (*nb)[2], (*v)[7], (*nb)[3]);
    }
    if (y + 5 < h) {
        store_quad_pair(comp, tx, ty + 2u, qstride, (*v)[8], (*nb)[4], (*v)[9], (*nb)[5]);
    }
    if (y + 7 < h) {
        store_quad_pair(comp, tx, ty + 3u, qstride, (*v)[10], (*nb)[6], (*v)[11], (*nb)[7]);
    }
}

fn store_scalar_8(comp: CompParams, x: i32, y: i32, out_width: i32, out_height: i32, v: ptr<function, array<f32, 16>>) {
    if (x < 0 || x >= out_width) {
        return;
    }
    let stride = comp.image_stride;
    let xu = u32(x);
    if (y >= 0 && y + 7 < out_height) {
        let row_base = comp.image_offset + u32(y) * stride + xu;
        pyramid[row_base + 0u * stride] = (*v)[4];
        pyramid[row_base + 1u * stride] = (*v)[5];
        pyramid[row_base + 2u * stride] = (*v)[6];
        pyramid[row_base + 3u * stride] = (*v)[7];
        pyramid[row_base + 4u * stride] = (*v)[8];
        pyramid[row_base + 5u * stride] = (*v)[9];
        pyramid[row_base + 6u * stride] = (*v)[10];
        pyramid[row_base + 7u * stride] = (*v)[11];
        return;
    }
    if (y + 0 >= 0 && y + 0 < out_height) { pyramid[comp.image_offset + u32(y + 0) * stride + xu] = (*v)[4]; }
    if (y + 1 >= 0 && y + 1 < out_height) { pyramid[comp.image_offset + u32(y + 1) * stride + xu] = (*v)[5]; }
    if (y + 2 >= 0 && y + 2 < out_height) { pyramid[comp.image_offset + u32(y + 2) * stride + xu] = (*v)[6]; }
    if (y + 3 >= 0 && y + 3 < out_height) { pyramid[comp.image_offset + u32(y + 3) * stride + xu] = (*v)[7]; }
    if (y + 4 >= 0 && y + 4 < out_height) { pyramid[comp.image_offset + u32(y + 4) * stride + xu] = (*v)[8]; }
    if (y + 5 >= 0 && y + 5 < out_height) { pyramid[comp.image_offset + u32(y + 5) * stride + xu] = (*v)[9]; }
    if (y + 6 >= 0 && y + 6 < out_height) { pyramid[comp.image_offset + u32(y + 6) * stride + xu] = (*v)[10]; }
    if (y + 7 >= 0 && y + 7 < out_height) { pyramid[comp.image_offset + u32(y + 7) * stride + xu] = (*v)[11]; }
}

// `nb` is the neighbouring column, used when an even column writes a quad.
fn publish(
    comp: CompParams,
    column: u32,
    x: i32,
    y: i32,
    out_width: i32,
    out_height: i32,
    v: ptr<function, array<f32, 16>>,
    nb: ptr<function, array<f32, 8>>,
) {
    if (comp.is_plane == 1u) {
        if (x >= 0 && y >= 0) {
            store_vertical_8(comp, comp.pad2, u32(x), u32(y), 0u, (*v)[4], (*v)[5], (*v)[6], (*v)[7], (*v)[8], (*v)[9], (*v)[10], (*v)[11]);
        }
        return;
    }
    if (comp.is_plane == 2u || comp.is_plane == 3u) {
        if ((column & 1u) == 0u) {
            store_quads_8(comp, x, y, v, nb);
        }
        return;
    }
    store_scalar_8(comp, x, y, out_width, out_height, v);
}
