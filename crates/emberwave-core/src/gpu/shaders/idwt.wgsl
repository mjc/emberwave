// Stage: inverse CDF 9/7 wavelet transform, one decomposition level.
//
// Upstream counterpart: shaders/idwt.comp. This is a reimplementation of the
// transform described in bitstream.md ("Wavelet transform", "Inverse DC
// shift"). It does not use upstream's textureGather and mirrored sampler
// trick: interleaved coordinates are mirrored explicitly (whole-sample
// symmetric extension, which keeps the parity of an index) and the sample is
// then fetched from the band its parity selects.
//
// One dispatch combines the LL, HL, LH and HH bands of a level into the LL band
// of the next finer level (or into the final plane). A workgroup produces a
// 64x32 tile of the output from its 72x40 interleaved neighbourhood (4 samples
// of extension on each side, one per lifting step): it lifts the rows of the
// neighbourhood, read straight from the bands, into workgroup memory, then the
// columns of the tile from there, and stores the tile.
// At the last level of a component the tile is quantised by `to_unorm8` or
// `to_unorm16` and stored straight into plane textures or packed into words.
// Workgroup z selects the colour component.
//
// This file is the front of the 8-bit shaders. idwt_packed.wgsl packs the last
// level four samples to a word. idwt_planes.wgsl stores it into an `r8unorm`
// texture per plane (decision D52). The 16-bit path uses idwt_fast.wgsl.

// The parameters of a level are shared with fdwt.wgsl, which goes the other
// way between the same places (gpu/layout.rs, TransformLevel).
struct CompParams {
    // Offsets of the LL, HL, LH and HH bands in the pyramid, in floats.
    band_offset: vec4<u32>,
    // Where the output goes, the level's image: the LL band of the next level
    // in the pyramid (offset and row length in floats), or at the last level
    // the packed 8-bit plane (offset and row length in words), if it is packed.
    image_offset: u32,
    image_stride: u32,
    // Size of the plane, which the last level crops its output to.
    plane_width: u32,
    plane_height: u32,
    // 1: the last level, which writes the plane; 0: an earlier level.
    is_plane: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct LevelParams {
    band_width: u32,
    band_height: u32,
    // Size of the output, twice that of the bands.
    image_width: u32,
    image_height: u32,
    comps: array<CompParams, 3>,
}

@group(0) @binding(0) var<storage, read_write> pyramid: array<f32>;
@group(0) @binding(2) var<uniform> params: LevelParams;

// Lifting step constants of the CDF 9/7 filter (ITU-T T.800, F.4.8.2).
const ALPHA: f32 = -1.586134342059924;
const BETA: f32 = -0.052980118572961;
const GAMMA: f32 = 0.882911075530934;
const DELTA: f32 = 0.443506852043971;
// Gain of the low-pass band; the high-pass band is scaled by its inverse.
const K: f32 = 1.230174104914001;
const INV_K: f32 = 0.8128930661159609;

// Output tile, and the extension of its neighbourhood on each side.
const TILE: u32 = 64u;
const TILE_H: u32 = 32u;
const APRON: i32 = 4;

// Shifts the DC level back (bitstream.md, "Inverse DC shift").
const DC_SHIFT: f32 = 0.5;

// Workgroup memory is not zeroed for this pipeline (decision D25): every
// element is written before it is read. The horizontal pass fills all of
// `horizontal` (40 * 64 = 2560 floats = 10240 bytes). Interior tiles first
// stage one band-row per warp in `band_even` / `band_odd` (8 * 36 floats)
// so a warp loads consecutive samples.
var<workgroup> horizontal: array<f32, 2560>;
var<workgroup> band_even: array<f32, 288>;
var<workgroup> band_odd: array<f32, 288>;

// Whole-sample symmetric mirroring: -1 -> 1, n -> n - 2.
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

// Inverse lifting of 16 interleaved samples whose first sample is at an even
// position. The gains are applied here. The 8 samples in the middle (4..12)
// are the valid result.
//
// This, and the loads and stores around it, is written out sample by sample
// on purpose. With loops over a variable index the compiler kept the array in
// memory instead of registers, and the whole transform took 4.5 times as long
// on an Apple GPU (ARCHITECTURE.md, decision D24).
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

// Stores one sample of the next level's LL band, if it lies inside the output.
fn store_ll(comp: CompParams, x: i32, y: i32, value: f32) {
    if (x < i32(params.image_width) && y < i32(params.image_height)) {
        pyramid[comp.image_offset + u32(y) * comp.image_stride + u32(x)] = value;
    }
}

// A final sample as an 8-bit level: the DC shift is added by the caller, the
// result clamped to [0, 1] and rounded like a UNORM store. WGSL's round()
// rounds ties to even on every backend.
fn to_unorm8(v: f32) -> f32 {
    return round(clamp(v, 0.0, 1.0) * 255.0);
}

fn horiz_segment(comp: CompParams, tile_x: i32, tile_y: i32, out_width: i32, out_height: i32, row: u32, segment: u32) {
    let raw_y = tile_y + i32(row) - APRON;
    let y = select(mirror(raw_y, out_height), raw_y, raw_y >= 0 && raw_y < out_height);
    let row_bands = u32(y & 1) << 1u;
    let band_row = u32(y >> 1u) * params.band_width;
    let even = comp.band_offset[row_bands] + band_row;
    let odd = comp.band_offset[row_bands | 1u] + band_row;
    let x0 = tile_x + i32(8u * segment) - APRON;
    var v: array<f32, 16>;
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
    lift(&v);
    let out = row * TILE + 8u * segment;
    horizontal[out + 0u] = v[4];
    horizontal[out + 1u] = v[5];
    horizontal[out + 2u] = v[6];
    horizontal[out + 3u] = v[7];
    horizontal[out + 4u] = v[8];
    horizontal[out + 5u] = v[9];
    horizontal[out + 6u] = v[10];
    horizontal[out + 7u] = v[11];
}

// Even and odd samples at one band x of an interior row. `y` is an output
// coordinate inside the image, so the band row needs no mirror.
fn band_pair(comp: CompParams, y: i32, band_x: u32) -> vec2<f32> {
    let row_bands = u32(y & 1) << 1u;
    let band_row = u32(y >> 1u) * params.band_width;
    let even = comp.band_offset[row_bands] + band_row;
    let odd = comp.band_offset[row_bands | 1u] + band_row;
    return vec2<f32>(pyramid[even + band_x], pyramid[odd + band_x]);
}

// Eight segments of one interior row, from the 36 staged band samples.
fn lift_staged_row(slot: u32, row: u32, segment: u32) {
    let local = 4u * segment;
    var v: array<f32, 16>;
    v[0] = band_even[slot + local + 0u];
    v[1] = band_odd[slot + local + 0u];
    v[2] = band_even[slot + local + 1u];
    v[3] = band_odd[slot + local + 1u];
    v[4] = band_even[slot + local + 2u];
    v[5] = band_odd[slot + local + 2u];
    v[6] = band_even[slot + local + 3u];
    v[7] = band_odd[slot + local + 3u];
    v[8] = band_even[slot + local + 4u];
    v[9] = band_odd[slot + local + 4u];
    v[10] = band_even[slot + local + 5u];
    v[11] = band_odd[slot + local + 5u];
    v[12] = band_even[slot + local + 6u];
    v[13] = band_odd[slot + local + 6u];
    v[14] = band_even[slot + local + 7u];
    v[15] = band_odd[slot + local + 7u];
    lift(&v);
    let out = row * TILE + 8u * segment;
    horizontal[out + 0u] = v[4];
    horizontal[out + 1u] = v[5];
    horizontal[out + 2u] = v[6];
    horizontal[out + 3u] = v[7];
    horizontal[out + 4u] = v[8];
    horizontal[out + 5u] = v[9];
    horizontal[out + 6u] = v[10];
    horizontal[out + 7u] = v[11];
}

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) tid: u32, @builtin(workgroup_id) wg: vec3<u32>) {
    let comp = params.comps[wg.z];
    let out_width = i32(params.image_width);
    let out_height = i32(params.image_height);
    let tile_x = i32(wg.x * TILE);
    let tile_y = i32(wg.y * TILE_H);
    init_levels(tid);

    // The neighbourhood of an interior tile sits inside the image, so the
    // band rows are contiguous and a warp can load them in order. Border
    // tiles keep the mirrored scalar path.
    let interior = tile_x >= APRON && tile_y >= APRON && tile_x + i32(TILE) + APRON <= out_width && tile_y + i32(TILE_H) + APRON <= out_height;
    if (interior) {
        let warp = tid >> 5u;
        let lane = tid & 31u;
        let bx0 = u32(tile_x - APRON) >> 1u;
        for (var phase: u32 = 0u; phase < 5u; phase++) {
            let row = warp + phase * 8u;
            let y = tile_y + i32(row) - APRON;
            let slot = warp * 36u;
            if (lane < 32u) {
                let pair = band_pair(comp, y, bx0 + lane);
                band_even[slot + lane] = pair.x;
                band_odd[slot + lane] = pair.y;
            }
            if (lane < 4u) {
                let pair = band_pair(comp, y, bx0 + 32u + lane);
                band_even[slot + 32u + lane] = pair.x;
                band_odd[slot + 32u + lane] = pair.y;
            }
            workgroupBarrier();
            if (lane < 8u) {
                lift_staged_row(slot, row, lane);
            }
            workgroupBarrier();
        }
    } else {
        let h_seg = tid % 8u;
        let row0 = 2u * ((tid % 128u) / 8u) + (tid / 128u);
        horiz_segment(comp, tile_x, tile_y, out_width, out_height, row0, h_seg);
        if (tid < 64u) {
            let row1 = 32u + 2u * ((tid % 32u) / 8u) + (tid / 32u);
            horiz_segment(comp, tile_x, tile_y, out_width, out_height, row1, h_seg);
        }
        workgroupBarrier();
    }

    // Vertical pass: 64 columns x 4 vertical segments = 256 invocations (100% active).
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
        store_vertical_8(
            comp,
            comp.pad2,
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
                let row_base = comp.image_offset + u32(y) * comp.image_stride + u32(x);
                let stride = comp.image_stride;
                pyramid[row_base + 0u * stride] = v[4];
                pyramid[row_base + 1u * stride] = v[5];
                pyramid[row_base + 2u * stride] = v[6];
                pyramid[row_base + 3u * stride] = v[7];
                pyramid[row_base + 4u * stride] = v[8];
                pyramid[row_base + 5u * stride] = v[9];
                pyramid[row_base + 6u * stride] = v[10];
                pyramid[row_base + 7u * stride] = v[11];
            } else {
                store_ll(comp, x, y + 0, v[4]);
                store_ll(comp, x, y + 1, v[5]);
                store_ll(comp, x, y + 2, v[6]);
                store_ll(comp, x, y + 3, v[7]);
                store_ll(comp, x, y + 4, v[8]);
                store_ll(comp, x, y + 5, v[9]);
                store_ll(comp, x, y + 6, v[10]);
                store_ll(comp, x, y + 7, v[11]);
            }
        }
    }

    finish_plane(comp, wg.z, u32(tile_x), u32(tile_y), tid, is_final);
}
