// Stage: the encoder's input from an RGB texture, before fdwt.wgsl
// (decision D65).
//
// Upstream has no counterpart: its encoders take Y'CbCr planes, and hosts
// convert their captures before. emberwave converts a picture of R', G' and
// B' as it packs it into the planes buffer, as gather.wgsl packs planes in
// textures: into the words the CPU writes for the planes that color.rs's
// rgb_to_ycbcr8 makes of the same picture, with the same integers
// (ForwardMatrix: 16-bit fixed point, sums of up to 8 pixels well within
// 32 bits). So a picture from an RGB texture encodes to the bytes of those
// planes.
//
// A texel of an 8-bit unorm texture loads as its byte divided by 255, and
// times 255 rounds back to the byte, as in gather.wgsl. A Bgra8Unorm texture
// loads in the order of an Rgba8Unorm one, so the shader takes both.

struct RgbParams {
    // Per component, Y, Cb and Cr in x, y and z: the plane's size in samples,
    // and where its rows start in the planes buffer and how many words each
    // takes.
    width: vec4<u32>,
    height: vec4<u32>,
    offset: vec4<u32>,
    stride: vec4<u32>,
    // ForwardMatrix's rows for Y, Cb and Cr: the factors of R', G' and B' in
    // x, y and z, and the offset of one pixel in w.
    y_row: vec4<i32>,
    cb_row: vec4<i32>,
    cr_row: vec4<i32>,
    // 1 where 4:2:0's chroma sits on the left of the two columns it covers,
    // which weighs three columns 1, 2 and 1; 0 where it sits between them,
    // which weighs two alike.
    left: u32,
    // 0 for non-linear 10-bit (Rgb10a2Unorm), 1 for linear scRGB (Rgba16Float).
    mode: u32,
    // 1 when linear scRGB input converts BT.709 primaries to BT.2020.
    gamut_bt2020: u32,
    // 1 when linear scRGB input uses SMPTE ST 2084 PQ OETF, 0 for sRGB OETF.
    transfer_pq: u32,
    // [kr, kb, 0.5 / (1 - kb), 0.5 / (1 - kr)] for 16-bit float conversion.
    luma_chroma: vec4<f32>,
    // [luma_offset, luma_scale, chroma_offset, chroma_scale] in [0, 65535].
    range_16: vec4<f32>,
}

@group(0) @binding(0) var<storage, read_write> planes: array<u32>;
@group(0) @binding(1) var<uniform> params: RgbParams;
@group(0) @binding(2) var picture: texture_2d<f32>;

// color.rs's FORWARD_BITS.
const FORWARD_BITS: u32 = 16u;

// R', G' and B' of the pixel at (x, y) as bytes; past the picture's edge, the
// edge's.
fn pixel(x: i32, y: u32) -> vec3<i32> {
    let size = textureDimensions(picture);
    let at = vec2<u32>(u32(clamp(x, 0, i32(size.x) - 1)), min(y, size.y - 1u));
    return vec3<i32>(round(saturate(textureLoad(picture, at, 0).rgb) * 255.0));
}

// ForwardMatrix::apply: `row` of the matrix for pixels whose R', G' and B',
// each weighed, add up to `sum`, the weights to 1 << shift. The products are
// spelt out rather than a dot(): totals pass 2^24, which a compiler that took
// an integer dot product through floats would round.
fn apply(row: vec4<i32>, sum: vec3<i32>, shift: u32) -> u32 {
    let total = row.x * sum.x + row.y * sum.y + row.z * sum.z + (row.w << shift)
        + (1i << (FORWARD_BITS + shift - 1u));
    return u32(clamp(total >> (FORWARD_BITS + shift), 0i, 255i));
}

// Samples x to x + 3 of a row as a word, the first in the low byte, with
// zeros for those at `width` and past it.
fn pack(samples: vec4<u32>, x: u32, width: u32) -> u32 {
    let inside = vec4<u32>(x) + vec4<u32>(0u, 1u, 2u, 3u) < vec4<u32>(width);
    let kept = select(vec4<u32>(0u), samples, inside);
    return kept.x | (kept.y << 8u) | (kept.z << 16u) | (kept.w << 24u);
}

// 4:4:4: one invocation per word of a row, of all three planes, which have
// one size: four pixels, and their Y, Cb and Cr.
@compute @workgroup_size(16, 16)
fn yuv444(@builtin(global_invocation_id) id: vec3<u32>) {
    let word = id.x;
    let row = id.y;
    if (word >= params.stride.x || row >= params.height.x) {
        return;
    }
    let x = 4u * word;
    var luma: vec4<u32>;
    var cb: vec4<u32>;
    var cr: vec4<u32>;
    for (var i = 0u; i < 4u; i++) {
        let rgb = pixel(i32(x + i), row);
        luma[i] = apply(params.y_row, rgb, 0u);
        cb[i] = apply(params.cb_row, rgb, 0u);
        cr[i] = apply(params.cr_row, rgb, 0u);
    }
    let width = params.width.x;
    planes[params.offset.x + row * params.stride.x + word] = pack(luma, x, width);
    planes[params.offset.y + row * params.stride.y + word] = pack(cb, x, width);
    planes[params.offset.z + row * params.stride.z + word] = pack(cr, x, width);
}

// 4:2:0: one invocation per word of a chroma row: four samples of Cb and of
// Cr, and the luma of the 8 columns and 2 rows they cover, two words of each
// row, or one at the end of a row of an odd count of words. The luma rows
// take at most twice the chroma rows' words, so every word is written.
@compute @workgroup_size(16, 16)
fn yuv420(@builtin(global_invocation_id) id: vec3<u32>) {
    let word = id.x;
    let chroma_row = id.y;
    if (word >= params.stride.y || chroma_row >= params.height.y) {
        return;
    }
    let x = 8u * word;
    let left = params.left == 1u;
    // Luma by row and word: row 0's two words, then row 1's.
    var luma: array<vec4<u32>, 4>;
    var sums: array<vec3<i32>, 4>;
    for (var r = 0u; r < 2u; r++) {
        let y = 2u * chroma_row + r;
        // The column before the chroma sample's two, for the left siting.
        var before = pixel(i32(x) - 1, y);
        for (var c = 0u; c < 4u; c++) {
            let even = pixel(i32(x + 2u * c), y);
            let odd = pixel(i32(x + 2u * c + 1u), y);
            let at = 2u * r + c / 2u;
            luma[at][(2u * c) % 4u] = apply(params.y_row, even, 0u);
            luma[at][(2u * c + 1u) % 4u] = apply(params.y_row, odd, 0u);
            if (left) {
                sums[c] += before + 2i * even + odd;
            } else {
                sums[c] += even + odd;
            }
            before = odd;
        }
    }
    let shift = select(2u, 3u, left);
    var cb: vec4<u32>;
    var cr: vec4<u32>;
    for (var c = 0u; c < 4u; c++) {
        cb[c] = apply(params.cb_row, sums[c], shift);
        cr[c] = apply(params.cr_row, sums[c], shift);
    }
    for (var r = 0u; r < 2u; r++) {
        let y = 2u * chroma_row + r;
        for (var w = 0u; w < 2u; w++) {
            let luma_word = 2u * word + w;
            if (y < params.height.x && luma_word < params.stride.x) {
                planes[params.offset.x + y * params.stride.x + luma_word] =
                    pack(luma[2u * r + w], x + 4u * w, params.width.x);
            }
        }
    }
    planes[params.offset.y + chroma_row * params.stride.y + word] = pack(cb, 4u * word, params.width.y);
    planes[params.offset.z + chroma_row * params.stride.z + word] = pack(cr, 4u * word, params.width.z);
}

const PQ_M1: f32 = 0.1593017578125;
const PQ_M2: f32 = 78.84375;
const PQ_C1: f32 = 0.8359375;
const PQ_C2: f32 = 18.8515625;
const PQ_C3: f32 = 18.6875;

fn bt709_to_bt2020(c: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        0.6274039 * c.r + 0.32928304 * c.g + 0.043313066 * c.b,
        0.06909729 * c.r + 0.9195404 * c.g + 0.011362316 * c.b,
        0.016391439 * c.r + 0.08801331 * c.g + 0.89559525 * c.b,
    );
}

fn pq_channel(scrgb: f32) -> f32 {
    let v = min(scrgb * (80.0 / 10000.0), 1.0);
    if (v <= 0.0) {
        return 0.0;
    }
    let l = pow(v, PQ_M1);
    return pow((PQ_C1 + PQ_C2 * l) / (1.0 + PQ_C3 * l), PQ_M2);
}

fn srgb_channel(linear: f32) -> f32 {
    let x = min(linear, 1.0);
    if (x <= 0.0031308) {
        return 12.92 * x;
    }
    return 1.055 * pow(x, 1.0 / 2.4) - 0.055;
}

fn sanitize_finite(v: f32) -> f32 {
    let is_finite = (bitcast<u32>(v) & 0x7f800000u) != 0x7f800000u;
    return select(0.0, v, is_finite);
}

fn pixel_rgb(x: i32, y: u32) -> vec3<f32> {
    let size = textureDimensions(picture);
    let at = vec2<u32>(u32(clamp(x, 0, i32(size.x) - 1)), min(y, size.y - 1u));
    let texel = textureLoad(picture, at, 0).rgb;
    if (params.mode == 0u) {
        return round(saturate(texel) * 1023.0) * (1.0 / 1023.0);
    }
    // Sanitize non-finite values before BT.2020 conversion so one NaN/Inf
    // channel does not corrupt finite channels, while preserving negative
    // wide-gamut scRGB values until after primary conversion.
    var c = vec3<f32>(
        sanitize_finite(texel.x),
        sanitize_finite(texel.y),
        sanitize_finite(texel.z),
    );
    if (params.gamut_bt2020 != 0u) {
        c = bt709_to_bt2020(c);
    }
    c = max(c, vec3<f32>(0.0));
    if (params.transfer_pq != 0u) {
        return vec3<f32>(pq_channel(c.x), pq_channel(c.y), pq_channel(c.z));
    }
    return vec3<f32>(srgb_channel(c.x), srgb_channel(c.y), srgb_channel(c.z));
}

fn luma_n(rgb: vec3<f32>) -> f32 {
    return rgb.g + params.luma_chroma.x * (rgb.r - rgb.g) + params.luma_chroma.y * (rgb.b - rgb.g);
}

fn luma_u16(rgb: vec3<f32>) -> u32 {
    let v = params.range_16.x + luma_n(rgb) * params.range_16.y;
    return u32(clamp(round(v), 0.0, 65535.0));
}

fn chroma_u16(rgb: vec3<f32>, is_cb: bool) -> u32 {
    let y = luma_n(rgb);
    let cn = select((rgb.r - y) * params.luma_chroma.w, (rgb.b - y) * params.luma_chroma.z, is_cb);
    let v = params.range_16.z + cn * params.range_16.w;
    return u32(clamp(round(v), 0.0, 65535.0));
}

fn pack16(s0: u32, s1: u32, x: u32, width: u32) -> u32 {
    let low = select(0u, s0, x < width);
    let high = select(0u, s1, x + 1u < width);
    return low | (high << 16u);
}

struct Row420 {
    luma_w0: u32,
    luma_w1: u32,
    c0: vec3<f32>,
    c1: vec3<f32>,
}

fn row420(x: u32, y: u32, left: bool) -> Row420 {
    let p_m1 = pixel_rgb(i32(x) - 1, y);
    let p0 = pixel_rgb(i32(x), y);
    let p1 = pixel_rgb(i32(x) + 1, y);
    let p2 = pixel_rgb(i32(x) + 2, y);
    let p3 = pixel_rgb(i32(x) + 3, y);
    let luma_w0 = pack16(luma_u16(p0), luma_u16(p1), x, params.width.x);
    let luma_w1 = pack16(luma_u16(p2), luma_u16(p3), x + 2u, params.width.x);
    let c0 = select(p0 + p1, p_m1 + 2.0 * p0 + p1, left);
    let c1 = select(p2 + p3, p1 + 2.0 * p2 + p3, left);
    return Row420(luma_w0, luma_w1, c0, c1);
}

// 4:4:4 16-bit: one invocation per word (2 samples) of a row of Y, Cb and Cr.
@compute @workgroup_size(16, 16)
fn yuv444_16(@builtin(global_invocation_id) id: vec3<u32>) {
    let word = id.x;
    let row = id.y;
    if (word >= params.stride.x || row >= params.height.x) {
        return;
    }
    let x = 2u * word;
    let rgb0 = pixel_rgb(i32(x), row);
    let rgb1 = pixel_rgb(i32(x + 1u), row);
    let width = params.width.x;
    planes[params.offset.x + row * params.stride.x + word] =
        pack16(luma_u16(rgb0), luma_u16(rgb1), x, width);
    planes[params.offset.y + row * params.stride.y + word] =
        pack16(chroma_u16(rgb0, true), chroma_u16(rgb1, true), x, width);
    planes[params.offset.z + row * params.stride.z + word] =
        pack16(chroma_u16(rgb0, false), chroma_u16(rgb1, false), x, width);
}

// 4:2:0 16-bit: one invocation per chroma word (2 chroma samples, covering 4x2 luma).
@compute @workgroup_size(16, 16)
fn yuv420_16(@builtin(global_invocation_id) id: vec3<u32>) {
    let word = id.x;
    let chroma_row = id.y;
    if (word >= params.stride.y || chroma_row >= params.height.y) {
        return;
    }
    let x = 4u * word;
    let left = params.left == 1u;
    let y0 = 2u * chroma_row;
    let y1 = y0 + 1u;
    let r0 = row420(x, y0, left);
    let r1 = row420(x, y1, left);

    let lw0 = 2u * word;
    let lw1 = lw0 + 1u;
    if (y0 < params.height.x) {
        if (lw0 < params.stride.x) {
            planes[params.offset.x + y0 * params.stride.x + lw0] = r0.luma_w0;
        }
        if (lw1 < params.stride.x) {
            planes[params.offset.x + y0 * params.stride.x + lw1] = r0.luma_w1;
        }
    }
    if (y1 < params.height.x) {
        if (lw0 < params.stride.x) {
            planes[params.offset.x + y1 * params.stride.x + lw0] = r1.luma_w0;
        }
        if (lw1 < params.stride.x) {
            planes[params.offset.x + y1 * params.stride.x + lw1] = r1.luma_w1;
        }
    }

    let inv_weight = select(0.25, 0.125, left);
    let avg0 = (r0.c0 + r1.c0) * inv_weight;
    let avg1 = (r0.c1 + r1.c1) * inv_weight;
    let cb0 = chroma_u16(avg0, true);
    let cb1 = chroma_u16(avg1, true);
    let cr0 = chroma_u16(avg0, false);
    let cr1 = chroma_u16(avg1, false);
    let cx = 2u * word;
    planes[params.offset.y + chroma_row * params.stride.y + word] =
        pack16(cb0, cb1, cx, params.width.y);
    planes[params.offset.z + chroma_row * params.stride.z + word] =
        pack16(cr0, cr1, cx, params.width.z);
}
