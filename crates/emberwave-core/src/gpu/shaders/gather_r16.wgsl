// Stage: the encoder's 16-bit UNORM input from textures, before fdwt.wgsl
// (decision D77).
//
// Upstream counterpart: the encoder's 16-bit / P010 input path, which reads
// three R16Unorm planes or biplanar P010 (R16Unorm luma + Rg16Unorm interleaved
// Cb/Cr). emberwave packs the picture into the encoder's 16-bit planes buffer,
// two 16-bit UNORM samples per u32 word (even column in low 16 bits, odd column
// in high 16 bits, rows padded to whole words with zeros): the words the CPU
// writes for a 16-bit picture in CPU memory (gpu/encoder.rs, upload_planes_u16).
// fdwt.wgsl reads them at `is_plane == 2u`, so a 16-bit picture in textures
// encodes to the same bytes as the same picture in CPU memory.
//
// A texel of a 16-bit UNORM texture loads as its u16 divided by 65535, and
// round(saturate(texel) * 65535.0) recovers the exact u16 for every value in
// 0..=65535 (including P010 10-bit levels stored in the upper 10 bits, c << 6).
// Each invocation uses one textureGather to fetch a 2x2 block of texels and
// packs one u32 word in each of two consecutive rows.

struct GatherParams {
    // Per component, Y, Cb and Cr in x, y and z: the plane's size in samples,
    // and where its rows start in the 16-bit planes buffer and how many words
    // each row takes (ceil(width / 2)).
    width: vec4<u32>,
    height: vec4<u32>,
    offset: vec4<u32>,
    stride: vec4<u32>,
    // The channel of the component in its texture: 0 red, 1 green.
    channel: vec4<u32>,
}

@group(0) @binding(0) var<storage, read_write> planes: array<u32>;
@group(0) @binding(1) var<uniform> params: GatherParams;
@group(0) @binding(2) var luma: texture_2d<f32>;
@group(0) @binding(3) var cb: texture_2d<f32>;
@group(0) @binding(4) var cr: texture_2d<f32>;
// For textureGather, which needs a sampler; its addressing clamps to the edge.
@group(0) @binding(5) var texels: sampler;

// Corner shared by texels (x, y) to (x + 1, y + 1) in normalised coordinates.
fn corner(size: vec2<u32>, x: u32, y: u32) -> vec2<f32> {
    return vec2<f32>(f32(x + 1u), f32(y + 1u)) / vec2<f32>(size);
}

// Texels (x, y) to (x + 1, y + 1) of a component recovered as exact 16-bit
// UNORM integers in textureGather's order: (x, y + 1), (x + 1, y + 1),
// (x + 1, y), (x, y).
fn gather_at(component: u32, x: u32, y: u32) -> vec4<u32> {
    let green = params.channel[component] == 1u;
    var texel: vec4<f32>;
    switch component {
        case 0u: {
            texel = textureGather(0, luma, texels, corner(textureDimensions(luma), x, y));
        }
        case 1u: {
            let at = corner(textureDimensions(cb), x, y);
            if (green) {
                texel = textureGather(1, cb, texels, at);
            } else {
                texel = textureGather(0, cb, texels, at);
            }
        }
        default: {
            let at = corner(textureDimensions(cr), x, y);
            if (green) {
                texel = textureGather(1, cr, texels, at);
            } else {
                texel = textureGather(0, cr, texels, at);
            }
        }
    }
    return vec4<u32>(round(saturate(texel) * 65535.0));
}

// Samples x and x + 1 of a row packed into one u32 word (even in low 16 bits,
// odd in high 16 bits), with zeros for samples at or past `width`.
fn pack2(s0: u32, s1: u32, x: u32, width: u32) -> u32 {
    let lo = select(0u, s0, x < width);
    let hi = select(0u, s1, x + 1u < width);
    return lo | (hi << 16u);
}

// One invocation per word of a pair of rows: two 16-bit samples of each row
// from one 2x2 textureGather. A workgroup covers 16 words of 32 rows (32 by 32
// samples); z selects the component.
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let component = id.z;
    let word = id.x;
    let row = 2u * id.y;
    let height = params.height[component];
    if (word >= params.stride[component] || row >= height) {
        return;
    }
    let width = params.width[component];
    let x = 2u * word;
    let block = gather_at(component, x, row);
    let at = params.offset[component] + row * params.stride[component] + word;
    planes[at] = pack2(block.w, block.z, x, width);
    if (row + 1u < height) {
        planes[at + params.stride[component]] = pack2(block.x, block.y, x, width);
    }
}
