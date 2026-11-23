// Stage: the store of the inverse transform's last level into r16unorm textures.
//
// Behind idwt_quads_lib.wgsl on the fast path, or idwt_f16.wgsl when the device
// has shader f16, for decodes into 16-bit normalised Y, Cb and Cr plane textures.
// Both fronts define DC_SHIFT. `to_unorm16` stays in this file: a test slices
// it out of the concatenated shader.

@group(0) @binding(3) var plane_y: texture_storage_2d<r16unorm, write>;
@group(0) @binding(4) var plane_cb: texture_storage_2d<r16unorm, write>;
@group(0) @binding(5) var plane_cr: texture_storage_2d<r16unorm, write>;

fn init_levels(_tid: u32) {}

// A final sample as a 16-bit UNORM level: the DC shift is added by the caller,
// the result clamped to [0, 1] and rounded ties to even.
fn to_unorm16(v: f32) -> f32 {
    return round(clamp(v, 0.0, 1.0) * 65535.0);
}

fn texel16(s: f32) -> vec4<f32> {
    return vec4<f32>(to_unorm16(s + DC_SHIFT) / 65535.0, 0.0, 0.0, 1.0);
}

// Stores 8 vertical samples directly from registers, avoiding workgroup staging
// and the second barrier.
fn store_vertical_8(
    comp: CompParams,
    component: u32,
    x: u32,
    y: u32,
    _at: u32,
    s0: f32,
    s1: f32,
    s2: f32,
    s3: f32,
    s4: f32,
    s5: f32,
    s6: f32,
    s7: f32,
) {
    if (x >= comp.plane_width || y >= comp.plane_height) {
        return;
    }
    let h = comp.plane_height;
    let full_y = y + 7u < h;
    switch (component) {
        case 0u: {
            if (full_y) {
                textureStore(plane_y, vec2<u32>(x, y + 0u), texel16(s0));
                textureStore(plane_y, vec2<u32>(x, y + 1u), texel16(s1));
                textureStore(plane_y, vec2<u32>(x, y + 2u), texel16(s2));
                textureStore(plane_y, vec2<u32>(x, y + 3u), texel16(s3));
                textureStore(plane_y, vec2<u32>(x, y + 4u), texel16(s4));
                textureStore(plane_y, vec2<u32>(x, y + 5u), texel16(s5));
                textureStore(plane_y, vec2<u32>(x, y + 6u), texel16(s6));
                textureStore(plane_y, vec2<u32>(x, y + 7u), texel16(s7));
            } else {
                if (y + 0u < h) { textureStore(plane_y, vec2<u32>(x, y + 0u), texel16(s0)); }
                if (y + 1u < h) { textureStore(plane_y, vec2<u32>(x, y + 1u), texel16(s1)); }
                if (y + 2u < h) { textureStore(plane_y, vec2<u32>(x, y + 2u), texel16(s2)); }
                if (y + 3u < h) { textureStore(plane_y, vec2<u32>(x, y + 3u), texel16(s3)); }
                if (y + 4u < h) { textureStore(plane_y, vec2<u32>(x, y + 4u), texel16(s4)); }
                if (y + 5u < h) { textureStore(plane_y, vec2<u32>(x, y + 5u), texel16(s5)); }
                if (y + 6u < h) { textureStore(plane_y, vec2<u32>(x, y + 6u), texel16(s6)); }
                if (y + 7u < h) { textureStore(plane_y, vec2<u32>(x, y + 7u), texel16(s7)); }
            }
        }
        case 1u: {
            if (full_y) {
                textureStore(plane_cb, vec2<u32>(x, y + 0u), texel16(s0));
                textureStore(plane_cb, vec2<u32>(x, y + 1u), texel16(s1));
                textureStore(plane_cb, vec2<u32>(x, y + 2u), texel16(s2));
                textureStore(plane_cb, vec2<u32>(x, y + 3u), texel16(s3));
                textureStore(plane_cb, vec2<u32>(x, y + 4u), texel16(s4));
                textureStore(plane_cb, vec2<u32>(x, y + 5u), texel16(s5));
                textureStore(plane_cb, vec2<u32>(x, y + 6u), texel16(s6));
                textureStore(plane_cb, vec2<u32>(x, y + 7u), texel16(s7));
            } else {
                if (y + 0u < h) { textureStore(plane_cb, vec2<u32>(x, y + 0u), texel16(s0)); }
                if (y + 1u < h) { textureStore(plane_cb, vec2<u32>(x, y + 1u), texel16(s1)); }
                if (y + 2u < h) { textureStore(plane_cb, vec2<u32>(x, y + 2u), texel16(s2)); }
                if (y + 3u < h) { textureStore(plane_cb, vec2<u32>(x, y + 3u), texel16(s3)); }
                if (y + 4u < h) { textureStore(plane_cb, vec2<u32>(x, y + 4u), texel16(s4)); }
                if (y + 5u < h) { textureStore(plane_cb, vec2<u32>(x, y + 5u), texel16(s5)); }
                if (y + 6u < h) { textureStore(plane_cb, vec2<u32>(x, y + 6u), texel16(s6)); }
                if (y + 7u < h) { textureStore(plane_cb, vec2<u32>(x, y + 7u), texel16(s7)); }
            }
        }
        default: {
            if (full_y) {
                textureStore(plane_cr, vec2<u32>(x, y + 0u), texel16(s0));
                textureStore(plane_cr, vec2<u32>(x, y + 1u), texel16(s1));
                textureStore(plane_cr, vec2<u32>(x, y + 2u), texel16(s2));
                textureStore(plane_cr, vec2<u32>(x, y + 3u), texel16(s3));
                textureStore(plane_cr, vec2<u32>(x, y + 4u), texel16(s4));
                textureStore(plane_cr, vec2<u32>(x, y + 5u), texel16(s5));
                textureStore(plane_cr, vec2<u32>(x, y + 6u), texel16(s6));
                textureStore(plane_cr, vec2<u32>(x, y + 7u), texel16(s7));
            } else {
                if (y + 0u < h) { textureStore(plane_cr, vec2<u32>(x, y + 0u), texel16(s0)); }
                if (y + 1u < h) { textureStore(plane_cr, vec2<u32>(x, y + 1u), texel16(s1)); }
                if (y + 2u < h) { textureStore(plane_cr, vec2<u32>(x, y + 2u), texel16(s2)); }
                if (y + 3u < h) { textureStore(plane_cr, vec2<u32>(x, y + 3u), texel16(s3)); }
                if (y + 4u < h) { textureStore(plane_cr, vec2<u32>(x, y + 4u), texel16(s4)); }
                if (y + 5u < h) { textureStore(plane_cr, vec2<u32>(x, y + 5u), texel16(s5)); }
                if (y + 6u < h) { textureStore(plane_cr, vec2<u32>(x, y + 6u), texel16(s6)); }
                if (y + 7u < h) { textureStore(plane_cr, vec2<u32>(x, y + 7u), texel16(s7)); }
            }
        }
    }
}

fn finish_plane(_comp: CompParams, _component: u32, _tile_x: u32, _tile_y: u32, _tid: u32, _is_final: bool) {}
