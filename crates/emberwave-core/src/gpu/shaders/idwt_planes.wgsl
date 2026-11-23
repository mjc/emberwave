// Stage: the store of the inverse transform's last level, into textures.
//
// Behind idwt.wgsl, in place of idwt_packed.wgsl, for decodes into the Y, Cb
// and Cr textures of a caller, on adapters that can write 8-bit storage
// textures (decision D52). As upstream's last level does (shaders/idwt.comp),
// it stores the samples straight into a UNORM texture per plane, which saves
// writing the packed planes and copying them out (decision D41). It is not
// one of the portable shaders: WebGPU does not guarantee `r8unorm` storage.

@group(0) @binding(3) var plane_y: texture_storage_2d<r8unorm, write>;
@group(0) @binding(4) var plane_cb: texture_storage_2d<r8unorm, write>;
@group(0) @binding(5) var plane_cr: texture_storage_2d<r8unorm, write>;

fn init_levels(_tid: u32) {}

fn texel8(s: f32) -> vec4<f32> {
    return vec4<f32>(to_unorm8(s + DC_SHIFT) / 255.0, 0.0, 0.0, 1.0);
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
                textureStore(plane_y, vec2<u32>(x, y + 0u), texel8(s0));
                textureStore(plane_y, vec2<u32>(x, y + 1u), texel8(s1));
                textureStore(plane_y, vec2<u32>(x, y + 2u), texel8(s2));
                textureStore(plane_y, vec2<u32>(x, y + 3u), texel8(s3));
                textureStore(plane_y, vec2<u32>(x, y + 4u), texel8(s4));
                textureStore(plane_y, vec2<u32>(x, y + 5u), texel8(s5));
                textureStore(plane_y, vec2<u32>(x, y + 6u), texel8(s6));
                textureStore(plane_y, vec2<u32>(x, y + 7u), texel8(s7));
            } else {
                if (y + 0u < h) { textureStore(plane_y, vec2<u32>(x, y + 0u), texel8(s0)); }
                if (y + 1u < h) { textureStore(plane_y, vec2<u32>(x, y + 1u), texel8(s1)); }
                if (y + 2u < h) { textureStore(plane_y, vec2<u32>(x, y + 2u), texel8(s2)); }
                if (y + 3u < h) { textureStore(plane_y, vec2<u32>(x, y + 3u), texel8(s3)); }
                if (y + 4u < h) { textureStore(plane_y, vec2<u32>(x, y + 4u), texel8(s4)); }
                if (y + 5u < h) { textureStore(plane_y, vec2<u32>(x, y + 5u), texel8(s5)); }
                if (y + 6u < h) { textureStore(plane_y, vec2<u32>(x, y + 6u), texel8(s6)); }
                if (y + 7u < h) { textureStore(plane_y, vec2<u32>(x, y + 7u), texel8(s7)); }
            }
        }
        case 1u: {
            if (full_y) {
                textureStore(plane_cb, vec2<u32>(x, y + 0u), texel8(s0));
                textureStore(plane_cb, vec2<u32>(x, y + 1u), texel8(s1));
                textureStore(plane_cb, vec2<u32>(x, y + 2u), texel8(s2));
                textureStore(plane_cb, vec2<u32>(x, y + 3u), texel8(s3));
                textureStore(plane_cb, vec2<u32>(x, y + 4u), texel8(s4));
                textureStore(plane_cb, vec2<u32>(x, y + 5u), texel8(s5));
                textureStore(plane_cb, vec2<u32>(x, y + 6u), texel8(s6));
                textureStore(plane_cb, vec2<u32>(x, y + 7u), texel8(s7));
            } else {
                if (y + 0u < h) { textureStore(plane_cb, vec2<u32>(x, y + 0u), texel8(s0)); }
                if (y + 1u < h) { textureStore(plane_cb, vec2<u32>(x, y + 1u), texel8(s1)); }
                if (y + 2u < h) { textureStore(plane_cb, vec2<u32>(x, y + 2u), texel8(s2)); }
                if (y + 3u < h) { textureStore(plane_cb, vec2<u32>(x, y + 3u), texel8(s3)); }
                if (y + 4u < h) { textureStore(plane_cb, vec2<u32>(x, y + 4u), texel8(s4)); }
                if (y + 5u < h) { textureStore(plane_cb, vec2<u32>(x, y + 5u), texel8(s5)); }
                if (y + 6u < h) { textureStore(plane_cb, vec2<u32>(x, y + 6u), texel8(s6)); }
                if (y + 7u < h) { textureStore(plane_cb, vec2<u32>(x, y + 7u), texel8(s7)); }
            }
        }
        default: {
            if (full_y) {
                textureStore(plane_cr, vec2<u32>(x, y + 0u), texel8(s0));
                textureStore(plane_cr, vec2<u32>(x, y + 1u), texel8(s1));
                textureStore(plane_cr, vec2<u32>(x, y + 2u), texel8(s2));
                textureStore(plane_cr, vec2<u32>(x, y + 3u), texel8(s3));
                textureStore(plane_cr, vec2<u32>(x, y + 4u), texel8(s4));
                textureStore(plane_cr, vec2<u32>(x, y + 5u), texel8(s5));
                textureStore(plane_cr, vec2<u32>(x, y + 6u), texel8(s6));
                textureStore(plane_cr, vec2<u32>(x, y + 7u), texel8(s7));
            } else {
                if (y + 0u < h) { textureStore(plane_cr, vec2<u32>(x, y + 0u), texel8(s0)); }
                if (y + 1u < h) { textureStore(plane_cr, vec2<u32>(x, y + 1u), texel8(s1)); }
                if (y + 2u < h) { textureStore(plane_cr, vec2<u32>(x, y + 2u), texel8(s2)); }
                if (y + 3u < h) { textureStore(plane_cr, vec2<u32>(x, y + 3u), texel8(s3)); }
                if (y + 4u < h) { textureStore(plane_cr, vec2<u32>(x, y + 4u), texel8(s4)); }
                if (y + 5u < h) { textureStore(plane_cr, vec2<u32>(x, y + 5u), texel8(s5)); }
                if (y + 6u < h) { textureStore(plane_cr, vec2<u32>(x, y + 6u), texel8(s6)); }
                if (y + 7u < h) { textureStore(plane_cr, vec2<u32>(x, y + 7u), texel8(s7)); }
            }
        }
    }
}

fn finish_plane(_comp: CompParams, _component: u32, _tile_x: u32, _tile_y: u32, _tid: u32, _is_final: bool) {}
