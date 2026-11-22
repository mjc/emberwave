// Stage: the store of the inverse transform's last level into r16unorm textures.
//
// Behind idwt.wgsl, for decodes into 16-bit normalised Y, Cb and Cr plane
// textures on devices with TEXTURE_FORMAT_16BIT_NORM.

@group(0) @binding(3) var plane_y: texture_storage_2d<r16unorm, write>;
@group(0) @binding(4) var plane_cb: texture_storage_2d<r16unorm, write>;
@group(0) @binding(5) var plane_cr: texture_storage_2d<r16unorm, write>;

// A final sample as a 16-bit UNORM level: the DC shift is added by the caller,
// the result clamped to [0, 1] and rounded ties to even.
fn to_unorm16(v: f32) -> f32 {
    return round(clamp(v, 0.0, 1.0) * 65535.0);
}

fn to_level(v: f32) -> f32 {
    return to_unorm16(v);
}

// Stores a 16-bit level into the texture of `component`.
fn store_level(component: u32, at: vec2<u32>, level: f32) {
    let texel = vec4<f32>(level / 65535.0, 0.0, 0.0, 1.0);
    switch (component) {
        case 0u: {
            textureStore(plane_y, at, texel);
        }
        case 1u: {
            textureStore(plane_cb, at, texel);
        }
        default: {
            textureStore(plane_cr, at, texel);
        }
    }
}

// Every invocation stores four samples of a row of the tile inside the plane.
fn store_plane(comp: CompParams, component: u32, tile_x: u32, tile_y: u32, tid: u32) {
    let row = tid / 8u;
    let column = 4u * (tid % 8u);
    let y = tile_y + row;
    let x = tile_x + column;
    let at = row * TILE + column;
    if (y < comp.plane_height) {
        if (x < comp.plane_width) {
            store_level(component, vec2<u32>(x, y), levels[at]);
        }
        if (x + 1u < comp.plane_width) {
            store_level(component, vec2<u32>(x + 1u, y), levels[at + 1u]);
        }
        if (x + 2u < comp.plane_width) {
            store_level(component, vec2<u32>(x + 2u, y), levels[at + 2u]);
        }
        if (x + 3u < comp.plane_width) {
            store_level(component, vec2<u32>(x + 3u, y), levels[at + 3u]);
        }
    }
}
