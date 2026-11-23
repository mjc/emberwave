// Entry point for the fast 16-bit inverse on devices whose subgroup size is
// not fixed at 32. Odd columns park their eight outputs in `odd_column`, then
// every invocation hits one barrier and even columns read the partner.
//
// `odd_column` is 32 rows by 32 odd columns, 4096 bytes. Together with
// `horizontal` (10240) that is 14336 bytes. Each element is written once,
// by the odd column of that row, before the barrier that precedes the reads.

var<workgroup> odd_column: array<f32, 1024>;

fn odd_at(segment: u32, row: u32, slot: u32) -> u32 {
    return (8u * segment + row) * 32u + slot;
}

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) tid: u32, @builtin(workgroup_id) wg: vec3<u32>) {
    let packed_id = tile_ids[wg.x];
    let tile_ix = packed_id & 0xFFFFu;
    let tile_iy = (packed_id >> 16u) & 0x1FFu;
    let mask = (packed_id >> 25u) & 7u;
    let z = (packed_id >> 28u) & 3u;
    let tile_mode = packed_id >> 30u;
    let comp = params.comps[z];
    let tile_x = i32(tile_ix * TILE);
    let tile_y = i32(tile_iy * TILE_H);
    let out_width = i32(params.image_width);
    let out_height = i32(params.image_height);
    init_levels(tid);

    run_rows(comp, tile_x, tile_y, out_width, out_height, tid, tile_mode, mask);
    workgroupBarrier();

    let column = tid % TILE;
    let segment = tid / TILE;
    var v: array<f32, 16>;
    lift_column(column, segment, &v);

    if ((column & 1u) == 1u) {
        let slot = column >> 1u;
        odd_column[odd_at(segment, 0u, slot)] = v[4];
        odd_column[odd_at(segment, 1u, slot)] = v[5];
        odd_column[odd_at(segment, 2u, slot)] = v[6];
        odd_column[odd_at(segment, 3u, slot)] = v[7];
        odd_column[odd_at(segment, 4u, slot)] = v[8];
        odd_column[odd_at(segment, 5u, slot)] = v[9];
        odd_column[odd_at(segment, 6u, slot)] = v[10];
        odd_column[odd_at(segment, 7u, slot)] = v[11];
    }
    workgroupBarrier();

    var nb: array<f32, 8>;
    nb[0] = 0.0;
    nb[1] = 0.0;
    nb[2] = 0.0;
    nb[3] = 0.0;
    nb[4] = 0.0;
    nb[5] = 0.0;
    nb[6] = 0.0;
    nb[7] = 0.0;
    if ((column & 1u) == 0u) {
        let slot = column >> 1u;
        nb[0] = odd_column[odd_at(segment, 0u, slot)];
        nb[1] = odd_column[odd_at(segment, 1u, slot)];
        nb[2] = odd_column[odd_at(segment, 2u, slot)];
        nb[3] = odd_column[odd_at(segment, 3u, slot)];
        nb[4] = odd_column[odd_at(segment, 4u, slot)];
        nb[5] = odd_column[odd_at(segment, 5u, slot)];
        nb[6] = odd_column[odd_at(segment, 6u, slot)];
        nb[7] = odd_column[odd_at(segment, 7u, slot)];
    }

    let x = tile_x + i32(column);
    let y = tile_y + i32(8u * segment);
    publish(comp, column, x, y, out_width, out_height, &v, &nb);
    finish_plane(comp, comp.pad2, u32(tile_x), u32(tile_y), tid, comp.is_plane == 1u);
}
