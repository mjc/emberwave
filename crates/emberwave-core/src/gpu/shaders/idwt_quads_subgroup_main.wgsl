// Entry point for the fast 16-bit inverse when the subgroup size is fixed
// at 32. `subgroupShuffleXor` of 1 swaps adjacent columns of one segment:
// a subgroup covers columns 0..31 or 32..63. Every lane executes the shuffles,
// including plane tiles. There is no second barrier and no `odd_column`.

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

    var nb: array<f32, 8>;
    nb[0] = subgroupShuffleXor(v[4], 1u);
    nb[1] = subgroupShuffleXor(v[5], 1u);
    nb[2] = subgroupShuffleXor(v[6], 1u);
    nb[3] = subgroupShuffleXor(v[7], 1u);
    nb[4] = subgroupShuffleXor(v[8], 1u);
    nb[5] = subgroupShuffleXor(v[9], 1u);
    nb[6] = subgroupShuffleXor(v[10], 1u);
    nb[7] = subgroupShuffleXor(v[11], 1u);

    let x = tile_x + i32(column);
    let y = tile_y + i32(8u * segment);
    publish(comp, column, x, y, out_width, out_height, &v, &nb);
    finish_plane(comp, comp.pad2, u32(tile_x), u32(tile_y), tid, comp.is_plane == 1u);
}
