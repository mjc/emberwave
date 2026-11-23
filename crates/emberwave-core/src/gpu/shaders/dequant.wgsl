// Stage: coefficient decoding and dequantisation.
//
// Upstream counterpart: shaders/wavelet_dequant.comp (the first GPU stage of the
// PyroWave decoder). This is a reimplementation of the algorithm in
// bitstream.md ("Decoding 8x8 blocks", "Dequantization") without the Vulkan
// features upstream needs (subgroup scans, 8/16-bit storage): the prefix sums
// use workgroup memory and the payload is read as u32 words (ARCHITECTURE.md,
// decision D4).
//
// Two dispatches run in the same compute pass using a CPU-built worklist
// (`dequant_work`) and per-block dirty-ballot bitmask (`dirty_ballots`):
// - `clear_blocks`: zeroes 32x32 blocks that were dirty in an earlier frame
//   and are missing in this frame (skipped for concealed detail blocks, D66).
// - `main`: decodes only the received 32x32 blocks, skipping zero writes for
//   8x8 sub-blocks that are inactive in both the previous and current ballots.
//
// One workgroup of 128 invocations processes one 32x32 block. Invocation
// `tid` = 8 * (8x8 block, row-major) + (4x2 subblock) decodes the 8 values of
// its subblock.

struct BandInfo {
    first_block: u32,
    blocks_x: u32,
    width: u32,
    height: u32,
    dst_offset: u32,
    // The band's decomposition level and component (0 Y, 1 Cb, 2 Cr).
    level: u32,
    component: u32,
    pad: u32,
}

struct DequantParams {
    band_count: u32,
    block_count: u32,
    grid_width: u32,
    // 1 for a frame drawn in part that the decoder conceals.
    conceal: u32,
    // Transmitted bands in block index order.
    bands: array<BandInfo, 48>,
}

@group(0) @binding(0) var<storage, read> payload: array<u32>;
@group(0) @binding(1) var<storage, read> dequant_work: array<u32>;
@group(0) @binding(2) var<storage, read_write> pyramid: array<f32>;
@group(0) @binding(3) var<uniform> params: DequantParams;
@group(0) @binding(4) var<storage, read_write> dirty_ballots: array<u32>;

// Side of a coded block.
const BLOCK_SIZE: u32 = 32u;

// Workgroup memory is not zeroed for this pipeline (decision D25): each
// variable is written in full before the barrier that precedes its reads.
var<workgroup> shared_prev_ballot: u32;
// Packed per 8x8 block: control (bits 0..15), qscale (16..23), plane cost (24..31).
var<workgroup> block8_meta: array<u32, 16>;
// Eight 4-bit subblock sign counts (0..=8) packed into one u32 per 8x8 block.
var<workgroup> block8_packed_counts: array<atomic<u32>, 16>;

fn byte_at(byte_offset: u32) -> u32 {
    return (payload[byte_offset >> 2u] >> ((byte_offset & 3u) * 8u)) & 0xffu;
}

// `byte_offset` must be even.
fn half_at(byte_offset: u32) -> u32 {
    return (payload[byte_offset >> 2u] >> ((byte_offset & 2u) * 8u)) & 0xffffu;
}

// Block32x32Scale: (8 + m) * 2^(1 - (quant_code >> 3)), built from the exponent
// bits so that it is exact, like the reference formula.
fn block_scale(quant_code: u32) -> f32 {
    let exponent_bits = (128u - (quant_code >> 3u)) << 23u;
    return f32(8u + (quant_code & 7u)) * bitcast<f32>(exponent_bits);
}

// Block8x8Scale: code / 8 + 0.25.
fn block8_scale(code: u32) -> f32 {
    return f32(code) * 0.125 + 0.25;
}

// (m + 0.5) * scale32 * scale8, in the same order as the CPU decoder, negated
// if the sign bit is set.
fn signed_value(magnitude: u32, scale: f32, negative: u32) -> f32 {
    let value = (f32(magnitude) + 0.5) * scale;
    return select(value, -value, negative != 0u);
}

// Stores one coefficient, if it lies inside its band.
fn store(band: BandInfo, x: u32, y: u32, value: f32) {
    if (x < band.width && y < band.height) {
        pyramid[band.dst_offset + y * band.width + x] = value;
    }
}

@compute @workgroup_size(128)
fn clear_blocks(@builtin(local_invocation_index) tid: u32, @builtin(workgroup_id) wg: vec3<u32>) {
    let clear_offset = dequant_work[1];
    let clear_count = dequant_work[clear_offset];
    let grid_w = min(clear_count, 65535u);
    let item_index = wg.y * grid_w + wg.x;
    if (item_index >= clear_count) {
        return;
    }

    let item = dequant_work[clear_offset + 1u + item_index];
    let block_index = item & 0x00ffffffu;
    let band_slot = (item >> 24u) & 0x3fu;
    let band = params.bands[band_slot];
    let relative = block_index - band.first_block;
    let origin_x = (relative % band.blocks_x) * BLOCK_SIZE;
    let origin_y = (relative / band.blocks_x) * BLOCK_SIZE;

    if (tid == 0u) {
        dirty_ballots[block_index] = 0u;
    }

    let block8 = tid >> 3u;
    let subblock = tid & 7u;
    let x0 = origin_x + 8u * (block8 & 3u) + 4u * (subblock >> 2u);
    let y0 = origin_y + 8u * (block8 >> 2u) + 2u * (subblock & 3u);

    if (x0 + 3u < band.width && y0 + 1u < band.height) {
        let row0 = band.dst_offset + y0 * band.width + x0;
        let row1 = row0 + band.width;
        pyramid[row0] = 0.0;
        pyramid[row0 + 1u] = 0.0;
        pyramid[row0 + 2u] = 0.0;
        pyramid[row0 + 3u] = 0.0;
        pyramid[row1] = 0.0;
        pyramid[row1 + 1u] = 0.0;
        pyramid[row1 + 2u] = 0.0;
        pyramid[row1 + 3u] = 0.0;
    } else {
        for (var dx = 0u; dx < 4u; dx = dx + 1u) {
            store(band, x0 + dx, y0, 0.0);
            store(band, x0 + dx, y0 + 1u, 0.0);
        }
    }
}

@compute @workgroup_size(128)
fn main(@builtin(local_invocation_index) tid: u32, @builtin(workgroup_id) wg: vec3<u32>) {
    let decode_count = dequant_work[0];
    let grid_w = min(decode_count, 65535u);
    let item_index = wg.y * grid_w + wg.x;
    let valid = item_index < decode_count;
    // Clamp out-of-range workgroups (only possible when decode_count > 65535) so
    // all invocations stay in uniform control flow across the workgroup barriers.
    let clamped_item = select(0u, item_index, valid);

    let item_w0 = dequant_work[2u + 2u * clamped_item];
    let offset_words = dequant_work[3u + 2u * clamped_item];
    let block_index = item_w0 & 0x00ffffffu;
    let band_slot = (item_w0 >> 24u) & 0x3fu;
    let band = params.bands[band_slot];
    let relative = block_index - band.first_block;
    let origin_x = (relative % band.blocks_x) * BLOCK_SIZE;
    let origin_y = (relative / band.blocks_x) * BLOCK_SIZE;

    let ballot = payload[offset_words] & 0xffffu;
    let quant_code = payload[offset_words + 1u] & 0xffu;
    let active_count = countOneBits(ballot);
    let base = offset_words * 4u;
    let tables_end = 8u + 3u * active_count;

    // Phase A: read the previous dirty ballot, parse the active 8x8 tables into
    // packed workgroup metadata, and clear the packed sign-count accumulators.
    if (tid == 0u) {
        var prev_b = 0u;
        if (valid) {
            prev_b = dirty_ballots[block_index];
        }
        shared_prev_ballot = prev_b;
    }
    if (tid < 16u) {
        var b8_meta = 0u;
        if (((ballot >> tid) & 1u) != 0u) {
            let index = countOneBits(ballot & ((1u << tid) - 1u));
            let control = half_at(base + 8u + 2u * index);
            let qscale = byte_at(base + 8u + 2u * active_count + index);
            let cost = countOneBits(control & 0x5555u)
                + 2u * countOneBits((control & 0xaaaau) >> 1u)
                + 8u * (qscale & 0xfu);
            b8_meta = control | (qscale << 16u) | (cost << 24u);
        }
        block8_meta[tid] = b8_meta;
        atomicStore(&block8_packed_counts[tid], 0u);
    }
    workgroupBarrier();

    let prev_ballot = shared_prev_ballot;
    if (valid && tid == 0u) {
        dirty_ballots[block_index] = ballot;
    }

    // This invocation's 4x2 subblock: raster index of its 8x8 block and
    // subblock number inside it.
    let block8 = tid >> 3u;
    let subblock = tid & 7u;

    // Where the bit-planes of this 8x8 block start, and where the sign
    // bits start (right after the last bit-plane of the block).
    var plane_start = tables_end;
    var planes_total = 0u;
    var active_bits = ballot;
    while (active_bits != 0u) {
        let r = firstTrailingBit(active_bits);
        active_bits = active_bits & (active_bits - 1u);
        let cost = block8_meta[r] >> 24u;
        plane_start = plane_start + select(0u, cost, r < block8);
        planes_total = planes_total + cost;
    }
    let sign_base_bit = (tables_end + planes_total) * 8u;

    // Phase B: decode this subblock's magnitudes from packed u32 payload words.
    // The eight values stay in named scalar variables so they remain in registers
    // (ARCHITECTURE.md, decision D24).
    let b8_meta = block8_meta[block8];
    let control = b8_meta & 0xffffu;
    let qscale = (b8_meta >> 16u) & 0xffu;

    var m0 = 0u;
    var m1 = 0u;
    var m2 = 0u;
    var m3 = 0u;
    var m4 = 0u;
    var m5 = 0u;
    var m6 = 0u;
    var m7 = 0u;
    var nz_mask = 0u;
    var count = 0u;
    var scale = 0.0;

    if (((ballot >> block8) & 1u) != 0u && control != 0u) {
        let base_planes = qscale & 0xfu;
        let below = control & ((1u << (2u * subblock)) - 1u);
        let extra_before = countOneBits(below & 0x5555u) + 2u * countOneBits((below & 0xaaaau) >> 1u);
        let byte_offset = base + plane_start + extra_before + base_planes * subblock;
        let planes = base_planes + ((control >> (2u * subblock)) & 3u);

        if (planes > 0u) {
            var word_idx = byte_offset >> 2u;
            var byte_in_word = byte_offset & 3u;
            var packed_word = payload[word_idx] >> (byte_in_word * 8u);
            var bytes_left = 4u - byte_in_word;

            // Most significant plane first; bit v of a plane byte is value v.
            for (var plane = planes; plane > 0u; plane = plane - 1u) {
                if (bytes_left == 0u) {
                    word_idx = word_idx + 1u;
                    packed_word = payload[word_idx];
                    bytes_left = 4u;
                }
                let bits = packed_word & 0xffu;
                packed_word = packed_word >> 8u;
                bytes_left = bytes_left - 1u;
                nz_mask = nz_mask | bits;
                m0 = (m0 << 1u) | (bits & 1u);
                m1 = (m1 << 1u) | ((bits >> 1u) & 1u);
                m2 = (m2 << 1u) | ((bits >> 2u) & 1u);
                m3 = (m3 << 1u) | ((bits >> 3u) & 1u);
                m4 = (m4 << 1u) | ((bits >> 4u) & 1u);
                m5 = (m5 << 1u) | ((bits >> 5u) & 1u);
                m6 = (m6 << 1u) | ((bits >> 6u) & 1u);
                m7 = (m7 << 1u) | ((bits >> 7u) & 1u);
            }
            count = countOneBits(nz_mask);
            if (nz_mask != 0u) {
                scale = block_scale(quant_code) * block8_scale(qscale >> 4u);
            }
        }
    }

    // Pack the 8 subblock counts (0..=8 each) into one u32 per 8x8 block so a
    // single barrier suffices before SWAR nibble summation.
    if (count != 0u) {
        atomicOr(&block8_packed_counts[block8], count << (subblock * 4u));
    }
    workgroupBarrier();

    var v0 = 0.0;
    var v1 = 0.0;
    var v2 = 0.0;
    var v3 = 0.0;
    var v4 = 0.0;
    var v5 = 0.0;
    var v6 = 0.0;
    var v7 = 0.0;

    if (count != 0u) {
        var exclusive = 0u;
        var rem = ballot & ((1u << block8) - 1u);
        while (rem != 0u) {
            let r = firstTrailingBit(rem);
            rem = rem & (rem - 1u);
            let p = atomicLoad(&block8_packed_counts[r]);
            let s1 = (p & 0x0f0f0f0fu) + ((p >> 4u) & 0x0f0f0f0fu);
            exclusive = exclusive + ((s1 * 0x01010101u) >> 24u);
        }
        let mask = (1u << (subblock * 4u)) - 1u;
        let below_counts = atomicLoad(&block8_packed_counts[block8]) & mask;
        let sb1 = (below_counts & 0x0f0f0f0fu) + ((below_counts >> 4u) & 0x0f0f0f0fu);
        exclusive = exclusive + ((sb1 * 0x01010101u) >> 24u);

        // Read the sign bits of this invocation (at most 8, so at most two words).
        let sign_bit = sign_base_bit + exclusive;
        let word = offset_words + (sign_bit >> 5u);
        let shift = sign_bit & 31u;
        var sign_bits = payload[word] >> shift;
        if (shift != 0u) {
            sign_bits = sign_bits | (payload[word + 1u] << (32u - shift));
        }

        var sign_index = 0u;
        if (m0 != 0u) {
            v0 = signed_value(m0, scale, (sign_bits >> sign_index) & 1u);
            sign_index = sign_index + 1u;
        }
        if (m1 != 0u) {
            v1 = signed_value(m1, scale, (sign_bits >> sign_index) & 1u);
            sign_index = sign_index + 1u;
        }
        if (m2 != 0u) {
            v2 = signed_value(m2, scale, (sign_bits >> sign_index) & 1u);
            sign_index = sign_index + 1u;
        }
        if (m3 != 0u) {
            v3 = signed_value(m3, scale, (sign_bits >> sign_index) & 1u);
            sign_index = sign_index + 1u;
        }
        if (m4 != 0u) {
            v4 = signed_value(m4, scale, (sign_bits >> sign_index) & 1u);
            sign_index = sign_index + 1u;
        }
        if (m5 != 0u) {
            v5 = signed_value(m5, scale, (sign_bits >> sign_index) & 1u);
            sign_index = sign_index + 1u;
        }
        if (m6 != 0u) {
            v6 = signed_value(m6, scale, (sign_bits >> sign_index) & 1u);
            sign_index = sign_index + 1u;
        }
        if (m7 != 0u) {
            v7 = signed_value(m7, scale, (sign_bits >> sign_index) & 1u);
        }
    }

    // Only write if this subblock has non-zero coefficients or its 8x8 block had
    // values in the previous frame; otherwise pyramid already holds 0.0.
    let block8_was_dirty = ((prev_ballot >> block8) & 1u) != 0u;
    if (valid && (count != 0u || block8_was_dirty)) {
        let x0 = origin_x + 8u * (block8 & 3u) + 4u * (subblock >> 2u);
        let y0 = origin_y + 8u * (block8 >> 2u) + 2u * (subblock & 3u);
        if (x0 + 3u < band.width && y0 + 1u < band.height) {
            let row0 = band.dst_offset + y0 * band.width + x0;
            let row1 = row0 + band.width;
            pyramid[row0] = v0;
            pyramid[row0 + 1u] = v2;
            pyramid[row0 + 2u] = v4;
            pyramid[row0 + 3u] = v6;
            pyramid[row1] = v1;
            pyramid[row1 + 1u] = v3;
            pyramid[row1 + 2u] = v5;
            pyramid[row1 + 3u] = v7;
        } else {
            // Values run down the columns of the 4x2 subblock.
            store(band, x0 + 0u, y0 + 0u, v0);
            store(band, x0 + 0u, y0 + 1u, v1);
            store(band, x0 + 1u, y0 + 0u, v2);
            store(band, x0 + 1u, y0 + 1u, v3);
            store(band, x0 + 2u, y0 + 0u, v4);
            store(band, x0 + 2u, y0 + 1u, v5);
            store(band, x0 + 3u, y0 + 0u, v6);
            store(band, x0 + 3u, y0 + 1u, v7);
        }
    }
}
