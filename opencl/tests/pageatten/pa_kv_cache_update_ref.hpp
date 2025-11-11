
/*
 * Copyright (c) 2020-2025, Intel Corporation
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */
#include <cm/cm.h>
#include <cm/cmtl.h>

// -------------------------------------------------------------------
//   K_HEAD_SIZE, V_HEAD_SIZE
//   KV_HEADS_NUM
//   PAGED_ATTENTION_BLOCK_SIZE
//   ADJUSTED_K_HEAD_SIZE, ADJUSTED_V_HEAD_SIZE
//
//   KV_CACHE_COMPRESSION_PER_CHANNEL
//   KV_CACHE_COMPRESSION_PER_TOKEN
//
// by-channel group:
//   GROUP_NUM  // K_HEAD_SIZE % GROUP_NUM == 0
// -------------------------------------------------------------------


#ifndef GROUP_NUM
#define GROUP_NUM 1
#endif

#if defined(KV_CACHE_COMPRESSION_PER_TOKEN) && defined(KV_CACHE_COMPRESSION_PER_CHANNEL)
#error "Define either KV_CACHE_COMPRESSION_PER_TOKEN or KV_CACHE_COMPRESSION_PER_CHANNEL, not both."
#endif

#ifdef KV_CACHE_COMPRESSION_PER_CHANNEL
#define COMP_BYTES_PER_GROUP (sizeof(half) * 2)     // [scale_inv, zp]
#define COMP_BYTES_TOTAL     (COMP_BYTES_PER_GROUP * GROUP_NUM)
#define ADJUSTED_PAGED_ATTENTION_BLOCK_SIZE (PAGED_ATTENTION_BLOCK_SIZE + COMP_BYTES_TOTAL)
#endif

#if KV_CACHE_COMPRESSION_PER_TOKEN || KV_CACHE_COMPRESSION_PER_CHANNEL
CM_INLINE void quantize_and_store_per_token(vector<half, K_HEAD_SIZE> data, uchar* out, uint out_offset, uint token_pos) {
    uint scale_offset = out_offset + K_HEAD_SIZE * PAGED_ATTENTION_BLOCK_SIZE + token_pos * sizeof(half);

    half max_val = cm_reduced_max<half>(data);
    half min_val = cm_reduced_min<half>(data);

    half scale_val = half(0.0);
    half zp_val    = half(0.0);
    if (max_val == min_val) {
        scale_val = half(0.0);
        zp_val    = max_val;
    } else {
        scale_val = half(255.0) / (max_val - min_val);
        zp_val    = (half(0.0) - min_val) * scale_val;
    }

    vector<half, K_HEAD_SIZE>  acc = cm_mul<half>(data, scale_val) + zp_val;
    vector<uchar, K_HEAD_SIZE> q   = cm_rnde<uchar, K_HEAD_SIZE>(acc);
    cm_ptr_store<uint32_t, K_HEAD_SIZE / 4>(
        (uint32_t*)(out + out_offset + token_pos * K_HEAD_SIZE),
        0, q.format<uint32_t>());

    half *out_scale_zp = (half*)(out + scale_offset);
    out_scale_zp[0] = (max_val - min_val) / half(255.0);
    out_scale_zp[PAGED_ATTENTION_BLOCK_SIZE] = zp_val;
}
#endif

#ifdef KV_CACHE_COMPRESSION_PER_CHANNEL
CM_INLINE void compute_group_comp_from_minmax(
    half vmin_h, half vmax_h,
    half &scale_val_out, half &zp_val_out, half &scale_inv_out)
{
    half diff = vmax_h - vmin_h;
    if (diff == half(0.0)) {
        scale_val_out = half(0.0);
        zp_val_out    = vmax_h;
        scale_inv_out = half(0.0);
        return;
    }

    float diff_f = (float)diff;
    float s = 255.0f / diff_f;
    float inv = diff_f / 255.0f;

    scale_val_out = half(s);
    zp_val_out    = half(-vmin_h * scale_val_out);
    scale_inv_out = half(inv);
}
#endif

extern "C" _GENX_MAIN_ void pa_kv_cache_update(
    const half*    key           [[type("svmptr_t")]],
    const half*    value         [[type("svmptr_t")]],
    const int32_t* past_lens     [[type("svmptr_t")]],
    const int32_t* block_indices [[type("svmptr_t")]],
    const int32_t* block_indices_begins [[type("svmptr_t")]],
    const int32_t* subsequence_begins   [[type("svmptr_t")]],
#if KV_CACHE_COMPRESSION_PER_TOKEN || KV_CACHE_COMPRESSION_PER_CHANNEL
    uint8_t*       key_cache     [[type("svmptr_t")]],
    uint8_t*       value_cache   [[type("svmptr_t")]],
#else
    half*          key_cache     [[type("svmptr_t")]],
    half*          value_cache   [[type("svmptr_t")]],
#endif
    uint32_t key_pitch,
    uint32_t value_pitch,
    uint32_t batch_size_in_sequences,

    // prefill only
    const int32_t* blocked_indexes_start [[type("svmptr_t")]],
    const int32_t* blocked_indexes_end   [[type("svmptr_t")]],
    const int32_t* gws_seq_indexes_correspondence [[type("svmptr_t")]],
    const int      is_prefill_stage
) {
    const uint head_idx  = cm_group_id(1);
    const uint token_gid = cm_global_id(2);

#ifdef KV_CACHE_COMPRESSION_PER_CHANNEL
    constexpr uint GROUP_SIZE = (K_HEAD_SIZE % GROUP_NUM == 0) ? (K_HEAD_SIZE / GROUP_NUM) : 0;
    static_assert(GROUP_SIZE != 0, "K_HEAD_SIZE must be divisible by GROUP_NUM");

    auto comp_ptr_of_group = [&](uchar* base_u8, uint block_k_base_offset_bytes, uint g)->half* {
        const uint comp_base = block_k_base_offset_bytes + K_HEAD_SIZE * PAGED_ATTENTION_BLOCK_SIZE;
        return reinterpret_cast<half*>(base_u8 + comp_base + g * 2 * sizeof(half)); // [scale_inv, zp]
    };

    auto key_by_channel_update_one_token =
        [&](vector<half, K_HEAD_SIZE> key_vec, uchar* out_u8, uint block_k_base_offset_bytes, uint token_pos) {
            for (uint g = 0; g < GROUP_NUM; ++g) {
                const uint h_beg = g * GROUP_SIZE;
                vector<half, GROUP_SIZE> grp_new;
                #pragma unroll
                for (uint k = 0; k < GROUP_SIZE; ++k)
                    grp_new(k) = key_vec(h_beg + k);

                half old_scale_inv = half(1.0);
                half old_zp = half(0.0);
                if (token_pos > 0) {
                    half* comp_old = comp_ptr_of_group(out_u8, block_k_base_offset_bytes, g);
                    old_scale_inv = comp_old[0];
                    old_zp = comp_old[1];
                }

                half vmax_h = half(-65504.0f);
                half vmin_h = half(65504.0f);

                if (token_pos > 0) {
                    #pragma unroll
                    for (uint t = 0; t < token_pos; ++t) {
                        const uint row_base = block_k_base_offset_bytes + t * K_HEAD_SIZE + h_beg;
                        vector<uint32_t, GROUP_SIZE / 4> u32 =
                            cm_ptr_load<uint32_t, GROUP_SIZE / 4>((uint32_t*)(out_u8 + row_base), /*Offset=*/0);
                        vector<uchar, GROUP_SIZE> u8_grp = u32.format<uchar>();

                        vector<half, GROUP_SIZE> u8_h = (vector<half, GROUP_SIZE>)u8_grp;
                        vector<half, GROUP_SIZE> v_dq = cm_mul<half>(u8_h - old_zp, old_scale_inv);

                        half vmax_t = cm_reduced_max<half>(v_dq);
                        half vmin_t = cm_reduced_min<half>(v_dq);
                        vmax_h = (vmax_t > vmax_h) ? vmax_t : vmax_h;
                        vmin_h = (vmin_t < vmin_h) ? vmin_t : vmin_h;
                    }
                }

                half vmax_t = cm_reduced_max<half>(grp_new);
                half vmin_t = cm_reduced_min<half>(grp_new);
                vmax_h = (vmax_t > vmax_h) ? vmax_t : vmax_h;
                vmin_h = (vmin_t < vmin_h) ? vmin_t : vmin_h;

                half scale_val, zp_val, scale_inv;
                compute_group_comp_from_minmax(vmin_h, vmax_h, scale_val, zp_val, scale_inv);

                if (token_pos > 0) {
                    #pragma unroll
                    for (uint t = 0; t < token_pos; ++t) {
                        const uint row_base = block_k_base_offset_bytes + t * K_HEAD_SIZE + h_beg;

                        vector<uint32_t, GROUP_SIZE / 4> u32_old =
                            cm_ptr_load<uint32_t, GROUP_SIZE / 4>((uint32_t*)(out_u8 + row_base), /*Offset=*/0);
                        vector<uchar, GROUP_SIZE> u8_old = u32_old.format<uchar>();

                        vector<half, GROUP_SIZE> u8_h = (vector<half, GROUP_SIZE>)u8_old;
                        vector<half, GROUP_SIZE> v_dq = cm_mul<half>(u8_h - old_zp, old_scale_inv);
                        vector<half, GROUP_SIZE> v_q = cm_mul<half>(v_dq, scale_val) + zp_val;

                        vector<uchar, GROUP_SIZE> u8_new = cm_rnde<uchar, GROUP_SIZE>(v_q);
                        cm_ptr_store<uint32_t, GROUP_SIZE / 4>((uint32_t*)(out_u8 + row_base),
                                                               /*Offset=*/0,
                                                               u8_new.format<uint32_t>());
                    }
                }

                const uint row_base = block_k_base_offset_bytes + token_pos * K_HEAD_SIZE + h_beg;
                vector<half, GROUP_SIZE> v_q = cm_mul<half>(grp_new, scale_val) + zp_val;
                vector<uchar, GROUP_SIZE> u8_new = cm_rnde<uchar, GROUP_SIZE>(v_q);
                cm_ptr_store<uint32_t, GROUP_SIZE / 4>((uint32_t*)(out_u8 + row_base),
                                                       /*Offset=*/0,
                                                       u8_new.format<uint32_t>());

                {
                    half* comp_new = comp_ptr_of_group(out_u8, block_k_base_offset_bytes, g);
                    comp_new[0] = scale_inv;
                    comp_new[1] = zp_val;
                }
            }
        };

        auto key_by_channel_prefill_window = [&](const half* key_in_base,
                                                uint key_in_offset_half,
                                                uint key_in_stride_half,
                                                uchar* out_u8,
                                                uint block_k_base_offset_bytes,
                                                uint token_start_pos,
                                                uint tokens_num) {
            // Process a window of tokens in one shot:
            // Pass 1: compute global min/max across the window
            // Pass 2: quantize using the same comp and store
            auto do_window = [&](uint win_tokens, uint dst_row_offset) {
                CM_STATIC_ERROR((GROUP_SIZE % 4) == 0, "CM:e:GROUP_SIZE must be multiple of 4 for u32 loads/stores");

                for (uint g = 0; g < GROUP_NUM; ++g) {
                    const uint h_beg = g * GROUP_SIZE;

                    // --- Pass 1: compute (vmin, vmax) over the entire window using cm_reduced_* ---
                    half vmin_h = half(65504.0f);
                    half vmax_h = half(-65504.0f);

                    // Merge group offset into in_off for easier pointer arithmetic
                    uint in_off = key_in_offset_half + h_beg;
                    #pragma unroll
                    for (uint t = 0; t < win_tokens; ++t) {
                        vector<half, GROUP_SIZE> v_grp;
                        v_grp.format<int>() = cm_ptr_load<int, (GROUP_SIZE * sizeof(half)) / 4>(
                            (int*)key_in_base,
                            (in_off + t * key_in_stride_half) * (int)sizeof(half));

                        half row_max = cm_reduced_max<half>(v_grp);
                        half row_min = cm_reduced_min<half>(v_grp);
                        vmax_h = (row_max > vmax_h) ? row_max : vmax_h;
                        vmin_h = (row_min < vmin_h) ? row_min : vmin_h;
                    }

                    // Compute quantization parameters:
                    // scale_val / zp_val are used for quantization
                    // scale_inv will be stored for dequant
                    half scale_val, zp_val, scale_inv;
                    compute_group_comp_from_minmax(vmin_h, vmax_h, scale_val, zp_val, scale_inv);

                    // --- Pass 2: apply quantization and store back as packed u32 blocks ---
                    in_off = key_in_offset_half + h_beg;
                    #pragma unroll
                    for (uint t = 0; t < win_tokens; ++t) {
                        const uint row_base = block_k_base_offset_bytes + (dst_row_offset + t) * K_HEAD_SIZE + h_beg;

                        vector<half, GROUP_SIZE> v_grp;
                        v_grp.format<int>() = cm_ptr_load<int, (GROUP_SIZE * sizeof(half)) / 4>(
                            (int*)key_in_base,
                            (in_off + t * key_in_stride_half) * (int)sizeof(half));

                        // acc = v_grp * scale_val + zp_val
                        vector<half, GROUP_SIZE> acc = cm_mul<half>(v_grp, scale_val) + zp_val;
                        vector<uchar, GROUP_SIZE> q = cm_rnde<uchar, GROUP_SIZE>(acc);

                        // Store GROUP_SIZE values in one vector store (packed as uint32_t)
                        cm_ptr_store<uint32_t, GROUP_SIZE / 4>((uint32_t*)(out_u8 + row_base),
                                                            /*Offset=*/0,
                                                            q.format<uint32_t>());
                    }

                    // Store comp parameters: [scale_inv, zp]
                    half* comp_ptr = comp_ptr_of_group(out_u8, block_k_base_offset_bytes, g);
                    comp_ptr[0] = scale_inv;
                    comp_ptr[1] = zp_val;
                }
            };

            if (token_start_pos == 0) {
                // Block boundary case: compute window stats once and quantize the whole window
                do_window(tokens_num, /*dst_row_offset=*/0);
            } else {
                // Mixed mode: use per-token incremental update for historical + new tokens
                uint in_off = key_in_offset_half;
                #pragma unroll
                for (uint j = 0; j < tokens_num; ++j, in_off += key_in_stride_half) {
                    vector<half, K_HEAD_SIZE> key_vec;
                    key_vec.format<int>() =
                        cm_ptr_load<int, (K_HEAD_SIZE * sizeof(half)) / 4>((int*)key_in_base, in_off * (int)sizeof(half));

                    key_by_channel_update_one_token(key_vec, out_u8, block_k_base_offset_bytes, token_start_pos + j);
                }
            }
        };

#endif // KV_CACHE_COMPRESSION_PER_CHANNEL
    if (!is_prefill_stage) {
        // ---- decode (2nd+ tokens) ----
        if (token_gid >= (uint)subsequence_begins[batch_size_in_sequences]) return;

        uint subseq_idx = 0;
        for (uint i = 0; i < batch_size_in_sequences; ++i) {
            if (token_gid >= (uint)subsequence_begins[i] && token_gid < (uint)subsequence_begins[i+1]) { subseq_idx = i; break; }
        }
        const uint subseq_beg = (uint)subsequence_begins[subseq_idx];
        const uint past_len   = (uint)past_lens[subseq_idx];

        const uint block_idx_in_seq = (past_len + token_gid - subseq_beg) / PAGED_ATTENTION_BLOCK_SIZE;
        const uint token_pos_in_blk = (past_len + token_gid - subseq_beg) % PAGED_ATTENTION_BLOCK_SIZE;
        const uint blk_offset = (uint)block_indices_begins[subseq_idx] + block_idx_in_seq;
        const uint phys_block = (uint)block_indices[blk_offset];

        uint key_in_off_half   = token_gid * key_pitch   + head_idx * K_HEAD_SIZE;
        uint value_in_off_half = token_gid * value_pitch + head_idx * V_HEAD_SIZE;

        // ---- Key ----
#ifdef KV_CACHE_COMPRESSION_PER_CHANNEL
        constexpr uint DATA_BYTES_PER_BLOCK   = PAGED_ATTENTION_BLOCK_SIZE * K_HEAD_SIZE;
        constexpr uint STRIDE_BYTES_PER_HEAD  = DATA_BYTES_PER_BLOCK + COMP_BYTES_TOTAL;
        uint block_k_base_offset_bytes = (phys_block * KV_HEADS_NUM + head_idx) * STRIDE_BYTES_PER_HEAD;

        if (token_pos_in_blk == 0) {
            // Block boundary: compute + quantize the remaining new tokens in this block window
            const uint new_total   = (uint)subsequence_begins[subseq_idx+1] - (uint)subsequence_begins[subseq_idx];
            const uint new_offset  = token_gid;
            const uint remain_new  = (new_total > new_offset) ? (new_total - new_offset) : 0;
            const uint win_tokens  = (remain_new > PAGED_ATTENTION_BLOCK_SIZE) ? PAGED_ATTENTION_BLOCK_SIZE : remain_new;

            if (win_tokens > 0) {
                const uint block_first_token_gid = token_gid - token_pos_in_blk;
                const uint block_key_in_off_half = block_first_token_gid * key_pitch + head_idx * K_HEAD_SIZE;
                key_by_channel_prefill_window(
                    key, block_key_in_off_half, key_pitch,
                    (uchar*)key_cache, block_k_base_offset_bytes,
                    /*token_start_pos=*/0, /*tokens_num=*/win_tokens);
            }
        }
#else
        const uint block_k_base_offset =
            (phys_block * KV_HEADS_NUM + head_idx) * ADJUSTED_K_HEAD_SIZE * PAGED_ATTENTION_BLOCK_SIZE;
        const uint key_out_off_half = block_k_base_offset + token_pos_in_blk * K_HEAD_SIZE;

        vector<half, K_HEAD_SIZE> key_data;
        key_data.format<int>() = cm_ptr_load<int, K_HEAD_SIZE/2>((int*)key, key_in_off_half * (int)sizeof(half));
    #if KV_CACHE_COMPRESSION_PER_TOKEN
        quantize_and_store_per_token(key_data, (uchar*)key_cache, block_k_base_offset, token_pos_in_blk);
    #else
        cm_ptr_store<int, K_HEAD_SIZE/2>((int*)key_cache, key_out_off_half * (int)sizeof(half), key_data.format<int>());
    #endif
#endif

        // ---- Value ---- (Use per-token compression or no compression as configured)
        const uint block_v_base_offset =
            (phys_block * KV_HEADS_NUM + head_idx) * ADJUSTED_V_HEAD_SIZE * PAGED_ATTENTION_BLOCK_SIZE;
        const uint value_out_off_half = block_v_base_offset + token_pos_in_blk * V_HEAD_SIZE;

        vector<half, V_HEAD_SIZE> value_vec;
        value_vec.format<int>() = cm_ptr_load<int, V_HEAD_SIZE/2>((int*)value, value_in_off_half * (int)sizeof(half));
    #if KV_CACHE_COMPRESSION_PER_TOKEN || KV_CACHE_COMPRESSION_PER_CHANNEL
        quantize_and_store_per_token(value_vec, (uchar*)value_cache, block_v_base_offset, token_pos_in_blk);
    #else
        cm_ptr_store<int, V_HEAD_SIZE/2>((int*)value_cache, value_out_off_half * (int)sizeof(half), value_vec.format<int>());
    #endif
    } else {
        // --------------------- First prefill (batched blocks) ---------------------
        const uint phys_block_idx = cm_global_id(0);

        const uint subseq_idx = (uint)gws_seq_indexes_correspondence[phys_block_idx];
        const uint subseq_beg = (uint)subsequence_begins[subseq_idx];
        const uint past_len   = (uint)past_lens[subseq_idx];

        const uint block_start_pos = (uint)blocked_indexes_start[phys_block_idx];
        const uint block_end_pos   = (uint)blocked_indexes_end[phys_block_idx];
        const uint tokens_num      = block_end_pos - block_start_pos;

        const uint token_start_key = (past_len + block_start_pos - subseq_beg) % PAGED_ATTENTION_BLOCK_SIZE;
        const uint token_start_val = token_start_key;

        const uint seq_block_idx = (past_len + block_start_pos - subseq_beg) / PAGED_ATTENTION_BLOCK_SIZE;
        const uint blk_offset    = (uint)block_indices_begins[subseq_idx] + seq_block_idx;
        const uint phys_block    = (uint)block_indices[blk_offset];

        uint key_in_off_half   = block_start_pos * key_pitch   + head_idx * K_HEAD_SIZE;
        uint value_in_off_half = block_start_pos * value_pitch + head_idx * V_HEAD_SIZE;

#ifdef KV_CACHE_COMPRESSION_PER_CHANNEL
        constexpr uint DATA_BYTES_PER_BLOCK  = PAGED_ATTENTION_BLOCK_SIZE * K_HEAD_SIZE;
        constexpr uint STRIDE_BYTES_PER_HEAD = DATA_BYTES_PER_BLOCK + COMP_BYTES_TOTAL;
        uint block_k_base_offset_bytes = (phys_block * KV_HEADS_NUM + head_idx) * STRIDE_BYTES_PER_HEAD;

        if (token_start_key == 0) {
            // Starting from offset 0: full-block or partial-block both use one-shot window stats + quant
            key_by_channel_prefill_window(
                key, key_in_off_half, key_pitch,
                (uchar*)key_cache, block_k_base_offset_bytes,
                /*token_start_pos=*/0, /*tokens_num=*/tokens_num);
        } else {
            // With offset: mixed mode
            key_by_channel_prefill_window(
                key, key_in_off_half, key_pitch,
                (uchar*)key_cache, block_k_base_offset_bytes,
                token_start_key, tokens_num);
        }
#else
        const uint block_k_base_offset = (phys_block * KV_HEADS_NUM + head_idx) * ADJUSTED_K_HEAD_SIZE * PAGED_ATTENTION_BLOCK_SIZE;
        const uint key_out_off_half    = block_k_base_offset + token_start_key;

        if (tokens_num == PAGED_ATTENTION_BLOCK_SIZE) {
            for (uint t = 0; t < PAGED_ATTENTION_BLOCK_SIZE; ++t) {
                vector<half, K_HEAD_SIZE> key_vec;
                key_vec.format<int>() =
                    cm_ptr_load<int, K_HEAD_SIZE/2>((int*)key, (key_in_off_half + t*key_pitch) * (int)sizeof(half));
        #if KV_CACHE_COMPRESSION_PER_TOKEN
                quantize_and_store_per_token(key_vec, (uchar*)key_cache, block_k_base_offset, t);
        #else
                cm_ptr_store<int, K_HEAD_SIZE/2>(
                    (int*)key_cache,
                    (key_out_off_half + t * K_HEAD_SIZE) * (int)sizeof(half),
                    key_vec.format<int>());
        #endif
            }
        } else {
            for (uint j = 0; j < tokens_num; ++j) {
                vector<half, K_HEAD_SIZE> key_vec;
                key_vec.format<int>() =
                    cm_ptr_load<int, K_HEAD_SIZE/2>((int*)key, (key_in_off_half + j*key_pitch) * (int)sizeof(half));
        #if KV_CACHE_COMPRESSION_PER_TOKEN
                quantize_and_store_per_token(key_vec, (uchar*)key_cache, block_k_base_offset, token_start_key + j);
        #else
                cm_ptr_store<int, K_HEAD_SIZE/2>(
                    (int*)key_cache,
                    (block_k_base_offset + (token_start_key + j) * K_HEAD_SIZE) * (int)sizeof(half),
                    key_vec.format<int>());
        #endif
            }
        }
#endif // KV_CACHE_COMPRESSION_PER_CHANNEL

        // ---- Value prefill (same per-token path) ----
        const uint block_v_base_offset = (phys_block * KV_HEADS_NUM + head_idx) * ADJUSTED_V_HEAD_SIZE * PAGED_ATTENTION_BLOCK_SIZE;
        const uint value_out_off       = block_v_base_offset + token_start_val * V_HEAD_SIZE;

        if (tokens_num == PAGED_ATTENTION_BLOCK_SIZE) {
            for (uint t = 0; t < PAGED_ATTENTION_BLOCK_SIZE; ++t) {
                vector<half, V_HEAD_SIZE> val_vec;
                val_vec.format<int>() =
                    cm_ptr_load<int, V_HEAD_SIZE/2>((int*)value, (value_in_off_half + t*value_pitch) * (int)sizeof(half));
        #if KV_CACHE_COMPRESSION_PER_TOKEN || KV_CACHE_COMPRESSION_PER_CHANNEL
                quantize_and_store_per_token(val_vec, (uchar*)value_cache, block_v_base_offset, t);
        #else
                cm_ptr_store<int, V_HEAD_SIZE/2>(
                    (int*)value_cache,
                    (value_out_off + t * V_HEAD_SIZE) * (int)sizeof(half),
                    val_vec.format<int>());
        #endif
            }
        } else {
            for (uint j = 0; j < tokens_num; ++j) {
                vector<half, V_HEAD_SIZE> val_vec;
                val_vec.format<int>() =
                    cm_ptr_load<int, V_HEAD_SIZE/2>((int*)value, (value_in_off_half + j*value_pitch) * (int)sizeof(half));
        #if KV_CACHE_COMPRESSION_PER_TOKEN || KV_CACHE_COMPRESSION_PER_CHANNEL
                quantize_and_store_per_token(val_vec, (uchar*)value_cache, block_v_base_offset, token_start_val + j);
        #else
                cm_ptr_store<int, V_HEAD_SIZE/2>(
                    (int*)value_cache,
                    (block_v_base_offset + (token_start_val + j) * V_HEAD_SIZE) * (int)sizeof(half),
                    val_vec.format<int>());
        #endif
            }
        }
    }
}
