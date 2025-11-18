
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

#ifndef ATTR
#define ATTR [[type("svmptr_t")]]
#define ATTR_BUF [[type("buffer_t")]]
#endif

#ifndef ENABLE_KV_CACHE_INCREMENTAL_DECODE
#define ENABLE_KV_CACHE_INCREMENTAL_DECODE 1
#endif

#ifndef USE_SLM
#define USE_SLM 0
#endif

#ifndef GROUP_NUM
#define GROUP_NUM 1
#endif

#if defined(KV_CACHE_COMPRESSION_PER_TOKEN) && defined(KV_CACHE_COMPRESSION_PER_CHANNEL)
#error "Define either KV_CACHE_COMPRESSION_PER_TOKEN or KV_CACHE_COMPRESSION_PER_CHANNEL, not both."
#endif

#ifdef KV_CACHE_COMPRESSION_PER_CHANNEL
    #define COMP_BYTES_PER_GROUP                (sizeof(half) * 2)  // [scale_inv, zp]
    #define COMP_BYTES_TOTAL                    (COMP_BYTES_PER_GROUP * GROUP_NUM)
    #define ADJUSTED_PAGED_ATTENTION_BLOCK_SIZE (PAGED_ATTENTION_BLOCK_SIZE + COMP_BYTES_TOTAL)
#endif

#if USE_SLM
    #define SLM_RESERVE_BYTES (PAGED_ATTENTION_BLOCK_SIZE * K_HEAD_SIZE * sizeof(half))
    #define GENX_ENTRY _GENX_MAIN_WITH_SLM(SLM_RESERVE_BYTES)
#else
  #define GENX_ENTRY _GENX_MAIN_
#endif

#if KV_CACHE_COMPRESSION_PER_CHANNEL
CM_INLINE bool is_decode_block_leader(uint block_idx_in_seq, uint token_pos_in_blk, uint past_len) {
    const uint B  = PAGED_ATTENTION_BLOCK_SIZE;
    const uint sb = past_len / B;
    const uint sp = past_len % B;
    if (block_idx_in_seq == sb) return token_pos_in_blk == sp;
    return token_pos_in_blk == 0;
}

CM_INLINE void compute_group_comp(half vmin_h, half vmax_h, half& scale_inv_out, half& zp_out) {
    if (vmax_h == vmin_h) {
        scale_inv_out = half(0.0f);
        zp_out        = vmin_h;
        return;
    }
    const half  range_h = vmax_h - vmin_h;
    const float scale_f = 255.0f / float(range_h);
    const float zp_f    = -float(vmin_h) * scale_f;
    scale_inv_out = half(1.0f / scale_f);
    zp_out        = half(zp_f);
}
#endif

#if KV_CACHE_COMPRESSION_PER_TOKEN || KV_CACHE_COMPRESSION_PER_CHANNEL
CM_INLINE void quantize_and_store_per_token(vector<half, K_HEAD_SIZE> data,
                                            uchar* out,
                                            uint out_offset,   // per-head base in bytes
                                            uint token_pos) {
    const uint data_bytes   = K_HEAD_SIZE * PAGED_ATTENTION_BLOCK_SIZE;
    const uint scale_offset = out_offset + data_bytes + token_pos * sizeof(half);

    const half max_val = cm_reduced_max<half>(data);
    const half min_val = cm_reduced_min<half>(data);

    half scale_val = half(0.0f);
    half zp_val    = half(0.0f);
    if (max_val == min_val) {
        scale_val = half(0.0f);
        zp_val    = max_val;
    } else {
        scale_val = half(255.0) / (max_val - min_val);
        zp_val    = (half(0.0) - min_val) * scale_val;
    }

    vector<half,  K_HEAD_SIZE> acc = cm_mul<half>(data, scale_val) + zp_val;
    vector<uchar, K_HEAD_SIZE> q   = cm_rnde<uchar, K_HEAD_SIZE>(acc);
    cm_ptr_store<uint32_t, K_HEAD_SIZE / 4>(
        (uint32_t*)(out + out_offset + token_pos * K_HEAD_SIZE),
        0,
        q.format<uint32_t>());

    half* out_scale_zp = (half*)(out + scale_offset);
    out_scale_zp[0] = (max_val - min_val) / half(255.0);  // scale_inv[token_pos]
    out_scale_zp[PAGED_ATTENTION_BLOCK_SIZE] = zp_val;    // zp[token_pos]
}
#endif

// ==================== Kernel entry ====================
extern "C" GENX_ENTRY
void pa_kv_cache_update(const half* key [[type("svmptr_t")]],
                        const half* value [[type("svmptr_t")]],
                        const int32_t* past_lens [[type("svmptr_t")]],
                        const int32_t* block_indices [[type("svmptr_t")]],
                        const int32_t* block_indices_begins [[type("svmptr_t")]],
                        const int32_t* subsequence_begins [[type("svmptr_t")]],
#if KV_CACHE_COMPRESSION_PER_TOKEN || KV_CACHE_COMPRESSION_PER_CHANNEL
                        uint8_t* key_cache [[type("svmptr_t")]],
                        uint8_t* value_cache [[type("svmptr_t")]],
#else
                        half* key_cache [[type("svmptr_t")]],
                        half* value_cache [[type("svmptr_t")]],
#endif
                        uint32_t key_pitch,
                        uint32_t value_pitch,
                        uint32_t batch_size_in_sequences,
                        // prefill only
                        const int32_t* blocked_indexes_start [[type("svmptr_t")]],
                        const int32_t* blocked_indexes_end [[type("svmptr_t")]],
                        const int32_t* gws_seq_indexes_correspondence [[type("svmptr_t")]],
                        const int is_prefill_stage
                        ) {

#if USE_SLM
    cm_slm_init(SLM_RESERVE_BYTES);
#endif

    const uint head_idx  = cm_group_id(1);
    const uint token_gid = cm_global_id(2);

#ifdef KV_CACHE_COMPRESSION_PER_CHANNEL
    static_assert(K_HEAD_SIZE % GROUP_NUM == 0, "K_HEAD_SIZE must be divisible by GROUP_NUM");
    constexpr uint GROUP_SIZE = K_HEAD_SIZE / GROUP_NUM;
    static_assert((GROUP_SIZE % 4) == 0, "GROUP_SIZE must be multiple of 4 for aligned stores");

    auto comp_ptr_of_group = [&](uchar* base_u8, uint block_k_base_offset_bytes, uint g) -> half* {
        const uint comp_base = block_k_base_offset_bytes + K_HEAD_SIZE * PAGED_ATTENTION_BLOCK_SIZE;
        return reinterpret_cast<half*>(base_u8 + comp_base + g * 2 * sizeof(half));  // [scale_inv, zp]
    };

    auto key_by_channel_prefill_window = [&](const half* key_in_base,
                                             uint key_in_offset_half,
                                             uint key_in_stride_half,
                                             uchar* out_u8,
                                             uint block_k_base_offset_bytes,
                                             uint token_start_pos,
                                             uint tokens_num) {
        if (token_start_pos != 0) {
            uint in_off = key_in_offset_half;
            #pragma unroll
            for (uint j = 0; j < tokens_num; ++j, in_off += key_in_stride_half) {
                vector<half, K_HEAD_SIZE> key_vec;
                cm_svm_block_read((svmptr_t)((const uchar*)key_in_base + in_off * sizeof(half)), key_vec);
                #pragma unroll
                for (uint g = 0; g < GROUP_NUM; ++g) {
                    const uint h_beg = g * GROUP_SIZE;
                    auto grp = key_vec.template select<GROUP_SIZE, 1>(h_beg);
                    half vmin_h = cm_reduced_min<half>(grp);
                    half vmax_h = cm_reduced_max<half>(grp);
                    half s_inv, zp;
                    compute_group_comp(vmin_h, vmax_h, s_inv, zp);

                    const uint row = block_k_base_offset_bytes + (token_start_pos + j) * K_HEAD_SIZE + h_beg;
                    vector<half, GROUP_SIZE> acc =
                        cm_mul<half>(grp, half(1.0f / (float)s_inv)) + zp;
                    vector<uchar, GROUP_SIZE> q = cm_rnde<uchar, GROUP_SIZE>(acc);
                    cm_ptr_store<uint32_t, GROUP_SIZE / 4>(
                        (uint32_t*)((uchar*)out_u8 + row), 0, q.format<uint32_t>());

                    half* comp = comp_ptr_of_group(out_u8, block_k_base_offset_bytes, g);
                    comp[0] = s_inv;
                    comp[1] = zp;
                }
            }
            return;
        }

        const uint rows = (tokens_num > PAGED_ATTENTION_BLOCK_SIZE)
                            ? PAGED_ATTENTION_BLOCK_SIZE
                            : tokens_num;

#if USE_SLM
        const uint slm_bytes = rows * K_HEAD_SIZE * sizeof(half);
        const uint slm_base  = cm_slm_alloc(slm_bytes);

        // Pass-0: DDR -> SLM
        for (uint t = 0, in_off = key_in_offset_half; t < rows; ++t, in_off += key_in_stride_half) {
            vector<half, K_HEAD_SIZE> row_h;
            cm_svm_block_read((svmptr_t)((const uchar*)key_in_base + in_off * sizeof(half)), row_h);
            cm_slm_block_write(0, slm_base + t * K_HEAD_SIZE * sizeof(half), row_h);
        }
#endif

        // Pass-1: get min/max
        vector<half, GROUP_NUM> vmin_g(half( 65504.0f));
        vector<half, GROUP_NUM> vmax_g(half(-65504.0f));
        for (uint t = 0; t < rows; ++t) {
            vector<half, K_HEAD_SIZE> row_h;
        #if USE_SLM
            cm_slm_block_read(0, slm_base + t * K_HEAD_SIZE * sizeof(half), row_h);
        #else
            const uint in_off = key_in_offset_half + t * key_in_stride_half;
            cm_svm_block_read((svmptr_t)((const uchar*)key_in_base + in_off * sizeof(half)), row_h);
        #endif
            #pragma unroll
            for (uint g = 0; g < GROUP_NUM; ++g) {
                const uint h_beg = g * GROUP_SIZE;
                auto grp = row_h.template select<GROUP_SIZE, 1>(h_beg);
                half rmax = cm_reduced_max<half>(grp);
                half rmin = cm_reduced_min<half>(grp);
                vmax_g(g) = (rmax > vmax_g(g)) ? rmax : vmax_g(g);
                vmin_g(g) = (rmin < vmin_g(g)) ? rmin : vmin_g(g);
            }
        }
        if (rows < PAGED_ATTENTION_BLOCK_SIZE) {
            #pragma unroll
            for (uint g = 0; g < GROUP_NUM; ++g) {
                if (half(0.0f) < vmin_g(g)) vmin_g(g) = half(0.0f);
                if (half(0.0f) > vmax_g(g)) vmax_g(g) = half(0.0f);
            }
        }

        // Pass-2: comp
        vector<half, GROUP_NUM> scale_inv_g, zp_g;
        #pragma unroll
        for (uint g = 0; g < GROUP_NUM; ++g) {
            half s_inv, zp;
            compute_group_comp(vmin_g(g), vmax_g(g), s_inv, zp);
            scale_inv_g(g) = s_inv;
            zp_g(g)        = zp;
        }

        // Pass-3: write block
        for (uint t = 0; t < PAGED_ATTENTION_BLOCK_SIZE; ++t) {
            const bool is_pad = (t >= rows);
            vector<half, K_HEAD_SIZE> row_h;
            if (!is_pad) {
            #if USE_SLM
                cm_slm_block_read(0, slm_base + t * K_HEAD_SIZE * sizeof(half), row_h);
            #else
                const uint in_off = key_in_offset_half + t * key_in_stride_half;
                cm_svm_block_read((svmptr_t)((const uchar*)key_in_base + in_off * sizeof(half)), row_h);
            #endif
            } else {
                row_h = vector<half, K_HEAD_SIZE>(half(0.0f));
            }
            #pragma unroll
            for (uint g = 0; g < GROUP_NUM; ++g) {
                const uint h_beg = g * GROUP_SIZE;
                auto grp = row_h.template select<GROUP_SIZE, 1>(h_beg);
                vector<half, GROUP_SIZE> acc =
                    cm_mul<half>(grp, half(1.0f / (float)scale_inv_g(g))) + zp_g(g);
                vector<uchar, GROUP_SIZE> q = cm_rnde<uchar, GROUP_SIZE>(acc);
                const uint row_base = block_k_base_offset_bytes + t * K_HEAD_SIZE + h_beg;
                cm_ptr_store<uint32_t, GROUP_SIZE / 4>(
                    (uint32_t*)((uchar*)out_u8 + row_base), 0, q.format<uint32_t>());
            }
        }

        // Pass-4: write comp
        #pragma unroll
        for (uint g = 0; g < GROUP_NUM; ++g) {
            half* comp_ptr = comp_ptr_of_group(out_u8, block_k_base_offset_bytes, g);
            comp_ptr[0] = scale_inv_g(g);
            comp_ptr[1] = zp_g(g);
        }
    };
#endif // KV_CACHE_COMPRESSION_PER_CHANNEL

    // ===================== Decode / Prefill =====================
    if (!is_prefill_stage) {
        if (token_gid >= (uint)subsequence_begins[batch_size_in_sequences]) return;

        uint subseq_idx = 0;
        for (uint i = 0; i < batch_size_in_sequences; ++i) {
            if (token_gid >= (uint)subsequence_begins[i] &&
                token_gid <  (uint)subsequence_begins[i + 1]) {
                subseq_idx = i;
                break;
            }
        }
        const uint subseq_beg = (uint)subsequence_begins[subseq_idx];
        const uint past_len   = (uint)past_lens[subseq_idx];
        if (token_gid < past_len) return;

        const uint seq_pos          = token_gid - subseq_beg;
        const uint block_idx_in_seq = seq_pos / PAGED_ATTENTION_BLOCK_SIZE;
        const uint token_pos_in_blk = seq_pos % PAGED_ATTENTION_BLOCK_SIZE;
        const uint blk_offset       = (uint)block_indices_begins[subseq_idx] + block_idx_in_seq;
        const uint phys_block       = (uint)block_indices[blk_offset];

        const uint key_in_off_half   = token_gid * key_pitch   + head_idx * K_HEAD_SIZE;
        const uint value_in_off_half = token_gid * value_pitch + head_idx * V_HEAD_SIZE;

#ifdef KV_CACHE_COMPRESSION_PER_CHANNEL
        constexpr uint DATA_BYTES_PER_BLOCK  = PAGED_ATTENTION_BLOCK_SIZE * K_HEAD_SIZE;
        constexpr uint STRIDE_BYTES_PER_HEAD = DATA_BYTES_PER_BLOCK + COMP_BYTES_TOTAL;

        const uint subseq_len_total = (uint)(subsequence_begins[subseq_idx + 1] - subseq_beg);
        const uint new_tokens_total = (subseq_len_total > past_len)
                                        ? (subseq_len_total - past_len)
                                        : 0;

        const bool partial_hist_block = ((past_len % PAGED_ATTENTION_BLOCK_SIZE) != 0) && (block_idx_in_seq == (past_len / PAGED_ATTENTION_BLOCK_SIZE));
        if (new_tokens_total > 0 &&
            is_decode_block_leader(block_idx_in_seq, token_pos_in_blk, past_len)) {
            // ================= Incremental decode attempt (per new token line) =================
#if ENABLE_KV_CACHE_INCREMENTAL_DECODE
            if (!partial_hist_block) {
                const uint base_bytes_try = (phys_block * KV_HEADS_NUM + head_idx) * STRIDE_BYTES_PER_HEAD;
                const uint comp_bytes_try = base_bytes_try + K_HEAD_SIZE * PAGED_ATTENTION_BLOCK_SIZE;
                const uint row_bytes_try  = base_bytes_try + token_pos_in_blk * K_HEAD_SIZE;

                // Read new half row
                vector<half, K_HEAD_SIZE> new_row_half;
                cm_svm_block_read((svmptr_t)((const uchar*)key + key_in_off_half * sizeof(half)), new_row_half);

                bool need_full_requant = false;
                vector<char, GROUP_NUM> group_expand(0);
                #pragma unroll
                for (uint g = 0; g < GROUP_NUM; ++g) {
                    half* comp_ptr = (half*)((uchar*)key_cache + comp_bytes_try + g * 2 * sizeof(half));
                    const half s_inv = comp_ptr[0];
                    const half zp    = comp_ptr[1];
                    // Recover theoretical range
                    const half vmin_old = -zp * s_inv;
                    const half vmax_old = vmin_old + half(255.0f) * s_inv;
                    const uint h_beg = g * GROUP_SIZE;
                    auto grp = new_row_half.select<GROUP_SIZE,1>(h_beg);
                    half rmin_new = cm_reduced_min<half>(grp);
                    half rmax_new = cm_reduced_max<half>(grp);
                    // Degenerate case: s_inv == 0 means constant row previously.
                    if ( (rmin_new < vmin_old) || (rmax_new > vmax_old) || (s_inv == half(0.0f) && (rmin_new != vmin_old || rmax_new != vmax_old)) ) {
                        group_expand(g) = 1;
                        need_full_requant = true;
                    }
                }
                if (!need_full_requant) {
                    // Quantize & store only this new line using old comp
                    #pragma unroll
                    for (uint g = 0; g < GROUP_NUM; ++g) {
                        half* comp_ptr = (half*)((uchar*)key_cache + comp_bytes_try + g * 2 * sizeof(half));
                        const half s_inv = comp_ptr[0];
                        const half zp    = comp_ptr[1];
                        const uint h_beg = g * GROUP_SIZE;
                        auto grp = new_row_half.select<GROUP_SIZE,1>(h_beg);
                        vector<half, GROUP_SIZE> acc = cm_mul<half>(grp, half(1.0f / (float)s_inv)) + zp; // v * scale + zp with scale = 1/s_inv
                        vector<uchar, GROUP_SIZE> q  = cm_rnde<uchar, GROUP_SIZE>(acc);
                        cm_ptr_store<uint32_t, GROUP_SIZE/4>((uint32_t*)((uchar*)key_cache + row_bytes_try + h_beg), 0, q.format<uint32_t>());
                    }
                    // Skip full-block recompute path
                    goto BY_CHANNEL_VALUE_PATH;
                }
            }
#endif // ENABLE_KV_CACHE_INCREMENTAL_DECODE
            const uint total_tokens = past_len + new_tokens_total;

            const uint blk = block_idx_in_seq;
            const uint blk_offset_iter = (uint)block_indices_begins[subseq_idx] + blk;
            const uint phys_block_iter = (uint)block_indices[blk_offset_iter];
            const uint base_bytes      = (phys_block_iter * KV_HEADS_NUM + head_idx) * STRIDE_BYTES_PER_HEAD;
            const uint token_bytes     = base_bytes;
            const uint comp_bytes      = base_bytes + K_HEAD_SIZE * PAGED_ATTENTION_BLOCK_SIZE;

            const uint block_first_tok_global =
                blk * PAGED_ATTENTION_BLOCK_SIZE;
            const uint block_tok_end_logical =
                cm_min<uint>(block_first_tok_global + PAGED_ATTENTION_BLOCK_SIZE, total_tokens);
            const uint block_tok_count = block_tok_end_logical - block_first_tok_global;
            const bool have_prev       = (block_first_tok_global < past_len);

            vector<half, GROUP_NUM> vmin_g(half( 65504.0f));
            vector<half, GROUP_NUM> vmax_g(half(-65504.0f));
            for (uint t = 0; t < block_tok_count; ++t) {
                const uint global_tok_rel = block_first_tok_global + t;
                if (global_tok_rel < past_len && have_prev) {
                    vector<uint32_t, K_HEAD_SIZE / 4> u32 =
                        cm_ptr_load<uint32_t, K_HEAD_SIZE / 4>(
                            (uint32_t*)((uchar*)key_cache + token_bytes + t * K_HEAD_SIZE), 0);
                    vector<uchar, K_HEAD_SIZE> u8_row = u32.format<uchar>();
                    #pragma unroll
                    for (uint g = 0; g < GROUP_NUM; ++g) {
                        half* comp_old = (half*)((uchar*)key_cache +
                                                 comp_bytes + g * 2 * sizeof(half));
                        const half s_inv = comp_old[0];
                        const half zp    = comp_old[1];
                        const uint h_beg = g * GROUP_SIZE;
                        auto u8_grp = u8_row.template select<GROUP_SIZE, 1>(h_beg);
                        vector<half, GROUP_SIZE> u8_h = (vector<half, GROUP_SIZE>)u8_grp;
                        vector<half, GROUP_SIZE> v    = cm_mul<half>(u8_h - zp, s_inv);
                        half rmax = cm_reduced_max<half>(v);
                        half rmin = cm_reduced_min<half>(v);
                        vmax_g(g) = (rmax > vmax_g(g)) ? rmax : vmax_g(g);
                        vmin_g(g) = (rmin < vmin_g(g)) ? rmin : vmin_g(g);
                    }
                } else {
                    const uint t_gid_rel = subseq_beg + global_tok_rel;
                    const uint in_off_h  = t_gid_rel * key_pitch + head_idx * K_HEAD_SIZE;
                    vector<half, K_HEAD_SIZE> row_half;
                    cm_svm_block_read(
                        (svmptr_t)((const uchar*)key + in_off_h * sizeof(half)),
                        row_half);
                    #pragma unroll
                    for (uint g = 0; g < GROUP_NUM; ++g) {
                        const uint h_beg = g * GROUP_SIZE;
                        auto grp = row_half.template select<GROUP_SIZE, 1>(h_beg);
                        half rmax = cm_reduced_max<half>(grp);
                        half rmin = cm_reduced_min<half>(grp);
                        vmax_g(g) = (rmax > vmax_g(g)) ? rmax : vmax_g(g);
                        vmin_g(g) = (rmin < vmin_g(g)) ? rmin : vmin_g(g);
                    }
                }
            }
            if (block_tok_count < PAGED_ATTENTION_BLOCK_SIZE) {
                #pragma unroll
                for (uint g = 0; g < GROUP_NUM; ++g) {
                    if (half(0.0f) < vmin_g(g)) vmin_g(g) = half(0.0f);
                    if (half(0.0f) > vmax_g(g)) vmax_g(g) = half(0.0f);
                }
            }
            vector<half, GROUP_NUM> scale_inv_g, zp_g;
            #pragma unroll
            for (uint g = 0; g < GROUP_NUM; ++g) {
                half s_inv, zp;
                compute_group_comp(vmin_g(g), vmax_g(g), s_inv, zp);
                scale_inv_g(g) = s_inv;
                zp_g(g)        = zp;
            }

            for (uint t = 0; t < PAGED_ATTENTION_BLOCK_SIZE; ++t) {
                const bool is_pad = (t >= block_tok_count);
                vector<half, K_HEAD_SIZE> row_half;
                if (!is_pad) {
                    const uint global_tok_rel = block_first_tok_global + t;
                    const uint t_gid_rel      = subseq_beg + global_tok_rel;
                    const uint in_off_h       = t_gid_rel * key_pitch + head_idx * K_HEAD_SIZE;
                    cm_svm_block_read(
                        (svmptr_t)((const uchar*)key + in_off_h * sizeof(half)),
                        row_half);
                } else {
                    row_half = vector<half, K_HEAD_SIZE>(half(0.0f));
                }
                #pragma unroll
                for (uint g = 0; g < GROUP_NUM; ++g) {
                    const uint h_beg = g * GROUP_SIZE;
                    auto grp = row_half.template select<GROUP_SIZE, 1>(h_beg);
                    vector<half, GROUP_SIZE> acc =
                        cm_mul<half>(grp, half(1.0f / (float)scale_inv_g(g))) + zp_g(g);
                    vector<uchar, GROUP_SIZE> q = cm_rnde<uchar, GROUP_SIZE>(acc);
                    const uint row_base = token_bytes + t * K_HEAD_SIZE + h_beg;
                    cm_ptr_store<uint32_t, GROUP_SIZE / 4>(
                        (uint32_t*)((uchar*)key_cache + row_base), 0, q.format<uint32_t>());
                }
            }
            #pragma unroll
            for (uint g = 0; g < GROUP_NUM; ++g) {
                half* comp_new = (half*)((uchar*)key_cache +
                                         comp_bytes + g * 2 * sizeof(half));
                comp_new[0] = scale_inv_g(g);
                comp_new[1] = zp_g(g);
            }
        }
#else
        const uint block_k_base_offset =
            (phys_block * KV_HEADS_NUM + head_idx) *
            ADJUSTED_K_HEAD_SIZE * PAGED_ATTENTION_BLOCK_SIZE;
        const uint key_out_off_half =
            block_k_base_offset + token_pos_in_blk * K_HEAD_SIZE;
        vector<half, K_HEAD_SIZE> key_data;
        cm_svm_block_read(
            (svmptr_t)((const uchar*)key + key_in_off_half * sizeof(half)),
            key_data);
    #if KV_CACHE_COMPRESSION_PER_TOKEN
        quantize_and_store_per_token(
            key_data, (uchar*)key_cache, block_k_base_offset, token_pos_in_blk);
    #else
        cm_svm_block_write(
            (svmptr_t)((uchar*)key_cache + key_out_off_half * sizeof(half)),
            key_data);
    #endif
#endif // KV_CACHE_COMPRESSION_PER_CHANNEL

BY_CHANNEL_VALUE_PATH:
        const uint block_v_base_offset =
            (phys_block * KV_HEADS_NUM + head_idx) *
            ADJUSTED_V_HEAD_SIZE * PAGED_ATTENTION_BLOCK_SIZE;
        const uint value_out_off_half =
            block_v_base_offset + token_pos_in_blk * V_HEAD_SIZE;
        vector<half, V_HEAD_SIZE> value_vec;
        cm_svm_block_read(
            (svmptr_t)((const uchar*)value + value_in_off_half * sizeof(half)),
            value_vec);
    #if KV_CACHE_COMPRESSION_PER_TOKEN || KV_CACHE_COMPRESSION_PER_CHANNEL
        quantize_and_store_per_token(
            value_vec, (uchar*)value_cache, block_v_base_offset, token_pos_in_blk);
    #else
        cm_svm_block_write(
            (svmptr_t)((uchar*)value_cache + value_out_off_half * sizeof(half)),
            value_vec);
    #endif

    } else {
        // --------------------- prefill ---------------------
        const uint phys_block_idx = cm_global_id(0);
        const uint subseq_idx     = (uint)gws_seq_indexes_correspondence[phys_block_idx];
        const uint subseq_beg     = (uint)subsequence_begins[subseq_idx];
        const uint past_len       = (uint)past_lens[subseq_idx];

        const uint block_start_pos = (uint)blocked_indexes_start[phys_block_idx];
        const uint block_end_pos   = (uint)blocked_indexes_end[phys_block_idx];
        const uint tokens_num      = block_end_pos - block_start_pos;

        const uint token_start_key =
            (past_len + block_start_pos - subseq_beg) % PAGED_ATTENTION_BLOCK_SIZE;
        const uint token_start_val = token_start_key;

        const uint seq_block_idx =
            (past_len + block_start_pos - subseq_beg) / PAGED_ATTENTION_BLOCK_SIZE;
        const uint blk_offset =
            (uint)block_indices_begins[subseq_idx] + seq_block_idx;
        const uint phys_block =
            (uint)block_indices[blk_offset];

        const uint key_in_off_half =
            block_start_pos * key_pitch   + head_idx * K_HEAD_SIZE;
        const uint value_in_off_half =
            block_start_pos * value_pitch + head_idx * V_HEAD_SIZE;

#ifdef KV_CACHE_COMPRESSION_PER_CHANNEL
        constexpr uint DATA_BYTES_PER_BLOCK  = PAGED_ATTENTION_BLOCK_SIZE * K_HEAD_SIZE;
        constexpr uint STRIDE_BYTES_PER_HEAD = DATA_BYTES_PER_BLOCK + COMP_BYTES_TOTAL;
        const uint block_k_base_offset_bytes =
            (phys_block * KV_HEADS_NUM + head_idx) * STRIDE_BYTES_PER_HEAD;
        key_by_channel_prefill_window(
            key, key_in_off_half, key_pitch,
            (uchar*)key_cache, block_k_base_offset_bytes,
            token_start_key, tokens_num);
#else
        const uint block_k_base_offset =
            (phys_block * KV_HEADS_NUM + head_idx) *
            ADJUSTED_K_HEAD_SIZE * PAGED_ATTENTION_BLOCK_SIZE;
        for (uint n = 0; n < tokens_num; ++n) {
            const uint in_off  = key_in_off_half + n * key_pitch;
            const uint out_tok = token_start_key + n;
            vector<half, K_HEAD_SIZE> key_vec;
            cm_svm_block_read(
                (svmptr_t)((const uchar*)key + in_off * sizeof(half)),
                key_vec);
        #if KV_CACHE_COMPRESSION_PER_TOKEN
            quantize_and_store_per_token(
                key_vec, (uchar*)key_cache, block_k_base_offset, out_tok);
        #else
            const uint out_off_half =
                block_k_base_offset + out_tok * K_HEAD_SIZE;
            cm_svm_block_write(
                (svmptr_t)((uchar*)key_cache + out_off_half * sizeof(half)),
                key_vec);
        #endif
        }
#endif // KV_CACHE_COMPRESSION_PER_CHANNEL
        const uint block_v_base_offset =
            (phys_block * KV_HEADS_NUM + head_idx) *
            ADJUSTED_V_HEAD_SIZE * PAGED_ATTENTION_BLOCK_SIZE;
        for (uint n = 0; n < tokens_num; ++n) {
            const uint in_off  = value_in_off_half + n * value_pitch;
            const uint out_tok = token_start_val + n;
            vector<half, V_HEAD_SIZE> val_vec;
            cm_svm_block_read(
                (svmptr_t)((const uchar*)value + in_off * sizeof(half)),
                val_vec);
        #if KV_CACHE_COMPRESSION_PER_TOKEN || KV_CACHE_COMPRESSION_PER_CHANNEL
            quantize_and_store_per_token(
                val_vec, (uchar*)value_cache, block_v_base_offset, out_tok);
        #else
            const uint out_off_half =
                block_v_base_offset + out_tok * V_HEAD_SIZE;
            cm_svm_block_write(
                (svmptr_t)((uchar*)value_cache + out_off_half * sizeof(half)),
                val_vec);
        #endif
        }
    }
}
