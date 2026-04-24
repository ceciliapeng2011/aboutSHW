/*******************************************************************************
 * Copyright (c) 2018-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/

namespace KERNEL_NAME {
#include "estimate.hpp"

#define ABS(x) (x) < 0 ? -(x) : (x)

// Per-subsequence metadata field offsets (16 ints per subsequence for alignment)
#define XATTN_META_SUBSEQ_Q_BEGIN    0
#define XATTN_META_SUBSEQ_Q_LEN     1
#define XATTN_META_M                 2
#define XATTN_META_N                 3
#define XATTN_META_Q_STRIDE_PAD      4
#define XATTN_META_N_KQ_GROUPS       5
#define XATTN_META_Q_BLOCK_PAD       6
#define XATTN_META_K_BLOCK_PAD       7
#define XATTN_META_CAUSAL_START      8
#define XATTN_META_Q_START_STRIDED   9
#define XATTN_META_BUF_OFF_KQ_MAX   10
#define XATTN_META_BUF_OFF_EXP_SUM  11
#define XATTN_META_BUF_OFF_MASK     12
#define XATTN_META_BUF_OFF_MASK_WG  13
#define XATTN_META_BLOCK_IDX_BEGIN  14
#define XATTN_META_WG_OFFSET        15
#define XATTN_META_STRIDE           16

CM_INLINE void get_mn(uint& id_wg_m, uint& id_wg_n, uint M, uint N, int slice_no, int slice, const int BLOCK_WG_M, const int BLOCK_WG_N) {
    uint id_wg_mn = cm_group_id(0) / WALK_HQ;
    if (slice_no == 0) {
        if (slice == 0) {
            // loop M first, N is shared, total = N/256*M+N
            uint WG_MN = (M + BLOCK_WG_M - 1) / BLOCK_WG_M;
            id_wg_m = id_wg_mn % WG_MN;
            id_wg_n = id_wg_mn / WG_MN;
        } else {
            // loop N first, M is shared, total = M/128*N+M
            uint WG_MN = (N + BLOCK_WG_N - 1) / BLOCK_WG_N;
            id_wg_n = id_wg_mn % WG_MN;
            id_wg_m = id_wg_mn / WG_MN;
        }
    } else {
        uint wg_x = slice > 0 ? N / BLOCK_WG_N : M / BLOCK_WG_M;
        uint slice_no_abs = ABS(slice_no);
        uint slice_abs = ABS(slice);
        int id_wg_mn_in_reminder = (int)id_wg_mn - (int)(slice_no_abs * slice_abs * wg_x);
        uint slice_idx;
        // in [slice_no x slice]
        if (id_wg_mn_in_reminder < 0) {
            slice_idx = id_wg_mn / (slice_abs * wg_x);
            uint rem_in_slice = id_wg_mn % (slice_abs * wg_x);
            uint x = rem_in_slice % slice_abs;
            uint y = rem_in_slice / slice_abs;
            id_wg_m = slice > 0 ? x + slice_idx * slice_abs : y;
            id_wg_n = slice < 0 ? x + slice_idx * slice_abs : y;
        } else {
            uint slice_rem = slice_abs + (slice_no > 0 ? 1 : -1);
            slice_idx = id_wg_mn_in_reminder / (slice_rem * wg_x);
            uint rem_in_slice = id_wg_mn_in_reminder % (slice_rem * wg_x);
            uint x = rem_in_slice % slice_rem;
            uint y = rem_in_slice / slice_rem;
            id_wg_m = slice > 0 ? x + slice_idx * slice_rem + slice_no_abs * slice_abs : y;
            id_wg_n = slice < 0 ? x + slice_idx * slice_rem + slice_no_abs * slice_abs : y;
        }
    }
}

#define CONCAT_IMPL(a, b) KERNEL_NAME::gemm_qk
#define CONCAT(x, y) CONCAT_IMPL(x, y)
#define FUNC CONCAT(BLOCK_SG_M, BLOCK_SG_N)

#if MULTI_SUBSEQ

#ifndef CM_HAS_LSC_UNTYPED_2D
#error "MULTI_SUBSEQ requires Xe2 (CM_HAS_LSC_UNTYPED_2D) — Xe1 SurfaceIndex path not yet supported"
#endif

extern "C" _GENX_MAIN_ void gemm_qk(
    #ifdef CM_HAS_LSC_UNTYPED_2D
    svmptr_t key_cache ATTR,
    svmptr_t query ATTR,
    #else
    SurfaceIndex key_cache [[type("buffer_t")]],
    SurfaceIndex query [[type("buffer_t")]],
    #endif
    svmptr_t block_indices ATTR,
    svmptr_t xattn_subseq_meta ATTR,
    svmptr_t kq_max_wg ATTR,
    #ifdef CM_HAS_LSC_UNTYPED_2D
    svmptr_t kq_exp_partial_sum ATTR,
    #else
    SurfaceIndex kq_exp_partial_sum [[type("buffer_t")]],
    #endif
    uint K, uint query_stride, int num_subseqs) {
    const uint BLOCK_WG_M = BLOCK_SG_M * SG_M;
    const uint BLOCK_WG_N = BLOCK_SG_N * SG_N;
    uint hq = cm_group_id(2) * WALK_HQ;
    hq += cm_group_id(0) & (WALK_HQ - 1);
    if (hq >= HQ) return;
    uint hk = hq / (HQ / HK);
    const uint slm_size = SG_N * BLOCK_WG_M * sizeof(SOFTMAX_TYPE);
    cm_slm_init(slm_size);
    auto slm = cm_slm_alloc(slm_size);

    static_assert(HQ % HK == 0, "HQ must be multiple of HK");

    uint wg_mn_flat = cm_group_id(0) / WALK_HQ;

    // Find which subsequence this workgroup belongs to via linear scan of wg_offset prefix sums
    int subseq_id = 0;
    int* meta = (int*)xattn_subseq_meta;
    for (int s = 1; s < num_subseqs; s++) {
        int wg_off_next = meta[s * XATTN_META_STRIDE + XATTN_META_WG_OFFSET];
        if ((int)wg_mn_flat >= wg_off_next)
            subseq_id = s;
    }

    int meta_base = subseq_id * XATTN_META_STRIDE;
    uint M = (uint)meta[meta_base + XATTN_META_M];
    uint N = (uint)meta[meta_base + XATTN_META_N];
    uint q_start_strided = (uint)meta[meta_base + XATTN_META_Q_START_STRIDED];
    int block_index_begin = meta[meta_base + XATTN_META_BLOCK_IDX_BEGIN];
    uint subseq_wg_offset = (uint)meta[meta_base + XATTN_META_WG_OFFSET];
    uint subseq_q_begin = (uint)meta[meta_base + XATTN_META_SUBSEQ_Q_BEGIN];

    // Compute local (id_wg_m, id_wg_n) within this subsequence's workgroup grid
    uint wg_local = wg_mn_flat - subseq_wg_offset;
    uint WG_M_count = (M + BLOCK_WG_M - 1) / BLOCK_WG_M;
    uint id_wg_m = wg_local % WG_M_count;
    uint id_wg_n = wg_local / WG_M_count;

    // Offset output buffers by per-subsequence byte offsets
    uint buf_off_kq_max = (uint)meta[meta_base + XATTN_META_BUF_OFF_KQ_MAX];
    uint buf_off_exp_sum = (uint)meta[meta_base + XATTN_META_BUF_OFF_EXP_SUM];

    #ifdef CM_HAS_LSC_UNTYPED_2D
    // key cache: [block, HK, KV_BLOCK_SIZE, HEAD_SIZE_KEY]
#if KV_CACHE_COMPRESSION
    key_cache += hk * (KV_BLOCK_SIZE * HEAD_SIZE_KEY * (uint)sizeof(char));
#else
    key_cache += hk * (KV_BLOCK_SIZE * HEAD_SIZE_KEY * (uint)sizeof(half));
#endif
    query += (subseq_q_begin * (query_stride / STRIDE) + hq * HEAD_SIZE) * (uint)sizeof(half);
    #endif

    // Per-head output buffer offsets
    uint m_pad = (M + BLOCK_WG_M - 1) / BLOCK_WG_M * BLOCK_WG_M;
    uint n_groups = (N + BLOCK_WG_N - 1) / BLOCK_WG_N;
    kq_max_wg += buf_off_kq_max + hq * n_groups * m_pad * (uint)sizeof(SOFTMAX_TYPE);

    const uint sum_per_n_token_in_block = BLOCK_SIZE / STRIDE;
    const uint n_after_sum_in_group = BLOCK_WG_N / sum_per_n_token_in_block;
    const uint n_after_sum_pad = n_after_sum_in_group * n_groups;
    const uint offset_partial_sum = buf_off_exp_sum + hq * n_after_sum_pad * m_pad * (uint)sizeof(SOFTMAX_TYPE);
    #ifdef CM_HAS_LSC_UNTYPED_2D
    kq_exp_partial_sum += offset_partial_sum;
    #endif

    FUNC(id_wg_m, id_wg_n, hq, slm, key_cache, query, block_indices, block_index_begin, kq_max_wg, kq_exp_partial_sum, M, N, K, query_stride, q_start_strided, offset_partial_sum);
}

#else // !MULTI_SUBSEQ — original single-subsequence path

extern "C" _GENX_MAIN_ void gemm_qk(
    #ifdef CM_HAS_LSC_UNTYPED_2D
    svmptr_t key_cache ATTR,
    svmptr_t query ATTR,
    #else
    SurfaceIndex key_cache [[type("buffer_t")]],
    SurfaceIndex query [[type("buffer_t")]],
    #endif
    svmptr_t block_indices ATTR,
    svmptr_t block_indices_begins ATTR,
    svmptr_t kq_max_wg ATTR,
    #ifdef CM_HAS_LSC_UNTYPED_2D
    svmptr_t kq_exp_partial_sum ATTR,
    #else
    SurfaceIndex kq_exp_partial_sum [[type("buffer_t")]],
    #endif
    uint M, uint N, uint K, uint query_stride, int slice_no, int slice, uint q_start_strided) {
    const uint BLOCK_WG_M = BLOCK_SG_M * SG_M;
    const uint BLOCK_WG_N = BLOCK_SG_N * SG_N;
    uint hq = cm_group_id(2) * WALK_HQ;
    hq += cm_group_id(0) & (WALK_HQ - 1);
    if (hq >= HQ) return;
    uint hk = hq / (HQ / HK);
    const uint slm_size = SG_N * BLOCK_WG_M * sizeof(SOFTMAX_TYPE);
    cm_slm_init(slm_size);
    auto slm = cm_slm_alloc(slm_size);

    static_assert(HQ % HK == 0, "HQ must be multiple of HK");

    uint id_wg_m, id_wg_n;
    get_mn(id_wg_m, id_wg_n, M, N, slice_no, slice, BLOCK_WG_M, BLOCK_WG_N);

    #ifdef CM_HAS_LSC_UNTYPED_2D
    // key cache: [block, HQ, KV_BLOCK_SIZE, HEAD_SIZE_KEY]
#if KV_CACHE_COMPRESSION
    key_cache += hk * (KV_BLOCK_SIZE * HEAD_SIZE_KEY * (uint)sizeof(char));
#else
    key_cache += hk * (KV_BLOCK_SIZE * HEAD_SIZE_KEY * (uint)sizeof(half));
#endif
    query += hq * HEAD_SIZE * (uint)sizeof(half);
    #endif


    // kq_max: [hq, m_pad]
    // kq_max_wg: [hq, n_groups, m_pad]
    // kq_exp_partial_sum: [hq, m_pad, n_groups*BLOCK_WG_M/(BLOCK_SIZE/STRIDE)]
    uint m_pad = (M + BLOCK_WG_M - 1) / BLOCK_WG_M * BLOCK_WG_M;
    uint n_groups = (N + BLOCK_WG_N - 1) / BLOCK_WG_N;
    kq_max_wg += hq * n_groups * m_pad * (uint)sizeof(SOFTMAX_TYPE);

    const uint sum_per_n_token_in_block = BLOCK_SIZE / STRIDE;
    const uint n_after_sum_in_group = BLOCK_WG_N / sum_per_n_token_in_block;
    const uint n_after_sum_pad = n_after_sum_in_group * n_groups;
    const uint offset_partial_sum = hq * n_after_sum_pad * m_pad * (uint)sizeof(SOFTMAX_TYPE);
    #ifdef CM_HAS_LSC_UNTYPED_2D
    kq_exp_partial_sum += offset_partial_sum;
    #endif

    int block_index_begin = ((int*)block_indices_begins)[0];

    FUNC(id_wg_m, id_wg_n, hq, slm, key_cache, query, block_indices, block_index_begin, kq_max_wg, kq_exp_partial_sum, M, N, K, query_stride, q_start_strided, offset_partial_sum);
}

#endif // MULTI_SUBSEQ

}  // NAMESPACE
