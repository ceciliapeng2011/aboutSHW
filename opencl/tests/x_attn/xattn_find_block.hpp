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
#include "find_block.hpp"

#ifndef ATTR
#define ATTR [[type("svmptr_t")]]
#define ATTR_BUF [[type("buffer_t")]]
#endif

// Shared metadata field offsets (same as xattn_gemm_qk.hpp)
#ifndef XATTN_META_STRIDE
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
#endif

#if MULTI_SUBSEQ

extern "C" _GENX_MAIN_ void find_block(
    svmptr_t kq_max_wg ATTR,
    svmptr_t kq_exp_partial_sum ATTR,
    svmptr_t block_mask ATTR,
    svmptr_t xattn_subseq_meta ATTR,
    float thresh,
    svmptr_t kq_sum ATTR
) {
    // GWS: [max_q_block_pad, HQ, num_subseqs]
    const int TOKEN_IN_BLOCK = BLOCK_SIZE / STRIDE;
    const int TOKEN_SHARE_MAX = BLOCK_SHARE_MAX / TOKEN_IN_BLOCK;
    uint m = cm_group_id(0);
    uint hq = cm_group_id(1);
    uint b = cm_group_id(2);

    int* meta = (int*)xattn_subseq_meta;
    int meta_base = b * XATTN_META_STRIDE;

    uint q_len = (uint)meta[meta_base + XATTN_META_SUBSEQ_Q_LEN];
    uint q_stride = (uint)meta[meta_base + XATTN_META_M];
    uint q_stride_pad = (uint)meta[meta_base + XATTN_META_Q_STRIDE_PAD];
    uint q_block_pad = (uint)meta[meta_base + XATTN_META_Q_BLOCK_PAD];
    uint k_block_pad = (uint)meta[meta_base + XATTN_META_K_BLOCK_PAD];
    uint causal_start_index = (uint)meta[meta_base + XATTN_META_CAUSAL_START];

    // Early-return for padding workgroups beyond this subsequence's q_block_pad
    if (m >= q_block_pad) return;

    // Apply per-subsequence buffer offsets
    uint buf_off_kq_max = (uint)meta[meta_base + XATTN_META_BUF_OFF_KQ_MAX];
    uint buf_off_exp_sum = (uint)meta[meta_base + XATTN_META_BUF_OFF_EXP_SUM];
    uint buf_off_mask = (uint)meta[meta_base + XATTN_META_BUF_OFF_MASK];

    kq_max_wg += buf_off_kq_max + hq * (k_block_pad / TOKEN_SHARE_MAX) * q_stride_pad * (uint)sizeof(SOFTMAX_TYPE);
    kq_exp_partial_sum += buf_off_exp_sum + hq * q_stride_pad * k_block_pad * (uint)sizeof(SOFTMAX_TYPE);
    block_mask += buf_off_mask + hq * q_block_pad * k_block_pad;

    const uint slm_size = NUM_THREADS * 16 * sizeof(ushort);
    cm_slm_init(slm_size);
    auto slm = cm_slm_alloc(slm_size);

    find(slm, m, kq_max_wg, kq_exp_partial_sum, block_mask, q_len, q_stride, q_stride_pad, k_block_pad, thresh, causal_start_index
#if DEBUG_ACC == 1
    , kq_sum
#endif
    );
}

#else // !MULTI_SUBSEQ — original single-subsequence path

extern "C" _GENX_MAIN_ void find_block(
    svmptr_t kq_max_wg ATTR,
    svmptr_t kq_exp_partial_sum ATTR,
    svmptr_t block_mask ATTR,
    uint q_len,
    uint q_stride,
    uint q_stride_pad,
    uint q_block_pad,
    uint k_block_pad,
    uint causal_start_index,
    float thresh
#if DEBUG_ACC == 1
    , svmptr_t kq_sum ATTR
#endif
) {
    // kq_max_wg:          [b, hq, n_groups, q_stride_pad]
    // kq_exp_partial_sum: [b, hq, q_stride_pad, k_block_pad]
    // kq_sum:             [b, hq, q_stride_pad/TOKEN_IN_BLOCK, k_block_pad]
    // block_mask:         [b, hq, q_stride_pad/TOKEN_IN_BLOCK, k_block_pad]
    // [1, 32, 256], [1, 32, 64, 256], [1, 32, 256, 64 * 16], A_sum:[1, 32, 32, 64 * 16]
    // global:            [q_block_pad, hq, b]
    const int TOKEN_IN_BLOCK = BLOCK_SIZE / STRIDE;
    const int TOKEN_SHARE_MAX = BLOCK_SHARE_MAX / TOKEN_IN_BLOCK;
    uint m = cm_group_id(0);
    uint hq = cm_group_id(1);
    uint b = cm_group_id(2);
    kq_max_wg += (b * HQ + hq) * (k_block_pad / TOKEN_SHARE_MAX) * q_stride_pad * (uint)sizeof(SOFTMAX_TYPE);
    kq_exp_partial_sum += (b * HQ + hq) * q_stride_pad * k_block_pad * (uint)sizeof(SOFTMAX_TYPE);
#if DEBUG_ACC == 1
    kq_sum += (b * HQ + hq) * (q_stride_pad / TOKEN_IN_BLOCK) * k_block_pad * (uint)sizeof(half);
#endif
    block_mask += (b * HQ + hq) * q_block_pad * k_block_pad;

    const uint slm_size = NUM_THREADS * 16 * sizeof(ushort);
    cm_slm_init(slm_size);
    auto slm = cm_slm_alloc(slm_size);

    find(slm, m, kq_max_wg, kq_exp_partial_sum, block_mask, q_len, q_stride, q_stride_pad, k_block_pad, thresh, causal_start_index
#if DEBUG_ACC == 1
    , kq_sum
#endif
    );
}

#endif // MULTI_SUBSEQ

}  // NAMESPACE
