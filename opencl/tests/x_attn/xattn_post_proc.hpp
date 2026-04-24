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
#include <cm/cm.h>
#include <cm/cmtl.h>

#ifndef ATTR
#define ATTR [[type("svmptr_t")]]
#define ATTR_BUF [[type("buffer_t")]]
#endif

// Shared metadata field offsets
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

CM_INLINE void post_proc_impl(svmptr_t block_mask, svmptr_t merged_block_mask, uint q_stride_pad, uint q_block_pad, uint k_block_pad) {
    const int TOKEN_IN_BLOCK = BLOCK_SIZE / STRIDE;
    uint m_mereged = cm_group_id(0);
    uint hq = cm_group_id(1);

    int j = 0;
    {
        constexpr int STEP = 32;
        for (; j + STEP <= k_block_pad; j += STEP) {
            vector<uchar, STEP> new_mask = cm_ptr_load<int, STEP / 4>((int*)block_mask, j).format<uchar>();
            #pragma unroll
            for (int i = 1; i < MERGED_Q_NUM; i++) {
                if (m_mereged * MERGED_Q_NUM + i < q_stride_pad / TOKEN_IN_BLOCK) {
                    vector<uchar, STEP> cur_mask = cm_ptr_load<int, STEP / 4>((int*)block_mask, j + i * k_block_pad).format<uchar>();
                    new_mask |= cur_mask;
                }
            }
            cm_ptr_store<int, STEP / 4>((int*)merged_block_mask, j, new_mask.format<int>());
        }
    }
    {
        constexpr int STEP = 16;
        for (; j < k_block_pad; j += STEP) {
            vector<uchar, STEP> new_mask = cm_ptr_load<int, STEP / 4>((int*)block_mask, j).format<uchar>();
            #pragma unroll
            for (int i = 1; i < MERGED_Q_NUM; i++) {
                if (m_mereged * MERGED_Q_NUM + i < q_stride_pad / TOKEN_IN_BLOCK) {
                    vector<uchar, STEP> cur_mask = cm_ptr_load<int, STEP / 4>((int*)block_mask, j + i * k_block_pad).format<uchar>();
                    new_mask |= cur_mask;
                }
            }
            cm_ptr_store<int, STEP / 4>((int*)merged_block_mask, j, new_mask.format<int>());
        }
    }
}

#if MULTI_SUBSEQ

extern "C" _GENX_MAIN_ void post_proc_mask(svmptr_t block_mask ATTR, svmptr_t merged_block_mask ATTR, svmptr_t xattn_subseq_meta ATTR) {
    // GWS: [max(q_block_pad/MERGED_Q_NUM), HQ, num_subseqs]
    uint b = cm_group_id(2);
    uint hq = cm_group_id(1);
    uint m_mereged = cm_group_id(0);

    int* meta = (int*)xattn_subseq_meta;
    int meta_base = b * XATTN_META_STRIDE;

    uint q_stride_pad = (uint)meta[meta_base + XATTN_META_Q_STRIDE_PAD];
    uint q_block_pad = (uint)meta[meta_base + XATTN_META_Q_BLOCK_PAD];
    uint k_block_pad = (uint)meta[meta_base + XATTN_META_K_BLOCK_PAD];
    uint buf_off_mask = (uint)meta[meta_base + XATTN_META_BUF_OFF_MASK];
    uint buf_off_mask_wg = (uint)meta[meta_base + XATTN_META_BUF_OFF_MASK_WG];

    uint merged_q_blocks = (q_block_pad + MERGED_Q_NUM - 1) / MERGED_Q_NUM;

    // Early-return for padding workgroups beyond this subsequence
    if (m_mereged >= merged_q_blocks) return;

    svmptr_t mask_base = block_mask + buf_off_mask + hq * q_block_pad * k_block_pad;
    mask_base += m_mereged * MERGED_Q_NUM * k_block_pad;
    svmptr_t merged_base = merged_block_mask + buf_off_mask_wg + hq * merged_q_blocks * k_block_pad;
    merged_base += m_mereged * k_block_pad;

    post_proc_impl(mask_base, merged_base, q_stride_pad, q_block_pad, k_block_pad);
}

#else // !MULTI_SUBSEQ — original single-subsequence path

extern "C" _GENX_MAIN_ void post_proc_mask(svmptr_t block_mask ATTR, svmptr_t merged_block_mask ATTR, uint q_stride_pad, uint q_block_pad, uint k_block_pad) {
    // block_mask:                [b, hq, q_block_pad, k_block_pad]
    // merged_block_mask:         [b, hq, q_block_pad/MERGED_Q_NUM, k_block_pad]
    // global:                    [q_block_pad/MERGED_Q_NUM, hq, b]
    uint m_mereged = cm_group_id(0);
    uint hq = cm_group_id(1);
    uint b = cm_group_id(2);
    block_mask += (b * HQ + hq) * q_block_pad * k_block_pad;
    block_mask += m_mereged * MERGED_Q_NUM * k_block_pad;
    merged_block_mask += (b * HQ + hq) * cm_group_count(0) * k_block_pad;
    merged_block_mask += m_mereged * k_block_pad;

    post_proc_impl(block_mask, merged_block_mask, q_stride_pad, q_block_pad, k_block_pad);
}

#endif // MULTI_SUBSEQ

}  // NAMESPACE
