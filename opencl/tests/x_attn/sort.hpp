/*
 * Copyright (c) 2020-2023, Intel Corporation
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

#if 0
template<typename T, int N>
void show(const vector<T, N> mat, bool is_hex=true) {
    printf("vector [%d]:\n[", N);
    for(int n = 0; n < N; n ++) {
        if (is_hex)
            printf("%x,", mat[n]);
        else
            printf("%d,", mat[n]);
    }
    printf("]\n");
}

template<typename T, int M, int N>
void show(const matrix<T, M, N> mat) {
    printf("Matrix [%d, %d]:\n", M, N);
    for(int m = 0; m < M; m ++) {
        printf("\t[");
        for(int n = 0; n < N; n ++) {
            printf("%8.4f,", mat[m][n]);
        }
        printf("],\n");
    }
    printf("]\n");
}
#endif

// https://gpuopen.com/download/Introduction_to_GPU_Radix_Sort.pdf
//
// Sortable-key width tracks TYPE's own width (via KeyT below), so this radix
// sort works for any sign-magnitude float-like TYPE -- currently `half`
// (16-bit) and `float` (32-bit) are the only instantiations (see
// find_block.hpp's `sort<SOFTMAX_TYPE>` call). Previously this was hardcoded
// for 16-bit keys: the float<->sortable-uint conversion only ran for `half`,
// and the counting/reorder/copy-out passes always read/wrote 16-bit `ushort`
// chunks regardless of TYPE. For `float` (SOFTMAX_TYPE='float', the value
// every x_attn caller actually uses) this silently reinterpreted each 4-byte
// score as two independent 16-bit keys, scrambling the sort -- a
// never-exercised bug since no x_attn test ran the non-causal branch (the
// only caller of this function) with causal=False. Index arrays
// (sorted_index/sort_tmp) stay 16-bit regardless of TYPE -- they hold
// positions in [0, n), which fits ushort for all realistic block counts.
template<typename TYPE>
CM_INLINE void sort(uint slm, svmptr_t src, svmptr_t sorted_value, svmptr_t sorted_index, svmptr_t sort_tmp, uint n) {
    using KeyT = typename std::conditional<sizeof(TYPE) == 4, unsigned, ushort>::type;
    constexpr int NUM_PASSES = (int)sizeof(KeyT) * 8 / 4;   // 4 bits/pass radix sort
    constexpr int INT_CHUNKS = (NUM_THREADS * (int)sizeof(KeyT)) / (int)sizeof(int);

    const ushort THREADS = NUM_THREADS;
    vector<unsigned, THREADS> seq_u32;
    vector<ushort, THREADS> seq;
    cmtl::cm_vector_assign(seq.select_all(), 0, 1);
    cmtl::cm_vector_assign(seq_u32.select_all(), 0, 1);
    svmptr_t sorted_src = src;
    svmptr_t sorted_tmp = sorted_value;
    svmptr_t cur_idx = sorted_index;
    svmptr_t cur_idx_tmp = sort_tmp;
    uint iter = (n + THREADS - 1) / THREADS;
    vector<unsigned, THREADS> offset_src;
    cmtl::cm_vector_assign(offset_src.select_all(), 0, iter);
    auto to_sortable = [] (vector_ref<TYPE, THREADS> in, vector_ref<KeyT, THREADS> out) {
        static const KeyT HIGH_BIT = KeyT(1) << (sizeof(KeyT) * 8 - 1);
        auto in_u = in.format<KeyT>();
        auto mask = (in_u & HIGH_BIT) != 0;
        vector<KeyT, THREADS> m;
        m.merge((KeyT)~KeyT(0), HIGH_BIT, mask);
        out = in_u ^ m;
    };
    auto from_sortable = [] (vector_ref<KeyT, THREADS> in, vector_ref<TYPE, THREADS> out) {
        static const KeyT HIGH_BIT = KeyT(1) << (sizeof(KeyT) * 8 - 1);
        auto mask = (in & HIGH_BIT) != 0;
        vector<KeyT, THREADS> m;
        m.merge(HIGH_BIT, (KeyT)~KeyT(0), mask);
        out.format<KeyT>() = in ^ m;
    };
    {
        // TYPE -> sortable KeyT, in place (always -- for any sign-magnitude
        // float type, not just half)
        vector<KeyT, THREADS> data;
        vector<TYPE, THREADS> data_f;
        int i;
        for (i = 0; i + THREADS <= n / THREADS * THREADS; i += THREADS) {
            data_f.format<int>() = cm_ptr_load<int, INT_CHUNKS>((int*)src, i * (int)sizeof(TYPE));
            to_sortable(data_f, data);
            cm_ptr_store<int, INT_CHUNKS>((int*)src, i * (int)sizeof(KeyT), data.format<int>());
        }
        if (i < n) {
            auto pos = seq_u32 + i;
            SIMD_IF_BEGIN (pos < n) {
                data_f = cm_ptr_load<TYPE>((TYPE*)src, pos * (uint)sizeof(TYPE));
                to_sortable(data_f, data);
                cm_ptr_store<KeyT>((KeyT*)src, pos * (uint)sizeof(KeyT), data);
            } SIMD_IF_END;
        }
    }
    {
        // generate idx
        vector<ushort, THREADS> data;
        int i;
        for (i = 0; i + THREADS <= n / THREADS * THREADS; i += THREADS) {
            data = seq + i;
            cm_ptr_store<int, THREADS / 2>((int*)cur_idx, i * (int)sizeof(short), data.format<int>());
        }
        if (i < n) {
            auto pos = seq_u32 + i;
            data = seq + i;

            SIMD_IF_BEGIN (pos < n) {
                cm_ptr_store<ushort>((ushort*)cur_idx, pos * (uint)sizeof(ushort), data);
            } SIMD_IF_END;
        }
    }
    // 4 bits per pass, NUM_PASSES passes to cover all of KeyT's bits (4 for
    // 16-bit half keys, 8 for 32-bit float keys)
    for (int pass = 0; pass < NUM_PASSES; pass++) {
        {
            // slm layout: short [16][work items] = 16*32*2 bytes -- bucket
            // histogram counts, always 16 buckets of ushort regardless of
            // KeyT's width
            vector<int, 64> data = 0;
            for (int i = 0; i < 16 * THREADS * sizeof(ushort); i += 256)
                cm_slm_block_write(slm, i, data);
        }
        {
            // counting phase
            vector<KeyT, THREADS> data;
            for (int i = 0; i < iter; i++) {
                data = cm_ptr_load<KeyT>((KeyT*)sorted_src, (offset_src + i) * (uint)sizeof(KeyT), (offset_src + i) < n);
                vector<ushort, THREADS> bits = 0xf - ((data >> (pass * 4)) & 0xf);
                vector<ushort, THREADS> addr = bits * THREADS + seq;
                vector<ushort, THREADS> total;
                cm_slm_read(slm, addr, total);
                total += 1;
                cm_slm_write(slm, addr, total);
            }
        }
        {
            // prefix sum
            vector<ushort, 16> local_prefix = 0;

            vector<ushort, 16> seq_prefix;
            cmtl::cm_vector_assign(seq_prefix.select_all(), 0, THREADS);

            #pragma unroll
            for (ushort i = 0; i < THREADS; i++) {
                auto prev = local_prefix;
                vector<ushort, 16> hist;
                vector<ushort, 16> addr = seq_prefix + i;
                cm_slm_read(slm, addr, hist);
                local_prefix += hist;
                cm_slm_write(slm, addr, prev);
            }
            // Hillis-Steele scan
            vector<ushort, 16> local_tmp;
            local_tmp.select<15, 1>(1) = local_prefix.select<15, 1>(1) + local_prefix.select<15, 1>(0); local_tmp[0] = local_prefix[0];
            local_prefix.select<14, 1>(2) = local_tmp.select<14, 1>(2) + local_tmp.select<14, 1>(0); local_prefix.select<2, 1>(0) = local_tmp.select<2, 1>(0);
            local_tmp.select<12, 1>(4) = local_prefix.select<12, 1>(4) + local_prefix.select<12, 1>(0); local_tmp.select<4, 1>(0) = local_prefix.select<4, 1>(0);
            local_prefix.select<8, 1>(8) = local_tmp.select<8, 1>(8) + local_tmp.select<8, 1>(0); local_prefix.select<8, 1>(0) = local_tmp.select<8, 1>(0);
            vector<ushort, 16 * THREADS> data;
            cm_slm_block_read(slm, 0, data);
            #pragma unroll
            for (int i = 1; i < 16; i++) {
                data.select<THREADS, 1>(i * THREADS) += local_prefix[i - 1];
            }
            cm_slm_block_write(slm, 0, data);
        }
        {
            // reorder
            vector<KeyT, THREADS> data;
            for (int i = 0; i < iter; i++) {
                data = cm_ptr_load<KeyT>((KeyT*)sorted_src, (offset_src + i) * (uint)sizeof(KeyT), (offset_src + i) < n);
                vector<ushort, THREADS> bits = 0xf - ((data >> (pass * 4)) & 0xf);
                vector<ushort, THREADS> addr = bits * THREADS + seq;
                vector<ushort, THREADS> index;
                cm_slm_read(slm, addr, index);
                // Two separate byte offsets: the key/value array is KeyT-wide
                // (tracks TYPE), but the index array is always ushort-wide.
                vector<unsigned, THREADS> offset_val = index * (uint)sizeof(KeyT);
                cm_ptr_store<KeyT>((KeyT*)sorted_tmp, offset_val, data, index < n);

                vector<ushort, THREADS> idx_data = cm_ptr_load<ushort>((ushort*)cur_idx, (offset_src + i) * (uint)sizeof(ushort));
                vector<unsigned, THREADS> offset_idx = index * (uint)sizeof(ushort);
                cm_ptr_store<ushort>((ushort*)cur_idx_tmp, offset_idx, idx_data, (offset_src + i) < n);

                index += 1;
                cm_slm_write(slm, addr, index);
            }
            auto tmp = sorted_src;
            sorted_src = sorted_tmp;
            sorted_tmp = tmp;
            tmp = cur_idx;
            cur_idx = cur_idx_tmp;
            cur_idx_tmp = tmp;
        }
    }
    {
        // copy to output: KeyT (sortable) -> TYPE, always (for any
        // sign-magnitude float type). NUM_PASSES is even for every supported
        // TYPE (KeyT's bit-width is a multiple of 8), so the sorted keys end
        // up back in `src` (not `sorted_src`) after the ping-pong loop above.
        vector<KeyT, THREADS> data;
        vector<TYPE, THREADS> data_f;
        int i;
        for (i = 0; i + THREADS <= n / THREADS * THREADS; i += THREADS) {
            data.format<int>() = cm_ptr_load<int, INT_CHUNKS>((int*)src, i * (int)sizeof(KeyT));
            from_sortable(data, data_f);
            cm_ptr_store<int, INT_CHUNKS>((int*)sorted_value, i * (int)sizeof(TYPE), data_f.format<int>());
        }
        if (i < n) {
            auto pos = seq_u32 + i;
            SIMD_IF_BEGIN (pos < n) {
                data = cm_ptr_load<KeyT>((KeyT*)src, pos * (uint)sizeof(KeyT));
                from_sortable(data, data_f);
                cm_ptr_store<TYPE>((TYPE*)sorted_value, pos * (uint)sizeof(TYPE), data_f);
            } SIMD_IF_END;
        }
    }
}
