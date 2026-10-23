// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Shared dense [B,H,S,D] addressing helpers for the SDPA-CM kernel family
// (cm_sdpa_opt_prefill.cm, cm_sdpa_opt_decode.cm). Include this inside each kernel's
// `namespace KERNEL_NAME { ... }` block, mirroring how cm_sdpa_common.hpp is included.
//
// Q_TRANSPOSE_ORDER/K_TRANSPOSE_ORDER/V_TRANSPOSE_ORDER/OUTPUT_TRANSPOSE_ORDER select between
// OCL SDPA's identity layout [B,H,S,D] (heads outer, TransposeSDPAMatcher's default) and the
// {0,2,1,3}-fused heads<->seq swap [B,S,H,D] (heads inner) -- the only two orders the OCL kernel
// itself supports (sdpa_gen_opt.cpp's OUTPUT_TRANSPOSE_ORDER_PRESENT assert). head_size stays
// innermost/contiguous either way, matching the transpose-fusion invariant.

#define TRANSPOSE_ORDER_IDENTITY 0  // [B,H,S,D]
#define TRANSPOSE_ORDER_SWAPPED  1  // [B,S,H,D]

#ifndef Q_TRANSPOSE_ORDER
#define Q_TRANSPOSE_ORDER TRANSPOSE_ORDER_IDENTITY
#endif
#ifndef K_TRANSPOSE_ORDER
#define K_TRANSPOSE_ORDER TRANSPOSE_ORDER_IDENTITY
#endif
#ifndef V_TRANSPOSE_ORDER
#define V_TRANSPOSE_ORDER TRANSPOSE_ORDER_IDENTITY
#endif
#ifndef OUTPUT_TRANSPOSE_ORDER
#define OUTPUT_TRANSPOSE_ORDER TRANSPOSE_ORDER_IDENTITY
#endif

// Byte pitch between consecutive tokens for a [.., num_heads, .., head_size] tensor, given a
// compile-time transpose order. SWAPPED (heads inner) -> tokens are num_heads*head_size apart.
// IDENTITY (heads outer) -> tokens are head_size apart (heads live in separate seq_len*head_size
// chunks, handled by get_head_byte_offset below).
//
// elem_size defaults to sizeof(half) (every existing caller's behavior, unchanged). An int8-packed
// cache buffer (see KV_CACHE_COMPRESSION in cm_sdpa_opt_prefill_dualsource.cm) packs 1
// byte/element instead of 2, so its own pitch/offset must be computed with elem_size=sizeof(uchar)
// -- callers opt in explicitly by passing the second template argument.
template<int order, int elem_size = sizeof(half)>
inline uint get_seq_pitch_bytes(int seq_len, int num_heads, int head_size) {
    if constexpr (order == TRANSPOSE_ORDER_SWAPPED)
        return num_heads * head_size * elem_size;
    else
        return head_size * elem_size;
}

// Byte offset of head `h`'s sub-tensor within one batch's chunk.
template<int order, int elem_size = sizeof(half)>
inline uint get_head_byte_offset(int h, int seq_len, int head_size) {
    if constexpr (order == TRANSPOSE_ORDER_SWAPPED)
        return h * head_size * elem_size;
    else
        return (uint)h * seq_len * head_size * elem_size;
}

// Byte size of one batch's chunk (invariant to transpose order -- same total elements either way).
inline uint get_batch_byte_stride(int seq_len, int num_heads, int head_size, int elem_size = sizeof(half)) {
    return (uint)seq_len * num_heads * head_size * elem_size;
}
