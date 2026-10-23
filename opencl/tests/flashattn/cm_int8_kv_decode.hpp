// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// int8 (uint8, affine scale + zero-point) KV-cache dequantization -- the SAME scheme
// opencl/tests/pageatten/cm_pa_xe2.hpp's pa_lsc_u8 already uses (KV_CACHE_COMPRESSION: 1=BY_TOKEN,
// 2=BY_CHANNEL) -- see that function's
// load_slm_KV_active for the precedent this ports verbatim. No gather/LUT: dequant is a flat
// `(raw - zp) * scale` per element, fully unrolled per row (pa_lsc_u8's own comment notes a CM
// compiler bug with a type-converting subtract+tail inside a real loop, so #pragma unroll is kept
// here too).
//
// Deliberately a standalone header (no #include of cm_attention_common.hpp or cm_sdpa_common.hpp):
// draft_block_mask_pool.cm needs this too but must not pull in SDPA-specific JIT/DPAS machinery
// (that header hard-requires CMFLA_SCALE_FACTOR and several REG_N/REG_K/q_step/kv_step-gated
// static_asserts this plain elementwise mean-pool kernel has no reason to define) -- this way both
// call sites share ONE implementation instead of two local copies.
//
// Layout convention (see int8_kv_ref.py, the host-side encode reference): three flat tensors per
// quantized K or V -- raw uint8 data [..., kv_len, head_size], half scale [..., kv_len/granularity,
// head_size or 1], half zero-point [same shape as scale]. NOT pageatten's single packed blob --
// sdpa_cm has no on-device cache-update/page-relocation kernel, so scale/zp never need to travel
// physically attached to the data; they're just three separate flat SVM buffers.
//
// Deliberately NO include guard: this project's CM headers (cm_sdpa_common.hpp,
// cm_sdpa_opt_common.hpp) are routinely #included twice per kernel -- once at file (global) scope
// and once inside `namespace KERNEL_NAME { ... }` -- so each kernel gets its own namespace-scoped
// copy of every template/function here (no cross-kernel ODR collisions when multiple kernels are
// compiled together). An include guard would silently suppress the second (namespace-scoped) copy.
#include <cm/cm.h>
#include <cm/cmtl.h>

// BY_CHANNEL: scale/zp are a length-COLS vector, one value per CHANNEL (head_size column) covered
// by this tile, broadcast across all ROWS rows -- valid as long as every row in the tile belongs to
// the same quantization sub-block (true whenever the caller's row-tile height, e.g. kv_step, evenly
// divides the sub_block_size scale/zp were computed over -- see int8_kv_ref.py's
// DEFAULT_SUB_BLOCK_SIZE).
template <int ROWS, int COLS>
CM_INLINE void int8_decode_by_channel(matrix_ref<uint8_t, ROWS, COLS> raw,
                                       vector_ref<half, COLS> scale,
                                       vector_ref<half, COLS> zp,
                                       matrix_ref<half, ROWS, COLS> out) {
#pragma unroll
    for (int r = 0; r < ROWS; r++) {
        out[r] = raw[r] - zp;
        out[r] = cm_mul<half>(out[r], scale);
    }
}

// BY_TOKEN: scale/zp are a length-ROWS vector, one value PER ROW (per KV token) -- used for V
// always, and for K when the caller selects BY_TOKEN granularity.
template <int ROWS, int COLS>
CM_INLINE void int8_decode_by_token(matrix_ref<uint8_t, ROWS, COLS> raw,
                                     vector_ref<half, ROWS> scale,
                                     vector_ref<half, ROWS> zp,
                                     matrix_ref<half, ROWS, COLS> out) {
#pragma unroll
    for (int r = 0; r < ROWS; r++) {
        out[r] = raw[r] - zp[r];
        out[r] = cm_mul<half>(out[r], scale[r]);
    }
}

// 1-row vector form for kernels with no matrix/DPAS tiling at all (draft_block_mask_pool.cm's
// per-token mean-pooling loop) -- by-channel only, since pooling only ever quantizes K, and K
// pooling reads one token (one row) at a time.
template <int N>
CM_INLINE void int8_decode_by_channel_row(vector_ref<uint8_t, N> raw,
                                           vector_ref<half, N> scale,
                                           vector_ref<half, N> zp,
                                           vector_ref<half, N> out) {
    out = raw - zp;
    out = cm_mul<half>(out, scale);
}
