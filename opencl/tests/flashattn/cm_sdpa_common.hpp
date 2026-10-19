// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cm_attention_common.hpp"

// Number of kv sub-tiles (kv_step rows each) processed per online-softmax update in
// sdpa_kernel_lsc_prefetch. Larger amortizes the rO rescale + softmax bookkeeping over
// more DPAS work at the cost of GRF pressure. Overridable via -DCMFLA_KV_BLK=N.
#ifndef CMFLA_KV_BLK
#    define CMFLA_KV_BLK 1
#endif

// online_softmax_update_tree and transpose_St_to_P_half are defined in cm_attention_common.hpp

// fp8e4m3 ("hf8") -> fp16 in-register decode, bit-identical to the real OpenVINO intel_gpu
// plugin's _intel_convert_hf8_to_f16 (XeTLA-derived) scalar reference -- see
// thirdparty/openvino/src/plugins/intel_gpu/src/kernel_selector/cl_kernels/include/f8_utils.cl
// (read-only reference; not included/linked, this is an independent CM re-implementation since
// that file is plain OpenCL C, a different toolchain from CM/ESIMD).
//
// e4m3 fields: sign(1) | exp(4, bias 7) | mant(3). fp16 fields: sign(1) | exp(5, bias 15) |
// mant(10). Three cases, vectorized via predicated merge (house style: see cur_sum.merge(...)
// above and online_softmax_update_tree's branch-free reductions):
//   * exp==0xF && mant==0x7  -> NaN                       (hf8's single NaN encoding)
//   * exp==0   && mant==0    -> +-0
//   * exp==0   && mant!=0    -> denormal hf8: renormalize by counting leading zeros of the
//                                3-bit mantissa (lz_count in {0,1,2}; mant is never 0 here) and
//                                shifting the implicit bit into place.
//   * otherwise               -> normal: just re-bias the exponent (+8, since -7+15==+8) and
//                                left-shift the 3-bit mantissa into fp16's 10-bit field.
// All-ushort arithmetic is used throughout (including the denormal path's `exp - 7 + 15`, done
// as `exp + 8` directly) so this never relies on signed/unsigned promotion rules -- every
// element stays in-range or wraps and un-wraps consistently in 16-bit modular arithmetic.
//
// Declared up front (ahead of sdpa_accumulate_kv_source below, its only caller) because this
// compiler resolves an unqualified call inside a template body against declarations visible at
// the template's definition point, not at instantiation -- a forward reference here is a hard
// "undeclared identifier" error, unlike full two-phase C++ lookup.
template <int N>
CM_INLINE vector<half, N> hf8_e4m3_decode_to_f16(vector_ref<uchar, N> raw) {
    vector<uchar, N> sign_u8 = (raw >> 7) & 1;
    vector<uchar, N> exp_u8 = (raw >> 3) & 0xF;
    vector<uchar, N> mant_u8 = raw & 0x7;
    vector<ushort, N> sign = sign_u8;
    vector<ushort, N> exp = exp_u8;
    vector<ushort, N> mant = mant_u8;

    vector<unsigned short, N> is_nan = (exp == 0xF) & (mant == 0x7);
    vector<unsigned short, N> is_zero = (exp == 0) & (mant == 0);
    vector<unsigned short, N> is_denorm = (exp == 0) & (mant != 0);

    // lz_count: leading zeros of the 3-bit mantissa, MSB-first (only meaningful where
    // is_denorm, but computed branch-free for every lane).
    vector<ushort, N> lz_count = 2;
    lz_count.merge((ushort)1, mant > 1);
    lz_count.merge((ushort)0, mant > 3);

    vector<ushort, N> base_exp = exp + 8;  // == exp - 7 + 15 (normal-case re-bias)
    vector<ushort, N> dst_exp = base_exp;
    dst_exp.merge(base_exp - lz_count, is_denorm);

    vector<ushort, N> dst_mant = mant;
    vector<ushort, N> denorm_mant = (mant << (lz_count + 1)) & 0x7;
    dst_mant.merge(denorm_mant, is_denorm);

    vector<ushort, N> dst_val = (dst_exp << 10) | (dst_mant << 7);
    dst_val.merge((ushort)0, is_zero);
    dst_val.merge((ushort)0x7fff, is_nan);

    vector<ushort, N> bits = (sign << 15) | dst_val;
    return bits.format<half>();
}

// fp8e4m3 ("hf8") -> fp16 LUT-gather decode: an ADDITIONAL decode path alongside
// hf8_e4m3_decode_to_f16 above, which stays completely untouched (same input/output contract,
// bit-exact against the same oracle -- see hf8_e4m3_ref.py's build_hf8_decode_lut_f16). Only
// sdpa_kernel_lsc below (when instantiated with KVElemT=uchar) ever calls this
// (decode-once-per-work-group, measured faster than the per-work-item bit-manipulation decode --
// see that function's doc comment); the dual-source ring kernel
// (sdpa_accumulate_kv_source/sdpa_kernel_lsc_prefetch_2src) is out of scope and always uses the
// bit-manipulation decode.
//
// fp8e4m3 has only 256 possible byte values, so the entire decode is representable as a 256-entry
// half lookup table, built once on the HOST (hf8_e4m3_ref.py's build_hf8_decode_lut_f16) and
// uploaded as a plain half[256] array (512 bytes) alongside K/V -- see hf8_lut_base below, threaded
// down from KV_FP8E4M3's extra kernel param in cm_sdpa_opt_prefill.cm.
//
// Layout: NOT packed into uint32 (the originally-expected fallback for a toolchain with no
// sub-dword gather). A direct compile+run probe against this toolchain's
// /cm/builtin/cm/cm_lsc.h (not just this repo's own prior gather usages in cm_pa_xe1.hpp /
// x_attn/estimate.hpp, which only ever gather at uint/4-byte granularity via the SurfaceIndex-
// bound cm_load<...>(SurfaceIndex, offsets[, pred]) overload set) found that cm_ptr_load -- the
// raw-pointer sibling of that overload set, already used elsewhere in this repo for non-gather
// block loads, see find_block.hpp's cm_ptr_load<uint,N,...>(ptr, scalar_offset) -- also supports a
// genuine per-lane offset VECTOR directly against a plain T* pointer at true 2-byte (DataSize::U16)
// granularity, confirmed against the CM language spec
// (/home/ceciliapeng/CM/CM_spec/cmlangspec/cmlangspec.html) and verified bit-exact end to end on
// real hardware. No predicate vector is needed either (the Pred parameter defaults to all-true),
// so this is a plain 2-argument call. That made the uint32-packed/extract-low-half-of-a-dword
// fallback originally expected to be necessary unneeded.
template <int N>
CM_INLINE vector<half, N> hf8_e4m3_decode_to_f16_lut(vector_ref<uchar, N> raw, svmptr_t lut_base) {
    // Widen to uint before the *sizeof(half) multiply: raw is 0..255, so plain uchar arithmetic
    // would overflow (255*2==510) well before a later widening could save it -- go via
    // vector<uint, N> from the start so the multiply itself happens at full width.
    vector<uint, N> idx = raw;
    vector<uint, N> offsets = idx * (uint)sizeof(half);
    return cm_ptr_load<half, VectorSize::N1, DataSize::U16, CacheHint::Cached, CacheHint::Cached>(
        reinterpret_cast<half*>(lut_base), offsets);
}

#ifdef CM_HAS_LSC_UNTYPED_2D
// sdpa_kernel_lsc: flash-attention body shared by the plain fp16 KV-cache path (KVElemT=half,
// the default) and the fp8e4m3 KV-cache path (KVElemT=uchar). The two used to be separate
// functions (sdpa_kernel_lsc / sdpa_kernel_lsc_fp8) that differed ONLY in load_slm_KV below -- see
// cm_pa_xe2.hpp's pa_lsc_u8/load_slm_KV_active for the architectural precedent this mirrors (int8
// affine dequant staged into SLM once per work-group instead of once per work-item), ported here
// to fp8e4m3 LUT-gather decode. A prior per-work-item decode path (every one of the
// wg_local_size=16 work-items in a work-group independently decoding the SAME underlying K/V
// bytes at every kv_pos) was measured slower than this decode-once-per-work-group approach and has
// since been removed -- load_slm_KV decodes each (kv_pos, head_size column range) byte range to
// half EXACTLY ONCE per work-group when KVElemT=uchar -- half the work-group's lanes decode+stage
// K, the other half decode+stage V -- and every lane then reads the already-decoded half data back
// out of SLM via the unmodified (type-agnostic) ugemm_KQ/ugemm_PV0/ugemm_PV1 helpers below: by the
// time those run, SLM already holds plain half regardless of K/V's original on-the-wire dtype, so
// no change is needed there. When KVElemT=half, load_slm_KV instead loads K/V directly as half
// (no decode), i.e. byte-for-byte the original sdpa_kernel_lsc behavior.
//
// Always uses hf8_e4m3_decode_to_f16_lut (the measured-faster LUT-gather decode -- see that
// function's doc comment above) rather than the bit-manipulation decode, for the KVElemT=uchar
// case. See KV_FP8E4M3 in cm_sdpa_opt_prefill.cm for that call site.
//
// Causal-only, no dense attn_mask (this is still a focused perf experiment on the decode-once
// hypothesis for the fp8 case, not a feature-parity requirement with sdpa_kernel_lsc_prefetch's
// fuller causal+attn_mask+sparse_block_mask surface).
//
// sparse_block_size (default 1, dense): when >1, KV chunks the mask says to skip have their ENTIRE
// load_slm_KV+barrier-pipeline stage skipped outright (no decode, no DPAS), mirroring
// sdpa_kernel_lsc_prefetch's / cm_pa_xe2.hpp's pa_lsc_u8's OPTIMIZED_SPARSE_PIPELINE block-granular
// skip. The barrier-synchronized LOAD/skip decision is always made at 256-token (wg_seq_len)
// granularity -- the coarsest unit every lane in a work-group can safely agree on in one shot --
// regardless of sparse_block_size:
//
//   * sparse_block_size == 256 (== wg_seq_len): q_block = q_start/256 is IDENTICAL for every lane
//     in a work-group by construction (every lane's q_start falls inside the same 256-wide range),
//     so sparse_mask_base already points every lane at the SAME mask row -- the skip decision is
//     read-identical across the whole work-group with no merging needed, and skip-LOAD and
//     skip-COMPUTE collapse into the one decision. sparse_mask_wg_lo_base/sparse_mask_wg_hi_base
//     are unused in this case.
//
//   * sparse_block_size == 128 (half a work-group's span): two DIFFERENT lanes in the SAME
//     work-group can map to two DIFFERENT q_blocks (q_block = q_start/128 differs between
//     wg_local_id < 8 and >= 8), so sparse_mask_base -- this lane's OWN row -- is NOT
//     work-group-uniform, and using it directly for the barrier-pipelined skip-LOAD decision would
//     let lanes disagree on whether to run a given chunk's barriers at all: the exact
//     lockstep-violation deadlock class that hung this kernel's causal-only predecessor at
//     multi-work-group shapes (see kv_stop_wg's own doc comment in cm_sdpa_opt_prefill.cm). Instead
//     this case uses a DIFFERENT, already work-group-uniform pair of rows for the skip-LOAD
//     decision -- sparse_mask_wg_lo_base/sparse_mask_wg_hi_base, the two q_blocks (q_block=2*wg_id
//     and 2*wg_id+1) this work-group's 256-token span actually covers, computed by the caller from
//     wg_id (not q_start, hence identical for every lane) -- and skips a 256-token chunk's LOAD
//     only when NEITHER of those two rows wants ANY of its two 128-token halves (an OR over 2 rows
//     x 2 sub-blocks = 4 bits). A chunk that's active for only ONE of the two q_blocks therefore
//     still gets loaded for the WHOLE work-group (wasted bandwidth for the half that didn't need
//     it, but still correct and still a net win whenever anything is skippable), and each lane then
//     separately applies skip-COMPUTE at its own finer 128-token granularity using its OWN
//     sparse_mask_base (purely numeric -- masking this lane's own St contribution to -3.4e38 -- so
//     no barrier/control-flow divergence risk, unlike the load decision). This two-level mask
//     split mirrors cm_pa_xe2.hpp's pa_lsc_u8 sparse_mask_base (skip-COMPUTE) /
//     wg_sparse_mask_base (skip-LOAD) distinction in its legacy (non-OPTIMIZED_SPARSE_PIPELINE)
//     path, specialized here to the fixed 2x relationship between sparse_block_size=128 and
//     wg_seq_len=256 this kernel's dispatch convention (wg_local_size=16, q_step=16) always has.
template <bool use_causal_mask, int num_heads, int num_kv_heads, int head_size, int is_qkv_fused = 0,
          int sparse_block_size = 1, typename KVElemT = half>
void sdpa_kernel_lsc(uint slm_K,
                     uint slm_V,
                     int wg_local_id,
                     int local_size,
                     int q_start,
                     int kv_stop,
                     int q_len,
                     int kv_len,
                     uint q_pitch_bytes,
                     uint k_pitch_bytes,
                     uint v_pitch_bytes,
                     svmptr_t q_base [[type("svmptr_t")]],
                     svmptr_t k_base [[type("svmptr_t")]],
                     svmptr_t v_base [[type("svmptr_t")]],
                     svmptr_t o_base [[type("svmptr_t")]],
                     // Output row pitch in bytes, defaulting to the original hardcoded
                     // num_heads*head_size*sizeof(half) (heads-inner-per-token layout) so existing
                     // KVElemT=half callers see identical behavior. Pass a runtime value instead
                     // for cm_sdpa_opt_prefill.cm's OUTPUT_TRANSPOSE_ORDER callers, regardless of
                     // whether that's the identity (heads-outer) or swapped (heads-inner) layout --
                     // mirrors sdpa_kernel_lsc_prefetch's own o_pitch_bytes param.
                     uint o_pitch_bytes = num_heads * head_size * sizeof(half),
                     // 256-entry half LUT base for hf8_e4m3_decode_to_f16_lut (index == raw
                     // fp8e4m3 byte value), only read when KVElemT=uchar -- see hf8_e4m3_ref.py's
                     // build_hf8_decode_lut_f16 (host side) and KV_FP8E4M3's extra kernel param in
                     // cm_sdpa_opt_prefill.cm (device side, where this pointer comes from).
                     svmptr_t hf8_lut_base [[type("svmptr_t")]] = 0,
                     // XAttention/block-sparsity mask row for THIS WORK-ITEM's own q_block
                     // (only read when sparse_block_size>1). Already offset by the caller to
                     // this (batch,q_head,q_block) row, same convention as
                     // sdpa_kernel_lsc_prefetch's sparse_mask_base. Used for skip-COMPUTE at
                     // sparse_block_size==128 (see sparse_block_size's doc comment above); at
                     // sparse_block_size==256 it's already work-group-uniform and doubles as
                     // the skip-LOAD row too (sparse_mask_wg_lo/hi_base below are unused then).
                     svmptr_t sparse_mask_base [[type("svmptr_t")]] = 0,
                     // The two q_block rows (q_block=2*wg_id and 2*wg_id+1) this WORK-GROUP's
                     // 256-token span covers -- only meaningful (and only ever non-null) when
                     // sparse_block_size==128; computed by the caller from wg_id so every lane
                     // sees the same two pointers. Used ONLY for the work-group-uniform
                     // skip-LOAD decision -- see sparse_block_size's doc comment above. hi may
                     // be 0 if this work-group's span only has one valid q_block left (q_len
                     // not a multiple of 256).
                     svmptr_t sparse_mask_wg_lo_base [[type("svmptr_t")]] = 0,
                     svmptr_t sparse_mask_wg_hi_base [[type("svmptr_t")]] = 0) {
    static_assert(std::is_same<KVElemT, half>::value || std::is_same<KVElemT, uchar>::value,
                  "KVElemT must be half (plain fp16 KV-cache) or uchar (fp8e4m3-packed KV-cache, "
                  "LUT-decoded once per work-group in load_slm_KV)");
    constexpr bool fp8_kv = std::is_same<KVElemT, uchar>::value;

    uint o_pitch = o_pitch_bytes;
    // round up head_size to multiple of 16
    // block_2d_desc will automatically handle the tailing block
    constexpr int padded_head_size = (head_size + 16 - 1) / 16 * 16;
    vector<float, q_step> cur_max;
    vector<float, q_step> cur_sum;

    cur_max = -3e38f;
    cur_sum = 0;
    constexpr int num_P_tiles = REG_N / REG_M;
    matrix<half, padded_head_size / REG_K, REG_K * REG_N> rQ;
    matrix<float, padded_head_size / REG_N * num_P_tiles, REG_M * REG_N> rO;
    rO = 0.0f;  // Zero the accumulator: the first softmax block scales rO by max_comp==0, and 0*NaN==NaN if the GRF holds stale NaN/Inf bits.

    auto q_tokens_left = q_len;
    static_assert(q_step == REG_N);
    static_assert(kv_step == REG_K);

    if (q_tokens_left < 0)
        q_tokens_left = 0;
    if (q_tokens_left > q_step)
        q_tokens_left = q_step;

    if (q_tokens_left > 0) {
        lsc::block_2d_desc<uint, 1, REG_N, REG_K / 2> b2dQ(reinterpret_cast<uint*>(q_base),
                                                           q_tokens_left - 1,
                                                           head_size * sizeof(half) - 1,
                                                           q_pitch_bytes - 1,
                                                           0,
                                                           0);
#    pragma unroll
        for (int k = 0, ri = 0; k < padded_head_size / 2; k += REG_K / 2, ri++) {
            cm_load<lsc::Transpose>(rQ[ri].format<uint>(), b2dQ.set_block_x(k));
            rQ[ri].format<half>() = cm_mul<half>(rQ[ri].format<half>(), (half)q_prescale);
        }
    }

    // K/V element type is KVElemT: plain half for the fp16 KV-cache path (unchanged descriptor),
    // or fp8e4m3-packed uchar (1 byte/elem, decoded to half exactly once per work-group in
    // load_slm_KV below) for the fp8 path.
    lsc::block_2d_desc<KVElemT, 1, kv_step, REG_K> b2dK(k_base, kv_stop - 1, head_size * sizeof(KVElemT) - 1, k_pitch_bytes - 1, 0, 0);
    lsc::block_2d_desc<KVElemT, 1, REG_K, REG_N> b2dV(v_base, kv_stop - 1, head_size * sizeof(KVElemT) - 1, v_pitch_bytes - 1, 0, 0);

    int causal_left = q_start;

    // SLM always holds decoded half regardless of K/V's on-the-wire dtype, so the buffer size and
    // quad-buffering scheme are identical for both KVElemT instantiations.
    constexpr uint slm_buff_size = kv_step * head_size * sizeof(half);
    int slm_buff_id_write = 0;
    int slm_buff_id_read = 0;

    // The ONLY place the two KVElemT instantiations differ: a plain half load straight into SLM
    // (KVElemT=half), or a load of raw fp8e4m3 bytes + LUT-decode-to-half before the SLM write
    // (KVElemT=uchar, "decode-once-per-work-group" -- see this function's doc comment above).
    // Everything below this lambda (the sparse_block_size dispatch, softmax, output store) is
    // shared verbatim between both element types.
    auto load_slm_KV = [&](int kv_pos) {
        if (kv_pos < kv_stop) {
            uint slm_offset = (slm_buff_id_write & 3) * slm_buff_size;
            slm_buff_id_write++;
            if (wg_local_id < local_size / 2) {
                if constexpr (fp8_kv) {
                    // K (simple case, decode-once-per-WG): load raw fp8e4m3 bytes, LUT-decode to
                    // half, write plain (no VNNI) half into SLM.
                    vector<uchar, kv_step * REG_K> temp0_u8;
                    vector<half, kv_step * REG_K> temp0;
                    b2dK.set_block_y(kv_pos);
                    for (int k = REG_K * wg_local_id; k < padded_head_size; k += REG_K * (local_size / 2)) {
                        cm_load<lsc::Normal>(temp0_u8, b2dK.set_block_x(k));
                        temp0 = hf8_e4m3_decode_to_f16_lut<kv_step * REG_K>(temp0_u8, hf8_lut_base);
                        cm_slm_block_write(slm_K, slm_offset + k * kv_step * sizeof(half), temp0);
                    }
                } else {
                    vector<half, kv_step * REG_K> temp0;
                    b2dK.set_block_y(kv_pos);
                    for (int k = REG_K * wg_local_id; k < padded_head_size; k += REG_K * (local_size / 2)) {
                        cm_load<lsc::Normal>(temp0, b2dK.set_block_x(k));
                        cm_slm_block_write(slm_K, slm_offset + k * kv_step * sizeof(half), temp0);
                    }
                }
            } else {
                if constexpr (fp8_kv) {
                    // V (harder case, decode-once-per-WG): hardware cm_load<lsc::VNNI> only knows
                    // how to half-pair-interleave 2-byte (half) source elements, so the raw
                    // fp8e4m3 bytes are loaded as a PLAIN (non-VNNI) 2D block, LUT-decoded to half,
                    // then the exact VNNI2 interleave hardware would have produced from a half
                    // source is reproduced manually via select<>() -- verbatim copy of
                    // sdpa_accumulate_kv_source's V-path pattern above (see that function's doc
                    // comment for the full rationale), just staged into SLM instead of consumed
                    // immediately by DPAS.
                    matrix<uchar, REG_K, REG_N> Vmat_u8;
                    matrix<half, REG_K, REG_N> Vmat_nat;
                    matrix<half, REG_K / 2, REG_N * 2> temp2;
                    b2dV.set_block_y(kv_pos);
#    pragma unroll
                    for (int k = REG_N * (wg_local_id - (local_size / 2)); k < padded_head_size; k += REG_N * (local_size / 2)) {
                        cm_load<lsc::Normal>(Vmat_u8.format<uchar>(), b2dV.set_block_x(k));
                        Vmat_nat.format<half>() = hf8_e4m3_decode_to_f16_lut<REG_K * REG_N>(Vmat_u8.format<uchar>(), hf8_lut_base);
                        temp2.select<REG_K / 2, 1, REG_N, 2>(0, 0) = Vmat_nat.select<REG_K / 2, 2, REG_N, 1>(0, 0);
                        temp2.select<REG_K / 2, 1, REG_N, 2>(0, 1) = Vmat_nat.select<REG_K / 2, 2, REG_N, 1>(1, 0);
                        cm_slm_block_write(slm_V, slm_offset + k * REG_K * sizeof(half), temp2.format<half>());
                    }
                } else {
                    vector<half, REG_K * REG_N> temp2;
                    b2dV.set_block_y(kv_pos);
#    pragma unroll
                    for (int k = REG_N * (wg_local_id - (local_size / 2)); k < padded_head_size; k += REG_N * (local_size / 2)) {
                        cm_load<lsc::VNNI>(temp2, b2dV.set_block_x(k));
                        cm_slm_block_write(slm_V, slm_offset + k * REG_K * sizeof(half), temp2);
                    }
                }
            }
        }
    };

    if constexpr (sparse_block_size == 1) {
        // Dense path: byte-for-byte unchanged from before sparse_block_size existed.
        load_slm_KV(0);
        load_slm_KV(kv_step);
        cm_slm_fence(CM_LOCAL_BARRIER);
        cm_sbarrier(1);

        for (int kv_pos = 0; kv_pos < kv_stop; kv_pos += kv_step, slm_buff_id_read++) {
            //  load0, load1, signal1,
            //  [wait2, signal2, load2, read0]
            //  [wait3, signal3, load3, read1]
            //  [wait4, signal4, load4, read2]
            //  [wait5, signal5, load5, read3]
            //
            //  after wait4, all workers have reached signal3, so:
            //     - all workers have finished load2 & read0.
            //     - we can start to load 4 into SLM slot 0 (i & 3) safely
            //     - we can start to read 2 ((i-2) & 3) safely

            cm_fence(CM_LOCAL_BARRIER);
            cm_sbarrier(0);
            if (kv_pos + kv_step < kv_stop)
                cm_sbarrier(1);

            load_slm_KV(kv_pos + kv_step * 2);

            {
                uint slm_offset = (slm_buff_id_read & 3) * slm_buff_size;
                //# St = k @ Qt
                matrix<float, kv_step, q_step> St = ugemm_KQ(slm_K, rQ, slm_offset);

                if constexpr (use_causal_mask) {
                    // since kv_step == q_step == 16, causal_left is n*kv_step
                    if (causal_left == 0) {
                        apply_causal_mask<1>(St);
                    } else if (causal_left < 0) {
                        St = -3.4e38f;
                    }
                    causal_left -= kv_step;
                } else {
                    int kv_tokens = kv_stop - kv_pos;
                    // LSC ensures no overflow-access, but mask off k-tails attn-score is still required
                    for (int p = kv_tokens; p < kv_step; p++)
                        St[p] = -3.4e38f;
                }

                auto max_comp = cm_online_softmax_update(St, cur_max, cur_sum);

                matrix<half, REG_N, REG_K> P;
                Transpose2DMatrix(St, P);

                if (kv_pos == 0)
                    ugemm_PV0(slm_V, P, rO, slm_offset);
                else
                    ugemm_PV1(slm_V, P, max_comp, rO, slm_offset);
            }
        }
        // cm_sbarrier(0);
    } else if constexpr (sparse_block_size == 256) {
        // Block-granular sparse path, sparse_block_size == wg_seq_len (see sparse_block_size's doc
        // comment above). Restructured into an outer kv_blk loop (one sparse_block_size-sized
        // chunk per iteration): a skipped chunk `continue`s past its ENTIRE load_slm_KV+barrier
        // pipeline (no decode, no DPAS -- the actual perf win), while an active chunk runs the
        // exact same per-kv_step barrier-pipelined body the dense path above uses, just scoped to
        // [kv_blk, blk_end) and with its own fresh quad-buffer ring (reset to 0, mirroring
        // cm_pa_xe2.hpp's pa_lsc_u8 OPTIMIZED_SPARSE_PIPELINE's "reset ring counters per active
        // block").
        constexpr int sb_shift = 8;

        for (int kv_blk = 0; kv_blk < kv_stop; kv_blk += sparse_block_size) {
            int blk_end = kv_blk + sparse_block_size;
            if (blk_end > kv_stop)
                blk_end = kv_stop;

            // sparse_mask_base already points every lane at the SAME (batch,q_head,q_block) row
            // (see this function's doc comment), so every lane in the work-group evaluates `skip`
            // identically and either ALL take the `continue` below or ALL enter the active-chunk
            // pipeline -- no lane can diverge from the rest at this branch.
            bool skip = sparse_mask_base && !((const bool*)sparse_mask_base)[(uint)kv_blk >> sb_shift];
            if (skip) {
                // Match the per-step decrement the active path below would have applied over the
                // same number of kv_step iterations (ceil((blk_end-kv_blk)/kv_step) steps, each
                // decrementing by kv_step) -- using the nominal sparse_block_size here is exact for
                // every chunk except a final partial tail chunk, and a wrong causal_left after the
                // very last chunk is harmless (the loop ends there; no later chunk ever reads it).
                if constexpr (use_causal_mask)
                    causal_left -= sparse_block_size;
                continue;
            }

            // Full work-group barrier before this active chunk starts overwriting the SLM ring
            // buffer: the steady-state per-kv_step cm_sbarrier/cm_fence pairing below only proves
            // "every lane has finished reading slot N-2" *within a single uninterrupted run of the
            // loop* (as in the dense path above) -- it says nothing about a lane that already
            // raced ahead past a `continue`'d skip chunk while another lane is still finishing this
            // chunk's PREDECESSOR active chunk's tail reads (ugemm_KQ/ugemm_PV read SLM data other
            // lanes wrote, so a stale straggler reader here would see corrupted data, not just its
            // own). A plain cm_barrier() (every lane blocks until all lanes arrive, unlike the
            // split cm_sbarrier pair) closes that gap unconditionally and cheaply -- once per
            // up-to-256-token chunk, not per 16-token step -- removing any doubt instead of
            // relying on an unverified assumption about cm_sbarrier's cross-chunk semantics.
            cm_slm_fence(CM_LOCAL_BARRIER);
            cm_barrier();

            slm_buff_id_write = 0;
            slm_buff_id_read = 0;

            load_slm_KV(kv_blk);
            load_slm_KV(kv_blk + kv_step);
            cm_slm_fence(CM_LOCAL_BARRIER);
            cm_sbarrier(1);

            for (int kv_pos = kv_blk; kv_pos < blk_end; kv_pos += kv_step, slm_buff_id_read++) {
                cm_fence(CM_LOCAL_BARRIER);
                cm_sbarrier(0);
                if (kv_pos + kv_step < blk_end)
                    cm_sbarrier(1);

                // Bounded by kv_stop (not blk_end): may prefetch a step or two into the NEXT
                // chunk's KV range (same contiguous buffer, no page-table indirection unlike
                // cm_pa_xe2.hpp's precedent, so this is always an in-bounds, merely-redundant read
                // if that next chunk turns out to be skipped) -- the resulting SLM write is simply
                // never consumed by THIS chunk's loop (which stops at blk_end) and gets overwritten
                // by the next active chunk's own ring reset above.
                load_slm_KV(kv_pos + kv_step * 2);

                {
                    uint slm_offset = (slm_buff_id_read & 3) * slm_buff_size;
                    //# St = k @ Qt
                    matrix<float, kv_step, q_step> St = ugemm_KQ(slm_K, rQ, slm_offset);

                    if constexpr (use_causal_mask) {
                        // since kv_step == q_step == 16, causal_left is n*kv_step
                        if (causal_left == 0) {
                            apply_causal_mask<1>(St);
                        } else if (causal_left < 0) {
                            St = -3.4e38f;
                        }
                        causal_left -= kv_step;
                    } else {
                        int kv_tokens = kv_stop - kv_pos;
                        // LSC ensures no overflow-access, but mask off k-tails attn-score is still required
                        for (int p = kv_tokens; p < kv_step; p++)
                            St[p] = -3.4e38f;
                    }

                    auto max_comp = cm_online_softmax_update(St, cur_max, cur_sum);

                    matrix<half, REG_N, REG_K> P;
                    Transpose2DMatrix(St, P);

                    // kv_pos==0 implies kv_blk==0 (kv_pos ranges over [kv_blk, blk_end)), i.e. this
                    // is still exactly "the very first KV token this work-item ever processes",
                    // same condition the dense path above uses -- unaffected by sparsity. If chunk
                    // 0 itself is skipped, every later chunk's first iteration instead takes the
                    // ugemm_PV1 branch with rO still at its zero-init value: rescaling 0 by
                    // max_comp (finite, since cm_online_softmax_update never produces NaN/Inf here)
                    // is exactly 0 again, so PV1 is bit-identical to PV0 in that case too -- no
                    // separate "first ACTIVE chunk" tracking needed.
                    if (kv_pos == 0)
                        ugemm_PV0(slm_V, P, rO, slm_offset);
                    else
                        ugemm_PV1(slm_V, P, max_comp, rO, slm_offset);
                }
            }
        }
    } else {
        // sparse_block_size == 128: half a work-group's 256-token span -- see sparse_block_size's
        // doc comment above for the full two-level (skip-LOAD vs skip-COMPUTE) rationale this
        // implements. static_assert below documents this is the only other value reaching here (the
        // outer if constexpr chain above already handles 1 and 256).
        static_assert(sparse_block_size == 128, "sparse_block_size must be 1, 128, or 256");
        // Both the per-lane row (sparse_mask_base) and the work-group-merged rows
        // (sparse_mask_wg_lo/hi_base) are laid out at the mask's NATIVE sparse_block_size==128
        // granularity (one bool per 128 kv tokens) -- the "load decision is effectively at 256
        // granularity" is achieved below by OR-ing TWO adjacent shift-7 indices (kv_blk>>7 and
        // (kv_blk+128)>>7) together, not by using a different shift amount.
        constexpr int sb_shift = 7;

        // Barrier-chunk stride is ALWAYS 256 (wg_seq_len) here, not sparse_block_size (128) --
        // the skip-LOAD decision below must be made once per chunk by every lane identically, and
        // 256 is the coarsest unit every lane in the work-group can agree on without merging (see
        // doc comment). The finer 128-token granularity is applied purely as numeric skip-COMPUTE
        // masking inside the active chunk's kv_step loop below, which carries no barrier risk.
        constexpr int blk_stride = 256;

        for (int kv_blk = 0; kv_blk < kv_stop; kv_blk += blk_stride) {
            int blk_end = kv_blk + blk_stride;
            if (blk_end > kv_stop)
                blk_end = kv_stop;

            // Work-group-uniform skip-LOAD decision: OR across the WG's two sparse_block_size=128
            // sub-halves of this chunk (kv_blk and kv_blk+128) AND across the WG's own two q_block
            // rows (lo, hi -- computed by the caller from wg_id, so every lane in the work-group
            // reads the exact same two pointers and therefore reaches the exact same `skip` value;
            // no lane can diverge here). A null row (base==0, e.g. no mask supplied, or hi==0 when
            // this work-group's span only has one valid q_block left) contributes "nothing to
            // load" rather than "everything", matching the dense/bypass convention elsewhere in
            // this function.
            auto chunk_needed = [&](svmptr_t base) -> bool {
                if (!base)
                    return false;
                bool need = ((const bool*)base)[(uint)kv_blk >> sb_shift];
                if (kv_blk + 128 < blk_end)
                    need = need || ((const bool*)base)[(uint)(kv_blk + 128) >> sb_shift];
                return need;
            };
            bool skip = !(chunk_needed(sparse_mask_wg_lo_base) || chunk_needed(sparse_mask_wg_hi_base));
            if (skip) {
                // Same fixed-size decrement discipline as the sparse_block_size==256 branch above
                // (a wrong causal_left after the very last, possibly-partial chunk is harmless: the
                // loop ends there, nothing reads it again).
                if constexpr (use_causal_mask)
                    causal_left -= blk_stride;
                continue;
            }

            // Full work-group barrier before reusing SLM for this active chunk -- same rationale as
            // the sparse_block_size==256 branch's identical barrier above (protects against a
            // straggler lane still reading the PREVIOUS active chunk's tail while another lane
            // starts overwriting the same ring-buffer slots after a `continue`'d skip in between).
            cm_slm_fence(CM_LOCAL_BARRIER);
            cm_barrier();

            slm_buff_id_write = 0;
            slm_buff_id_read = 0;

            load_slm_KV(kv_blk);
            load_slm_KV(kv_blk + kv_step);
            cm_slm_fence(CM_LOCAL_BARRIER);
            cm_sbarrier(1);

            for (int kv_pos = kv_blk; kv_pos < blk_end; kv_pos += kv_step, slm_buff_id_read++) {
                cm_fence(CM_LOCAL_BARRIER);
                cm_sbarrier(0);
                if (kv_pos + kv_step < blk_end)
                    cm_sbarrier(1);

                // Bounded by kv_stop, may prefetch into the next chunk -- same rationale as the
                // sparse_block_size==256 branch above (same contiguous buffer, no page-table
                // indirection, so always in-bounds and merely redundant if wasted).
                load_slm_KV(kv_pos + kv_step * 2);

                {
                    uint slm_offset = (slm_buff_id_read & 3) * slm_buff_size;
                    //# St = k @ Qt
                    matrix<float, kv_step, q_step> St = ugemm_KQ(slm_K, rQ, slm_offset);

                    if constexpr (use_causal_mask) {
                        // since kv_step == q_step == 16, causal_left is n*kv_step
                        if (causal_left == 0) {
                            apply_causal_mask<1>(St);
                        } else if (causal_left < 0) {
                            St = -3.4e38f;
                        }
                        causal_left -= kv_step;
                    } else {
                        int kv_tokens = kv_stop - kv_pos;
                        // LSC ensures no overflow-access, but mask off k-tails attn-score is still required
                        for (int p = kv_tokens; p < kv_step; p++)
                            St[p] = -3.4e38f;
                    }

                    // Skip-COMPUTE: THIS lane's own 128-token-granularity mask bit, read from its
                    // own (q_block-specific) row -- applied AFTER the causal/tail masking above so
                    // it can only ever add masking, never remove it. Pure numeric op on this lane's
                    // own register data (St), no control flow or barrier involvement, so unlike the
                    // skip-LOAD decision above it's safe for every lane to evaluate independently
                    // even though sparse_mask_base genuinely differs between the two halves of this
                    // work-group at sparse_block_size==128 (see doc comment above).
                    if (sparse_mask_base && !((const bool*)sparse_mask_base)[(uint)kv_pos >> sb_shift])
                        St = -3.4e38f;

                    auto max_comp = cm_online_softmax_update(St, cur_max, cur_sum);

                    matrix<half, REG_N, REG_K> P;
                    Transpose2DMatrix(St, P);

                    // Same "kv_pos==0 implies first-ever-write, PV1 is bit-identical to PV0 when
                    // rO is still zero" reasoning as the sparse_block_size==256 branch above.
                    if (kv_pos == 0)
                        ugemm_PV0(slm_V, P, rO, slm_offset);
                    else
                        ugemm_PV1(slm_V, P, max_comp, rO, slm_offset);
                }
            }
        }
    }
    if (q_tokens_left == 0)
        return;

    //# save cur_O/cur_sum.transpose(0, 1)
    matrix<half, num_P_tiles * REG_M, REG_N> cur_O_f16;
    // Guard the all-masked-row case (every KV block skipped by a sparse block mask, or an
    // otherwise-empty attended range): cur_sum stays exactly 0, and an unguarded cm_inv(0) -> +Inf
    // combines with the (also-zero) rO accumulator into 0*Inf == NaN. Match the CPU reference
    // (ov::reference::block_sparse_attention's `sum > 0.0f ? 1.0f/sum : 0.0f`) by forcing the
    // reciprocal to 0 wherever the sum was 0, so an empty row's output is 0 instead of NaN.
    vector<unsigned short, q_step> cur_sum_is_zero = (cur_sum == 0.0f);
    cur_sum = cm_inv(cur_sum);
    cur_sum.merge(0.0f, cur_sum_is_zero);

    lsc::block_2d_desc<half, 1, REG_M, REG_N> b2dO(o_base, q_tokens_left - 1, head_size * sizeof(half) - 1, o_pitch - 1, 0, 0);

#    pragma unroll
    for (int k = 0, ri = 0; k < padded_head_size; k += REG_N, ri += num_P_tiles) {
#    pragma unroll
        for (int p = 0; p < num_P_tiles; p++) {
            auto cO = rO[ri + p].format<float, REG_M, REG_N>();
#    pragma unroll
            for (int r = 0; r < cO.n_rows(); r++) {
                cur_O_f16[r + p * REG_M] = cm_mul<float>(cO.row(r), cur_sum[r + p * REG_M]);
            }
        }
        b2dO.set_block_x(k);
        cm_store(b2dO.set_block_y(0), cur_O_f16.format<half, num_P_tiles, REG_M * REG_N>().row(0));
        cm_store(b2dO.set_block_y(REG_M), cur_O_f16.format<half, num_P_tiles, REG_M * REG_N>().row(1));
    }
}

template <bool use_causal_mask,
          int num_heads,
          int num_kv_heads,
          int head_size,
          int is_qkv_fused,
          int wg_local_size,
          bool has_attn_mask = false,
          int sparse_block_size = 1>
void sdpa_kernel_lsc_prefetch(int wg_local_id,
                              int q_start,
                              int kv_stop,
                              int q_len,
                              int kv_len,
                              uint q_pitch_bytes,
                              uint k_pitch_bytes,
                              uint v_pitch_bytes,
                              svmptr_t q_base [[type("svmptr_t")]],
                              svmptr_t k_base [[type("svmptr_t")]],
                              svmptr_t v_base [[type("svmptr_t")]],
                              svmptr_t o_base [[type("svmptr_t")]],
                              // Defaults to today's heads-inner-per-token layout (unchanged behavior for
                              // existing callers, e.g. cm_sdpa_vlen.cm). Pass head_size*sizeof(half) instead
                              // for a heads-outer ([B,H,S,D]) output layout, mirroring how
                              // k_pitch_bytes/v_pitch_bytes already let callers pick a layout.
                              uint o_pitch_bytes = num_heads * head_size * sizeof(half),
                              // Runtime Q pre-scale, defaulting to the compile-time q_prescale (==
                              // scale_factor*log2e or scale_factor, per CMFLA_Q_SCALED_BY_LOG2) so existing
                              // callers see identical behavior. Pass a value derived from a runtime `scale`
                              // input instead to override CMFLA_SCALE_FACTOR.
                              float q_prescale_rt = q_prescale,
                              // Optional additive attn_mask [Sq,Sk] (only read when has_attn_mask=true);
                              // mask values are added to the raw St score pre-scaled by q_prescale_rt,
                              // matching how Q itself is pre-scaled (sdpa_opt.cl's load_attn_mask semantics).
                              svmptr_t mask_base [[type("svmptr_t")]] = 0,
                              uint mask_pitch_bytes = 0,
                              // XAttention/block-sparsity mask (only read when sparse_block_size>1): one
                              // bool per sparse_block_size-sized KV chunk for this (batch,head,q_block) row,
                              // 1=keep/0=skip. Already offset by the caller to point at this work-item's row
                              // -- see cm_sdpa_opt_prefill.cm / block_sparse_attention_prefill.cm.
                              svmptr_t sparse_mask_base [[type("svmptr_t")]] = 0) {
    uint o_pitch = o_pitch_bytes;
    // round up head_size to multiple of 16
    // block_2d_desc will automatically handle the tailing block
    constexpr int padded_head_size = (head_size + 16 - 1) / 16 * 16;

    vector<float, q_step> cur_max;
    vector<float, q_step> cur_sum;

    cur_max = -3e38f;
    cur_sum = 0;
    constexpr int num_P_tiles = REG_N / REG_M;
    matrix<half, padded_head_size / REG_K, REG_K * REG_N> rQ;
    matrix<float, padded_head_size / REG_N * num_P_tiles, REG_M * REG_N> rO;
    rO = 0.0f;

    auto q_tokens_left = q_len;  // - q_start;
    static_assert(q_step == REG_N);
    static_assert(kv_step == REG_K);

    if (q_tokens_left < 0)
        q_tokens_left = 0;
    if (q_tokens_left > q_step)
        q_tokens_left = q_step;

    if (q_tokens_left > 0) {
        lsc::block_2d_desc<uint, 1, REG_N, REG_K / 2> b2dQ(reinterpret_cast<uint*>(q_base),
                                                           q_tokens_left - 1,
                                                           head_size * sizeof(half) - 1,
                                                           q_pitch_bytes - 1,
                                                           0,
                                                           0);
#    pragma unroll
        for (int k = 0, ri = 0; k < padded_head_size / 2; k += REG_K / 2, ri++) {
            cm_load<lsc::Transpose>(rQ[ri].format<uint>(), b2dQ.set_block_x(k));
            rQ[ri].format<half>() = cm_mul<half>(rQ[ri].format<half>(), (half)q_prescale_rt);
        }
    }

    lsc::block_2d_desc<half, 1, kv_step, REG_K> b2dK(k_base, kv_stop - 1, head_size * sizeof(half) - 1, k_pitch_bytes - 1, 0, 0);
    lsc::block_2d_desc<half, 1, REG_K, REG_N> b2dV(v_base, kv_stop - 1, head_size * sizeof(half) - 1, v_pitch_bytes - 1, 0, 0);
    // Mask surface is [Sq,Sk] (q rows outer, kv cols inner) -- same row range (q_tokens_left,
    // starting at this tile's q_start via the caller's pointer offset) as b2dQ above, so a plain
    // (non-transpose) load followed by an on-chip Transpose2DMatrix gives the [kv,q] layout St
    // needs, without requiring an LSC hardware-transpose descriptor for the mask.
    lsc::block_2d_desc<half, 1, q_step, kv_step> b2dMask(mask_base, q_tokens_left - 1, kv_len * sizeof(half) - 1, mask_pitch_bytes - 1, 0, 0);

    static_assert(wg_local_size == 16);
    lsc::block_2d_desc<half, 1, kv_step / wg_local_size, REG_K> prefetch_K(k_base, kv_stop - 1, head_size * sizeof(half) - 1, k_pitch_bytes - 1, 0, 0);
    lsc::block_2d_desc<half, 1, REG_K / wg_local_size, REG_N> prefetch_V(v_base, kv_stop - 1, head_size * sizeof(half) - 1, v_pitch_bytes - 1, 0, 0);

    int causal_left = q_start;

    // KV-blocking factor: number of kv tiles (each kv_step rows) processed per online-
    // softmax update. The per-iteration rO rescale (REG_M*num_rO_tiles muls) and the
    // softmax bookkeeping amortize over KV_BLK tiles, cutting the ALU-bound consume cost.
    // The K@Q^T and the St->P transpose are still per sub-tile (they don't amortize).
    constexpr int KV_BLK = CMFLA_KV_BLK;
    constexpr int num_K = kv_step / REG_M;
    constexpr int BLK_ROWS = KV_BLK * kv_step;

    // Warm up first K and V tiles before the loop so kv_pos=0 finds them in cache.
    prefetch_K.set_block_y(wg_local_id);
    prefetch_V.set_block_y(wg_local_id);
    cm_prefetch<CacheHint::Cached, CacheHint::Cached>(prefetch_K.set_block_x(0));
    cm_prefetch<CacheHint::Cached, CacheHint::Cached>(prefetch_V.set_block_x(0));
#    pragma unroll
    for (int ri = 1; ri < padded_head_size / REG_K; ri++) {
        cm_prefetch<CacheHint::Cached, CacheHint::Cached>(prefetch_K.set_block_x(ri * REG_K));
        cm_prefetch<CacheHint::Cached, CacheHint::Cached>(prefetch_V.set_block_x(ri * REG_N));
    }

    for (int kv_base = 0; kv_base < kv_stop; kv_base += BLK_ROWS) {
        // ---- block-sparsity: skip this whole KV_BLK chunk (no load, no DPAS) when its
        // sparse_block_size-sized bucket is masked out. causal_left still has to be decremented by
        // the chunk's nominal length so later (non-skipped) chunks see the right causal offset --
        // mirrors cm_pa_xe2.hpp's skip_compute bookkeeping exactly.
        if constexpr (sparse_block_size > 1) {
            constexpr int sb_shift = (sparse_block_size == 128) ? 7 : (sparse_block_size == 256) ? 8 : -1;
            static_assert(sb_shift > 0, "sparse_block_size must be 128 or 256");
            static_assert(sparse_block_size % BLK_ROWS == 0,
                          "sparse_block_size must be a multiple of KV_BLK*kv_step so a chunk never straddles a sparse block");
            if (sparse_mask_base && !((const bool*)sparse_mask_base)[(uint)kv_base >> sb_shift]) {
                if constexpr (use_causal_mask)
                    causal_left -= BLK_ROWS;
                continue;
            }
        }
        //# St = K @ Q^T for KV_BLK stacked sub-tiles -> [KV_BLK*kv_step, q_step]
        matrix<float, BLK_ROWS, q_step> St;
        auto St2 = St.format<float, KV_BLK * num_K, REG_M * REG_N>();
#    pragma unroll
        for (int b = 0; b < KV_BLK; b++) {
            int kv_pos = kv_base + b * kv_step;
            matrix<half, num_K, REG_M * REG_K> Kmat;
            // Prefetch K of the matching tile one block ahead and V of this tile, so V
            // is warm by the PV phase and the next block's K is warm by its K phase.
            prefetch_K.set_block_y(wg_local_id + kv_pos + BLK_ROWS);
            prefetch_V.set_block_y(wg_local_id + kv_pos);
            cm_prefetch<CacheHint::Cached, CacheHint::Cached>(prefetch_K.set_block_x(0));
            cm_prefetch<CacheHint::Cached, CacheHint::Cached>(prefetch_V.set_block_x(0));

            b2dK.set_block_y(kv_pos);
            cm_load<lsc::Normal>(Kmat.format<half>(), b2dK.set_block_x(0));
#    pragma unroll
            for (int k = 0; k < num_K; k++)
                St2.row(b * num_K + k) =
                    cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, SystolicDepth, RepeatCount, float>(0, rQ[0].format<int32_t>(), Kmat[k].format<int32_t>());

#    pragma unroll
            for (int ri = 1; ri < padded_head_size / REG_K; ri++) {
                cm_prefetch<CacheHint::Cached, CacheHint::Cached>(prefetch_K.set_block_x(ri * REG_K));
                cm_prefetch<CacheHint::Cached, CacheHint::Cached>(prefetch_V.set_block_x(ri * REG_N));
                cm_load<lsc::Normal>(Kmat.format<half>(), b2dK.set_block_x(ri * REG_K));
#    pragma unroll
                for (int k = 0; k < num_K; k++) {
                    St2.row(b * num_K + k) = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, SystolicDepth, RepeatCount, float>(St2.row(b * num_K + k),
                                                                                                                          rQ[ri].format<int32_t>(),
                                                                                                                          Kmat[k].format<int32_t>());
                }
            }
        }

        // ---- optional additive attn_mask, added in the same [kv,q] domain St lives in ----
        if constexpr (has_attn_mask) {
#    pragma unroll
            for (int b = 0; b < KV_BLK; b++) {
                int kv_pos = kv_base + b * kv_step;
                matrix<half, q_step, kv_step> Mraw;
                cm_load<lsc::Normal>(Mraw.format<half>(), b2dMask.set_block_x(kv_pos));
                matrix<half, kv_step, q_step> Mt;
                Transpose2DMatrix(Mraw.select<q_step, 1, kv_step, 1>(0, 0), Mt);
                auto Stb = St.select<kv_step, 1, q_step, 1>(b * kv_step, 0);
#    pragma unroll
                for (int r = 0; r < kv_step; r++) {
                    vector<float, q_step> mrow = Mt.row(r);
                    Stb.row(r) = cm_add<float>(Stb.row(r), cm_mul<float>(mrow, q_prescale_rt));
                }
            }
        }

        // ---- mask per sub-tile (causal or kv-tail) ----
        if constexpr (use_causal_mask) {
#    pragma unroll
            for (int b = 0; b < KV_BLK; b++) {
                auto Stb = St.select<kv_step, 1, q_step, 1>(b * kv_step, 0);
                int cl = causal_left - b * kv_step;
                if (cl == 0) {
                    apply_causal_mask<1>(Stb);
                } else if (cl < 0) {
                    Stb = -3.4e38f;
                }
            }
            causal_left -= BLK_ROWS;
        } else {
            int kv_tokens = kv_stop - kv_base;
            // LSC ensures no overflow-access, but mask off k-tail attn-score is still required
            for (int p = kv_tokens; p < BLK_ROWS; p++)
                St[p] = -3.4e38f;
        }

        // ---- one online-softmax update over the whole block ----
        auto max_comp = cm_online_softmax_update(St, cur_max, cur_sum);

        // ---- rescale rO (skip first iter only when KV blocking amortizes the branch) ----
        // The kv_base=0 rescale is mathematically redundant for every head size: max_comp=0
        // and rO is still zero.  The skip, however, adds a runtime kv_base branch to every
        // outer KV block.  Enable it only for KV_BLK>=2, where fewer outer iterations make
        // the branch cost small enough to be profitable; KV_BLK=1 paths such as Omni HD=72
        // keep the straight-line rescale because forcing the branch measured as a regression.
        if constexpr (KV_BLK >= 2) {
            if (kv_base > 0) {
#    pragma unroll
                for (int t = 0; t < padded_head_size / REG_N * num_P_tiles; t++) {
                    auto cO = rO[t].format<float, REG_M, REG_N>();
#    pragma unroll
                    for (int r = 0; r < REG_M; r++)
                        cO.row(r) = cm_mul<float>(cO.row(r), max_comp[r + (t % num_P_tiles) * REG_M]);
                }
            }
        } else {
#    pragma unroll
            for (int t = 0; t < padded_head_size / REG_N * num_P_tiles; t++) {
                auto cO = rO[t].format<float, REG_M, REG_N>();
#    pragma unroll
                for (int r = 0; r < REG_M; r++)
                    cO.row(r) = cm_mul<float>(cO.row(r), max_comp[r + (t % num_P_tiles) * REG_M]);
            }
        }

        // ---- transpose each sub-tile and accumulate P@V into rO ----
        constexpr int num_Vchunks = padded_head_size / REG_N;
#    pragma unroll
        for (int b = 0; b < KV_BLK; b++) {
            matrix<half, REG_N, REG_K> P;
            transpose_St_to_P_half(St.select<kv_step, 1, q_step, 1>(b * kv_step, 0), P);
            auto P2 = P.format<half, num_P_tiles, REG_M * REG_K>();
            b2dV.set_block_y(kv_base + b * kv_step);
#    pragma unroll
            for (int ci = 0, ri = 0; ci < num_Vchunks; ci++, ri += num_P_tiles) {
                matrix<half, REG_K / 2, REG_N * 2> Vmat;
                cm_load<lsc::VNNI>(Vmat.format<half>(), b2dV.set_block_x(ci * REG_N));
#    pragma unroll
                for (int p = 0; p < num_P_tiles; p++) {
                    rO[ri + p] = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, SystolicDepth, RepeatCount>(rO[ri + p].format<float>(),
                                                                                                       Vmat.format<int32_t>(),
                                                                                                       P2.row(p).format<int32_t>());
                }
            }
        }
    }
    if (q_tokens_left == 0)
        return;

    //# save cur_O/cur_sum.transpose(0, 1)
    matrix<half, num_P_tiles * REG_M, REG_N> cur_O_f16;
    // Guard the all-masked-row case (every KV block skipped by a sparse block mask, or an
    // otherwise-empty attended range): cur_sum stays exactly 0, and an unguarded cm_inv(0) -> +Inf
    // combines with the (also-zero) rO accumulator into 0*Inf == NaN. Match the CPU reference
    // (ov::reference::block_sparse_attention's `sum > 0.0f ? 1.0f/sum : 0.0f`) by forcing the
    // reciprocal to 0 wherever the sum was 0, so an empty row's output is 0 instead of NaN.
    vector<unsigned short, q_step> cur_sum_is_zero = (cur_sum == 0.0f);
    cur_sum = cm_inv(cur_sum);
    cur_sum.merge(0.0f, cur_sum_is_zero);

    lsc::block_2d_desc<half, 1, REG_M, REG_N> b2dO(o_base, q_tokens_left - 1, head_size * sizeof(half) - 1, o_pitch - 1, 0, 0);

#    pragma unroll
    for (int k = 0, ri = 0; k < padded_head_size; k += REG_N, ri += num_P_tiles) {
#    pragma unroll
        for (int p = 0; p < num_P_tiles; p++) {
            auto cO = rO[ri + p].format<float, REG_M, REG_N>();
#    pragma unroll
            for (int r = 0; r < cO.n_rows(); r++) {
                cur_O_f16[r + p * REG_M] = cm_mul<float>(cO.row(r), cur_sum[r + p * REG_M]);
            }
        }
        b2dO.set_block_x(k);
        cm_store(b2dO.set_block_y(0), cur_O_f16.format<half, num_P_tiles, REG_M * REG_N>().row(0));
        cm_store(b2dO.set_block_y(REG_M), cur_O_f16.format<half, num_P_tiles, REG_M * REG_N>().row(1));
    }
}

// ============================================================================================
// Two-source variant: ring (history, persistent) + current (this step's own new K/V) are two
// INDEPENDENT buffers that are never concatenated in memory -- the online-softmax accumulators
// (cur_max/cur_sum/rO) are shared across both sources, so attending to "ring then current" is
// numerically identical to attending to concat(ring, current) in one pass. See
// flashvsr_dit_kvcache_int8_3slot_fused_multisource_design_zh.md Sec 3.3/10.4 (the real
// BlockSparseAttention/DraftBlockMask primitives' `input[1]=key_cache.<layer>` +
// `input[2]=k_new_f16` ABI this mirrors) and cm_sdpa_opt_prefill_multisource.cm.
//
// Deliberately NOT generalized to causal masking or the dense additive attn_mask: FlashVSR's own
// self-attention never uses either (only the sparse block-table mask), so this stays its own
// small, independently-testable code path alongside (not merged into) sdpa_kernel_lsc_prefetch
// above, which remains untouched.

// hf8_e4m3_decode_to_f16 (fp8e4m3 -> fp16 in-register decode) is declared near the top of this
// file, ahead of sdpa_kernel_lsc_prefetch above -- see its doc comment there for the full
// rationale/algorithm description and why it has to precede both of its callers.

// Processes one source's full KV range [0, kv_stop) into the SHARED cur_max/cur_sum/rO
// accumulators. kv_stop<=0 (an empty source, e.g. chunk0's not-yet-populated ring, or warmup's
// absent "current" split) is a no-op -- constructing a block_2d_desc with height kv_stop-1 would
// underflow for kv_stop==0, so this is checked before any descriptor is built.
// sparse_mask_base must already be offset by the caller to this source's own block-index range
// within one combined mask buffer (ring blocks first, current blocks second).
//
// KVElemT: half for today's fp16 K/V (default -- unchanged behavior) or uchar for an
// fp8e4m3-packed source, decoded to half in-register right before each DPAS (see
// hf8_e4m3_decode_to_f16 above). Only the ring source is ever invoked with KVElemT=uchar --
// current always stays half, see sdpa_kernel_lsc_prefetch_2src's two call sites below.
template <int head_size, int wg_local_size, int sparse_block_size, typename KVElemT = half>
CM_INLINE void sdpa_accumulate_kv_source(
    int wg_local_id,
    int kv_stop,
    uint k_pitch_bytes,
    uint v_pitch_bytes,
    svmptr_t k_base [[type("svmptr_t")]],
    svmptr_t v_base [[type("svmptr_t")]],
    svmptr_t sparse_mask_base [[type("svmptr_t")]],
    matrix_ref<half, ((head_size + 16 - 1) / 16 * 16) / REG_K, REG_K * REG_N> rQ,
    vector_ref<float, q_step> cur_max,
    vector_ref<float, q_step> cur_sum,
    matrix_ref<float, ((head_size + 16 - 1) / 16 * 16) / REG_N * (REG_N / REG_M), REG_M * REG_N> rO) {
    static_assert(sparse_block_size > 1, "dense dual-source is out of scope -- use sdpa_kernel_lsc_prefetch for single-source dense attention");
    static_assert(std::is_same<KVElemT, half>::value || std::is_same<KVElemT, uchar>::value,
                  "KVElemT must be half (fp16 K/V, default) or uchar (fp8e4m3-packed ring K/V)");
    if (kv_stop <= 0)
        return;

    constexpr int padded_head_size = (head_size + 16 - 1) / 16 * 16;
    constexpr int num_P_tiles = REG_N / REG_M;
    constexpr bool fp8_kv = std::is_same<KVElemT, uchar>::value;

    lsc::block_2d_desc<KVElemT, 1, kv_step, REG_K> b2dK(k_base, kv_stop - 1, head_size * sizeof(KVElemT) - 1, k_pitch_bytes - 1, 0, 0);
    lsc::block_2d_desc<KVElemT, 1, REG_K, REG_N> b2dV(v_base, kv_stop - 1, head_size * sizeof(KVElemT) - 1, v_pitch_bytes - 1, 0, 0);

    static_assert(wg_local_size == 16);
    lsc::block_2d_desc<KVElemT, 1, kv_step / wg_local_size, REG_K> prefetch_K(k_base, kv_stop - 1, head_size * sizeof(KVElemT) - 1, k_pitch_bytes - 1, 0, 0);
    lsc::block_2d_desc<KVElemT, 1, REG_K / wg_local_size, REG_N> prefetch_V(v_base, kv_stop - 1, head_size * sizeof(KVElemT) - 1, v_pitch_bytes - 1, 0, 0);

    constexpr int KV_BLK = CMFLA_KV_BLK;
    constexpr int num_K = kv_step / REG_M;
    constexpr int BLK_ROWS = KV_BLK * kv_step;

    // Warm up first K and V tiles before the loop so kv_base=0 finds them in cache (identical
    // mechanism to sdpa_kernel_lsc_prefetch above, just scoped to this source's own range).
    prefetch_K.set_block_y(wg_local_id);
    prefetch_V.set_block_y(wg_local_id);
    cm_prefetch<CacheHint::Cached, CacheHint::Cached>(prefetch_K.set_block_x(0));
    cm_prefetch<CacheHint::Cached, CacheHint::Cached>(prefetch_V.set_block_x(0));
#    pragma unroll
    for (int ri = 1; ri < padded_head_size / REG_K; ri++) {
        cm_prefetch<CacheHint::Cached, CacheHint::Cached>(prefetch_K.set_block_x(ri * REG_K));
        cm_prefetch<CacheHint::Cached, CacheHint::Cached>(prefetch_V.set_block_x(ri * REG_N));
    }

    for (int kv_base = 0; kv_base < kv_stop; kv_base += BLK_ROWS) {
        // ---- block-sparsity: skip this whole KV_BLK chunk (no load, no DPAS) when its
        // sparse_block_size-sized bucket is masked out -- indexed relative to THIS source's own
        // kv_base (sparse_mask_base already points at this source's row in the combined mask).
        {
            constexpr int sb_shift = (sparse_block_size == 128) ? 7 : (sparse_block_size == 256) ? 8 : -1;
            static_assert(sb_shift > 0, "sparse_block_size must be 128 or 256");
            static_assert(sparse_block_size % BLK_ROWS == 0,
                          "sparse_block_size must be a multiple of KV_BLK*kv_step so a chunk never straddles a sparse block");
            if (sparse_mask_base && !((const bool*)sparse_mask_base)[(uint)kv_base >> sb_shift]) {
                continue;
            }
        }

        //# St = K @ Q^T for KV_BLK stacked sub-tiles -> [KV_BLK*kv_step, q_step]
        matrix<float, BLK_ROWS, q_step> St;
        auto St2 = St.format<float, KV_BLK * num_K, REG_M * REG_N>();
#    pragma unroll
        for (int b = 0; b < KV_BLK; b++) {
            int kv_pos = kv_base + b * kv_step;
            matrix<half, num_K, REG_M * REG_K> Kmat;
            prefetch_K.set_block_y(wg_local_id + kv_pos + BLK_ROWS);
            prefetch_V.set_block_y(wg_local_id + kv_pos);
            cm_prefetch<CacheHint::Cached, CacheHint::Cached>(prefetch_K.set_block_x(0));
            cm_prefetch<CacheHint::Cached, CacheHint::Cached>(prefetch_V.set_block_x(0));

            b2dK.set_block_y(kv_pos);
            if constexpr (fp8_kv) {
                // K path (simple case): raw fp8e4m3 bytes load as a plain 2D block (same tile
                // shape as the half path, 1 byte/elem instead of 2), decoded to half in-register
                // right before cm_dpas -- no separate reorder kernel, no extra buffer.
                matrix<uchar, num_K, REG_M * REG_K> Kmat_u8;
                cm_load<lsc::Normal>(Kmat_u8.format<uchar>(), b2dK.set_block_x(0));
                Kmat.format<half>() = hf8_e4m3_decode_to_f16<num_K * REG_M * REG_K>(Kmat_u8.format<uchar>());
            } else {
                cm_load<lsc::Normal>(Kmat.format<half>(), b2dK.set_block_x(0));
            }
#    pragma unroll
            for (int k = 0; k < num_K; k++)
                St2.row(b * num_K + k) =
                    cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, SystolicDepth, RepeatCount, float>(0, rQ[0].format<int32_t>(), Kmat[k].format<int32_t>());

#    pragma unroll
            for (int ri = 1; ri < padded_head_size / REG_K; ri++) {
                cm_prefetch<CacheHint::Cached, CacheHint::Cached>(prefetch_K.set_block_x(ri * REG_K));
                cm_prefetch<CacheHint::Cached, CacheHint::Cached>(prefetch_V.set_block_x(ri * REG_N));
                if constexpr (fp8_kv) {
                    matrix<uchar, num_K, REG_M * REG_K> Kmat_u8;
                    cm_load<lsc::Normal>(Kmat_u8.format<uchar>(), b2dK.set_block_x(ri * REG_K));
                    Kmat.format<half>() = hf8_e4m3_decode_to_f16<num_K * REG_M * REG_K>(Kmat_u8.format<uchar>());
                } else {
                    cm_load<lsc::Normal>(Kmat.format<half>(), b2dK.set_block_x(ri * REG_K));
                }
#    pragma unroll
                for (int k = 0; k < num_K; k++) {
                    St2.row(b * num_K + k) = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, SystolicDepth, RepeatCount, float>(St2.row(b * num_K + k),
                                                                                                                          rQ[ri].format<int32_t>(),
                                                                                                                          Kmat[k].format<int32_t>());
                }
            }
        }

        // ---- kv-tail mask (no causal mask in this kernel family -- see file header comment) ----
        {
            int kv_tokens = kv_stop - kv_base;
            for (int p = kv_tokens; p < BLK_ROWS; p++)
                St[p] = -3.4e38f;
        }

        // ---- one online-softmax update over the whole block ----
        auto max_comp = cm_online_softmax_update(St, cur_max, cur_sum);

        // ---- rescale rO ----
        if constexpr (KV_BLK >= 2) {
            if (kv_base > 0) {
#    pragma unroll
                for (int t = 0; t < padded_head_size / REG_N * num_P_tiles; t++) {
                    auto cO = rO[t].format<float, REG_M, REG_N>();
#    pragma unroll
                    for (int r = 0; r < REG_M; r++)
                        cO.row(r) = cm_mul<float>(cO.row(r), max_comp[r + (t % num_P_tiles) * REG_M]);
                }
            }
        } else {
#    pragma unroll
            for (int t = 0; t < padded_head_size / REG_N * num_P_tiles; t++) {
                auto cO = rO[t].format<float, REG_M, REG_N>();
#    pragma unroll
                for (int r = 0; r < REG_M; r++)
                    cO.row(r) = cm_mul<float>(cO.row(r), max_comp[r + (t % num_P_tiles) * REG_M]);
            }
        }

        // ---- transpose each sub-tile and accumulate P@V into rO ----
        constexpr int num_Vchunks = padded_head_size / REG_N;
#    pragma unroll
        for (int b = 0; b < KV_BLK; b++) {
            matrix<half, REG_N, REG_K> P;
            transpose_St_to_P_half(St.select<kv_step, 1, q_step, 1>(b * kv_step, 0), P);
            auto P2 = P.format<half, num_P_tiles, REG_M * REG_K>();
            b2dV.set_block_y(kv_base + b * kv_step);
#    pragma unroll
            for (int ci = 0, ri = 0; ci < num_Vchunks; ci++, ri += num_P_tiles) {
                matrix<half, REG_K / 2, REG_N * 2> Vmat;
                if constexpr (fp8_kv) {
                    // V path (harder case): hardware cm_load<lsc::VNNI> only knows how to
                    // half-pair-interleave 2-byte (half) source elements -- there is no 1-byte
                    // VNNI transform to ask the LSC unit for. So: load the raw fp8e4m3 bytes as a
                    // PLAIN (non-VNNI) 2D block (REG_K rows x REG_N cols, same shape as the K
                    // path), decode every byte to half, then MANUALLY reproduce in software the
                    // exact VNNI2 interleave hardware would have produced from a half source --
                    // out_row[i].lo16 = in_row[2*i], out_row[i].hi16 = in_row[2*i+1] (pairs of
                    // adjacent source *rows* interleaved into each output dword). This is the
                    // same select<>-based interleave sdpa_kernel_lsc's non-LSC-2D fallback path
                    // already uses above (search "temp_vnni" earlier in this file) to build a
                    // VNNI2 tile by hand from already-decoded half data -- reused here verbatim,
                    // just renamed, so no SLM round-trip is needed for a first correct version.
                    matrix<uchar, REG_K, REG_N> Vmat_u8;
                    cm_load<lsc::Normal>(Vmat_u8.format<uchar>(), b2dV.set_block_x(ci * REG_N));
                    matrix<half, REG_K, REG_N> Vmat_nat;
                    Vmat_nat.format<half>() = hf8_e4m3_decode_to_f16<REG_K * REG_N>(Vmat_u8.format<uchar>());
                    Vmat.select<REG_K / 2, 1, REG_N, 2>(0, 0) = Vmat_nat.select<REG_K / 2, 2, REG_N, 1>(0, 0);
                    Vmat.select<REG_K / 2, 1, REG_N, 2>(0, 1) = Vmat_nat.select<REG_K / 2, 2, REG_N, 1>(1, 0);
                } else {
                    cm_load<lsc::VNNI>(Vmat.format<half>(), b2dV.set_block_x(ci * REG_N));
                }
#    pragma unroll
                for (int p = 0; p < num_P_tiles; p++) {
                    rO[ri + p] = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, SystolicDepth, RepeatCount>(rO[ri + p].format<float>(),
                                                                                                       Vmat.format<int32_t>(),
                                                                                                       P2.row(p).format<int32_t>());
                }
            }
        }
    }
}

// fp8_ring_kv: compile-time switch (see KV_RING_FP8E4M3 in cm_sdpa_opt_prefill_multisource.cm)
// selecting an fp8e4m3-packed ring K/V (decoded to half in-register, see
// sdpa_accumulate_kv_source/hf8_e4m3_decode_to_f16 above) instead of today's plain half ring.
// Only the ring call below is templated on it -- current (this step's own new K/V) always stays
// half (see the scope note at this file's "Two-source variant" header comment).
template <int num_heads, int num_kv_heads, int head_size, int wg_local_size, int sparse_block_size, bool fp8_ring_kv = false>
void sdpa_kernel_lsc_prefetch_2src(int wg_local_id,
                                   int q_len,
                                   uint q_pitch_bytes,
                                   svmptr_t q_base [[type("svmptr_t")]],
                                   svmptr_t o_base [[type("svmptr_t")]],
                                   int kv_len_ring,
                                   uint k_pitch_bytes_ring,
                                   uint v_pitch_bytes_ring,
                                   svmptr_t k_base_ring [[type("svmptr_t")]],
                                   svmptr_t v_base_ring [[type("svmptr_t")]],
                                   svmptr_t sparse_mask_ring [[type("svmptr_t")]],
                                   int kv_len_cur,
                                   uint k_pitch_bytes_cur,
                                   uint v_pitch_bytes_cur,
                                   svmptr_t k_base_cur [[type("svmptr_t")]],
                                   svmptr_t v_base_cur [[type("svmptr_t")]],
                                   svmptr_t sparse_mask_cur [[type("svmptr_t")]],
                                   uint o_pitch_bytes = num_heads * head_size * sizeof(half),
                                   float q_prescale_rt = q_prescale) {
    uint o_pitch = o_pitch_bytes;
    constexpr int padded_head_size = (head_size + 16 - 1) / 16 * 16;

    vector<float, q_step> cur_max;
    vector<float, q_step> cur_sum;

    cur_max = -3e38f;
    cur_sum = 0;
    constexpr int num_P_tiles = REG_N / REG_M;
    matrix<half, padded_head_size / REG_K, REG_K * REG_N> rQ;
    matrix<float, padded_head_size / REG_N * num_P_tiles, REG_M * REG_N> rO;
    rO = 0.0f;

    auto q_tokens_left = q_len;
    static_assert(q_step == REG_N);
    static_assert(kv_step == REG_K);

    if (q_tokens_left < 0)
        q_tokens_left = 0;
    if (q_tokens_left > q_step)
        q_tokens_left = q_step;

    if (q_tokens_left > 0) {
        lsc::block_2d_desc<uint, 1, REG_N, REG_K / 2> b2dQ(reinterpret_cast<uint*>(q_base),
                                                           q_tokens_left - 1,
                                                           head_size * sizeof(half) - 1,
                                                           q_pitch_bytes - 1,
                                                           0,
                                                           0);
#    pragma unroll
        for (int k = 0, ri = 0; k < padded_head_size / 2; k += REG_K / 2, ri++) {
            cm_load<lsc::Transpose>(rQ[ri].format<uint>(), b2dQ.set_block_x(k));
            rQ[ri].format<half>() = cm_mul<half>(rQ[ri].format<half>(), (half)q_prescale_rt);
        }
    }

    // ---- ring, then current: numerically equivalent to one pass over concat(ring, current) ----
    sdpa_accumulate_kv_source<head_size, wg_local_size, sparse_block_size,
                               typename std::conditional<fp8_ring_kv, uchar, half>::type>(
        wg_local_id, kv_len_ring, k_pitch_bytes_ring, v_pitch_bytes_ring,
        k_base_ring, v_base_ring, sparse_mask_ring, rQ, cur_max, cur_sum, rO);
    sdpa_accumulate_kv_source<head_size, wg_local_size, sparse_block_size>(
        wg_local_id, kv_len_cur, k_pitch_bytes_cur, v_pitch_bytes_cur,
        k_base_cur, v_base_cur, sparse_mask_cur, rQ, cur_max, cur_sum, rO);

    if (q_tokens_left == 0)
        return;

    //# save cur_O/cur_sum.transpose(0, 1)
    matrix<half, num_P_tiles * REG_M, REG_N> cur_O_f16;
    // Guard the all-masked-row case (every KV block skipped across both sources): see
    // sdpa_kernel_lsc_prefetch's identical guard above for the NaN-avoidance rationale.
    vector<unsigned short, q_step> cur_sum_is_zero = (cur_sum == 0.0f);
    cur_sum = cm_inv(cur_sum);
    cur_sum.merge(0.0f, cur_sum_is_zero);

    lsc::block_2d_desc<half, 1, REG_M, REG_N> b2dO(o_base, q_tokens_left - 1, head_size * sizeof(half) - 1, o_pitch - 1, 0, 0);

#    pragma unroll
    for (int k = 0, ri = 0; k < padded_head_size; k += REG_N, ri += num_P_tiles) {
#    pragma unroll
        for (int p = 0; p < num_P_tiles; p++) {
            auto cO = rO[ri + p].format<float, REG_M, REG_N>();
#    pragma unroll
            for (int r = 0; r < cO.n_rows(); r++) {
                cur_O_f16[r + p * REG_M] = cm_mul<float>(cO.row(r), cur_sum[r + p * REG_M]);
            }
        }
        b2dO.set_block_x(k);
        cm_store(b2dO.set_block_y(0), cur_O_f16.format<half, num_P_tiles, REG_M * REG_N>().row(0));
        cm_store(b2dO.set_block_y(REG_M), cur_O_f16.format<half, num_P_tiles, REG_M * REG_N>().row(1));
    }
}

#else  // CM_HAS_LSC_UNTYPED_2D

template <bool use_causal_mask, int num_heads, int num_kv_heads, int head_size, int is_qkv_fused = 0>
void sdpa_kernel(uint slm_K,
                 uint slm_V,
                 int wg_local_id,
                 int local_size,
                 int q_start,
                 int kv_stop,
                 int q_len,
                 int kv_len,
                 SurfaceIndex query [[type("buffer_t")]],
                 SurfaceIndex key [[type("buffer_t")]],
                 SurfaceIndex value [[type("buffer_t")]],
                 SurfaceIndex output [[type("buffer_t")]],
                 uint q_pitch_bytes,
                 uint k_pitch_bytes,
                 uint v_pitch_bytes,
                 uint q_off,
                 uint k_off,
                 uint v_off,
                 uint o_off) {
    constexpr int padded_head_size = (head_size + 16 - 1) / 16 * 16;
    constexpr uint o_pitch = (num_heads * head_size * sizeof(half));

    vector<float, q_step> cur_max;
    vector<float, q_step> cur_sum;

    cur_max = -3e38f;
    cur_sum = 0;

    matrix<half, padded_head_size / REG_K, REG_K * REG_N> rQ;
    auto q_tokens_left = q_len;
    static_assert(q_step == REG_N);
    static_assert(kv_step == REG_K);
    static_assert((head_size % 8) == 0);

    if (q_tokens_left < 0)
        q_tokens_left = 0;
    if (q_tokens_left > q_step)
        q_tokens_left = q_step;

    if (q_tokens_left > 0) {
        // load as many as possible given one address
        if constexpr (head_size == 128 || head_size == 64) {
            matrix<uint, q_step, head_size / 2> QmatI32;
            cm_load_2d(QmatI32, query, q_off, q_pitch_bytes);
#    pragma unroll
            for (int k = 0, ri = 0; k < head_size / 2; k += REG_K / 2, ri++) {
                Transpose2DMatrix(QmatI32.select<q_step, 1, REG_K / 2, 1>(0, k), rQ[ri].format<uint, REG_K / 2, q_step>());
                rQ[ri].format<half>() = cm_mul<half>(rQ[ri].format<half>(), (half)q_prescale);
            }
        } else {
            constexpr int num_full_blocks = head_size / REG_K;
            int i = 0;
#    pragma unroll
            for (; i < num_full_blocks; i++) {
                matrix<uint, q_step, REG_K / 2> QmatI32;
                int k = i * REG_K;
                cm_load_2d(QmatI32, query, q_off + k * sizeof(uint) / 2, q_pitch_bytes);
                Transpose2DMatrix(QmatI32, rQ[i].format<uint, REG_K / 2, q_step>());
                rQ[i].format<half>() = cm_mul<half>(rQ[i].format<half>(), (half)q_prescale);
            }

            // if with tail, load with head_size_tail
            // following code will be optimized out when head_size_tail = 0
            if constexpr (head_size % REG_K > 0) {
                int k = num_full_blocks * REG_K;
                matrix<uint, q_step, REG_K / 2> QmatI32;
                cm_load_2d_with_tail<q_step, REG_K / 2, (head_size % REG_K) / 2>(QmatI32, query, q_off + k * sizeof(half), q_pitch_bytes);
                Transpose2DMatrix(QmatI32, rQ[num_full_blocks].format<uint, REG_K / 2, q_step>());
                rQ[num_full_blocks].format<half>() = cm_mul<half>(rQ[num_full_blocks].format<half>(), (half)q_prescale);
            }
        }
    }

    constexpr int num_P_tiles = REG_N / REG_M;
    matrix<float, padded_head_size / REG_N * num_P_tiles, REG_M * REG_N> rO;
    rO = 0.0f;  // Zero the accumulator: the first softmax block scales rO by max_comp==0, and 0*NaN==NaN if the GRF holds stale NaN/Inf bits.
    int causal_left = q_start;

    constexpr uint slm_buff_size = kv_step * padded_head_size * sizeof(half);
    int slm_buff_id_write = 0;
    int slm_buff_id_read = 0;

    auto load_slm_KV = [&](int kv_pos) {
        int kv_tokens = kv_stop - kv_pos;
        if (kv_tokens <= 0)
            return;
        // Calculate valid rows for this block (used to zero out garbage data)
        int kv_valid_rows = (kv_tokens >= kv_step) ? kv_step : kv_tokens;
        uint slm_offset = (slm_buff_id_write & 3) * slm_buff_size;
        slm_buff_id_write++;

        // non-tail branch is faster
        if (wg_local_id < local_size / 2) {
            matrix<half, 2 * REG_M, REG_K> temp;
            // head_size is split into blocks of <half, REG_K>
            constexpr int num_full_blocks = head_size / REG_K;
            int i = wg_local_id;
            for (; i < num_full_blocks; i += local_size / 2) {
                int k = i * REG_K;
                cm_load_2d(temp, key, k_off + k * sizeof(half), k_pitch_bytes);
                // Zero out unused K rows to prevent NaN from garbage data in KV cache
                // (Similar approach as PA kernel: cm_pa_common.hpp)
                for (int r = kv_valid_rows; r < kv_step; r++)
                    temp.row(r) = 0;
                cm_slm_block_write(slm_K, slm_offset + k * 2 * REG_M * sizeof(half), temp.format<half>());
            }
            // if with tail, load with head_size % REG_K
            // following code will be optimized out when head_size_tail = 0
            if constexpr (head_size % REG_K > 0) {
                int k = num_full_blocks * REG_K;
                cm_load_2d_with_tail<2 * REG_M, REG_K, head_size % REG_K>(temp, key, k_off + k * sizeof(half), k_pitch_bytes);
                // Zero out unused K rows
                for (int r = kv_valid_rows; r < kv_step; r++)
                    temp.row(r) = 0;
                cm_slm_block_write(slm_K, slm_offset + k * 2 * REG_M * sizeof(half), temp.format<half>());
            }
        } else {
            // read 16x16 XMX-B matrix (1x REG_N in Xe2, 2x REG_N in Xe1)
            constexpr int VK_STEP = 16;
            static_assert((VK_STEP % REG_N) == 0);
            matrix<half, REG_K, VK_STEP> temp2;
            matrix<half, REG_K / 2, REG_N * 2> temp_vnni;

            // head_size is split into blocks of <half, VK_STEP>
            constexpr int num_full_blocks = head_size / VK_STEP;
            int i = wg_local_id - local_size / 2;
            for (; i < num_full_blocks; i += local_size / 2) {
                int k = i * VK_STEP;
                cm_load_2d(temp2, value, v_off + k * sizeof(half), v_pitch_bytes);
                // Zero out unused V rows to prevent NaN from garbage data in KV cache
                // (Similar approach as PA kernel: cm_pa_common.hpp)
                for (int r = kv_valid_rows; r < kv_step; r++)
                    temp2.row(r) = 0;
#    pragma unroll
                for (int p = 0; p < VK_STEP / REG_N; p++) {
                    temp_vnni.select<REG_K / 2, 1, REG_N, 2>(0, 0) = temp2.select<REG_K / 2, 2, REG_N, 1>(0, p * REG_N);
                    temp_vnni.select<REG_K / 2, 1, REG_N, 2>(0, 1) = temp2.select<REG_K / 2, 2, REG_N, 1>(1, p * REG_N);

                    cm_slm_block_write(slm_V, slm_offset + (k + p * REG_N) * REG_K * sizeof(half), temp_vnni.format<half>());
                }
            }
            // if with tail, load with head_size_tail
            // following code will be optimized out when head_size_tail = 0
            if constexpr (head_size % VK_STEP > 0) {
                int k = num_full_blocks * VK_STEP;
                cm_load_2d_with_tail<REG_K, VK_STEP, head_size % VK_STEP>(temp2, value, v_off + k * sizeof(half), v_pitch_bytes);
                // Zero out unused V rows
                for (int r = kv_valid_rows; r < kv_step; r++)
                    temp2.row(r) = 0;
#    pragma unroll
                for (int p = 0; p < VK_STEP / REG_N; p++) {
                    temp_vnni.select<REG_K / 2, 1, REG_N, 2>(0, 0) = temp2.select<REG_K / 2, 2, REG_N, 1>(0, p * REG_N);
                    temp_vnni.select<REG_K / 2, 1, REG_N, 2>(0, 1) = temp2.select<REG_K / 2, 2, REG_N, 1>(1, p * REG_N);

                    cm_slm_block_write(slm_V, slm_offset + (k + p * REG_N) * REG_K * sizeof(half), temp_vnni.format<half>());
                }
            }
        }
        k_off += kv_step * k_pitch_bytes;
        v_off += kv_step * v_pitch_bytes;
    };

    load_slm_KV(0);
    load_slm_KV(kv_step);

    cm_slm_fence(CM_LOCAL_BARRIER);
    cm_sbarrier(1);

    for (int kv_pos = 0; kv_pos < kv_stop; kv_pos += kv_step, slm_buff_id_read++) {
        //
        //  load0->0, signal1,
        //  [load1->1, wait2, signal2, read0]
        //  [load2->2, wait3, signal3, read1]
        //  [load3->3, wait4, signal4, read2]
        //  [load4->0, wait5, signal5, read3]
        //
        //  after wait4, all workers have reached signal3, so:
        //     - all workers have finished load2 & read0.
        //     - we can start to load 4 into SLM slot 0 (i & 3) safely
        //     - we can start to read 2 ((i-2) & 3) safely
        //
        cm_fence(CM_LOCAL_BARRIER);
        cm_sbarrier(0);

        load_slm_KV(kv_pos + 2 * kv_step);

        if (kv_pos + kv_step < kv_stop)
            cm_sbarrier(1);

        uint slm_offset = (slm_buff_id_read & 3) * slm_buff_size;

        //# St = k @ Qt
        matrix<float, kv_step, q_step> St = ugemm_KQ(slm_K, rQ, slm_offset);

        if constexpr (use_causal_mask) {
            if (causal_left < kv_step) {
                vector<float, q_step> cmask = 0.0f;
                int p = causal_left + 1;
                int v = 0;
                for (; p < 0; p++) {
                    cmask[v] = -3.4e38f;
                    if (v < q_step - 1)
                        v++;
                }
                for (; p < kv_step; p++) {
                    cmask[v] = -3.4e38f;
                    St[p] = cm_add<float>(St[p], cmask);
                    if (v < q_step - 1)
                        v++;
                }
            }
            causal_left -= kv_step;
        }

        // mask off k-tails
        int kv_tokens = kv_stop - kv_pos;
        for (int p = kv_tokens; p < kv_step; p++)
            St[p] = -3.4e38f;

        auto max_comp = cm_online_softmax_update(St, cur_max, cur_sum);

        matrix<half, REG_N, REG_K> P;
        Transpose2DMatrix(St, P);

        if (kv_pos == 0)
            ugemm_PV0(slm_V, P, rO, slm_offset);
        else
            ugemm_PV1(slm_V, P, max_comp, rO, slm_offset);
    }

    if (q_tokens_left > 0) {
        //# save cur_O/cur_sum.transpose(0, 1)
        matrix<half, num_P_tiles * REG_M, REG_N> cur_O_f16;
        // Guard the all-masked-row case (every KV block skipped by a sparse block mask, or an
        // otherwise-empty attended range): cur_sum stays exactly 0, and an unguarded cm_inv(0) ->
        // +Inf combines with the (also-zero) rO accumulator into 0*Inf == NaN. Match the CPU
        // reference (ov::reference::block_sparse_attention's `sum > 0.0f ? 1.0f/sum : 0.0f`) by
        // forcing the reciprocal to 0 wherever the sum was 0, so an empty row's output is 0
        // instead of NaN.
        vector<unsigned short, q_step> cur_sum_is_zero = (cur_sum == 0.0f);
        cur_sum = cm_inv(cur_sum);
        cur_sum.merge(0.0f, cur_sum_is_zero);

#    pragma unroll
        for (int k = 0, ri = 0; k < head_size; k += REG_N, ri += num_P_tiles) {
#    pragma unroll
            for (int p = 0; p < num_P_tiles; p++) {
                auto cO = rO[ri + p].format<float, REG_M, REG_N>();
#    pragma unroll
                for (int r = 0; r < cO.n_rows(); r++) {
                    cur_O_f16[r + p * REG_M] = cm_mul<float>(cO.row(r), cur_sum[r + p * REG_M]);
                }
            }
            cm_store_2d(cur_O_f16, output, o_off + k * sizeof(half), o_pitch);
        }
    }
}

#endif  // !CM_HAS_LSC_UNTYPED_2D