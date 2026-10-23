// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cm_attention_common.hpp"
#include "cm_int8_kv_decode.hpp"

// Number of kv sub-tiles (kv_step rows each) processed per online-softmax update in
// sdpa_kernel_lsc_prefetch. Larger amortizes the rO rescale + softmax bookkeeping over
// more DPAS work at the cost of GRF pressure. Overridable via -DCMFLA_KV_BLK=N.
#ifndef CMFLA_KV_BLK
#    define CMFLA_KV_BLK 1
#endif

// online_softmax_update_tree and transpose_St_to_P_half are defined in cm_attention_common.hpp

#ifdef CM_HAS_LSC_UNTYPED_2D
// sdpa_kernel_lsc: flash-attention body shared by the plain fp16 KV-cache path (KVElemT=half,
// the default) and the int8-packed KV-cache path (KVElemT=uchar, affine uint8 quantization -- see
// cm_int8_kv_decode.hpp and KV_CACHE_COMPRESSION in cm_sdpa_opt_prefill.cm). Mirrors
// opencl/tests/pageatten/cm_pa_xe2.hpp's pa_lsc_u8/load_slm_KV_active: int8 affine dequant
// ((raw-zp)*scale, no gather/LUT) staged into SLM once per work-group instead of once per
// work-item -- load_slm_KV decodes each (kv_pos, head_size column range) byte range to half
// EXACTLY ONCE per work-group when KVElemT=uchar -- half the work-group's lanes decode+stage K,
// the other half decode+stage V -- and every lane then reads the already-decoded half data back
// out of SLM via the unmodified (type-agnostic) ugemm_KQ/ugemm_PV0/ugemm_PV1 helpers below: by the
// time those run, SLM already holds plain half regardless of K/V's original on-the-wire dtype, so
// no change is needed there. When KVElemT=half, load_slm_KV instead loads K/V directly as half
// (no decode), i.e. byte-for-byte the original sdpa_kernel_lsc behavior.
//
// K is quantized BY_CHANNEL (k_by_channel=true, one scale/zp pair per 16-token sub-block per
// head_size channel) or BY_TOKEN (k_by_channel=false, one pair per token), selected by the
// caller's KV_CACHE_COMPRESSION; V is always BY_TOKEN regardless of mode -- matches pageatten's own
// K/V asymmetry exactly (see kv_cache_quant_utils.py's quant_per_channel_parts/
// quant_per_token_parts, which int8_kv_ref.py thinly wraps as this kernel's host-side oracle).
//
// Causal-only, no dense attn_mask (this is still a focused perf experiment on the decode-once
// hypothesis for the quantized-KV case, not a feature-parity requirement with
// sdpa_kernel_lsc_prefetch's fuller causal+attn_mask+sparse_block_mask surface).
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
// sdpa_accumulate_kv_source_slm: the SLM-ring/barrier pipeline for ONE source's [0, kv_stop) KV
// range -- extracted from sdpa_kernel_lsc's body below (pure refactor, no functional change for
// sdpa_kernel_lsc's own single-source callers) so the dual-source kernel
// (sdpa_kernel_lsc_2src_slm) can call this SAME pipeline once per source (cache, then
// current), reusing one SLM ring-buffer allocation and one set of online-softmax accumulators
// across both calls -- giving the dual-source int8-cache path the same decode-once-per-work-group
// amortization this kernel already has, instead of the retired sdpa_accumulate_kv_source's
// per-lane-unamortized design.
//
// slm_buff_id_write/read are local state, reset to 0 at entry, and the SLM ring is (re-)primed
// from kv_pos=0 of THIS source's own [0, kv_stop) range every time this is called -- safe to call
// more than once in a row reusing the same slm_K/slm_V allocation, at the cost of one extra
// full-barrier per call (see the entry barrier below -- the same precedent already used further
// down for block-sparse chunk resumption). kv_stop<=0 (an empty source, e.g. chunk0's
// not-yet-populated cache) is a no-op after that entry barrier -- constructing a block_2d_desc with
// height kv_stop-1 would underflow for kv_stop==0, so this is checked before any descriptor is
// built, mirroring the retired sdpa_accumulate_kv_source's identical guard. causal_left and
// first_write are threaded by reference so a multi-segment caller
// (sdpa_kernel_lsc_2src_slm) can carry state across segments: first_write replaces a
// literal `kv_pos==0` check for the ugemm_PV0-vs-PV1 dispatch -- for a single call this is
// bit-identical to the old `kv_pos==0` check (both are true only at the very first real
// accumulation), but for a second segment called after a non-empty first one, first_write is
// already false, forcing ugemm_PV1 (required for correctness once rO holds real accumulated state,
// even if this segment's own local kv_pos starts back at 0).
template <bool use_causal_mask, int head_size, int sparse_block_size, typename KVElemT, bool k_by_channel>
CM_INLINE void sdpa_accumulate_kv_source_slm(
    uint slm_K,
    uint slm_V,
    int wg_local_id,
    int local_size,
    int kv_stop,
    uint k_pitch_bytes,
    uint v_pitch_bytes,
    svmptr_t k_base [[type("svmptr_t")]],
    svmptr_t v_base [[type("svmptr_t")]],
    svmptr_t k_scale_base [[type("svmptr_t")]],
    svmptr_t k_zp_base [[type("svmptr_t")]],
    svmptr_t v_scale_base [[type("svmptr_t")]],
    svmptr_t v_zp_base [[type("svmptr_t")]],
    svmptr_t sparse_mask_base [[type("svmptr_t")]],
    svmptr_t sparse_mask_wg_lo_base [[type("svmptr_t")]],
    svmptr_t sparse_mask_wg_hi_base [[type("svmptr_t")]],
    matrix_ref<half, ((head_size + 16 - 1) / 16 * 16) / REG_K, REG_K * REG_N> rQ,
    vector_ref<float, q_step> cur_max,
    vector_ref<float, q_step> cur_sum,
    matrix_ref<float, ((head_size + 16 - 1) / 16 * 16) / REG_N * (REG_N / REG_M), REG_M * REG_N> rO,
    int& causal_left,
    bool& first_write) {
    static_assert(std::is_same<KVElemT, half>::value || std::is_same<KVElemT, uchar>::value,
                  "KVElemT must be half (plain fp16 KV-cache) or uchar (int8-packed KV-cache, "
                  "affine-decoded once per work-group in load_slm_KV)");
    constexpr bool int8_kv = std::is_same<KVElemT, uchar>::value;
    constexpr int padded_head_size = (head_size + 16 - 1) / 16 * 16;

    // Entry barrier: makes it safe to call this function more than once in a row against the SAME
    // slm_K/slm_V allocation (see this function's doc comment) -- protects against a straggler
    // lane still reading the PREVIOUS call's final SLM buffer while this call's load_slm_KV(0)
    // starts overwriting the same ring slots. A plain cm_barrier() (not the split cm_sbarrier
    // scheme) is used, matching the identical precedent used below for block-sparse chunk
    // resumption. Run unconditionally (even when kv_stop<=0, below) and before any
    // single-source-vs-multi-segment distinction, so every call site -- including
    // sdpa_kernel_lsc's single call -- behaves identically with no special-casing: one harmless
    // extra barrier per call, negligible next to the per-tile barrier count inside the loops below.
    cm_slm_fence(CM_LOCAL_BARRIER);
    cm_barrier();

    if (kv_stop <= 0)
        return;

    // K/V element type is KVElemT: plain half for the fp16 KV-cache path (unchanged descriptor),
    // or int8-packed uchar (1 byte/elem, decoded to half exactly once per work-group in
    // load_slm_KV below) for the quantized path.
    lsc::block_2d_desc<KVElemT, 1, kv_step, REG_K> b2dK(k_base, kv_stop - 1, head_size * sizeof(KVElemT) - 1, k_pitch_bytes - 1, 0, 0);
    lsc::block_2d_desc<KVElemT, 1, REG_K, REG_N> b2dV(v_base, kv_stop - 1, head_size * sizeof(KVElemT) - 1, v_pitch_bytes - 1, 0, 0);

    // SLM always holds decoded half regardless of K/V's on-the-wire dtype, so the buffer size and
    // quad-buffering scheme are identical for both KVElemT instantiations.
    constexpr uint slm_buff_size = kv_step * head_size * sizeof(half);
    int slm_buff_id_write = 0;
    int slm_buff_id_read = 0;

    // The ONLY place the two KVElemT instantiations differ: a plain half load straight into SLM
    // (KVElemT=half), or a load of raw int8 bytes + affine-decode-to-half before the SLM write
    // (KVElemT=uchar, "decode-once-per-work-group" -- see this function's doc comment above).
    // Everything below this lambda (the sparse_block_size dispatch, softmax) is shared verbatim
    // between both element types.
    auto load_slm_KV = [&](int kv_pos) {
        if (kv_pos < kv_stop) {
            uint slm_offset = (slm_buff_id_write & 3) * slm_buff_size;
            slm_buff_id_write++;
            if (wg_local_id < local_size / 2) {
                if constexpr (int8_kv) {
                    // K (decode-once-per-WG): load raw int8 bytes, affine-decode to half -- by
                    // channel (one scale/zp pair per 16-token sub-block per head_size column,
                    // k_by_channel=true) or by token (one pair per row, k_by_channel=false), per
                    // cm_int8_kv_decode.hpp. Mirrors opencl/tests/pageatten/cm_pa_xe2.hpp's
                    // pa_lsc_u8 K-path (load_slm_KV_active), generalized from its fixed-size
                    // CMPA_BLOCK_SZ cache blocks to this kernel's flat [kv_len, head_size] K
                    // buffer (raw and decoded data use separate buffers here, unlike pa_lsc_u8's
                    // register-aliasing trick, to keep this new path simple to verify).
                    vector<half, kv_step> k_scale, k_zp;
                    if constexpr (!k_by_channel) {
                        cm_svm_block_read(reinterpret_cast<svmptr_t>(k_scale_base + (uint)kv_pos * sizeof(half)), k_scale);
                        cm_svm_block_read(reinterpret_cast<svmptr_t>(k_zp_base + (uint)kv_pos * sizeof(half)), k_zp);
                    }
                    matrix<uint8_t, kv_step, REG_K> quanKmat;
                    matrix<half, kv_step, REG_K> kmat;
                    b2dK.set_block_y(kv_pos);
                    for (int k = REG_K * wg_local_id; k < padded_head_size; k += REG_K * (local_size / 2)) {
                        if constexpr (k_by_channel) {
                            uint k_ch_off = (uint)(kv_pos / 16) * head_size + k;
                            cm_svm_block_read(reinterpret_cast<svmptr_t>(k_scale_base + k_ch_off * sizeof(half)), k_scale);
                            cm_svm_block_read(reinterpret_cast<svmptr_t>(k_zp_base + k_ch_off * sizeof(half)), k_zp);
                        }
                        cm_load<lsc::Normal>(quanKmat.format<uint8_t>(), b2dK.set_block_x(k));
                        if constexpr (k_by_channel)
                            int8_decode_by_channel<kv_step, REG_K>(quanKmat, k_scale, k_zp, kmat);
                        else
                            int8_decode_by_token<kv_step, REG_K>(quanKmat, k_scale, k_zp, kmat);
                        cm_slm_block_write(slm_K, slm_offset + k * kv_step * sizeof(half), kmat.format<half>());
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
                if constexpr (int8_kv) {
                    // V (always BY_TOKEN, scale/zp per row, loaded once for the whole tile):
                    // hardware cm_load<lsc::VNNI> only knows how to half-pair-interleave 2-byte
                    // (half) source elements, so the raw int8 bytes are loaded as a PLAIN
                    // (non-VNNI) 2D block, affine-decoded to half, then the exact VNNI2 interleave
                    // hardware would have produced from a half source is reproduced manually via
                    // select<>() -- the same manual-VNNI approach this path has always used.
                    vector<half, kv_step> v_scale, v_zp;
                    cm_svm_block_read(reinterpret_cast<svmptr_t>(v_scale_base + (uint)kv_pos * sizeof(half)), v_scale);
                    cm_svm_block_read(reinterpret_cast<svmptr_t>(v_zp_base + (uint)kv_pos * sizeof(half)), v_zp);
                    matrix<uchar, REG_K, REG_N> Vmat_u8;
                    matrix<half, REG_K, REG_N> Vmat_nat;
                    matrix<half, REG_K / 2, REG_N * 2> temp2;
                    b2dV.set_block_y(kv_pos);
#    pragma unroll
                    for (int k = REG_N * (wg_local_id - (local_size / 2)); k < padded_head_size; k += REG_N * (local_size / 2)) {
                        cm_load<lsc::Normal>(Vmat_u8.format<uchar>(), b2dV.set_block_x(k));
                        int8_decode_by_token<REG_K, REG_N>(Vmat_u8, v_scale, v_zp, Vmat_nat);
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

                if (first_write) {
                    ugemm_PV0(slm_V, P, rO, slm_offset);
                    first_write = false;
                } else {
                    ugemm_PV1(slm_V, P, max_comp, rO, slm_offset);
                }
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

                    // first_write replaces a literal `kv_pos==0` check (see this function's doc
                    // comment) -- if chunk 0 itself is skipped, every later chunk's first iteration
                    // still takes this same branch (first_write still true), with rO still at its
                    // zero-init value: PV0 and PV1 are mathematically identical there anyway
                    // (rescaling 0 by any finite max_comp is still 0), so no separate "first ACTIVE
                    // chunk" tracking is needed.
                    if (first_write) {
                        ugemm_PV0(slm_V, P, rO, slm_offset);
                        first_write = false;
                    } else {
                        ugemm_PV1(slm_V, P, max_comp, rO, slm_offset);
                    }
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

                    // Same first_write reasoning as the sparse_block_size==256 branch above.
                    if (first_write) {
                        ugemm_PV0(slm_V, P, rO, slm_offset);
                        first_write = false;
                    } else {
                        ugemm_PV1(slm_V, P, max_comp, rO, slm_offset);
                    }
                }
            }
        }
    }
}

template <bool use_causal_mask, int num_heads, int num_kv_heads, int head_size, int is_qkv_fused = 0,
          int sparse_block_size = 1, typename KVElemT = half, bool k_by_channel = false>
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
                     // Int8 dequant scale/zero-point SVM pointers, only read when KVElemT=uchar --
                     // see cm_int8_kv_decode.hpp. k_scale_base/k_zp_base are laid out
                     // [kv_len/16, head_size] when k_by_channel, else [kv_len]; v_scale_base/
                     // v_zp_base are always [kv_len] (V is always BY_TOKEN).
                     svmptr_t k_scale_base [[type("svmptr_t")]] = 0,
                     svmptr_t k_zp_base [[type("svmptr_t")]] = 0,
                     svmptr_t v_scale_base [[type("svmptr_t")]] = 0,
                     svmptr_t v_zp_base [[type("svmptr_t")]] = 0,
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
                  "KVElemT must be half (plain fp16 KV-cache) or uchar (int8-packed KV-cache, "
                  "affine-decoded once per work-group in load_slm_KV)");
    constexpr bool int8_kv = std::is_same<KVElemT, uchar>::value;

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

    int causal_left = q_start;
    bool first_write = true;

    sdpa_accumulate_kv_source_slm<use_causal_mask, head_size, sparse_block_size, KVElemT, k_by_channel>(
        slm_K, slm_V, wg_local_id, local_size, kv_stop, k_pitch_bytes, v_pitch_bytes,
        k_base, v_base, k_scale_base, k_zp_base, v_scale_base, v_zp_base,
        sparse_mask_base, sparse_mask_wg_lo_base, sparse_mask_wg_hi_base,
        rQ, cur_max, cur_sum, rO, causal_left, first_write);
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
// Two-source variant: cache (history, persistent) + current (this step's own new K/V) are two
// INDEPENDENT buffers that are never concatenated in memory -- the online-softmax accumulators
// (cur_max/cur_sum/rO) are shared across both sources, so attending to "cache then current" is
// numerically identical to attending to concat(cache, current) in one pass. See
// flashvsr_dit_kvcache_int8_3slot_fused_dualsource_design_zh.md Sec 3.3/10.4 (the real
// BlockSparseAttention/DraftBlockMask primitives' `input[1]=key_cache.<layer>` +
// `input[2]=k_new_f16` ABI this mirrors) and cm_sdpa_opt_prefill_dualsource.cm.
//
// Deliberately NOT generalized to causal masking or the dense additive attn_mask: FlashVSR's own
// self-attention never uses either (only the sparse block-table mask), so this stays its own
// small, independently-testable code path alongside (not merged into) sdpa_kernel_lsc_prefetch
// above, which remains untouched.
//
// Two entry points, mirroring the single-source fp16-vs-int8 fork (sdpa_kernel_lsc_prefetch above
// vs sdpa_kernel_lsc/sdpa_accumulate_kv_source_slm): sdpa_kernel_lsc_prefetch_2src (barrier-free,
// non-SLM, KV_CACHE_COMPRESSION==0 only -- nothing to decode-amortize, so no reason to pay for
// SLM/barrier sharing) and sdpa_kernel_lsc_2src_slm (SLM/barrier-pipelined, required
// whenever the cache is int8-packed). cm_sdpa_opt_prefill_dualsource.cm selects between them at
// compile time via #if KV_CACHE_COMPRESSION, same convention as cm_sdpa_opt_prefill.cm's own
// #if KV_CACHE_COMPRESSION split.
// ============================================================================================

// sdpa_accumulate_kv_source_prefetch: the barrier-free, non-SLM analog of
// sdpa_accumulate_kv_source_slm -- extracted from sdpa_kernel_lsc_prefetch's core loop (the
// descriptor setup through its "for (kv_base...)" loop) so sdpa_kernel_lsc_prefetch_2src can call
// this SAME per-segment pipeline once per source (cache, then current), reusing one set of
// online-softmax accumulators across both calls, numerically equivalent to one pass over
// concat(cache, current) -- same contract as sdpa_accumulate_kv_source_slm, just without any SLM
// staging or barriers, since this variant is only ever used when neither source needs int8 decode
// (KV_CACHE_COMPRESSION==0): every lane loads its own K/V directly from global memory into
// private registers (sdpa_kernel_lsc_prefetch's own approach), so there is no cross-lane shared
// state to protect and a block-sparse skip is a plain per-lane `continue` -- no wg-uniform
// skip-LOAD mask rows, no entry/chunk-boundary cm_barrier() needed at all (contrast
// sdpa_accumulate_kv_source_slm's entry barrier and block-sparse chunk-boundary barriers, both
// needed only because its SLM ring buffer IS shared across lanes).
//
// Deliberately narrower than sdpa_kernel_lsc_prefetch: no causal mask, no dense additive attn_mask,
// no work-group-uniform mask rows -- same established scope restriction as this file's "Two-source
// variant" section above (FlashVSR's own self-attention never needs causal or dense attn_mask, and
// this path has no barrier-pipelined skip-LOAD decision to keep WG-uniform in the first place).
//
// kv_stop<=0 (an empty source, e.g. chunk0's not-yet-populated cache) is a no-op, checked before any
// block_2d_desc is built (height kv_stop-1 would underflow for kv_stop==0) -- mirrors
// sdpa_accumulate_kv_source_slm's identical guard.
//
// first_write is threaded by reference exactly as sdpa_accumulate_kv_source_slm's own first_write,
// but here it replaces sdpa_kernel_lsc_prefetch's single-call-only `kv_base>0` rescale-skip check
// (its KV_BLK>=2 optimization: rO's rescale is a no-op at kv_base==0 only because rO is still its
// zero-initialized value there -- true for a single, self-contained call, but FALSE for a second
// segment's call, since current's own loop also starts at local kv_base==0 while rO may already
// hold cache's accumulated contribution). Changing the condition to
// `kv_base > 0 || !first_write` and updating first_write=false once per EXECUTED (non-block-sparse-
// skipped) outer iteration -- regardless of which KV_BLK branch ran -- fixes this: cache's call
// behaves identically to a single-segment call (first_write starts true), while current's call
// always rescales even at its own local kv_base==0 once cache has written anything real. A fully
// block-sparse-masked-out cache segment correctly leaves first_write true (current's first executed
// chunk still gets to skip-rescale, since rO really is still zero then) -- same property
// sdpa_accumulate_kv_source_slm's own first_write already has (see that function's doc comment).
template <int head_size, int sparse_block_size>
CM_INLINE void sdpa_accumulate_kv_source_prefetch(
    int wg_local_id,
    int kv_stop,
    uint k_pitch_bytes,
    uint v_pitch_bytes,
    svmptr_t k_base [[type("svmptr_t")]],
    svmptr_t v_base [[type("svmptr_t")]],
    // Per-lane sparse_block_mask row for THIS work-item's own q_block (already offset by the
    // caller) -- same convention as sdpa_kernel_lsc_prefetch's own sparse_mask_base. No wg_lo/hi
    // rows needed (see doc comment above).
    svmptr_t sparse_mask_base [[type("svmptr_t")]],
    matrix_ref<half, ((head_size + 16 - 1) / 16 * 16) / REG_K, REG_K * REG_N> rQ,
    vector_ref<float, q_step> cur_max,
    vector_ref<float, q_step> cur_sum,
    matrix_ref<float, ((head_size + 16 - 1) / 16 * 16) / REG_N * (REG_N / REG_M), REG_M * REG_N> rO,
    bool& first_write) {
    if (kv_stop <= 0)
        return;

    constexpr int padded_head_size = (head_size + 16 - 1) / 16 * 16;
    constexpr int wg_local_size = 16;

    lsc::block_2d_desc<half, 1, kv_step, REG_K> b2dK(k_base, kv_stop - 1, head_size * sizeof(half) - 1, k_pitch_bytes - 1, 0, 0);
    lsc::block_2d_desc<half, 1, REG_K, REG_N> b2dV(v_base, kv_stop - 1, head_size * sizeof(half) - 1, v_pitch_bytes - 1, 0, 0);

    static_assert(wg_local_size == 16);
    lsc::block_2d_desc<half, 1, kv_step / wg_local_size, REG_K> prefetch_K(k_base, kv_stop - 1, head_size * sizeof(half) - 1, k_pitch_bytes - 1, 0, 0);
    lsc::block_2d_desc<half, 1, REG_K / wg_local_size, REG_N> prefetch_V(v_base, kv_stop - 1, head_size * sizeof(half) - 1, v_pitch_bytes - 1, 0, 0);

    constexpr int num_P_tiles = REG_N / REG_M;
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
        // sparse_block_size-sized bucket is masked out -- plain per-lane continue, no barrier
        // needed (see this function's doc comment: nothing here is shared across lanes).
        if constexpr (sparse_block_size > 1) {
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

        // ---- mask kv-tail (no causal mask in this kernel family, see doc comment above) ----
        {
            int kv_tokens = kv_stop - kv_base;
            // LSC ensures no overflow-access, but mask off k-tail attn-score is still required
            for (int p = kv_tokens; p < BLK_ROWS; p++)
                St[p] = -3.4e38f;
        }

        // ---- one online-softmax update over the whole block ----
        auto max_comp = cm_online_softmax_update(St, cur_max, cur_sum);

        // ---- rescale rO (skip only on the very first executed block of the very first segment --
        // see this function's doc comment for why `|| !first_write` is required here, unlike
        // sdpa_kernel_lsc_prefetch's single-call-only `kv_base > 0` check) ----
        if constexpr (KV_BLK >= 2) {
            if (kv_base > 0 || !first_write) {
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
        first_write = false;

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
}

// sdpa_kernel_lsc_prefetch_2src: barrier-free dual-source entry point for the pure-fp16
// (KV_CACHE_COMPRESSION==0) case ONLY -- the int8-cache case keeps using
// sdpa_kernel_lsc_2src_slm unchanged. Calls sdpa_accumulate_kv_source_prefetch twice
// (cache, then current) sharing ONE set of online-softmax accumulators, numerically equivalent to
// sdpa_kernel_lsc_2src_slm / one pass over concat(cache, current), but with no SLM, no
// barriers, no scale/zp pointers (cache is never quantized here), no work-group-uniform mask rows
// (this path has no barrier-pipelined skip-LOAD decision to protect -- see
// sdpa_accumulate_kv_source_prefetch's doc comment). Safe to early-return for an inactive lane
// (q_tokens<=0) at the .cm call site exactly like sdpa_kernel_lsc_prefetch's own callers do
// (cm_sdpa_opt_prefill.cm's KV_CACHE_COMPRESSION==0 branch) -- there are no barriers here to strand
// a sibling lane on, unlike sdpa_kernel_lsc_2src_slm which must NOT early-return.
template <int head_size, int sparse_block_size>
void sdpa_kernel_lsc_prefetch_2src(int wg_local_id,
                                    int q_len,
                                    uint q_pitch_bytes,
                                    svmptr_t q_base [[type("svmptr_t")]],
                                    svmptr_t o_base [[type("svmptr_t")]],
                                    int kv_len_cache,
                                    uint k_pitch_bytes_cache,
                                    uint v_pitch_bytes_cache,
                                    svmptr_t k_base_cache [[type("svmptr_t")]],
                                    svmptr_t v_base_cache [[type("svmptr_t")]],
                                    svmptr_t sparse_mask_cache [[type("svmptr_t")]],
                                    int kv_len_cur,
                                    uint k_pitch_bytes_cur,
                                    uint v_pitch_bytes_cur,
                                    svmptr_t k_base_cur [[type("svmptr_t")]],
                                    svmptr_t v_base_cur [[type("svmptr_t")]],
                                    svmptr_t sparse_mask_cur [[type("svmptr_t")]],
                                    uint o_pitch_bytes,
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

    // ---- cache, then current: SAME online-softmax accumulators across both calls, numerically
    // equivalent to one pass over concat(cache, current) -- see sdpa_accumulate_kv_source_prefetch's
    // doc comment for the first_write-threading rescale-skip fix that makes this safe.
    bool first_write = true;
    sdpa_accumulate_kv_source_prefetch<head_size, sparse_block_size>(
        wg_local_id, kv_len_cache, k_pitch_bytes_cache, v_pitch_bytes_cache,
        k_base_cache, v_base_cache, sparse_mask_cache, rQ, cur_max, cur_sum, rO, first_write);
    sdpa_accumulate_kv_source_prefetch<head_size, sparse_block_size>(
        wg_local_id, kv_len_cur, k_pitch_bytes_cur, v_pitch_bytes_cur,
        k_base_cur, v_base_cur, sparse_mask_cur, rQ, cur_max, cur_sum, rO, first_write);

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

// sdpa_kernel_lsc_2src_slm: the dual-source entry point, SLM/barrier-pipelined (replacing
// a retired barrier-free sdpa_kernel_lsc_prefetch_2src/sdpa_accumulate_kv_source pair that paid its
// int8-cache decode cost once per lane instead of once per work-group -- a "KNOWN REMAINING PERF GAP"
// this function closes) by calling sdpa_accumulate_kv_source_slm once per source (cache, then
// current), reusing ONE shared SLM ring-buffer allocation (slm_K/slm_V, allocated by the caller,
// same convention as sdpa_kernel_lsc) and ONE set of online-softmax accumulators across both calls
// -- giving int8-cache
// the same decode-once-per-work-group amortization sdpa_kernel_lsc's KVElemT=uchar path already
// has, instead of paying the decode cost once per lane (16x over) like the retired
// sdpa_accumulate_kv_source did.
//
// cache_compression (0=NONE/half, 1=BY_TOKEN, 2=BY_CHANNEL, matching KV_CACHE_COMPRESSION in
// cm_sdpa_opt_prefill_dualsource.cm) selects the cache source's KVElemT and K granularity; current
// always stays half (see this file's "Two-source variant" header comment -- current is transient,
// quantizing it buys no persistent-memory win).
//
// sparse_mask_wg_lo/hi_cache and _cur: REQUIRED once this pipeline is barrier-pipelined (unlike the
// barrier-free sdpa_kernel_lsc_prefetch_2src, which never needed them) -- at SPARSE_BLOCK_SIZE==128
// two different lanes in the same work-group can map to two different q_blocks and legitimately
// disagree about which KV chunks to skip; the skip-LOAD decision inside
// sdpa_accumulate_kv_source_slm must be work-group-uniform or the barrier-pipelined lanes can
// diverge and deadlock -- same requirement, same fix (work-group-uniform OR-merged mask rows
// computed by the caller from wg_id) as sdpa_kernel_lsc's own SPARSE_BLOCK_SIZE==128 path. Each
// source needs its OWN pair since cache and current have independent block-mask ranges.
//
// Both sdpa_accumulate_kv_source_slm calls use use_causal_mask=false -- matches this kernel
// family's existing, deliberate scope (see this file's "Two-source variant" header comment: never
// generalized to causal masking since FlashVSR's own self-attention never needs it).
template <int num_heads, int num_kv_heads, int head_size, int sparse_block_size, int cache_compression = 0>
void sdpa_kernel_lsc_2src_slm(uint slm_K,
                                       uint slm_V,
                                       int wg_local_id,
                                       int local_size,
                                       int q_len,
                                       uint q_pitch_bytes,
                                       svmptr_t q_base [[type("svmptr_t")]],
                                       svmptr_t o_base [[type("svmptr_t")]],
                                       int kv_len_cache,
                                       uint k_pitch_bytes_cache,
                                       uint v_pitch_bytes_cache,
                                       svmptr_t k_base_cache [[type("svmptr_t")]],
                                       svmptr_t v_base_cache [[type("svmptr_t")]],
                                       svmptr_t sparse_mask_cache [[type("svmptr_t")]],
                                       svmptr_t sparse_mask_wg_lo_cache [[type("svmptr_t")]],
                                       svmptr_t sparse_mask_wg_hi_cache [[type("svmptr_t")]],
                                       // Int8 cache dequant scale/zero-point, only read when
                                       // cache_compression!=0 -- see cm_int8_kv_decode.hpp. Layout
                                       // matches sdpa_kernel_lsc's k_scale_base/k_zp_base/
                                       // v_scale_base/v_zp_base exactly, scoped to the cache source.
                                       svmptr_t k_scale_base_cache [[type("svmptr_t")]],
                                       svmptr_t k_zp_base_cache [[type("svmptr_t")]],
                                       svmptr_t v_scale_base_cache [[type("svmptr_t")]],
                                       svmptr_t v_zp_base_cache [[type("svmptr_t")]],
                                       int kv_len_cur,
                                       uint k_pitch_bytes_cur,
                                       uint v_pitch_bytes_cur,
                                       svmptr_t k_base_cur [[type("svmptr_t")]],
                                       svmptr_t v_base_cur [[type("svmptr_t")]],
                                       svmptr_t sparse_mask_cur [[type("svmptr_t")]],
                                       svmptr_t sparse_mask_wg_lo_cur [[type("svmptr_t")]],
                                       svmptr_t sparse_mask_wg_hi_cur [[type("svmptr_t")]],
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

    constexpr bool cache_int8 = (cache_compression != 0);
    constexpr bool cache_by_channel = (cache_compression == 2);
    using CacheElemT = typename std::conditional<cache_int8, uchar, half>::type;

    // Neither source uses causal masking, so these are write-only scratch -- the shared
    // first_write flag is the only state that genuinely carries across the two calls (see this
    // function's doc comment and sdpa_accumulate_kv_source_slm's own doc comment).
    int unused_causal_left_cache = 0;
    int unused_causal_left_cur = 0;
    bool first_write = true;

    // ---- cache, then current: SAME shared SLM ring buffer + online-softmax accumulators across
    // both calls, numerically equivalent to one pass over concat(cache, current) -- see
    // sdpa_accumulate_kv_source_slm's doc comment for the segment-boundary entry-barrier that
    // makes this safe.
    sdpa_accumulate_kv_source_slm<false, head_size, sparse_block_size, CacheElemT, cache_by_channel>(
        slm_K, slm_V, wg_local_id, local_size, kv_len_cache, k_pitch_bytes_cache, v_pitch_bytes_cache,
        k_base_cache, v_base_cache, k_scale_base_cache, k_zp_base_cache, v_scale_base_cache, v_zp_base_cache,
        sparse_mask_cache, sparse_mask_wg_lo_cache, sparse_mask_wg_hi_cache,
        rQ, cur_max, cur_sum, rO, unused_causal_left_cache, first_write);
    sdpa_accumulate_kv_source_slm<false, head_size, sparse_block_size, half, false>(
        slm_K, slm_V, wg_local_id, local_size, kv_len_cur, k_pitch_bytes_cur, v_pitch_bytes_cur,
        k_base_cur, v_base_cur, 0, 0, 0, 0,
        sparse_mask_cur, sparse_mask_wg_lo_cur, sparse_mask_wg_hi_cur,
        rQ, cur_max, cur_sum, rO, unused_causal_left_cur, first_write);

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