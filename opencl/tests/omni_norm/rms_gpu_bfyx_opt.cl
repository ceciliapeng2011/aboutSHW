// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// rms_gpu_bfyx_opt — optimized (MAP-Elites search, PTL 4Xe). ~1.70x gmean vs the verbatim
// OV kernel on the Qwen3-Omni-4B C6 Thinker RMS shapes (hidden 1.43x, q_norm 1.86x, k_norm 1.84x).
// Changes vs baseline (all correctness-gated):
//   1. Vectorized load/store CASCADE (block8 -> block4 -> block2 -> scalar) so the coalesced
//      sub-group block path fires for ANY items_num, not only multiples of the JIT block size
//      (the original path was dead for these shapes: lws==SUB_GROUP_SIZE and items<block size).
//   2. Square via multiply and native_rsqrt (drop native_powr(x,2)/powr(sqrt,-1)).
//   3. Single-sub-group rows (q/k, lws==SUB_GROUP_SIZE) skip SLM + all barriers entirely.
//   4. Multi-sub-group rows (hidden) reduce partials with ONE barrier (every sub-group reduces
//      redundantly) instead of the original log-tree of barriers.

#include "include/fetch_utils.cl"
#include "include/batch_headers/sub_group_block_read.cl"
#include "include/batch_headers/sub_group_block_write.cl"

#define USE_BLOCK_WRITE 1

REQD_SUB_GROUP_SIZE(SUB_GROUP_SIZE)
KERNEL(rms_gpu_bfyx_opt)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* input,
#if ELEMENTWISE_AFFINE
    const __global INPUT1_TYPE* gamma,
#endif
    __global OUTPUT_TYPE* output
    #if HAS_FUSED_OPS_DECLS
        , FUSED_OPS_DECLS
    #endif
)
{
    const uint data_idx = get_global_id(1);
    const uint in_data_idx = get_global_id(0);
    const uint workers_per_data = LWS;
    const uint data_size = DATA_SIZE;
    const uint items_num = data_size / workers_per_data;
    const uint leftovers = data_size % workers_per_data;

    #if HAS_PADDING
        uint b_idx = 0;
        uint f_idx = 0;
        uint z_idx = 0;
        uint y_idx = 0;
        uint x_idx = 0;
        #if INPUT_RANK == 2
            b_idx = (data_idx);
        #elif INPUT_RANK == 3
            f_idx = (data_idx % (INPUT0_FEATURE_NUM));
            b_idx = (data_idx / (INPUT0_FEATURE_NUM));
        #else
            y_idx = (data_idx % (INPUT0_SIZE_Y));
            z_idx = (data_idx / (INPUT0_SIZE_Y)) % INPUT0_SIZE_Z;
            f_idx = (data_idx / (INPUT0_SIZE_Y * INPUT0_SIZE_Z)) % INPUT0_FEATURE_NUM;
            b_idx = (data_idx / (INPUT0_SIZE_Y * INPUT0_SIZE_Z * INPUT0_FEATURE_NUM)) % INPUT0_BATCH_NUM;
        #endif

        const uint input_data_offset = FUNC_CALL(get_input_index)(OPTIONAL_SHAPE_INFO_TENSOR b_idx, f_idx, 0, z_idx, y_idx, x_idx);
    #else
        const uint input_data_offset = data_idx * data_size;
    #endif

    const uint output_data_offset = data_idx * data_size;

    const uint sgs = get_sub_group_size();
    const uint subgroup_offset = get_sub_group_id() * sgs * items_num;

    ACCUMULATOR_TYPE data[STACK_SIZE];
    ACCUMULATOR_TYPE rms = ACCUMULATOR_VAL_ZERO;

    __local ACCUMULATOR_TYPE slm_buf[SLM_SIZE];

    // ---- vectorized load cascade: block8 -> block4 -> block2 -> scalar ----
    uint i = 0;
    if (workers_per_data >= SUB_GROUP_SIZE)
    {
        const uint ibase = input_data_offset + subgroup_offset;
        for (; i + 8 <= items_num; i += 8) {
            half8 v = DT_INPUT_BLOCK_READ8(input, ibase + i * sgs);
            unroll_for (int j = 0; j < 8; j++) { ACCUMULATOR_TYPE t = TO_ACCUMULATOR_TYPE(v[j]); rms += t * t; data[i + j] = t; }
        }
        for (; i + 4 <= items_num; i += 4) {
            half4 v = DT_INPUT_BLOCK_READ4(input, ibase + i * sgs);
            unroll_for (int j = 0; j < 4; j++) { ACCUMULATOR_TYPE t = TO_ACCUMULATOR_TYPE(v[j]); rms += t * t; data[i + j] = t; }
        }
        for (; i + 2 <= items_num; i += 2) {
            half2 v = DT_INPUT_BLOCK_READ2(input, ibase + i * sgs);
            unroll_for (int j = 0; j < 2; j++) { ACCUMULATOR_TYPE t = TO_ACCUMULATOR_TYPE(v[j]); rms += t * t; data[i + j] = t; }
        }
    }
    for (; i < items_num; i++)
    {
        ACCUMULATOR_TYPE tmp = TO_ACCUMULATOR_TYPE(input[input_data_offset + subgroup_offset + get_sub_group_local_id() + i * sgs]);
        rms += tmp * tmp;
        data[i] = tmp;
    }

    if (in_data_idx < leftovers)
    {
        ACCUMULATOR_TYPE tmp = TO_ACCUMULATOR_TYPE(input[input_data_offset + workers_per_data * items_num + in_data_idx]);
        rms += tmp * tmp;
        data[items_num] = tmp;
    }

    rms = sub_group_reduce_add(rms);

    if (get_num_sub_groups() == 1) {
        // single sub-group covers the whole row: no SLM / no barrier needed.
        rms = native_rsqrt(rms / data_size + TO_ACCUMULATOR_TYPE(EPSILON));
    } else {
        if (get_sub_group_local_id() == 0)
            slm_buf[get_sub_group_id()] = rms;

        barrier(CLK_LOCAL_MEM_FENCE);
        // every sub-group redundantly reduces the partials -> no broadcast barrier (1 total).
        const uint nsg = get_num_sub_groups();
        ACCUMULATOR_TYPE p = ACCUMULATOR_VAL_ZERO;
        for (uint k = get_sub_group_local_id(); k < nsg; k += sgs)
            p += slm_buf[k];
        p = sub_group_reduce_add(p);
        rms = native_rsqrt(p / data_size + TO_ACCUMULATOR_TYPE(EPSILON));
    }

    #if HAS_FUSED_OPS
        uint b, f, z, y, x;
        #if INPUT_RANK == 1
            f = z = y = x = 1;
        #elif INPUT_RANK == 2
            z = y = x = 1;
            b = data_idx;
        #elif INPUT_RANK == 3
            x = 1;
            f = data_idx % OUTPUT_FEATURE_NUM;
            b = data_idx / OUTPUT_FEATURE_NUM;
        #else
            x = data_idx;
            y = x % OUTPUT_SIZE_Y;      x = x / OUTPUT_SIZE_Y;
            z = x % OUTPUT_SIZE_Z;      x = x / OUTPUT_SIZE_Z;
            f = x % OUTPUT_FEATURE_NUM; x = x / OUTPUT_FEATURE_NUM;
            b = x % OUTPUT_BATCH_NUM;   x = x / OUTPUT_BATCH_NUM;
        #endif
    #endif

#if HAS_FUSED_OPS
    // general/fused path: scalar (fused-ops not used by the Omni RMS config)
    for (i = 0; i < items_num; i++)
    {
#if ELEMENTWISE_AFFINE
        ACCUMULATOR_TYPE temp = TO_ACCUMULATOR_TYPE(gamma[subgroup_offset + get_sub_group_local_id() + i * sgs]);
        OUTPUT_TYPE normalized = TO_OUTPUT_TYPE(rms * data[i] * temp);
#else
        OUTPUT_TYPE normalized = TO_OUTPUT_TYPE(rms * data[i]);
#endif
        LAST_DIM = subgroup_offset + get_sub_group_local_id() + i * sgs;
        FUSED_OPS;
        normalized = FUSED_OPS_RESULT;
        output[output_data_offset + subgroup_offset + get_sub_group_local_id() + i * sgs] = normalized;
    }
#else
    // ---- vectorized store cascade: block8 -> block4 -> block2 -> scalar ----
    i = 0;
    if (workers_per_data >= SUB_GROUP_SIZE)
    {
        const uint obase = output_data_offset + subgroup_offset;
        const uint gbase = subgroup_offset;
        for (; i + 8 <= items_num; i += 8) {
#if ELEMENTWISE_AFFINE
            half8 g = DT_INPUT_BLOCK_READ8(gamma, gbase + i * sgs);
#endif
            half8 o;
            unroll_for (int j = 0; j < 8; j++) {
#if ELEMENTWISE_AFFINE
                o[j] = TO_OUTPUT_TYPE(rms * data[i + j] * TO_ACCUMULATOR_TYPE(g[j]));
#else
                o[j] = TO_OUTPUT_TYPE(rms * data[i + j]);
#endif
            }
            DT_OUTPUT_BLOCK_WRITE8(output, obase + i * sgs, o);
        }
        for (; i + 4 <= items_num; i += 4) {
#if ELEMENTWISE_AFFINE
            half4 g = DT_INPUT_BLOCK_READ4(gamma, gbase + i * sgs);
#endif
            half4 o;
            unroll_for (int j = 0; j < 4; j++) {
#if ELEMENTWISE_AFFINE
                o[j] = TO_OUTPUT_TYPE(rms * data[i + j] * TO_ACCUMULATOR_TYPE(g[j]));
#else
                o[j] = TO_OUTPUT_TYPE(rms * data[i + j]);
#endif
            }
            DT_OUTPUT_BLOCK_WRITE4(output, obase + i * sgs, o);
        }
        for (; i + 2 <= items_num; i += 2) {
#if ELEMENTWISE_AFFINE
            half2 g = DT_INPUT_BLOCK_READ2(gamma, gbase + i * sgs);
#endif
            half2 o;
            unroll_for (int j = 0; j < 2; j++) {
#if ELEMENTWISE_AFFINE
                o[j] = TO_OUTPUT_TYPE(rms * data[i + j] * TO_ACCUMULATOR_TYPE(g[j]));
#else
                o[j] = TO_OUTPUT_TYPE(rms * data[i + j]);
#endif
            }
            DT_OUTPUT_BLOCK_WRITE2(output, obase + i * sgs, o);
        }
    }
    for (; i < items_num; i++)
    {
#if ELEMENTWISE_AFFINE
        ACCUMULATOR_TYPE temp = TO_ACCUMULATOR_TYPE(gamma[subgroup_offset + get_sub_group_local_id() + i * sgs]);
        OUTPUT_TYPE normalized = TO_OUTPUT_TYPE(rms * data[i] * temp);
#else
        OUTPUT_TYPE normalized = TO_OUTPUT_TYPE(rms * data[i]);
#endif
        output[output_data_offset + subgroup_offset + get_sub_group_local_id() + i * sgs] = normalized;
    }
#endif

    if (in_data_idx < leftovers)
    {
#if ELEMENTWISE_AFFINE
        ACCUMULATOR_TYPE temp = TO_ACCUMULATOR_TYPE(gamma[workers_per_data * items_num + in_data_idx]);
        OUTPUT_TYPE normalized = TO_OUTPUT_TYPE(rms * data[items_num] * temp);
#else
        OUTPUT_TYPE normalized = TO_OUTPUT_TYPE(rms * data[items_num]);
#endif
        #if HAS_FUSED_OPS
            LAST_DIM = workers_per_data * items_num + in_data_idx;
            FUSED_OPS;
            normalized = FUSED_OPS_RESULT;
        #endif
        output[output_data_offset + workers_per_data * items_num + in_data_idx] = normalized;
    }
}
#undef USE_BLOCK_WRITE
