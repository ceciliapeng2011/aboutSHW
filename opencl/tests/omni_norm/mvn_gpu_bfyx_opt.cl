// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// mvn_gpu_bfyx_opt — optimized (MAP-Elites search, PTL 4Xe). ~1.16x gmean vs the verbatim
// OV kernel on the Qwen3-Omni-4B C6 vision-encoder MVN shapes (vit 1.17x, merger 1.15x).
// Changes vs baseline (all correctness-gated):
//   1. Read the input from global memory ONCE into registers (baseline read it 3x: once each
//      for the mean pass, the variance pass, and the normalize pass).
//   2. Single-pass mean+variance via sum and sum-of-squares (variance = E[x^2] - E[x]^2), so the
//      variance no longer needs a second data pass.
//   3. Drop work_group_broadcast: every WI derives mean/inv from the two reduced scalars.
//   4. Keep the well-optimized built-in work_group_reduce_add and add NO sub-group-size constraint
//      — forcing REQD_SUB_GROUP_SIZE or a manual SLM reduction was measured to REGRESS on this HW.

#include "include/batch_headers/fetch_data.cl"

#ifndef MVN_STACK_SIZE
#define MVN_STACK_SIZE 16
#endif

#if !IS_DYNAMIC
__attribute__((reqd_work_group_size(LWS, 1, 1)))
#endif
KERNEL (mvn_gpu_bfyx_opt)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* restrict output
#if HAS_FUSED_OPS_DECLS
    , FUSED_OPS_DECLS
#endif
) {
    const uint data_set_idx = get_global_id(1);
    const uint workers_per_data_set = LWS;
    const uint in_data_set_idx = get_global_id(0);
    const uint data_set_size = DATA_SET_SIZE;
    const uint data_sets_count = DATA_SETS_COUNT;
    const uint items_num = data_set_size / workers_per_data_set;
    const uint leftovers = data_set_size % workers_per_data_set;

    const uint data_set_offset = data_set_idx * data_set_size;
    const uint my_data_offset = data_set_offset + in_data_set_idx;
    uint iters_num = items_num;
    if (in_data_set_idx < leftovers)
        ++iters_num;

    float data[MVN_STACK_SIZE];
    float my_sum = 0.f;
    float my_sq = 0.f;
    for (uint i = 0; i < iters_num; ++i) {
        float v = (float)input[my_data_offset + i * workers_per_data_set];
        data[i] = v;
        my_sum += v;
        my_sq = fma(v, v, my_sq);
    }

    float red_sum = work_group_reduce_add(my_sum);
    float my_sum_mean = red_sum / data_set_size;

#if NORMALIZE_VARIANCE == 0
    for (uint i = 0; i < iters_num; ++i) {
        uint iteration_in_data_set_offset = i * workers_per_data_set;
        ACTIVATION_TYPE result = TO_ACTIVATION_TYPE(data[i]) - TO_ACTIVATION_TYPE(my_sum_mean);
#   if HAS_FUSED_OPS
        FUSED_OPS;
        output[my_data_offset + iteration_in_data_set_offset] = FUSED_OPS_RESULT;
#   else
        output[my_data_offset + iteration_in_data_set_offset] = TO_OUTPUT_TYPE(ACTIVATION(result, ACTIVATION_PARAMS));
#   endif
    }
#else
    float red_sq = work_group_reduce_add(my_sq);
    float my_variance = red_sq / data_set_size - my_sum_mean * my_sum_mean;
#   if defined EPS_OUTSIDE_SQRT
    float my_inv = native_powr(native_sqrt(my_variance) + (float)EPSILON, -1.f);
#   elif defined EPS_INSIDE_SQRT
    float my_inv = native_rsqrt(my_variance + (float)EPSILON);
#   else
    float my_inv = native_rsqrt(my_variance);
#   endif

    for (uint i = 0; i < iters_num; ++i) {
        uint iteration_in_data_set_offset = i * workers_per_data_set;
        ACTIVATION_TYPE result = (TO_ACTIVATION_TYPE(data[i]) - TO_ACTIVATION_TYPE(my_sum_mean)) * TO_ACTIVATION_TYPE(my_inv);
#   if HAS_FUSED_OPS
        FUSED_OPS;
        output[my_data_offset + iteration_in_data_set_offset] = FUSED_OPS_RESULT;
#   else
        output[my_data_offset + iteration_in_data_set_offset] = TO_OUTPUT_TYPE(ACTIVATION(result, ACTIVATION_PARAMS));
#   endif
    }
#endif
}
