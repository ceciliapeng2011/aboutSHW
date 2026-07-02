// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Minimal shim that reproduces the subset of OpenVINO intel_gpu kernel_selector
// JIT macros referenced by rms_gpu_bfyx_opt.cl and mvn_gpu_bfyx_opt.cl, so the
// verbatim OV kernels can be compiled stand-alone under the aboutSHW clops harness.
//
// It replaces these OV includes (which pull in the full tensor-jit machinery):
//   rms: include/fetch_utils.cl, include/batch_headers/sub_group_block_{read,write}.cl
//   mvn: include/batch_headers/fetch_data.cl
// The test loader strips those #include lines and prepends this shim.
//
// Only the FP16-in / FP16-out / FP32-accumulator configuration is provided (the
// Qwen3-Omni-4B case). gamma/beta affine for MVN is applied via OV fused-ops in
// production; here the MVN kernel tests the core normalization only.

#define KERNEL(name) __kernel void name
#define OPTIONAL_SHAPE_INFO_ARG
#define OPTIONAL_SHAPE_INFO_TENSOR

#define CAT_(a, b) a##b
#define CAT(a, b) CAT_(a, b)
#define MAKE_VECTOR_TYPE(T, N) CAT(T, N)

// OV common header macro (include/batch_headers/common.cl)
#define unroll_for __attribute__((opencl_unroll_hint)) for

// --- data types: FP16 in/out, FP32 accumulator ---
#define INPUT0_TYPE half
#define INPUT1_TYPE half
#define OUTPUT_TYPE half
#define ACCUMULATOR_TYPE float
#define ACCUMULATOR_VAL_ZERO 0.0f
#define TO_ACCUMULATOR_TYPE(x) ((float)(x))
#define TO_OUTPUT_TYPE(x) convert_half(x)

// --- MVN activation hooks (identity; gamma/beta are fused in OV) ---
#define ACTIVATION_TYPE float
#define TO_ACTIVATION_TYPE(x) ((float)(x))
#define ACTIVATION(x, params) (x)
#define ACTIVATION_PARAMS

// --- required subgroup size attribute ---
#define REQD_SUB_GROUP_SIZE(n) __attribute__((intel_reqd_sub_group_size(n)))

// --- FP16 subgroup block read/write via ushort (matches clops/linear_f16b1.py idiom) ---
#define DT_INPUT_BLOCK_READ(ptr, off)  as_half (intel_sub_group_block_read_us ((const __global ushort*)(ptr) + (off)))
#define DT_INPUT_BLOCK_READ2(ptr, off) as_half2(intel_sub_group_block_read_us2((const __global ushort*)(ptr) + (off)))
#define DT_INPUT_BLOCK_READ4(ptr, off) as_half4(intel_sub_group_block_read_us4((const __global ushort*)(ptr) + (off)))
#define DT_INPUT_BLOCK_READ8(ptr, off) as_half8(intel_sub_group_block_read_us8((const __global ushort*)(ptr) + (off)))

#define DT_OUTPUT_BLOCK_WRITE(ptr, off, val)  intel_sub_group_block_write_us ((__global ushort*)(ptr) + (off), as_ushort (val))
#define DT_OUTPUT_BLOCK_WRITE2(ptr, off, val) intel_sub_group_block_write_us2((__global ushort*)(ptr) + (off), as_ushort2(val))
#define DT_OUTPUT_BLOCK_WRITE4(ptr, off, val) intel_sub_group_block_write_us4((__global ushort*)(ptr) + (off), as_ushort4(val))
#define DT_OUTPUT_BLOCK_WRITE8(ptr, off, val) intel_sub_group_block_write_us8((__global ushort*)(ptr) + (off), as_ushort8(val))

// --- operands of the rms kernel's USE_BLOCK_WRITE test. Value is irrelevant: the
//     macro evaluates to 0 for every build because `& 0xF == 0` binds as
//     `& (0xF == 0)` in C, so the block-write path is always dead code. The
//     identifiers must merely exist for preprocessing. ---
#define OUTPUT_TYPE_SIZE 2
#define OUTPUT_FEATURE_PITCH 1
