#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Stand-alone unit / micro-benchmark for the OpenVINO intel_gpu RMSNorm kernel
# (rms_gpu_bfyx_opt.cl), reproducing EXACTLY what OV runs for the Qwen3-Omni-4B (C6)
# Thinker prefill shapes. No hypotheses — every JIT const / arg / flag below is taken
# from the verbose log + exec graph (see README "No-hypothesis rule").
#
# Verified from C6.verbose.log (kernel hash rms_gpu_bfyx_opt_12060.../14061.../12133...):
#   - args (set_kernel_arg): shape_info(64B), input, gamma, output  -> 4 args, NO residual
#   - is_dynamic=1 ; input/output tensors ":nopad" -> HAS_PADDING=0 (flat addressing)
#   - normalized dim is STATIC (log `?x?x2560`) -> DATA_SIZE is a compile constant
#   - dispatch (Enqueue kernel): gws=[LWS,rows,1] lws=[LWS,1,1]
#   - dynamic JIT (rms_kernel_bfyx_opt.cpp): LWS=get_local_size(0), SUBGROUP_BLOCK_SIZE=8,
#     STACK_SIZE=ceil_div(D,lws), SLM_SIZE=maxSlmSize, ELEMENTWISE_AFFINE=1 (gamma)
#
# Cases (Thinker, T=2556):
#   hidden : input_layernorm + post_attention_layernorm   [2556, 2560]     D=2560, rank 3
#   q_norm : per-head query RMSNorm (GQA, 32 q-heads)      [2556*32, 128]   D=128,  rank 4
#   k_norm : per-head key   RMSNorm (GQA,  8 kv-heads)     [2556*8,  128]   D=128,  rank 4
#
# Variants:
#   ov     : exact OV config (dynamic, shape_info arg, runtime LWS)      -> matches OV us
#   bucket : dynamic shape_info ABI, but bucket-specialized fixed LWS    -> step 1/3 probe
#   qk_specialized : bucket + compile-time one-subgroup row path for D=128 q/k
#   hidden_specialized : bucket + compile-time multi-subgroup row path for D=2560 hidden
#   generalized : dynamic ABI + generalized LWS rule + subgroup-row specialization;
#                 cached when stack<=16, reread fallback otherwise
#   bucket_tuned : bucket + static stack/subgroup constants              -> step 4 probe
#   static : same work, static-shape specialization (compile-const LWS)  -> optimization target
#
# Run inside the llm container (built_ov env):  python test_omni_rms.py

import os
import numpy as np
import torch
from clops import cl
from clops.utils import kernel_cache

HERE = os.path.dirname(os.path.abspath(__file__))
SUB_GROUP_SIZE = 16
MAX_LWS = 1024  # maxSlmSize = min(maxWorkGroupSize=1024, maxLocalMemSize/(2*2B)) on PTL Xe
CACHE_FLUSH_BYTES = 256 << 20  # rotate through >=256 MiB of buffers to defeat L2/SLC reuse
MAX_REGISTER_STACK = 16


def _pool_size(bytes_per_set):
    """#distinct buffer sets so the rotated footprint exceeds cache (kills reuse hits)."""
    return max(3, min(64, -(-CACHE_FLUSH_BYTES // bytes_per_set)))


def _stats_us(lat_ns):
    """min / median / std (us) from per-call device times (ns)."""
    a = np.sort(np.asarray(lat_ns, dtype=np.float64)) / 1e3
    return a[0], a[len(a) // 2], a.std()


def _read(name):
    return open(os.path.join(HERE, name)).read()


def _body(cl_name):
    """Verbatim OV kernel with the OV #include lines (shim replaces them) stripped."""
    return "\n".join(l for l in _read(cl_name).splitlines()
                     if not l.strip().startswith('#include "include/'))


SHIM = _read("ov_norm_shim.cl")
RMS_BODY = _body("rms_gpu_bfyx_opt.cl")

# Dynamic (=OV) defs: OV passes a shape_info buffer as the leading kernel arg and uses a
# runtime local size. HAS_PADDING=0 (tensors are :nopad), so shape_info is not read by
# the body; DATA_SIZE stays a compile constant (normalized dim is static). Only LWS is
# runtime. This mirrors rms_kernel_bfyx_opt.cpp's dynamic branch exactly.
RMS_DYN_DEFS = r"""
#undef OPTIONAL_SHAPE_INFO_ARG
#undef OPTIONAL_SHAPE_INFO_TENSOR
#define OPTIONAL_SHAPE_INFO_ARG const __global int* shape_info,
#define OPTIONAL_SHAPE_INFO_TENSOR shape_info,
#undef LWS
#define LWS (get_local_size(0))
"""

RMS_SHAPE_ARG_DEFS = r"""
#undef OPTIONAL_SHAPE_INFO_ARG
#undef OPTIONAL_SHAPE_INFO_TENSOR
#define OPTIONAL_SHAPE_INFO_ARG const __global int* shape_info,
#define OPTIONAL_SHAPE_INFO_TENSOR shape_info,
"""


def get_item_num_and_lws(data_size, max_lws=MAX_LWS):
    """Port of rms_kernel_bfyx_opt.cpp::get_item_num_and_lws (f16: local_mem_per_wi=4)."""
    lws, items = 1, data_size
    while (items > 8 or lws < items) and (2 * lws <= max_lws):
        lws *= 2
        items //= 2
    return items, lws


def generalized_lws(data_size, max_lws=MAX_LWS, target_items=8):
    """Largest power-of-two LWS that keeps at least target_items per WI."""
    lws = SUB_GROUP_SIZE
    limit = max(SUB_GROUP_SIZE, min(max_lws, data_size // target_items))
    while 2 * lws <= limit:
        lws *= 2
    return lws


def subgroup_block_size(items):
    if items >> 3:
        return 8
    if items >> 2:
        return 4
    if items >> 1:
        return 2
    return 1


def build(D, rank, eps, variant):
    items, lws = get_item_num_and_lws(D)
    base = (f"-DELEMENTWISE_AFFINE=1 -DSUB_GROUP_SIZE={SUB_GROUP_SIZE} "
            f"-DINPUT_RANK={rank} -DDATA_SIZE={D} -DEPSILON={eps}")
    if variant == "ov":
        # dynamic: SBS=8, STACK_SIZE=ceil_div(D,lws), SLM_SIZE=maxSlmSize; LWS runtime.
        stack = (D + lws - 1) // lws
        opts = base + f" -DSLM_SIZE={MAX_LWS} -DSTACK_SIZE={stack} -DSUBGROUP_BLOCK_SIZE=8"
        src = SHIM + "\n" + RMS_DYN_DEFS + "\n" + RMS_BODY
        disp = f"dyn sbs=8 stack={stack} LWS=rt"
    elif variant == "bucket":
        stack = (D + lws - 1) // lws
        opts = base + f" -DLWS={lws} -DSLM_SIZE={MAX_LWS} -DSTACK_SIZE={stack} -DSUBGROUP_BLOCK_SIZE=8"
        src = SHIM + "\n" + RMS_SHAPE_ARG_DEFS + "\n" + RMS_BODY
        disp = f"bucket sbs=8 stack={stack} LWS={lws}"
    elif variant == "qk_specialized":
        stack = (D + lws - 1) // lws
        opts = (base + f" -DLWS={lws} -DSLM_SIZE={MAX_LWS} -DSTACK_SIZE={stack} "
                f"-DSUBGROUP_BLOCK_SIZE=8 -DONE_SUBGROUP_ROW=1")
        src = SHIM + "\n" + RMS_SHAPE_ARG_DEFS + "\n" + RMS_BODY
        disp = f"qk one-sg stack={stack} LWS={lws}"
    elif variant == "hidden_specialized":
        stack = (D + lws - 1) // lws
        opts = (base + f" -DLWS={lws} -DSLM_SIZE={MAX_LWS} -DSTACK_SIZE={stack} "
                f"-DSUBGROUP_BLOCK_SIZE=8 -DMULTI_SUBGROUP_ROW=1")
        src = SHIM + "\n" + RMS_SHAPE_ARG_DEFS + "\n" + RMS_BODY
        disp = f"hidden multi-sg stack={stack} LWS={lws}"
    elif variant == "generalized":
        lws = generalized_lws(D)
        items = D // lws
        stack = (D + lws - 1) // lws
        reread = stack > MAX_REGISTER_STACK
        row_kind = "one-sg" if lws == SUB_GROUP_SIZE else "multi-sg"
        row_flag = "-DONE_SUBGROUP_ROW=1" if lws == SUB_GROUP_SIZE else "-DMULTI_SUBGROUP_ROW=1"
        opts = (base + f" -DLWS={lws} -DSLM_SIZE={MAX_LWS} -DSTACK_SIZE={min(stack, MAX_REGISTER_STACK)} "
                f"-DSUBGROUP_BLOCK_SIZE=8 {row_flag}")
        if reread:
            opts += " -DRMS_REREAD_INPUT=1"
        src = SHIM + "\n" + RMS_SHAPE_ARG_DEFS + "\n" + RMS_BODY
        mode = "reread" if reread else "cache"
        disp = f"generalized {row_kind} stack={stack} {mode} LWS={lws}"
    elif variant == "bucket_tuned":
        sbs = subgroup_block_size(items)
        stack = items + 1
        opts = base + (f" -DLWS={lws} -DSLM_SIZE={lws} -DSTACK_SIZE={stack} "
                       f"-DSUBGROUP_BLOCK_SIZE={sbs}")
        src = SHIM + "\n" + RMS_SHAPE_ARG_DEFS + "\n" + RMS_BODY
        disp = f"bucket sbs={sbs} stack={stack} LWS={lws}"
    else:  # static specialization (same work)
        sbs = subgroup_block_size(items)
        stack = items + 1
        opts = base + (f" -DLWS={lws} -DSLM_SIZE={lws} -DSTACK_SIZE={stack} "
                       f"-DSUBGROUP_BLOCK_SIZE={sbs}")
        src = SHIM + "\n" + RMS_BODY
        disp = f"sbs={sbs} stack={stack} LWS={lws}"
    return src, opts, lws, items, disp


def ref_rms(x, g, eps):
    xf = torch.from_numpy(x).float()
    var = xf.pow(2).mean(-1, keepdim=True)
    return (xf * torch.rsqrt(var + eps) * torch.from_numpy(g).float()).half().numpy()


def run_case(name, rows, D, rank, ov_ref_us, variant, eps=1e-6, iters=100):
    src, opts, lws, items, disp = build(D, rank, eps, variant)
    kernels = kernel_cache(src, opts)

    np.random.seed(0)
    x = np.random.uniform(-2.0, 2.0, [rows, D]).astype(np.float16)
    g = np.random.uniform(-1.0, 1.0, [D]).astype(np.float16)
    zeros = np.zeros_like(x)
    # Rotate through a pool of DISTINCT input/output buffers so back-to-back launches do
    # not re-read cache-resident data -> measures the true DRAM-bound cost, not L2/SLC hits.
    cold_gamma = True
    bytes_per_set = x.nbytes * 2 + g.nbytes
    pool = _pool_size(bytes_per_set)
    in_pool = [cl.tensor(x) for _ in range(pool)]
    out_pool = [cl.tensor(zeros) for _ in range(pool)]
    gamma_pool = [cl.tensor(g) for _ in range(pool)]
    # OV arg layout: [shape_info, input, gamma, output]. shape_info = 16 int32 (64 B),
    # unused by the body (HAS_PADDING=0) but present to match OV's dynamic signature.
    shape_info = [cl.tensor(np.zeros(16, np.int32))] if variant in (
        "ov", "bucket", "qk_specialized", "hidden_specialized", "generalized", "bucket_tuned") else []

    def _args(i):
        slot = i % pool
        return shape_info + [in_pool[slot], gamma_pool[slot], out_pool[slot]]

    kernels.enqueue("rms_gpu_bfyx_opt", [lws, rows], [lws, 1], *_args(0))
    cl.finish()
    cur = out_pool[0].numpy()
    ref = ref_rms(x, g, eps)
    ok = np.allclose(cur, ref, atol=1e-2, rtol=1e-2)
    if not ok:
        bad = np.abs(cur.astype(np.float32) - ref.astype(np.float32))
        print(f"  [{name}/{variant}] MAX ABS ERR = {bad.max():.4f}")

    # steady-state stats (warm up first; early launches run at a low GPU clock).
    for i in range(iters):
        kernels.enqueue("rms_gpu_bfyx_opt", [lws, rows], [lws, 1], *_args(i))
    cl.finish()
    for i in range(iters):
        kernels.enqueue("rms_gpu_bfyx_opt", [lws, rows], [lws, 1], *_args(i))
    mn, med, std = _stats_us(cl.finish())
    gbps = bytes_per_set / (med * 1e-6) / 1e9  # effective physical DRAM bandwidth at median

    print(f"  {name:7s} {variant:6s} rows={rows:6d} D={D:4d} | LWS={lws:4d} items={items} "
          f"{disp:22s} | acc={'OK ' if ok else 'FAIL'} | med={med:8.3f} min={mn:8.3f} "
          f"std={std:6.3f} us | {gbps:6.1f} GB/s (pool={pool}, gamma={'cold' if cold_gamma else 'hot'}) "
          f"(OV C6 {ov_ref_us:.0f} us)")
    return ok


def main():
    cl.profiling(True)
    print("=== OV rms_gpu_bfyx_opt.cl — Qwen3-Omni-4B C6 Thinker prefill (T=2556) ===")
    print("    ov = exact OV config; static = same work, static-shape specialization")
    print("    buffers rotated (cold cache) -> med/min/std us + effective DRAM GB/s")
    results = []
    # ov_ref_us = per-call device time from C6 CLIntercept trace (rms_gpu_profile.md).
    for name, rows, D, rank, ref in [("tail16", 4096, 272, 3, 0),
                                     ("hidden", 2556, 2560, 3, 722),
                                     ("q_norm", 2556 * 32, 128, 4, 1267),
                                     ("k_norm", 2556 * 8, 128, 4, 319),
                                     ("mid", 2048, 1152, 3, 0),
                                     ("wide", 1024, 8192, 3, 0),
                                     ("huge", 256, 32768, 3, 0)]:
        results.append(run_case(name, rows, D, rank, ref, "ov"))
        results.append(run_case(name, rows, D, rank, ref, "bucket"))
        if D == 2560:
            results.append(run_case(name, rows, D, rank, ref, "hidden_specialized"))
        if D == 128:
            results.append(run_case(name, rows, D, rank, ref, "qk_specialized"))
        results.append(run_case(name, rows, D, rank, ref, "generalized"))
        results.append(run_case(name, rows, D, rank, ref, "bucket_tuned"))
        results.append(run_case(name, rows, D, rank, ref, "static"))
    assert all(results), "accuracy check failed"
    print("accuracy: all good")


if __name__ == "__main__":
    main()
