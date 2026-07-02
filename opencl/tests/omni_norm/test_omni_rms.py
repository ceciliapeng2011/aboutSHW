#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Stand-alone unit / micro-benchmark for the OpenVINO intel_gpu RMSNorm kernel
# (rms_gpu_bfyx_opt.cl), reproducing the exact JIT constants and GWS/LWS host
# dispatch that OV derives for the Qwen3-Omni-4B (C6) prefill shapes.
#
# Cases (from C6.verbose.log, last iter):
#   hidden : input_layernorm + post_attention_layernorm   [2556, 2560]     D=2560
#   q_norm : per-head query RMSNorm (GQA, 32 q-heads)      [2556*32, 128]   D=128
#   k_norm : per-head key   RMSNorm (GQA,  8 kv-heads)     [2556*8,  128]   D=128
#
# JIT const / dispatch derivation mirrors:
#   rms_kernel_bfyx_opt.cpp :: get_item_num_and_lws() / SetDefault() / GetJitConstants()
#
# Run inside the llm container (built_ov env), e.g.:
#   python test_omni_rms.py

import os
import numpy as np
import torch
from clops import cl
from clops.utils import kernel_cache

HERE = os.path.dirname(os.path.abspath(__file__))
SUB_GROUP_SIZE = 16
MAX_LWS = 1024  # min(maxWorkGroupSize=1024, maxLocalMemSize/(2*2B) >> 1024) on PTL Xe


def load_kernel(cl_name):
    """Prepend the shim and strip the OV #include lines it replaces."""
    shim = open(os.path.join(HERE, "ov_norm_shim.cl")).read()
    body = open(os.path.join(HERE, cl_name)).read()
    body = "\n".join(l for l in body.splitlines()
                     if not l.strip().startswith('#include "include/'))
    return shim + "\n" + body


RMS_SRC = load_kernel("rms_gpu_bfyx_opt.cl")


def get_item_num_and_lws(data_size, max_lws=MAX_LWS):
    """Port of rms_kernel_bfyx_opt.cpp::get_item_num_and_lws (f16: local_mem_per_wi=4)."""
    lws, items = 1, data_size
    while (items > 8 or lws < items) and (2 * lws <= max_lws):
        lws *= 2
        items //= 2
    return items, lws


def subgroup_block_size(items):
    if items >> 3:
        return 8
    if items >> 2:
        return 4
    if items >> 1:
        return 2
    return 1


def build_opts(D, rank, eps):
    items, lws = get_item_num_and_lws(D)
    sbs = subgroup_block_size(items)
    stack = items + 1  # static path: dispatchData.itemsNum + 1
    opts = (f"-DELEMENTWISE_AFFINE=1 -DSUB_GROUP_SIZE={SUB_GROUP_SIZE} -DINPUT_RANK={rank} "
            f"-DDATA_SIZE={D} -DLWS={lws} -DSLM_SIZE={lws} -DSTACK_SIZE={stack} "
            f"-DSUBGROUP_BLOCK_SIZE={sbs} -DEPSILON={eps}")
    return opts, lws, items, sbs, stack


def ref_rms(x, g, eps):
    xf = torch.from_numpy(x).float()
    var = xf.pow(2).mean(-1, keepdim=True)
    out = xf * torch.rsqrt(var + eps) * torch.from_numpy(g).float()
    return out.half().numpy()


def run_case(name, rows, D, rank, ov_ref_us, eps=1e-6, iters=30):
    opts, lws, items, sbs, stack = build_opts(D, rank, eps)
    kernels = kernel_cache(RMS_SRC, opts)

    np.random.seed(0)
    x = np.random.uniform(-2.0, 2.0, [rows, D]).astype(np.float16)
    g = np.random.uniform(-1.0, 1.0, [D]).astype(np.float16)
    tx = cl.tensor(x)
    tg = cl.tensor(g)
    to = cl.tensor(np.zeros_like(x))

    # correctness
    kernels.enqueue("rms_gpu_bfyx_opt", [lws, rows], [lws, 1], tx, tg, to)
    cl.finish()
    cur = to.numpy()
    ref = ref_rms(x, g, eps)
    ok = np.allclose(cur, ref, atol=1e-2, rtol=1e-2)
    if not ok:
        bad = np.abs(cur.astype(np.float32) - ref.astype(np.float32))
        print(f"  [{name}] MAX ABS ERR = {bad.max():.4f} at {np.unravel_index(bad.argmax(), bad.shape)}")

    # timing
    for _ in range(iters):
        kernels.enqueue("rms_gpu_bfyx_opt", [lws, rows], [lws, 1], tx, tg, to)
    lat = cl.finish()
    avg_us = sum(lat) / len(lat) / 1e3

    print(f"  {name:8s} rows={rows:6d} D={D:4d} | LWS={lws:4d} items={items} "
          f"sbs={sbs} stack={stack} | acc={'OK ' if ok else 'FAIL'} | "
          f"{avg_us:7.3f} us/call  (OV C6 avg {ov_ref_us:.3f} us)")
    return ok


def main():
    cl.profiling(True)
    print("=== OV rms_gpu_bfyx_opt.cl — Qwen3-Omni-4B C6 prefill (T=2556) ===")
    results = []
    # ov_ref_us = per-call avg from C6 profiling (rms_gpu_profile.md)
    results.append(run_case("hidden", 2556,      2560, 3, ov_ref_us=722))
    results.append(run_case("q_norm", 2556 * 32, 128,  4, ov_ref_us=1267))
    results.append(run_case("k_norm", 2556 * 8,  128,  4, ov_ref_us=319))
    assert all(results), "accuracy check failed"
    print("accuracy: all good")


if __name__ == "__main__":
    main()
