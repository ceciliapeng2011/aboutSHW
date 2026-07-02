#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Stand-alone unit / micro-benchmark for the OpenVINO intel_gpu MVN (LayerNorm)
# kernel (mvn_gpu_bfyx_opt.cl), reproducing the exact JIT constants and GWS/LWS
# host dispatch that OV derives for the Qwen3-Omni-4B (C6) vision encoder shapes.
#
# Cases (from C6.verbose.log, last iter):
#   vit    : ViT block LayerNorm            [6864, 1152]   D=1152
#   merger : patch-Merger LayerNorm         [1716, 4608]   D=4608
#
# MVN mode = ACROSS_CHANNELS, NORMALIZE_VARIANCE=1. The base kernel computes the
# core normalization (x-mean)/sqrt(var+eps); gamma/beta are applied via OV
# fused-ops in production and are intentionally out of scope here.
#
# Dispatch derivation mirrors:
#   mvn_kernel_bfyx_opt.cpp :: SetDefault() / GetJitConstants()
#
# Run inside the llm container (built_ov env):
#   python test_omni_mvn.py

import os
import numpy as np
import torch
from clops import cl
from clops.utils import kernel_cache

HERE = os.path.dirname(os.path.abspath(__file__))
MAX_LWS = 1024  # min(maxWorkGroupSize=1024, maxLocalMemSize/(2*2B) >> 1024) on PTL Xe


def load_kernel(cl_name):
    shim = open(os.path.join(HERE, "ov_norm_shim.cl")).read()
    body = open(os.path.join(HERE, cl_name)).read()
    body = "\n".join(l for l in body.splitlines()
                     if not l.strip().startswith('#include "include/'))
    return shim + "\n" + body


MVN_SRC = load_kernel("mvn_gpu_bfyx_opt.cl")


def get_lws(data_set_size, max_lws=MAX_LWS):
    """Port of mvn_kernel_bfyx_opt.cpp::SetDefault LWS loop (f16: local_mem_per_wi=4)."""
    lws, items = 1, data_set_size
    while (items > 8 or lws < items) and (2 * lws <= max_lws):
        lws *= 2
        items //= 2
    return items, lws


def build_opts(D, eps):
    items, lws = get_lws(D)
    opts = (f"-DIS_DYNAMIC=0 -DNORMALIZE_VARIANCE=1 -DEPS_INSIDE_SQRT "
            f"-DLWS={lws} -DDATA_SET_SIZE={D} -DDATA_SETS_COUNT=0 -DEPSILON={eps}")
    # DATA_SETS_COUNT is unused by the kernel body (kept for parity); rows come via GWS.
    return opts, lws, items


def ref_mvn(x, eps):
    xf = torch.from_numpy(x).float()
    mean = xf.mean(-1, keepdim=True)
    var = xf.var(-1, unbiased=False, keepdim=True)  # biased: /N, matches kernel
    out = (xf - mean) / torch.sqrt(var + eps)
    return out.half().numpy()


def run_case(name, rows, D, ov_ref_us, eps=1e-6, iters=30):
    opts, lws, items = build_opts(D, eps)
    kernels = kernel_cache(MVN_SRC, opts)

    np.random.seed(0)
    x = np.random.uniform(-2.0, 2.0, [rows, D]).astype(np.float16)
    tx = cl.tensor(x)
    to = cl.tensor(np.zeros_like(x))

    # correctness
    kernels.enqueue("mvn_gpu_bfyx_opt", [lws, rows], [lws, 1], tx, to)
    cl.finish()
    cur = to.numpy()
    ref = ref_mvn(x, eps)
    ok = np.allclose(cur, ref, atol=2e-2, rtol=2e-2)
    if not ok:
        bad = np.abs(cur.astype(np.float32) - ref.astype(np.float32))
        print(f"  [{name}] MAX ABS ERR = {bad.max():.4f} at {np.unravel_index(bad.argmax(), bad.shape)}")

    # timing
    for _ in range(iters):
        kernels.enqueue("mvn_gpu_bfyx_opt", [lws, rows], [lws, 1], tx, to)
    lat = cl.finish()
    avg_us = sum(lat) / len(lat) / 1e3

    print(f"  {name:8s} rows={rows:6d} D={D:4d} | LWS={lws:4d} items={items} | "
          f"acc={'OK ' if ok else 'FAIL'} | {avg_us:7.3f} us/call  (OV C6 avg {ov_ref_us:.3f} us)")
    return ok


def main():
    cl.profiling(True)
    print("=== OV mvn_gpu_bfyx_opt.cl — Qwen3-Omni-4B C6 vision encoder ===")
    results = []
    # ov_ref_us = per-call avg from C6 profiling (mvn_gpu_profile.md)
    results.append(run_case("vit",    6864, 1152, ov_ref_us=833))
    results.append(run_case("merger", 1716, 4608, ov_ref_us=910))
    assert all(results), "accuracy check failed"
    print("accuracy: all good")


if __name__ == "__main__":
    main()
