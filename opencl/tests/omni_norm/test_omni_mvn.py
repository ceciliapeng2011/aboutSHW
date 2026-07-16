#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Stand-alone unit / micro-benchmark for the OpenVINO intel_gpu MVN (LayerNorm) kernel
# (mvn_gpu_bfyx_opt.cl), reproducing EXACTLY what OV runs for the Qwen3-Omni-4B (C6)
# vision-encoder shapes. No hypotheses — every JIT const / arg / flag is taken from the
# verbose log + exec graph (see README "No-hypothesis rule").
#
# Verified from C6.verbose.log (kernel hash mvn_gpu_bfyx_opt_17508.../9551...) and
# Model0_exec_graph.xml:
#   - args (set_kernel_arg): shape_info(64B), input, output, gamma, beta -> 5 args
#   - exec graph originalLayersNames = "MVN, Multiply(gamma), Add(beta)" -> fused affine
#   - is_dynamic=1 ; tensors ":nopad"
#   - normalized dim is STATIC (log `?x1152`) -> DATA_SET_SIZE is a compile constant
#   - dispatch: gws=[LWS,rows,1] lws=[LWS,1,1]
#   - mode ACROSS_CHANNELS, NORMALIZE_VARIANCE=1, EPS_INSIDE_SQRT
#
# Cases (vision encoder):
#   vit    : ViT block LayerNorm     [6864, 1152]   D=1152
#   merger : patch-Merger LayerNorm  [1716, 4608]   D=4608
#
# Variants:
#   ov     : exact OV config (dynamic, shape_info arg, fused gamma+beta, runtime LWS)
#   bucket : dynamic shape_info ABI, but bucket-specialized fixed LWS + reqd_work_group_size
#   generalized : dynamic ABI + generalized LWS rule; cached when stack<=16, reread fallback otherwise
#   static : same work, static-shape specialization
#
# NOTE: the fused epilogue reproduces OV's affine functionally (out = norm*gamma + beta,
# from the 5-arg layout + exec-graph Multiply/Add). OV auto-generates its FUSED_OPS macro
# from the fusion; the generated macro body itself was not dumped, so its exact codegen
# (e.g. any bounds guard) is not byte-reproduced — the math and I/O are identical.
#
# Run inside the llm container (built_ov env):  python test_omni_mvn.py

import os
import numpy as np
import torch
from clops import cl
from clops.utils import Colors, kernel_cache

HERE = os.path.dirname(os.path.abspath(__file__))
MAX_LWS = 1024  # min(maxWorkGroupSize=1024, maxLocalMemSize/(2*2B)) on PTL Xe
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
    return "\n".join(l for l in _read(cl_name).splitlines()
                     if not l.strip().startswith('#include "include/'))


SHIM = _read("ov_norm_shim.cl")
MVN_BODY = _body("mvn_gpu_bfyx_opt.cl")
MVN_WELFORD_BODY = _body("mvn_gpu_bfyx_opt_welford.cl")

# Fused affine epilogue (OV: MVN, Multiply(gamma), Add(beta)). Mirror the generated
# macro body from the dumped CL source as closely as possible to reduce the remaining
# vit delta: two fused ops (mul + add) with the same load/calc ordering and half
# conversions that OV emits. Gamma/beta are the two extra kernel args (arg3/arg4).
MVN_FUSED_DEFS = r"""
#define HAS_FUSED_OPS 1
#define HAS_FUSED_OPS_DECLS 1
#define OUTPUT_SIZE_X 1
#define OUTPUT_SIZE_Y 1
#define FUSED_OPS_DECLS const __global half* fused_gamma, const __global half* fused_beta
#define FUSED_OP0_LOAD \
    half eltwise0_data0 = fused_gamma[(in_data_set_idx + iteration_in_data_set_offset) / (OUTPUT_SIZE_X * OUTPUT_SIZE_Y)];
#define FUSED_OP0_ACTION \
    half result_out_0_tmp = result * eltwise0_data0;\
    half result_out_0 = convert_half(result_out_0_tmp);
#define FUSED_OP1_LOAD \
    half eltwise1_data0 = fused_beta[(in_data_set_idx + iteration_in_data_set_offset) / (OUTPUT_SIZE_X * OUTPUT_SIZE_Y)];
#define FUSED_OP1_ACTION \
    half result_out_1_tmp = result_out_0 + eltwise1_data0;\
    half result_out_1 = convert_half(result_out_1_tmp);
#define FUSED_OPS \
    FUSED_OP0_LOAD\
    FUSED_OP0_ACTION\
    FUSED_OP1_LOAD\
    FUSED_OP1_ACTION
#define FUSED_OPS_RESULT result_out_1
#define FUSED_OPS_PRELOAD \
    FUSED_OP0_LOAD\
    FUSED_OP1_LOAD
#define FUSED_OPS_CALC \
    FUSED_OP0_ACTION\
    FUSED_OP1_ACTION
#define FUSED_OPS_CAN_USE_PRELOAD 1
"""

# Dynamic (=OV): shape_info leading arg + runtime LWS. IS_DYNAMIC=1 drops the
# reqd_work_group_size attribute. DATA_SET_SIZE stays a compile constant (normalized dim
# static); shape_info is not read by the body.
MVN_DYN_DEFS = r"""
#undef OPTIONAL_SHAPE_INFO_ARG
#define OPTIONAL_SHAPE_INFO_ARG const __global int* shape_info,
#undef LWS
#define LWS (get_local_size(0))
"""

MVN_SHAPE_ARG_DEFS = r"""
#undef OPTIONAL_SHAPE_INFO_ARG
#define OPTIONAL_SHAPE_INFO_ARG const __global int* shape_info,
"""


def get_lws(data_set_size, max_lws=MAX_LWS):
    """Port of mvn_kernel_bfyx_opt.cpp::SetDefault LWS loop (f16: local_mem_per_wi=4)."""
    lws, items = 1, data_set_size
    while (items > 8 or lws < items) and (2 * lws <= max_lws):
        lws *= 2
        items //= 2
    return items, lws


def generalized_lws(data_set_size, max_lws=MAX_LWS, target_items=8):
    """Largest power-of-two LWS that keeps at least target_items per WI."""
    lws = 1
    limit = max(1, min(max_lws, data_set_size // target_items))
    while 2 * lws <= limit:
        lws *= 2
    return lws


def build(D, eps, variant):
    use_welford = variant.startswith("welford_")
    base_variant = variant[len("welford_"):] if use_welford else variant
    kernel_body = MVN_WELFORD_BODY if use_welford else MVN_BODY

    items, lws = get_lws(D)
    stack = (D + lws - 1) // lws
    base = (f"-DNORMALIZE_VARIANCE=1 -DEPS_INSIDE_SQRT -DEPSILON={eps} "
            f"-DDATA_SET_SIZE={D} -DDATA_SETS_COUNT=0")
    if base_variant == "ov":
        # Welford kernel declares local arrays sized by LWS; OpenCL requires that
        # size expression to be compile-time constant (runtime get_local_size is invalid).
        if use_welford:
            opts = base + f" -DIS_DYNAMIC=1 -DLWS={lws} -DMVN_STACK_SIZE={stack}"
            src = SHIM + "\n" + MVN_SHAPE_ARG_DEFS + "\n" + MVN_FUSED_DEFS + "\n" + kernel_body
        else:
            opts = base + f" -DIS_DYNAMIC=1 -DMVN_STACK_SIZE={stack}"
            src = SHIM + "\n" + MVN_DYN_DEFS + "\n" + MVN_FUSED_DEFS + "\n" + kernel_body
    elif base_variant == "bucket":
        opts = base + f" -DIS_DYNAMIC=0 -DLWS={lws} -DMVN_STACK_SIZE={stack}"
        src = SHIM + "\n" + MVN_SHAPE_ARG_DEFS + "\n" + MVN_FUSED_DEFS + "\n" + kernel_body
    elif base_variant in ("generalized", "gen_t16", "gen_hybrid", "gen_stack12"):
        if base_variant == "gen_t16":
            target_items = 16
            stack_cap = MAX_REGISTER_STACK
            tag = "gen t16"
        elif base_variant == "gen_hybrid":
            # Heuristic for larger normalized dims on 8Xe: use deeper per-WI work.
            target_items = 16 if D >= 8192 else 8
            stack_cap = MAX_REGISTER_STACK
            tag = f"gen hybrid t{target_items}"
        elif base_variant == "gen_stack12":
            target_items = 8
            stack_cap = 12
            tag = "gen stack12"
        else:
            target_items = 8
            stack_cap = MAX_REGISTER_STACK
            tag = "gen t8"

        lws = generalized_lws(D, target_items=target_items)
        items = D // lws
        stack = (D + lws - 1) // lws
        reread = stack > stack_cap
        opts = base + f" -DIS_DYNAMIC=0 -DLWS={lws} -DMVN_STACK_SIZE={min(stack, stack_cap)}"
        if reread:
            opts += " -DMVN_REREAD_INPUT=1"
        src = SHIM + "\n" + MVN_SHAPE_ARG_DEFS + "\n" + MVN_FUSED_DEFS + "\n" + kernel_body
        mode = "reread" if reread else "cache"
        disp = f"{tag} stack={stack} cap={stack_cap} {mode}"
    else:  # static specialization (same fused work)
        opts = base + f" -DIS_DYNAMIC=0 -DLWS={lws} -DMVN_STACK_SIZE={stack}"
        src = SHIM + "\n" + MVN_FUSED_DEFS + "\n" + kernel_body
        disp = f"static stack={stack}"
    if base_variant == "ov":
        disp = f"ov stack={stack}"
    elif base_variant == "bucket":
        disp = f"bucket stack={stack}"
    if use_welford:
        disp = f"welford {disp}"
    return src, opts, lws, items, disp


def ref_mvn(x, eps, gamma, beta):
    xf = torch.from_numpy(x).float()
    mean = xf.mean(-1, keepdim=True)
    var = xf.var(-1, unbiased=False, keepdim=True)  # biased (/N), matches kernel
    out = (xf - mean) / torch.sqrt(var + eps)
    out = out * torch.from_numpy(gamma).float() + torch.from_numpy(beta).float()
    return out.half().numpy()


def make_input(rows, D, distribution):
    if distribution == "uniform":
        return np.random.uniform(-2.0, 2.0, [rows, D]).astype(np.float16)
    if distribution == "cancellation":
        # Around 128, fp16 spacing is 0.125; tiny variance on a large offset amplifies
        # cancellation in E[x^2]-E[x]^2 when accumulated in fp32.
        base = np.float16(128.0)
        step = np.float16(0.125)
        jitter = np.random.choice(np.array([-1.0, 0.0, 1.0], dtype=np.float16), size=(rows, D))
        return (base + jitter * step).astype(np.float16)
    raise ValueError(f"unknown distribution: {distribution}")


def variance_cancellation_stats(x):
    x32 = x.astype(np.float32)
    mean32 = x32.mean(-1, keepdims=True)
    var_naive = (x32 * x32).mean(-1, keepdims=True) - mean32 * mean32

    x64 = x.astype(np.float64)
    mean64 = x64.mean(-1, keepdims=True)
    var_ref = ((x64 - mean64) ** 2).mean(-1, keepdims=True).astype(np.float32)

    rel = np.abs(var_naive - var_ref) / np.maximum(var_ref, 1e-12)
    return float((var_naive < 0).mean()), float(np.percentile(rel, 50)), float(np.percentile(rel, 99))


def _color(text, color):
    return f"{color}{text}{Colors.END}"


def print_pair_summary(records, base_variants, title):
    print(f"\n=== {title} per-pair summary (base vs welford) ===")
    print("    pair              base_med(us)  welford_med(us)   delta    acc passrate(base - welford)")
    for base in base_variants:
        welford = f"welford_{base}"
        base_rows = [r for r in records if r["variant"] == base]
        welford_rows = [r for r in records if r["variant"] == welford]
        if not base_rows or not welford_rows:
            continue

        base_med = float(np.mean([r["med_us"] for r in base_rows]))
        welford_med = float(np.mean([r["med_us"] for r in welford_rows]))
        delta_pct = (welford_med - base_med) / max(base_med, 1e-12) * 100.0
        delta_str = f"{delta_pct:+6.1f}%"
        delta_col = Colors.GREEN if delta_pct < 0 else Colors.RED

        base_by_name = {r["name"]: r for r in base_rows}
        welford_by_name = {r["name"]: r for r in welford_rows}
        common_names = sorted(set(base_by_name.keys()) & set(welford_by_name.keys()))
        base_pass = sum(1 for n in common_names if base_by_name[n]["ok"])
        welford_pass = sum(1 for n in common_names if welford_by_name[n]["ok"])
        pair_total = len(common_names)
        acc_str = f"{base_pass}/{pair_total} - {welford_pass}/{pair_total}"
        acc_col = Colors.GREEN if (base_pass == pair_total and welford_pass == pair_total) else Colors.YELLOW

        print(
            f"    {base:16s} {base_med:12.3f} {welford_med:16.3f} "
            f"{_color(delta_str, delta_col):>12s} {'':>9s} {_color(acc_str, acc_col):>9s}"
        )


def run_case(name, rows, D, ov_ref_us, variant, eps=1e-6, iters=100, distribution="uniform"):
    src, opts, lws, items, disp = build(D, eps, variant)
    kernels = kernel_cache(src, opts)
    base_variant = variant[len("welford_"):] if variant.startswith("welford_") else variant

    np.random.seed(0)
    x = make_input(rows, D, distribution)
    gamma = np.random.uniform(-1.0, 1.0, [D]).astype(np.float16)
    beta = np.random.uniform(-1.0, 1.0, [D]).astype(np.float16)
    zeros = np.zeros_like(x)
    # Rotate through a pool of DISTINCT input/output/fused buffers so back-to-back launches do
    # not re-read cache-resident data -> measures the true DRAM-bound cost, not L2/SLC hits.
    physical_bytes_per_call = x.nbytes * 2 + gamma.nbytes + beta.nbytes
    pool = _pool_size(physical_bytes_per_call)
    in_pool = [cl.tensor(x) for _ in range(pool)]
    out_pool = [cl.tensor(zeros) for _ in range(pool)]
    gamma_pool = [cl.tensor(gamma) for _ in range(pool)]
    beta_pool = [cl.tensor(beta) for _ in range(pool)]
    # OV arg layout: [shape_info, input, output, gamma, beta]. shape_info = 16 int32,
    # unused by the body but present to match OV's dynamic signature.
    shape_info = [cl.tensor(np.zeros(16, np.int32))] if base_variant in (
        "ov", "bucket", "generalized", "gen_t16", "gen_hybrid", "gen_stack12"
    ) else []

    def _args(i):
        slot = i % pool
        return shape_info + [in_pool[slot], out_pool[slot], gamma_pool[slot], beta_pool[slot]]

    kernels.enqueue("mvn_gpu_bfyx_opt", [lws, rows], [lws, 1], *_args(0))
    cl.finish()
    cur = out_pool[0].numpy()
    ref = ref_mvn(x, eps, gamma, beta)
    ok = np.allclose(cur, ref, atol=2e-2, rtol=2e-2)
    if not ok:
        bad = np.abs(cur.astype(np.float32) - ref.astype(np.float32))
        print(f"  [{name}/{variant}] MAX ABS ERR = {bad.max():.4f}")

    if distribution == "cancellation":
        neg_ratio, rel_p50, rel_p99 = variance_cancellation_stats(x)
        print(f"  [{name}/{variant}] cancellation stats: neg_var={neg_ratio:.3%} rel_err p50={rel_p50:.3f} p99={rel_p99:.3f}")
        # Keep a strict stress guard on the original adversarial shape only.
        if rows == 1716 and D == 4608:
            assert rel_p99 > 0.20, "cancellation distribution is not stressing E[x^2]-E[x]^2 enough"

    # steady-state stats (warm up first; early launches run at a low GPU clock).
    for i in range(iters):
        kernels.enqueue("mvn_gpu_bfyx_opt", [lws, rows], [lws, 1], *_args(i))
    cl.finish()
    for i in range(iters):
        kernels.enqueue("mvn_gpu_bfyx_opt", [lws, rows], [lws, 1], *_args(i))
    mn, med, std = _stats_us(cl.finish())
    gbps = physical_bytes_per_call / (med * 1e-6) / 1e9  # physical DRAM BW estimate
    stack = (D + lws - 1) // lws
    mode = "reread" if (base_variant in ("generalized", "gen_t16", "gen_hybrid", "gen_stack12") and stack > MAX_REGISTER_STACK) else "cache"

    print(f"  {name:7s} {variant:11s} rows={rows:6d} D={D:4d} | LWS={lws:4d} items={items} | "
          f"{disp:28s} | acc={'OK ' if ok else 'FAIL'} | med={med:8.3f} min={mn:8.3f} std={std:6.3f} us | "
          f"{gbps:6.1f} GB/s (pool={pool}, stack={stack}, mode={mode}) (OV C6 {ov_ref_us:.0f} us)")
    return {"name": name, "variant": variant, "distribution": distribution, "ok": ok, "med_us": float(med)}


def main():
    cl.profiling(True)
    print("=== OV mvn_gpu_bfyx_opt.cl — Qwen3-Omni-4B C6 vision encoder ===")
    print("    ov = exact OV config (fused gamma+beta); static = same work, static shapes")
    print("    input/output/gamma/beta buffers rotated -> med/min/std us + physical DRAM GB/s")
    print("    welford_* variants use mvn_gpu_bfyx_opt_welford.cl with the same launch ABI")
    results = []
    records = []
    base_variants = ("ov", "bucket", "generalized", "gen_t16", "gen_hybrid", "gen_stack12", "static")
    all_variants = base_variants + tuple(f"welford_{v}" for v in base_variants)
    # ov_ref_us = per-call device time from C6 CLIntercept trace (mvn_gpu_profile.md).
    cases = [("small", 4096, 257, 0),
             ("vit", 6864, 1152, 833),
             ("wide", 1024, 8192, 0),
             ("merger", 1716, 4608, 910),
             ("huge", 256, 32768, 0)]
    for name, rows, D, ref in cases:
        for variant in all_variants:
            rec = run_case(name, rows, D, ref, variant)
            results.append(rec["ok"])
            records.append(rec)

    # Adversarial distribution for the single-pass variance formula:
    # large mean + tiny fp16 jitter -> catastrophic cancellation in E[x^2]-E[x]^2.
    for name, rows, D, ref in cases:
        for variant in all_variants:
            records.append(run_case(name, rows, D, ref, variant, distribution="cancellation"))

    print_pair_summary([r for r in records if r["distribution"] == "uniform"], base_variants, "uniform")
    print_pair_summary([r for r in records if r["distribution"] == "cancellation"], base_variants, "cancellation")
    assert all(results), "accuracy check failed"
    print("accuracy: all good")


if __name__ == "__main__":
    main()
