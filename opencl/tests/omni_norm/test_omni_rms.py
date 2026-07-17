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
#   ov     : OV master host ABI (dynamic, shape_info arg, runtime LWS)
#   bucket : cecilia/opt/mvn_rms host ABI with shape_info arg and fixed generalized LWS
#   qk_specialized : branch ABI + explicit one-subgroup-row specialization for D=128 q/k
#   hidden_specialized : branch ABI + explicit multi-subgroup-row specialization for D=2560 hidden
#   gen_t8 : branch ABI + fixed t8 generalized LWS rule
#   gen_adaptive : shape-aware policy (t8/t16 + adaptive stack cap + adaptive block size)
#   bucket_tuned : branch ABI + alternative subgroup block / stack tuning probe
#   static : branch static-shape specialization
#
# Run inside the llm container (built_ov env):  python test_omni_rms.py

import os
import re
import argparse
import numpy as np
import torch
from clops import cl
from clops.utils import Colors, kernel_cache

HERE = os.path.dirname(os.path.abspath(__file__))
SUB_GROUP_SIZE = 16
MAX_LWS = 1024  # maxSlmSize = min(maxWorkGroupSize=1024, maxLocalMemSize/(2*2B)) on PTL Xe
CACHE_FLUSH_BYTES = 256 << 20  # rotate through >=256 MiB of buffers to defeat L2/SLC reuse
MAX_REGISTER_STACK = 16
SHAPE_T16_MIN = 1024
SHAPE_T16_MAX = 6144
DEFAULT_LOG_FILE = os.path.join(HERE, "test_omni_rms.nosummary.log")


RECORD_RE = re.compile(
    r"^\s*(?P<name>\S+)\s+(?P<variant>\S+)\s+rows=.*\|\s+"
    r"acc=(?P<acc>OK|FAIL)\s+\|\s+med=\s*(?P<med>[0-9.]+).*\|\s+"
    r"(?P<gbps>[0-9.]+)\s+GB/s"
)


def _pool_size(bytes_per_set):
    """#distinct buffer sets so the rotated footprint exceeds cache (kills reuse hits)."""
    return max(3, min(64, -(-CACHE_FLUSH_BYTES // bytes_per_set)))


def _stats_us(lat_ns):
    """min / median / std (us) from per-call device times (ns)."""
    a = np.sort(np.asarray(lat_ns, dtype=np.float64)) / 1e3
    return a[0], a[len(a) // 2], a.std()


def _color(text, color):
    return f"{color}{text}{Colors.END}"


def make_logger(file_obj=None):
    def _log(msg):
        print(msg)
        if file_obj is not None:
            file_obj.write(msg + "\n")
    return _log


def _read(name):
    return open(os.path.join(HERE, name)).read()


def _body(cl_name):
    """Verbatim OV kernel with the OV #include lines (shim replaces them) stripped."""
    return "\n".join(l for l in _read(cl_name).splitlines()
                     if not l.strip().startswith('#include "include/'))


SHIM = _read("ov_norm_shim.cl")
RMS_BODY = _body("rms_gpu_bfyx_opt.cl")
RMS_TWOPASS_BODY = _body("rms_gpu_bfyx_opt_twopass.cl")

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
    """Port of OV master rms_kernel_bfyx_opt.cpp::get_item_num_and_lws()."""
    lws, items = 1, data_size
    while (items > 8 or lws < items) and (2 * lws <= max_lws):
        lws *= 2
        items //= 2
    return items, lws


def generalized_lws(data_size, max_lws=MAX_LWS, target_items=8):
    """Port of cecilia/opt/mvn_rms get_generalized_lws()."""
    lws = SUB_GROUP_SIZE
    limit = max(SUB_GROUP_SIZE, min(max_lws, data_size // target_items))
    while 2 * lws <= limit:
        lws *= 2
    return lws


def branch_stack(data_size, lws, stack_cap=MAX_REGISTER_STACK):
    required = (data_size + lws - 1) // lws
    return required, min(required, stack_cap), required > stack_cap


def branch_slm_size(max_lws=MAX_LWS):
    return max(1, max_lws // SUB_GROUP_SIZE)


def subgroup_block_size(items):
    if items >> 3:
        return 8
    if items >> 2:
        return 4
    if items >> 1:
        return 2
    return 1


def adaptive_subgroup_block_size(items, preferred=8):
    return min(preferred, subgroup_block_size(items))


def adaptive_rms_stack_cap(data_size, base_cap):
    if data_size < 512 or data_size >= 8192:
        return min(base_cap, 12)
    return base_cap


def shape_aware_rms_policy(data_size, base_variant):
    """Return (target_items, preferred_sbs, stack_cap, tag)."""
    if base_variant == "gen_adaptive":
        # Adaptive RMS policy summary:
        # 1) Choose t16 only for wide-like rows in [6144, 16384), where it is typically
        #    best or tie-best on this platform.
        # 2) Use t8 elsewhere as the robust default.
        # 3) Keep preferred subgroup block size at 8, then cap by the kernel-derived limit.
        # 4) Use a tighter stack cap on <=2048 rows and clamp tiny (<512) / huge (>=8192)
        #    rows via adaptive_rms_stack_cap() to reduce pressure-driven regressions.
        # RMS empirical policy on LNL/Xe2:
        # - t8 is better or tie-best for most small/mid/hidden shapes.
        # - t16 helps wide-like shapes (e.g. D=8192).
        if 6144 <= data_size < 16384:
            target_items = 16
            stack_cap = adaptive_rms_stack_cap(data_size, MAX_REGISTER_STACK)
            return target_items, 8, stack_cap, "gen adaptive t16-wide"

        target_items = 8
        # Keep a tighter cap on <=2K rows to reduce sensitivity to pressure.
        base_cap = 12 if data_size <= 2048 else MAX_REGISTER_STACK
        stack_cap = adaptive_rms_stack_cap(data_size, base_cap)
        return target_items, 8, stack_cap, "gen adaptive t8-main"
    if base_variant == "gen_t8":
        return 8, 8, adaptive_rms_stack_cap(data_size, MAX_REGISTER_STACK), "gen t8"
    if base_variant == "gen_t16":
        return 16, 8, MAX_REGISTER_STACK, "gen t16"
    if base_variant == "gen_hybrid":
        target_items = 16 if data_size >= 2048 else 8
        stack_cap = adaptive_rms_stack_cap(data_size, MAX_REGISTER_STACK)
        return target_items, 8, stack_cap, f"gen hybrid t{target_items}"
    if base_variant == "gen_block4":
        return 8, 4, adaptive_rms_stack_cap(data_size, MAX_REGISTER_STACK), "gen block4"
    if base_variant == "gen_block2":
        return 8, 2, adaptive_rms_stack_cap(data_size, MAX_REGISTER_STACK), "gen block2"
    if base_variant == "gen_stack12":
        return 8, 8, adaptive_rms_stack_cap(data_size, 12), "gen stack12"

    return 8, 8, adaptive_rms_stack_cap(data_size, MAX_REGISTER_STACK), "gen t8"


def effective_rms_stack_cap(data_size, base_variant):
    if base_variant in ("gen_t8", "gen_adaptive", "gen_t16", "gen_hybrid", "gen_block4", "gen_block2", "gen_stack12"):
        _, _, stack_cap, _ = shape_aware_rms_policy(data_size, base_variant)
        return stack_cap
    if base_variant in ("bucket", "qk_specialized", "hidden_specialized", "static"):
        return adaptive_rms_stack_cap(data_size, MAX_REGISTER_STACK)
    return MAX_REGISTER_STACK


def build(D, rank, eps, variant):
    use_twopass = variant.startswith("twopass_")
    base_variant = variant[len("twopass_"):] if use_twopass else variant
    kernel_body = RMS_TWOPASS_BODY if use_twopass else RMS_BODY

    items, master_lws = get_item_num_and_lws(D)
    base = (f"-DELEMENTWISE_AFFINE=1 -DSUB_GROUP_SIZE={SUB_GROUP_SIZE} "
            f"-DINPUT_RANK={rank} -DDATA_SIZE={D} -DEPSILON={eps}")
    if base_variant == "ov":
        # dynamic: SBS=8, STACK_SIZE=ceil_div(D,lws), SLM_SIZE=maxSlmSize; LWS runtime.
        lws = master_lws
        stack = (D + lws - 1) // lws
        opts = base + f" -DSLM_SIZE={MAX_LWS} -DSTACK_SIZE={stack} -DSUBGROUP_BLOCK_SIZE=8"
        src = SHIM + "\n" + RMS_DYN_DEFS + "\n" + kernel_body
        disp = f"ov master sbs=8 stack={stack} LWS=rt"
    elif base_variant == "bucket":
        lws = generalized_lws(D)
        stack, capped_stack, reread = branch_stack(D, lws)
        stack_value = stack if use_twopass else capped_stack
        row_flag = "-DONE_SUBGROUP_ROW=1" if lws == SUB_GROUP_SIZE else "-DMULTI_SUBGROUP_ROW=1"
        row_kind = "one-sg" if lws == SUB_GROUP_SIZE else "multi-sg"
        opts = (base + f" -DLWS={lws} -DSLM_SIZE={branch_slm_size()} -DSTACK_SIZE={stack_value} "
                f"-DSUBGROUP_BLOCK_SIZE=8 {row_flag}")
        if reread and not use_twopass:
            opts += " -DRMS_REREAD_INPUT=1"
        src = SHIM + "\n" + RMS_SHAPE_ARG_DEFS + "\n" + kernel_body
        mode = "reread" if reread else "cache"
        disp = f"bucket branch {row_kind} sbs=8 stack={stack} cap={MAX_REGISTER_STACK} {mode} LWS={lws}"
    elif base_variant == "qk_specialized":
        lws = generalized_lws(D)
        stack, capped_stack, reread = branch_stack(D, lws)
        stack_value = stack if use_twopass else capped_stack
        opts = (base + f" -DLWS={lws} -DSLM_SIZE={branch_slm_size()} -DSTACK_SIZE={stack_value} "
                f"-DSUBGROUP_BLOCK_SIZE=8 -DONE_SUBGROUP_ROW=1")
        if reread and not use_twopass:
            opts += " -DRMS_REREAD_INPUT=1"
        src = SHIM + "\n" + RMS_SHAPE_ARG_DEFS + "\n" + kernel_body
        mode = "reread" if reread else "cache"
        disp = f"qk one-sg branch stack={stack} cap={MAX_REGISTER_STACK} {mode} LWS={lws}"
    elif base_variant == "hidden_specialized":
        lws = generalized_lws(D)
        stack, capped_stack, reread = branch_stack(D, lws)
        stack_value = stack if use_twopass else capped_stack
        opts = (base + f" -DLWS={lws} -DSLM_SIZE={branch_slm_size()} -DSTACK_SIZE={stack_value} "
                f"-DSUBGROUP_BLOCK_SIZE=8 -DMULTI_SUBGROUP_ROW=1")
        if reread and not use_twopass:
            opts += " -DRMS_REREAD_INPUT=1"
        src = SHIM + "\n" + RMS_SHAPE_ARG_DEFS + "\n" + kernel_body
        mode = "reread" if reread else "cache"
        disp = f"hidden multi-sg branch stack={stack} cap={MAX_REGISTER_STACK} {mode} LWS={lws}"
    elif base_variant in ("gen_t8", "gen_adaptive", "gen_t16", "gen_hybrid", "gen_block4", "gen_block2", "gen_stack12"):
        target_items, preferred_sbs, stack_cap, tag = shape_aware_rms_policy(D, base_variant)

        lws = generalized_lws(D, target_items=target_items)
        items = D // lws
        sbs = adaptive_subgroup_block_size(items, preferred=preferred_sbs)
        stack, capped_stack, reread = branch_stack(D, lws, stack_cap)
        stack_value = stack if use_twopass else capped_stack
        row_kind = "one-sg" if lws == SUB_GROUP_SIZE else "multi-sg"
        row_flag = "-DONE_SUBGROUP_ROW=1" if lws == SUB_GROUP_SIZE else "-DMULTI_SUBGROUP_ROW=1"
        opts = (base + f" -DLWS={lws} -DSLM_SIZE={branch_slm_size()} -DSTACK_SIZE={stack_value} "
                f"-DSUBGROUP_BLOCK_SIZE={sbs} {row_flag}")
        if reread and not use_twopass:
            opts += " -DRMS_REREAD_INPUT=1"
        src = SHIM + "\n" + RMS_SHAPE_ARG_DEFS + "\n" + kernel_body
        mode = "reread" if reread else "cache"
        disp = f"{tag} {row_kind} sbs={sbs} stack={stack} cap={stack_cap} {mode} LWS={lws}"
    elif base_variant == "bucket_tuned":
        lws = generalized_lws(D)
        items = D // lws
        sbs = subgroup_block_size(items)
        stack = items + 1
        row_flag = "-DONE_SUBGROUP_ROW=1" if lws == SUB_GROUP_SIZE else "-DMULTI_SUBGROUP_ROW=1"
        opts = base + (f" -DLWS={lws} -DSLM_SIZE={branch_slm_size()} -DSTACK_SIZE={stack} "
                       f"-DSUBGROUP_BLOCK_SIZE={sbs} {row_flag}")
        src = SHIM + "\n" + RMS_SHAPE_ARG_DEFS + "\n" + kernel_body
        disp = f"bucket tuned sbs={sbs} stack={stack} LWS={lws}"
    else:  # static specialization (same work)
        lws = generalized_lws(D)
        stack, capped_stack, reread = branch_stack(D, lws)
        stack_value = stack if use_twopass else capped_stack
        row_flag = "-DONE_SUBGROUP_ROW=1" if lws == SUB_GROUP_SIZE else "-DMULTI_SUBGROUP_ROW=1"
        row_kind = "one-sg" if lws == SUB_GROUP_SIZE else "multi-sg"
        opts = base + (f" -DLWS={lws} -DSLM_SIZE={branch_slm_size()} -DSTACK_SIZE={stack_value} "
                       f"-DSUBGROUP_BLOCK_SIZE=8 {row_flag}")
        if reread and not use_twopass:
            opts += " -DRMS_REREAD_INPUT=1"
        src = SHIM + "\n" + kernel_body
        mode = "reread" if reread else "cache"
        disp = f"static branch {row_kind} sbs=8 stack={stack} cap={MAX_REGISTER_STACK} {mode} LWS={lws}"
    if use_twopass:
        disp = f"twopass {disp}"
    return src, opts, lws, items, disp


def ref_rms(x, g, eps):
    xf = torch.from_numpy(x).float()
    var = xf.pow(2).mean(-1, keepdim=True)
    return (xf * torch.rsqrt(var + eps) * torch.from_numpy(g).float()).half().numpy()


def parse_cached_records(log_file):
    records = []
    with open(log_file, "r") as f:
        for line in f:
            m = RECORD_RE.match(line.rstrip("\n"))
            if m is None:
                continue
            records.append({
                "name": m.group("name"),
                "variant": m.group("variant"),
                "ok": m.group("acc") == "OK",
                "med_us": float(m.group("med")),
                "gbps": float(m.group("gbps")),
            })
    return records


def print_kernel_summary(records, base_variants):
    print("\n=== Matrix Summary (shape x ABI x kernel_impl) ===")
    print("    shape    ABI                  kernel_impl   med(us)   GB/s   acc")
    impls = (
        ("optimized", ""),
        ("twopass", "twopass_"),
    )
    grouped = sorted({r["name"] for r in records})
    for shape in grouped:
        for base in base_variants:
            rows = {}
            for impl_name, prefix in impls:
                variant = f"{prefix}{base}" if prefix else base
                rows[impl_name] = next(
                    (r for r in records if r["name"] == shape and r["variant"] == variant),
                    None,
                )
            if any(v is None for v in rows.values()):
                continue

            for impl_name, _ in impls:
                row = rows[impl_name]
                acc_col = Colors.GREEN if row["ok"] else Colors.YELLOW
                acc_str = _color("PASS" if row["ok"] else "FAIL", acc_col)
                print(
                    f"    {shape:7s} {base:20s} {impl_name:11s} "
                    f"{row['med_us']:8.3f} {row['gbps']:6.1f} {acc_str:>6s}"
                )


def print_requested_combo_view(records, output_format="table"):
    combos = [
        ("ov + twopass", "twopass_ov"),
        ("gen_t8 + optimized", "gen_t8"),
        ("gen_adaptive + optimized", "gen_adaptive"),
        ("gen_t16 + optimized", "gen_t16"),
        ("gen_hybrid + optimized", "gen_hybrid"),
        ("gen_stack12 + optimized", "gen_stack12"),
        ("bucket_tuned + optimized", "bucket_tuned"),
    ]

    grouped = sorted({r["name"] for r in records})

    if output_format in ("csv", "tsv"):
        sep = "," if output_format == "csv" else "\t"
        headers = ["shape"] + [name for name, _ in combos]
        print(sep.join(headers))
        for shape in grouped:
            fields = [shape]
            for _, variant in combos:
                row = next((r for r in records if r["name"] == shape and r["variant"] == variant), None)
                fields.append("" if row is None else f"{row['gbps']:.1f}")
            print(sep.join(fields))
        return

    print("\n=== Focused Combo Matrix (GB/s Only) ===")

    shape_w = 7
    col_w = 26
    row_header = f"{'shape':<{shape_w}s}"
    col_headers = " ".join(f"{name:>{col_w}s}" for name, _ in combos)
    print(f"    {row_header} {col_headers}")
    print(f"    {'-' * shape_w} {'-' * ((col_w + 1) * len(combos) - 1)}")

    for shape in grouped:
        raw_values = []
        for _, variant in combos:
            row = next((r for r in records if r["name"] == shape and r["variant"] == variant), None)
            raw_values.append(row["gbps"] if row is not None else None)

        present = [v for v in raw_values if v is not None]
        best = max(present) if present else None
        worst = min(present) if present else None

        values = []
        for v in raw_values:
            if v is None:
                values.append(f"{'-':>{col_w}s}")
                continue

            cell = f"{v:>{col_w}.1f}"
            if best is not None and worst is not None and best != worst:
                if v == best:
                    cell = _color(cell, Colors.GREEN)
                elif v == worst:
                    cell = _color(cell, Colors.RED)
            values.append(cell)

        print(f"    {shape:<{shape_w}s} {' '.join(values)}")


def print_gen_adaptive_policy_summary(cases):
    print("\n=== gen_adaptive Policy Summary ===")
    print("    shape     D      target_items  preferred_sbs  stack_cap  tag")
    seen = set()
    for name, _, D, _, _ in cases:
        # q_norm and k_norm share D=128; print each shape once.
        if (name, D) in seen:
            continue
        seen.add((name, D))
        target_items, preferred_sbs, stack_cap, tag = shape_aware_rms_policy(D, "gen_adaptive")
        print(f"    {name:7s} {D:6d} {target_items:13d} {preferred_sbs:14d} {stack_cap:10d}  {tag}")


def run_case(name, rows, D, rank, ov_ref_us, variant, eps=1e-6, iters=100, logger=print):
    src, opts, lws, items, disp = build(D, rank, eps, variant)
    kernels = kernel_cache(src, opts)
    if variant.startswith("twopass_"):
        base_variant = variant[len("twopass_"):]
    else:
        base_variant = variant

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
    shape_info = [cl.tensor(np.zeros(16, np.int32))] if base_variant in (
        "ov", "bucket", "qk_specialized", "hidden_specialized", "gen_t8", "gen_adaptive",
        "gen_t16", "gen_hybrid", "gen_block4", "gen_block2", "gen_stack12", "bucket_tuned") else []

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
        logger(f"  [{name}/{variant}] MAX ABS ERR = {bad.max():.4f}")

    # steady-state stats (warm up first; early launches run at a low GPU clock).
    for i in range(iters):
        kernels.enqueue("rms_gpu_bfyx_opt", [lws, rows], [lws, 1], *_args(i))
    cl.finish()
    for i in range(iters):
        kernels.enqueue("rms_gpu_bfyx_opt", [lws, rows], [lws, 1], *_args(i))
    mn, med, std = _stats_us(cl.finish())
    gbps = bytes_per_set / (med * 1e-6) / 1e9  # effective physical DRAM bandwidth at median
    stack = (D + lws - 1) // lws
    mode = "reread" if (base_variant in (
        "bucket", "qk_specialized", "hidden_specialized", "gen_t8", "gen_adaptive",
        "gen_t16", "gen_hybrid", "gen_block4", "gen_block2", "gen_stack12", "static"
    ) and stack > effective_rms_stack_cap(D, base_variant)) else "cache"

    logger(f"  {name:7s} {variant:11s} rows={rows:6d} D={D:4d} | LWS={lws:4d} items={items} | "
           f"{disp:40s} | acc={'OK ' if ok else 'FAIL'} | med={med:8.3f} min={mn:8.3f} "
           f"std={std:6.3f} us | {gbps:6.1f} GB/s (pool={pool}, stack={stack}, mode={mode}) "
           f"(OV C6 {ov_ref_us:.0f} us)")
    return {
        "name": name,
        "variant": variant,
        "ok": ok,
        "med_us": float(med),
        "gbps": float(gbps),
    }


def main():
    parser = argparse.ArgumentParser(description="RMS kernel benchmark harness")
    parser.add_argument("-f", "--force-rerun", action="store_true", help="Ignore cached log and rerun all benchmark cases")
    parser.add_argument("-l", "--log-file", default=DEFAULT_LOG_FILE,
                        help="Path to non-summary benchmark log cache (default: %(default)s)")
    parser.add_argument("-t", "--combo-format", choices=("table", "csv", "tsv"), default="table",
                        help="Focused combo matrix output format (default: %(default)s)")
    args = parser.parse_args()

    cl.profiling(True)
    base_variants = (
        "ov", "bucket", "qk_specialized", "hidden_specialized", "gen_t8", "gen_adaptive",
        "gen_t16", "gen_hybrid", "gen_block4", "gen_block2", "gen_stack12", "bucket_tuned", "static"
    )
    all_variants = base_variants + tuple(f"twopass_{v}" for v in base_variants)
    # ov_ref_us = per-call device time from C6 CLIntercept trace (rms_gpu_profile.md).
    cases = [("tail16", 4096, 272, 3, 0),
             ("hidden", 2556, 2560, 3, 722),
             ("q_norm", 2556 * 32, 128, 4, 1267),
             ("k_norm", 2556 * 8, 128, 4, 319),
             ("mid", 2048, 1152, 3, 0),
             ("wide", 1024, 8192, 3, 0),
             ("huge", 256, 32768, 3, 0)]

    if os.path.exists(args.log_file) and not args.force_rerun:
        print(f"Using cached non-summary log: {args.log_file}")
        records = parse_cached_records(args.log_file)
    else:
        print(f"Writing non-summary log to: {args.log_file}")
        records = []
        with open(args.log_file, "w") as log_f:
            logger = make_logger(log_f)
            logger("=== OV rms_gpu_bfyx_opt.cl — Qwen3-Omni-4B C6 Thinker prefill (T=2556) ===")
            logger("    ov = exact OV config; static = same work, static-shape specialization")
            logger("    buffers rotated (cold cache) -> med/min/std us + effective DRAM GB/s")
            logger("    twopass_* variants use rms_gpu_bfyx_opt_twopass.cl with the same launch ABI")

            for name, rows, D, rank, ref in cases:
                for variant in all_variants:
                    if variant.endswith("qk_specialized") and D != 128:
                        continue
                    if variant.endswith("hidden_specialized") and D != 2560:
                        continue
                    records.append(run_case(name, rows, D, rank, ref, variant, logger=logger))

    print_kernel_summary(records, base_variants)
    print_gen_adaptive_policy_summary(cases)
    print_requested_combo_view(records, output_format=args.combo_format)
    assert all(r["ok"] for r in records), "accuracy check failed"
    print("accuracy: all good")


if __name__ == "__main__":
    main()
