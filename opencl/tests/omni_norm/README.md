# Omni RMS / MVN norm kernel unit tests

Stand-alone micro-benchmarks for the two OpenVINO `intel_gpu` normalization kernels
that dominate Qwen3-Omni-4B norm time, so the kernels can be edited and re-measured
**outside** the full OV pipeline while iterating on optimizations.

- **RMSNorm** — Thinker LLM (`rms_gpu_bfyx_opt.cl`)
- **MVN / LayerNorm** — Vision encoder (`mvn_gpu_bfyx_opt.cl`)

Profiling context and optimization plans:
- `.../model_profile/qwen3-omni-4B/rms_gpu_profile.md`
- `.../model_profile/qwen3-omni-4B/mvn_gpu_profile.md`
- `.../model_profile/qwen3-omni-4B/mvn_analysis.md`

## No-hypothesis rule (READ FIRST)

**These tests reproduce OV's *observed* behavior, not a guess of it.** Every kernel
argument, JIT constant, dispatch size, dynamic flag, padding mode, and fusion below is
taken from a concrete source and cited:

- **verbose log** (`C6.verbose.log` / `logs/trace.1260x700x2.833_cn.mt1.verbose.log`,
  produced by `run_benchmark.sh 6 verbose`): `set_kernel_arg` (exact arg count & sizes),
  `Enqueue kernel ... gws/lws`, tensor descriptors (`:nopad`), `is_dynamic=`.
- **exec graph** (`exec_graphs/*.xml`): `primitiveType`, `originalLayersNames` (fusion).
- **kernel selectors** (`rms_kernel_bfyx_opt.cpp`, `mvn_kernel_bfyx_opt.cpp`): the JIT /
  dispatch derivation.

If a detail is not verified from one of these, it is **not** encoded as OV behavior —
it is called out explicitly (see the one such case: MVN's auto-generated `FUSED_OPS`
macro body). Do not add "it's probably…" reasoning to these tests or this doc.

## Files

| File | Purpose |
|---|---|
| `rms_gpu_bfyx_opt.cl` | **verbatim** copy of the OV kernel (`kernel_selector/cl_kernels/`) — do not edit except to optimize; re-`cp` to refresh |
| `mvn_gpu_bfyx_opt.cl` | **verbatim** copy of the OV kernel |
| `ov_norm_shim.cl` | minimal macro shim replacing the OV `#include`s (see below) |
| `test_omni_rms.py` | RMS cases `hidden`/`q_norm`/`k_norm`; variants `ov` (exact) + `static` |
| `test_omni_mvn.py` | MVN cases `vit`/`merger`; variants `ov` (exact) + `static` |
| `PROGRESS.md` | progress log + issues/causes found while building the tests |

The tests prepend `ov_norm_shim.cl` and strip the kernel's `#include "include/..."`
lines. The shim only covers the **FP16-in / FP16-out / FP32-accumulator** config (the
Omni case): `KERNEL`, datatype macros, `DT_INPUT_BLOCK_READ{,2,4,8}` /
`DT_OUTPUT_BLOCK_WRITE*` (via `as_halfN(intel_sub_group_block_read_usN(...))`),
`REQD_SUB_GROUP_SIZE`, `unroll_for`, and the MVN `ACTIVATION*` hooks. Everything else in
the kernels is unmodified OV source.

### Exact OV config reproduced by the `ov` variant

| | RMS (`rms:Multiply_*`) | MVN (`mvn:MVN_*`) |
|---|---|---|
| kernel args (verbose `set_kernel_arg`) | `shape_info, input, gamma, output` (4) | `shape_info, input, output, gamma, beta` (5) |
| fusion (exec graph) | none beyond gamma (`ELEMENTWISE_AFFINE`) | `MVN, Multiply(gamma), Add(beta)` |
| `is_dynamic` | 1 | 1 |
| padding (`:nopad`) | `HAS_PADDING=0` | `HAS_PADDING=0` |
| normalized dim | static → `DATA_SIZE` const | static → `DATA_SET_SIZE` const |
| `LWS` | `get_local_size(0)` (runtime) | `get_local_size(0)` (runtime) |
| RMS-only | `SUBGROUP_BLOCK_SIZE=8`, `STACK_SIZE=ceil_div(D,lws)`, `SLM_SIZE=maxSlmSize` | — |

`shape_info` is a 16×int32 (64 B) buffer passed as arg0 to match OV's dynamic
signature; with `HAS_PADDING=0` the body does not read it. The `static` variant does the
**same work** with compile-constant `LWS` (shape specialization) — an optimization
target, not an OV reproduction.

> The one detail NOT byte-reproduced: OV auto-generates the MVN `FUSED_OPS` macro from
> the fusion. The tests hand-write an equivalent (`out = norm*gamma + beta`) that is
> mathematically and I/O-identical (verified by the 5-arg layout + exec graph), but the
> generated macro's exact codegen is not dumped. This is the only non-verified item and
> is flagged in `test_omni_mvn.py`.

## Run (inside the `llm` container, `built_ov` env)

```bash
conda activate built_ov
source /ceciliapeng/VM/openvino/build_Release/install/setupvars.sh
cd /ceciliapeng/VM/aboutSHW/opencl/tests/omni_norm
python test_omni_rms.py
python test_omni_mvn.py
```

Each case prints accuracy (`np.allclose` vs a torch reference) and `us/call`,
next to the per-call average measured on the C6 profiling machine.

## JIT constants & GWS/LWS — how they are derived

The tests reproduce OV's host-side dispatch so the compiled kernel is byte-for-byte
equivalent to what the plugin builds. Both kernels share the same LWS heuristic
(`get_item_num_and_lws` in RMS, the identical loop in MVN's `SetDefault`):

```
local_mem_per_wi = 2 * sizeof(fp16) = 4 B
max_lws = min(maxWorkGroupSize=1024, maxLocalMemSize/4)   # = 1024 on PTL Xe
lws = 1; items = D
while ((items > 8 || lws < items) && 2*lws <= max_lws): lws *= 2; items //= 2
```

**GWS = `[LWS, rows]`, LWS = `[LWS, 1]`** — one work-group normalizes one row
(`get_global_id(1)` = row index, `get_global_id(0)` = worker within the row).
`rows` is the number of independent normalizations = product of all dims except the
normalized (last) dim.

### RMS (`rms_gpu_bfyx_opt.cl`), C6 prefill T=2556

Extra RMS consts: `STACK_SIZE = items+1`, `SUBGROUP_BLOCK_SIZE` = 8/4/2/1 by the
high bit of `items`, `SUB_GROUP_SIZE=16`, `SLM_SIZE=LWS`, `DATA_SIZE=D`, plus
`INPUT_RANK` (3 for hidden, 4 for per-head q/k).

| case | role | shape | D | rows | LWS | items | SUBGROUP_BLOCK_SIZE | STACK_SIZE |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `hidden` | input+post_attn LN | 2556×2560 | 2560 | 2556 | 512 | 5 | 4 | 6 |
| `q_norm` | GQA query norm (32 h) | 2556×32×128 | 128 | 81792 | 16 | 8 | 8 | 9 |
| `k_norm` | GQA key norm (8 h) | 2556×8×128 | 128 | 20448 | 16 | 8 | 8 | 9 |

`q_norm`/`k_norm` share an identical kernel body — they differ only in row count
(GWS). In OV they still hash differently because the base tensor-jit bakes
`INPUT0_FEATURE_NUM` (32 vs 8) into the source.

### MVN (`mvn_gpu_bfyx_opt.cl`), C6 vision encoder

Mode = `ACROSS_CHANNELS`, `NORMALIZE_VARIANCE=1`, `EPS_INSIDE_SQRT`, `IS_DYNAMIC=0`.
Consts: `LWS`, `DATA_SET_SIZE=D`, `DATA_SETS_COUNT` (unused by the body; rows via GWS).

| case | role | shape | D | rows | LWS | items |
|---|---|---|---:|---:|---:|---:|
| `vit` | ViT block LN | 6864×1152 | 1152 | 6864 | 256 | 4 |
| `merger` | patch-Merger LN | 1716×4608 | 4608 | 1716 | 1024 | 4 |

## Measured vs OV (same GPU, same container)

`clops` reports pure device time (`clGetEventProfilingInfo COMMAND_START/END`) — the
same basis as the C6 CLIntercept trace, so `ov`-variant numbers are directly comparable
to OV's. Tests **warm up then report the median** (the first tens of launches run at a
low GPU clock and are ~2x slower — a measurement artifact, not kernel behavior).

Steady-state median us (this run, 32-EU box):

| case | `ov` (exact) | OV C6 | Δ | `static` (same work) |
|---|---:|---:|---:|---:|
| rms hidden | 706 | 722 | −2% | 488 |
| rms q_norm | 1292 | 1267 | +2% | 1326 |
| rms k_norm | 320 | 319 | 0% | 322 |
| mvn vit | 685 | 833 | −18% | 649 |
| mvn merger | 838 | 910 | −8% | 799 |

- **RMS `ov` matches OV within ~2%** on all three cases (arg layout, dynamic flags,
  padding, dispatch all verified equal). Nothing is fused beyond gamma.
- **MVN `ov` matches merger (−8%); vit is −18%.** The only difference from OV in this
  variant is the hand-written `FUSED_OPS` macro body vs OV's auto-generated one (see the
  no-hypothesis note above) — the residual delta is attributed to that and **not**
  explained further here (no hypothesis).
- **`static`** does the same work with a compile-constant `LWS`; it is *not* an OV
  reproduction. For RMS `hidden` it is markedly faster (488 vs 706) — a data point that
  static-shape specialization would speed these kernels up. Use `static` as the
  optimization target and compare the same variant before/after a kernel change.

## Model structure: why the log shows far more RMS than `Model9_exec_graph.xml`

Verified from the exec graphs + verbose log:

- **`Model9` is the Talker, not the Thinker.** It has 891 layers, **65 RMS (all
  D=1280)**, 32 `scaled_dot_product_attention`, 131 `FullyConnected` — a 32-layer
  D=1280 transformer. The RMS that dominate the log are the **Thinker** (D=2560 hidden
  norms + D=128 GQA q/k norms), a different model. No dumped exec graph contains the
  Thinker's D=2560 RMS — its graph was not among the dumped files.
- **Exec graph = static nodes; log = executions.** Each RMS node runs once per forward
  pass; with prefill + decode step(s) × iterations the log shows many executions
  (`grep -c "execute rms_gpu_bfyx_opt"` ⇒ 580) of the per-pass node set.
- Thinker RMS per pass = **145** = 36 layers × 4 (`input_layernorm`, `q_norm`, `k_norm`,
  `post_attention_layernorm`) + 1 final. Kernel-hash counts in the log per pass:
  71 (D=2560 hidden) + 35 (q) + 35 (k) + 4 singletons = 145.

So counting Model9's 65 RMS against the log's Thinker RMS is comparing two different
models; the number that matches the log is the Thinker's 145/pass × #passes.

## Notes for optimization work

- Same GPU/container as C6; the `ov` variant reproduces OV's absolute us (RMS ~2%).
  When iterating on a kernel change, compare the **same variant** before/after.
- Cost structure (from `rms_gpu_profile.md` / `mvn_gpu_profile.md`): `q_norm ≫ k_norm`;
  ViT LN dominates MVN. RMS gap-to-peak is the **SLM barrier chain**
  (`rms_gpu_profile.md §3`); MVN's is **3-pass traffic** (`mvn_analysis.md §7.1`,
  Welford). Prototype in the verbatim `.cl` here, keep accuracy green, port back to OV.
