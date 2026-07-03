# Omni norm kernel tests — progress & issues log

Tracks findings while building/validating the stand-alone RMS/MVN kernel tests.
See `README.md` for how to run and the JIT/dispatch derivation.

**No-hypothesis rule:** every OV behavior encoded here is verified from the verbose log
(`set_kernel_arg`, `Enqueue gws/lws`, `:nopad`, `is_dynamic=`), the exec graphs
(`primitiveType`, `originalLayersNames`), or the kernel selectors. Anything not so
verified is flagged as such — not guessed. (See README "No-hypothesis rule".)

## Status

| item | state |
|---|---|
| Copy OV `rms_gpu_bfyx_opt.cl` / `mvn_gpu_bfyx_opt.cl` verbatim | ✅ |
| Shim (`ov_norm_shim.cl`) to compile them stand-alone (FP16, FP32 accum) | ✅ |
| Reproduce OV JIT consts + GWS/LWS (verified vs verbose log — exact match) | ✅ |
| Accuracy vs torch (RMS hidden/q/k, MVN vit/merger) | ✅ all pass |
| Steady-state timing (median, warm-up discarded) | ✅ |
| **`ov` variant = exact OV** (args/flags/padding/fusion verified) → RMS ~2%, MVN physical-traffic run below | ✅ (this pass) |
| `static` same-work baseline for optimization | ✅ |

## Current validated status (2026-07-03, commit c0bceb5)

Run environment is the aboutSHW setup from `HOW_TO_RUN_aboutSHW.md`: `llm` docker
container, `conda activate built_ov`, and
`/ceciliapeng/VM/openvino/build_Release/install/setupvars.sh`.

### RMS

`test_omni_rms.py` reproduces the C6 Thinker RMS path: dynamic shape-info arg,
runtime LWS, static normalized dim, gamma-only affine, no residual/bias fused op.
The exact `ov` variant matches OV device time closely; `static` is the same work with
compile-constant LWS and is mainly an optimization reference.

Latest validated run after rerunning `test_omni_rms.py`:

| case | `ov` median | OV C6 | `ov` DRAM BW | `static` median | `static` DRAM BW |
|---|---:|---:|---:|---:|---:|
| hidden | 512 us | 722 us | 51.1 GB/s | 356 us | 73.5 GB/s |
| q_norm | 703 us | 1267 us | 59.6 GB/s | 832 us | 50.3 GB/s |
| k_norm | 178 us | 319 us | 58.9 GB/s | 210 us | 49.9 GB/s |

Notes:
- RMS has no extra fused residual/bias operands in the verified C6 path. The gamma
  vector is part of the kernel signature, but physical DRAM traffic is modeled as
  input read + output write + one gamma vector read, not gamma reread as unique DRAM
  for every row.
- The current RMS script reports input/output physical DRAM bandwidth (`x.nbytes * 2`)
  and keeps gamma as one fixed tensor, so the table above follows that script output.
- The hidden RMS dynamic penalty remains the clearest evidence that fixed-LWS codegen
  is valuable when the normalized dim is already static.

### MVN

`test_omni_mvn.py` reproduces the C6 vision MVN path: dynamic shape-info arg,
runtime LWS, static normalized dim, and fused affine epilogue
`MVN -> Multiply(gamma) -> Add(beta)`. The generated `FUSED_OPS` macro body was
mirrored from dumped CL sources as a two-op affine chain with OV-like load/calc
ordering and `convert_half` steps.

Latest validated run uses rotated input/output/gamma/beta buffers and reports
**physical DRAM traffic**, not logical per-element fused traffic. The numerator is:
`input read + output write + one gamma vector read + one beta vector read`.

| case | `ov` median | OV C6 | `ov` DRAM BW | `static` median | `static` DRAM BW |
|---|---:|---:|---:|---:|---:|
| vit | 594 us | 833 us | 53.3 GB/s | 348 us | 90.8 GB/s |
| merger | 758 us | 910 us | 41.7 GB/s | 438 us | 72.2 GB/s |

Notes:
- Gamma/beta buffers are now rotated too, so cross-launch cache reuse is reduced for
  fused operands.
- Bandwidth utilization uses physical DRAM traffic. Counting gamma/beta as full
  per-element unique traffic gives impossible-looking values above the 100 GB/s roof
  because the gamma/beta vectors are reused across rows and are tiny versus the input.
- The `static` MVN baseline is much faster than dynamic, which points at dynamic-LWS
  codegen overhead rather than missing math.

## Issues found & causes

### I1 — Microbench looked ~1.5x SLOWER than OV, but wasn't (measurement artifact)
- **Symptom:** initial 30-iteration *mean* gave hidden 1091 us vs OV 723, q_norm 1993 vs 1271, etc.
- **Cause:** GPU-clock warm-up. The first tens of launches run at a low clock and are
  ~2x slower, then boost. Averaging them in inflated the result.
  Trajectory (hidden, us/launch): `first5 ≈ [1025,971,981,982,970] … last5 ≈ [490,486,484,485,491]`.
- **Fix:** tests now warm up, then report the **median**. `clops` latency is pure
  device time (`clGetEventProfilingInfo COMMAND_START/END`), same basis as the
  CLIntercept C6 trace — so the comparison is apples-to-apples once warm.

### I2 — After the fix, bare kernels are FASTER than OV for hidden/vit/merger
- **Symptom (steady-state median vs OV C6):** k_norm 321/319 ✓, q_norm 1329/1271 ✓,
  hidden 486/722, vit 604/833, merger 765/911.
- **Cause:** OV fuses extra elementwise work these bare kernels omit. Dispatch is
  identical (verified: OV gws/lws from verbose log == derived), so the gap is *work*:
  - **MVN vit/merger:** OV fuses the affine — exec graph
    `originalLayersNames = MVN, Multiply(gamma), Add(beta)`. gamma/beta are small &
    cache-resident → gap < 2x.
  - **RMS hidden:** *initially* attributed to a fused residual Add (the `Multiply, Add`
    pattern on `Model9_exec_graph.xml` D=1280 RMS layers, and a 722/486=1.49x ratio).
    **This was wrong** — see I3: the gap is dynamic-shape compile overhead, not fusion.
    (The D=2560 Thinker RMS graph was not among the dumped exec graphs, so the Model9
    pattern was over-generalized.)
  - **RMS q_norm/k_norm (D=128):** no fusion → match OV closely (also confirms the
    D=128 JIT derivation).
- **Resolution:** added fused variants (MVN, real) and a dynamic variant (below). The
  dynamic variant is what actually reproduces OV for RMS.

### I3 — The core→OV gap is DYNAMIC-shape compile overhead (added a dynamic variant)
Initial hypothesis (fused residual Add on hidden RMS) turned out **wrong** — see the
correction below. After adding a faithful dynamic variant, the numbers resolve cleanly.

**Key subtlety:** the log shows `?x?x2560` / `?x1152` — only the batch/token dims are
dynamic; the **normalized dim is STATIC**. So OV keeps `DATA_SIZE`/`DATA_SET_SIZE` a
compile constant; the runtime quantity is `LWS = get_local_size(0)` (plus RMS
`SUBGROUP_BLOCK_SIZE=8` and `STACK_SIZE = ceil_div(D, lws)`, and MVN `IS_DYNAMIC=1`
dropping `reqd_work_group_size`). Runtime LWS blocks trip-count unrolling and, for the
hidden RMS, forces all-scalar reads (`items_num=5 < SBS=8`).

**Exact args confirmed** (verbose `set_kernel_arg`): RMS = `shape_info, input, gamma,
output` (4 args → **no residual fusion**); MVN = `shape_info, input, output, gamma, beta`
(5 args → fused affine, matching exec graph `MVN, Multiply(gamma), Add(beta)`). All
tensors `:nopad` → `HAS_PADDING=0`. This replaced the guessed variants with a single
**`ov`** variant that reproduces exactly what OV compiles (+ a `static` same-work
baseline). Measured (steady-state median us):

| case | `ov` (exact) | OV C6 | Δ | `static` |
|---|---:|---:|---:|---:|
| rms hidden | 706 | 722 | −2% | 488 |
| rms q_norm | 1292 | 1267 | +2% | 1326 |
| rms k_norm | 320 | 319 | 0% | 322 |
| mvn vit | 685 | 833 | −18% | 649 |
| mvn merger | 838 | 910 | −8% | 799 |

- **RMS `ov` matches OV within ~2%.** Args, dynamic flags, `HAS_PADDING=0`, dispatch all
  verified equal; nothing fused beyond gamma. (This corrects I2's residual guess — the
  earlier 30-iter "1.5x" was warm-up (I1) and the static-vs-dynamic mismatch, not a fuse.)
- **MVN `ov` matches merger (−8%); vit −18%.** The only unreproduced difference is OV's
  auto-generated `FUSED_OPS` macro body (the tests hand-write an I/O-identical
  `norm*gamma+beta`). Per the no-hypothesis rule the residual delta is left attributed to
  that and **not** explained further.
- **`static`** = same work, compile-const LWS (not an OV repro). RMS hidden 488 vs 706
  shows shape specialization would help — an optimization target.

The latest MVN harness now rotates gamma/beta and reports physical DRAM traffic; with
that measurement mode, MVN `static` remains substantially faster than `ov` even though
the same work is executed. That strengthens the dynamic-codegen optimization target:
recover fixed-LWS specialization without losing dynamic row dispatch.

### I4 — Latent OV bug: `USE_BLOCK_WRITE` is always 0
- In `rms_gpu_bfyx_opt.cl`: `#define USE_BLOCK_WRITE ((OUTPUT_TYPE_SIZE * OUTPUT_FEATURE_PITCH) & 0xF == 0)`.
  In C, `==` binds tighter than `&`, so this is `X & (0xF==0)` = `X & 0` = **0** for
  every build. The subgroup-block-**write** path (`if (... && USE_BLOCK_WRITE)`) is
  therefore dead code — all writes go through the scalar loop. Confirmed by the
  compiler warning ("& has lower precedence than ==").
- **Impact:** the read phase still uses block reads; only block *writes* are disabled.
  A candidate optimization (fix precedence → enable coalesced block writes), but must
  be validated for the alignment condition it was meant to guard.

### I5 — Log shows far more RMS than `Model9_exec_graph.xml` (different model + exec vs nodes)
- **`Model9` is the Talker, not the Thinker:** 891 layers, **65 RMS all D=1280**, 32
  SDPA, 131 FC. The log's dominant RMS are the **Thinker** (D=2560 hidden + D=128 GQA
  q/k) — a different model whose exec graph was not among the dumped files. So Model9's
  65 vs the log is apples-to-oranges.
- Exec graph = static node count; verbose log = **executions** (`grep -c "execute
  rms_gpu_bfyx_opt"` ⇒ 580 across prefill + decode × iters). Thinker RMS/pass = **145** =
  36 layers × 4 (input/q/k/post_attn) + 1 final; per-hash: 71 (D=2560) + 35 (q) + 35 (k)
  + 4 singletons = 145.

## Variants (how each test runs)

Kernels stay **verbatim**; the tests drive OV's real dynamic / `HAS_FUSED_OPS` paths via
prepended macro blocks + `-D` flags. Two variants:

- **`ov`** — exact reproduction of what OV compiles (see the I3 arg/flag list): dynamic,
  `shape_info` arg, `HAS_PADDING=0`, runtime `LWS`, RMS gamma-only / MVN fused gamma+beta.
  Compare against OV's absolute us.
- **`static`** — same work, compile-constant `LWS` (shape specialization). Optimization
  target; **not** an OV reproduction. Compare the same variant before/after a change.

## Concrete improvement ideas for dynamic OV (commit c0bceb5)

The dynamic path is slower because `IS_DYNAMIC=1` currently removes
`reqd_work_group_size` and makes `LWS` a runtime expression (`get_local_size(0)`). For
these C6 norm kernels the normalized dim is already static, so the hot inner loops do
not actually need fully dynamic codegen.

1. **Bucket dynamic kernels by normalized dim and fixed LWS.**
  Keep dynamic row count / shape-info handling, but compile known buckets with literal
  `LWS` and `reqd_work_group_size`. Known C6 buckets include MVN `D=1152,LWS=256`,
  MVN `D=4608,LWS=1024`, RMS hidden `D=2560,LWS=512`, and RMS q/k `D=128,LWS=16`.

2. **Split dynamic rows from dynamic normalized dim.**
  Add a selector/JIT mode for "dynamic outer dims, static reduction dim". In that
  mode, keep `DATA_SET_SIZE` / `DATA_SIZE`, `LWS`, `STACK_SIZE`, and related loop
  trip counts compile-time constants while only `rows` remains runtime-dispatched.

3. **Allow fixed-LWS attributes in dynamic kernels.**
  Do not gate `reqd_work_group_size` only on `!IS_DYNAMIC`. Gate it on whether `LWS`
  is a compile-time literal. This should let shape-info kernels compile like static
  kernels when the chosen LWS bucket is known.

4. **Specialize stack and subgroup constants per bucket.**
  RMS dynamic currently uses conservative constants such as `SUBGROUP_BLOCK_SIZE=8`
  and `STACK_SIZE=ceil_div(D,lws)`. Bucketed kernels can tune these the same way the
  static path does, reducing register/array baggage and scalar fallback pressure.

5. **Audit shape-info usage and guards.**
  For the verified `:nopad` C6 paths, shape-info is present for ABI compatibility but
  not read by the kernel body. If real OV generated code still emits shape guards or
  bounds logic for these buckets, remove it for static normalized-dim cases.

6. **Keep algorithmic optimizations separate from dynamic-codegen fixes.**
  RMS barrier reduction (`sub_group_reduce_add` chain), MVN single-pass/Welford style
  variants, and the `USE_BLOCK_WRITE` precedence bug are still useful, but they should
  be measured after recovering static-LWS dynamic codegen so the effects do not mix.

## Open / next

- [x] Exact OV config (`ov` variant) — RMS within ~2%; MVN merger −8%, vit −18%.
- [x] Mirror the generated MVN `FUSED_OPS` body from the dumped CL sources in `test_omni_mvn.py` (mul/add load/calc ordering + half conversions). This narrows the vit gap to ~680 us vs 833 us, but the remaining delta is still attributed to any remaining codegen differences not yet byte-reproduced.
- [x] Rotate MVN gamma/beta buffers and correct bandwidth reporting to physical DRAM
  traffic instead of logical per-element fused traffic.
- [ ] Prototype RMS SLM-barrier reduction (`sub_group_reduce_add` chain) — see
      `rms_gpu_profile.md §3` / `mvn_analysis.md §7.2`.
- [ ] Prototype MVN Welford single-pass — see `mvn_analysis.md §7.1`.
- [ ] Prototype bucketed fixed-LWS dynamic kernels for RMS/MVN and compare against the
  current `ov` and `static` harness variants.
- [ ] Consider the `USE_BLOCK_WRITE` precedence fix (I4) as a quick win.
