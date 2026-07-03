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
| Bucketed fixed-LWS dynamic-ABI prototype (`bucket`) | ✅ measured |
| `static` same-work baseline for optimization | ✅ |

## Progress tracking rule

Validated-status sections are append-only snapshots. Do **not** overwrite an existing
`## Current validated status (date, commit ...)` section when new measurements are taken;
add a new section with the current commit and keep older sections for comparison.

## Current validated status (2026-07-03, commit c0bceb5)

Run environment is the aboutSHW setup from `HOW_TO_RUN_aboutSHW.md`: `llm` docker
container, `conda activate built_ov`, and
`/ceciliapeng/VM/openvino/build_Release/install/setupvars.sh`.

### RMS

`test_omni_rms.py` reproduces the C6 Thinker RMS path: dynamic shape-info arg,
runtime LWS, static normalized dim, gamma-only affine, no residual/bias fused op.
The exact `ov` variant matches OV device time closely; `static` is the same work with
compile-constant LWS and is mainly an optimization reference.

Latest validated run after adding the first step variants and rerunning `test_omni_rms.py`:

| case | `ov` median | `bucket` median | `bucket_tuned` median | `static` median | best step |
|---|---:|---:|---:|---:|---|
| hidden | 512 us | 356 us | 361 us | 356 us | fixed-LWS bucket, -30.5% vs `ov` |
| q_norm | 704 us | 463 us | 825 us | 829 us | fixed-LWS bucket, -34.2% vs `ov` |
| k_norm | 178 us | 117 us | 209 us | 209 us | fixed-LWS bucket, -34.0% vs `ov` |

Notes:
- RMS has no extra fused residual/bias operands in the verified C6 path. The gamma
  vector is part of the kernel signature, but physical DRAM traffic is modeled as
  input read + output write + one gamma vector read, not gamma reread as unique DRAM
  for every row.
- The current RMS script reports input/output physical DRAM bandwidth (`x.nbytes * 2`)
  and keeps gamma as one fixed tensor, so the table above follows that script output.
- The fixed-LWS dynamic-ABI `bucket` variant is the best RMS step in this snapshot. It
  keeps the shape-info arg but compiles literal `LWS`; that alone reaches hidden static
  performance and beats static-style constants for q/k.
- `bucket_tuned`/`static` use `STACK_SIZE=items+1`; for q/k this is 9 instead of the
  `bucket` value 8 and regresses heavily. Stack/subgroup tuning must be measured per
  bucket rather than copied wholesale from the old static path.

### MVN

`test_omni_mvn.py` reproduces the C6 vision MVN path: dynamic shape-info arg,
runtime LWS, static normalized dim, and fused affine epilogue
`MVN -> Multiply(gamma) -> Add(beta)`. The generated `FUSED_OPS` macro body was
mirrored from dumped CL sources as a two-op affine chain with OV-like load/calc
ordering and `convert_half` steps.

Latest validated run uses rotated input/output/gamma/beta buffers and reports
**physical DRAM traffic**, not logical per-element fused traffic. The numerator is:
`input read + output write + one gamma vector read + one beta vector read`.

| case | `ov` median | `bucket` median | `static` median | best step |
|---|---:|---:|---:|---|
| vit | 594 us | 349 us | 349 us | fixed-LWS bucket, -41.2% vs `ov` |
| merger | 761 us | 440 us | 439 us | fixed-LWS bucket, -42.2% vs `ov` |

Notes:
- Gamma/beta buffers are now rotated too, so cross-launch cache reuse is reduced for
  fused operands.
- Bandwidth utilization uses physical DRAM traffic. Counting gamma/beta as full
  per-element unique traffic gives impossible-looking values above the 100 GB/s roof
  because the gamma/beta vectors are reused across rows and are tiny versus the input.
- The `bucket` MVN variant keeps the dynamic shape-info ABI but recovers static-level
  performance, so the measured gap is fixed-LWS codegen/attribute loss rather than
  missing math or shape-info argument overhead.

## Current validated status (2026-07-03, commit d8d6e20)

Run environment is the aboutSHW setup from `HOW_TO_RUN_aboutSHW.md`: `llm` docker
container, `conda activate built_ov`, and
`/ceciliapeng/VM/openvino/build_Release/install/setupvars.sh`.

### RMS

`test_omni_rms.py` reproduces the C6 Thinker RMS path: dynamic shape-info arg,
runtime LWS, static normalized dim, gamma-only affine, no residual/bias fused op.
The exact `ov` variant matches OV device time closely; `static` is the same work with
compile-constant LWS and is mainly an optimization reference.

Latest validated run after adding the aboutSHW RMS specialization steps and rerunning `test_omni_rms.py`:

| case | `ov` | `bucket` | specialized | `bucket_tuned` | `static` | best kept step |
|---|---:|---:|---:|---:|---:|---|
| hidden | 513 us | 356 us | 341 us (`hidden_specialized`) | 356 us | 361 us | hidden multi-sg, −4.1% vs `bucket`, −33.5% vs `ov` |
| q_norm | 704 us | 465 us | 404 us (`qk_specialized`) | 833 us | 829 us | q/k one-sg, −13.0% vs `bucket`, −42.6% vs `ov` |
| k_norm | 178 us | 118 us | 100 us (`qk_specialized`) | 209 us | 210 us | q/k one-sg, −14.9% vs `bucket`, −43.6% vs `ov` |

Notes:
- RMS has no extra fused residual/bias operands in the verified C6 path. The gamma
  vector is part of the kernel signature, but physical DRAM traffic is modeled as
  input read + output write + one gamma vector read, not gamma reread as unique DRAM
  for every row.
- The current RMS script reports input/output physical DRAM bandwidth (`x.nbytes * 2`)
  and keeps gamma as one fixed tensor, so the table above follows that script output.
- The fixed-LWS dynamic-ABI `bucket` variant remains the baseline optimization. The new
  aboutSHW-only compile-time branch specializations improve further on top of it.
- `qk_specialized` compiles out SLM and the runtime `get_num_sub_groups()==1` branch for
  D=128 q/k rows. This is kept.
- `hidden_specialized` compiles out the runtime one-vs-many subgroup branch for D=2560
  hidden rows and uses the known multi-subgroup SLM path. This is kept.
- `bucket_tuned`/`static` use `STACK_SIZE=items+1`; for q/k this is 9 instead of the
  `bucket` value 8 and regresses heavily. Stack/subgroup tuning must be measured per
  bucket rather than copied wholesale from the old static path.

### MVN

`test_omni_mvn.py` reproduces the C6 vision MVN path: dynamic shape-info arg,
runtime LWS, static normalized dim, and fused affine epilogue
`MVN -> Multiply(gamma) -> Add(beta)`. The generated `FUSED_OPS` macro body was
mirrored from dumped CL sources as a two-op affine chain with OV-like load/calc
ordering and `convert_half` steps.

Latest validated run after adding the step variant uses rotated input/output/gamma/beta buffers and reports
**physical DRAM traffic**, not logical per-element fused traffic. The numerator is:
`input read + output write + one gamma vector read + one beta vector read`.

| case | `ov` median | `bucket` median | `static` median | best step |
|---|---:|---:|---:|---|
| vit | 594 us | 349 us | 349 us | fixed-LWS bucket, −41.2% vs `ov` |
| merger | 761 us | 440 us | 439 us | fixed-LWS bucket, −42.2% vs `ov` |

Notes:
- Gamma/beta buffers are now rotated too, so cross-launch cache reuse is reduced for
  fused operands.
- Bandwidth utilization uses physical DRAM traffic. Counting gamma/beta as full
  per-element unique traffic gives impossible-looking values above the 100 GB/s roof
  because the gamma/beta vectors are reused across rows and are tiny versus the input.
- The `bucket` MVN variant keeps the dynamic shape-info ABI but recovers static-level
  performance, so the measured gap is fixed-LWS codegen/attribute loss rather than
  missing math or shape-info argument overhead.

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
prepended macro blocks + `-D` flags. Variants:

- **`ov`** — exact reproduction of what OV compiles (see the I3 arg/flag list): dynamic,
  `shape_info` arg, `HAS_PADDING=0`, runtime `LWS`, RMS gamma-only / MVN fused gamma+beta.
  Compare against OV's absolute us.
- **`bucket`** — keeps the dynamic shape-info ABI but makes `LWS` compile-time literal,
  so the kernel gets fixed work-group metadata while row count remains runtime dispatch.
  This implements ideas 1-3 in the standalone harness.
- **`bucket_tuned`** — RMS-only probe: `bucket` plus static-style `STACK_SIZE` /
  `SUBGROUP_BLOCK_SIZE` constants. This implements idea 4 and currently regresses q/k.
- **`qk_specialized`** — RMS-only kept variant for D=128 q/k: `bucket` plus
  `ONE_SUBGROUP_ROW=1`, compiling out SLM and the runtime one-subgroup branch.
- **`hidden_specialized`** — RMS-only kept variant for D=2560 hidden: `bucket` plus
  `MULTI_SUBGROUP_ROW=1`, compiling directly to the known multi-subgroup SLM path.
- **`static`** — same work, compile-constant `LWS` (shape specialization). Optimization
  target; **not** an OV reproduction. Compare the same variant before/after a change.

## Concrete improvement ideas for aboutSHW (commit d8d6e20)

This section is aboutSHW-only. Actual OpenVINO selector/JIT changes are paused until the
standalone harness has a fully optimized and measured local winner.

| step | idea | status | result |
|---:|---|---|---|
| 1 | RMS q/k compile-time one-subgroup specialization (`ONE_SUBGROUP_ROW=1`) | ✅ kept | q_norm `465 -> 404 us`; k_norm `118 -> 100 us` vs `bucket` |
| 2 | RMS hidden compile-time multi-subgroup specialization (`MULTI_SUBGROUP_ROW=1`) | ✅ kept | hidden `356 -> 341 us` vs `bucket` |
| 3 | RMS static-style stack/subgroup constants (`bucket_tuned`) | ❌ abandoned as-is | q/k regress to ~`833/209 us`; do not copy `STACK_SIZE=items+1` blindly |
| 4 | MVN algorithmic kernel variant: one-pass sum/sumsq or Welford-style reduction | ⬜ not implemented | next MVN work item after RMS local variants settle |
| 5 | RMS hidden deeper multi-subgroup tuning | ⬜ not implemented | possible next RMS work: tune SLM reduction shape after `hidden_specialized` |
| 6 | Keep physical DRAM/cold-buffer reporting consistent | ✅ kept | MVN rotates input/output/gamma/beta; RMS currently rotates input/output and reports that model |

### aboutSHW step 1: RMS q/k one-subgroup specialization

Implementation:
- Added `ONE_SUBGROUP_ROW` in `rms_gpu_bfyx_opt.cl`.
- Added `qk_specialized` in `test_omni_rms.py` for D=128 q/k rows.
- The specialized path removes `__local` SLM allocation from the compiled q/k kernel and
  compiles directly to `sub_group_reduce_add -> native_rsqrt`, with no runtime
  `get_num_sub_groups()==1` branch.

Validation:
- `python -m py_compile test_omni_rms.py` passed.
- `python test_omni_rms.py` passed accuracy for all RMS cases.

Measured result:

| case | `bucket` | `qk_specialized` | status |
|---|---:|---:|---|
| q_norm | 465 us | 404 us | kept, −13.0% vs `bucket` |
| k_norm | 118 us | 100 us | kept, −14.9% vs `bucket` |

### aboutSHW step 2: RMS hidden multi-subgroup specialization

Implementation:
- Added `MULTI_SUBGROUP_ROW` in `rms_gpu_bfyx_opt.cl`.
- Added `hidden_specialized` in `test_omni_rms.py` for D=2560 hidden rows.
- The specialized path compiles directly to the known multi-subgroup SLM reduction and
  replaces runtime `get_num_sub_groups()` with `LWS / SUB_GROUP_SIZE`.

Validation:
- Same RMS run as step 1; syntax and accuracy passed.

Measured result:

| case | `bucket` | `hidden_specialized` | status |
|---|---:|---:|---|
| hidden | 356 us | 341 us | kept, −4.1% vs `bucket` |

### aboutSHW step 3: RMS static-style stack/subgroup constants

Status: abandoned as-is.

`bucket_tuned` reused static-style `STACK_SIZE=items+1`; for q/k this produces
`STACK_SIZE=9` and regresses badly versus the dynamic bucket's `STACK_SIZE=8`.

### aboutSHW next

The next aboutSHW-only optimization should be MVN algorithmic work, because RMS now has
two local branch-specialization wins on top of the fixed-LWS bucket. Start with an MVN
one-pass sum/sumsq or Welford-style variant and compare against current MVN `bucket`:
vit ~`349 us`, merger ~`440 us`.

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

### Step-by-step prototype results

Measurements below were run in the documented `llm`/`built_ov` aboutSHW environment;
`python -m py_compile` and accuracy passed for both scripts.

#### Step 1-3: bucket fixed LWS while preserving dynamic shape-info ABI

Implementation in the harness:
- RMS/MVN add a `bucket` variant that prepends the same shape-info kernel argument as
  `ov`, but compiles with literal `-DLWS=<bucket_lws>` and `IS_DYNAMIC=0`, restoring
  `reqd_work_group_size` in kernels that gate it on `!IS_DYNAMIC`.
- Row count remains runtime dispatch through `gws=[LWS, rows]`; only the normalized
  dim and selected LWS are bucket-specialized.

Implementation in OpenVINO (`/home/intel/ceciliapeng/VM/openvino`, commit `8212005dfb`):
- `mvn_kernel_bfyx_opt.cpp`: when tensors are dynamic but the reduction dim JIT string
  is static, compute the same bucket LWS as the static path and emit literal `LWS`
  instead of `get_local_size(0)`.
- `mvn_gpu_bfyx_opt.cl`: keep `IS_DYNAMIC` for ABI/shape-info behavior, but allow
  `reqd_work_group_size(LWS,1,1)` when `FIXED_LWS_DYNAMIC` is true.
- `rms_kernel_bfyx_opt.cpp`: when the dynamic data-size JIT string is static, compute
  bucket LWS and emit literal `LWS` while preserving the shape-info ABI and existing
  dynamic stack/subgroup constants (`STACK_SIZE=ceil_div(D,LWS)`, `SUBGROUP_BLOCK_SIZE=8`).

Build validation:
- `cmake --build /ceciliapeng/VM/openvino/build_Release --target openvino_intel_gpu_plugin -j$(nproc)`
  inside the `llm`/`built_ov` environment compiled the touched RMS/MVN objects and linked
  the GPU kernel selector/graph libraries.
- The aboutSHW RMS/MVN scripts still measure their local CL harness variants; they passed
  `py_compile` and accuracy after this patch, but they are not an end-to-end validation of
  the real OpenVINO selector change.

Result: this is the main win.

| kernel | case | `ov` | `bucket` | delta |
|---|---|---:|---:|---:|
| RMS | hidden | 512 us | 356 us | −30.5% |
| RMS | q_norm | 704 us | 463 us | −34.2% |
| RMS | k_norm | 178 us | 117 us | −34.0% |
| MVN | vit | 594 us | 349 us | −41.2% |
| MVN | merger | 761 us | 440 us | −42.2% |

Conclusion: dynamic row dispatch is not the expensive part in this harness. The expensive
part is losing compile-time LWS / fixed work-group metadata.

#### Step 4: specialize stack/subgroup constants per bucket

Implementation in the harness:
- RMS adds `bucket_tuned`, which keeps the shape-info ABI and fixed LWS but swaps in the
  previous static-style `STACK_SIZE=items+1` and `SUBGROUP_BLOCK_SIZE=subgroup_block_size(items)`.

Result: not a blanket win.

| case | `bucket` | `bucket_tuned` | result |
|---|---:|---:|---|
| hidden | 356 us | 361 us | neutral/slightly worse |
| q_norm | 463 us | 825 us | worse |
| k_norm | 117 us | 209 us | worse |

Conclusion: keep the dynamic bucket constants for RMS q/k (`STACK_SIZE=8`) instead of the
old static-style `STACK_SIZE=9`. Per-bucket tuning is still valid, but every constant must
be measured; the fixed-LWS change should land independently.

#### Step 5: audit shape-info usage and guards

Implementation in the harness:
- `bucket` preserves the shape-info ABI while matching static performance for MVN and
  RMS hidden. That isolates shape-info as not the bottleneck when the body does not read it.

Conclusion: for verified `:nopad` C6 norm paths, the shape-info argument can remain for ABI
compatibility. The real target is avoiding runtime `LWS` and broad `IS_DYNAMIC` codegen.

#### Step 6: keep algorithmic optimizations separate

No new algorithmic kernel changes were mixed into this pass. The bucket result should be
treated as the first OV-facing optimization target; RMS barrier changes, MVN algorithmic
rewrites, and `USE_BLOCK_WRITE` should be measured after fixed-LWS dynamic codegen.

### Remaining validation for the OpenVINO patch

- Re-run the original Qwen3-Omni C6 OpenVINO benchmark/log capture against the rebuilt
  `/ceciliapeng/VM/openvino/build_Release/install` runtime and confirm verbose log / dumped
  kernel JIT shows literal `LWS` for the dynamic static-reduction RMS/MVN buckets.
- Compare the real C6 RMS/MVN hash timings before/after the selector patch. Expected
  direction from the harness: dynamic RMS/MVN should move toward the `bucket` timings, but
  exact model-level gain must be measured in the full OV graph.
- If the real dumped MVN fused code still carries dynamic fused-op boundary checks, audit
  whether they remain necessary for these static normalized-dim `:nopad` buckets.

## Open / next

- [x] Exact OV config (`ov` variant) — RMS within ~2%; MVN merger −8%, vit −18%.
- [x] Mirror the generated MVN `FUSED_OPS` body from the dumped CL sources in `test_omni_mvn.py` (mul/add load/calc ordering + half conversions). This narrows the vit gap to ~680 us vs 833 us, but the remaining delta is still attributed to any remaining codegen differences not yet byte-reproduced.
- [x] Rotate MVN gamma/beta buffers and correct bandwidth reporting to physical DRAM
  traffic instead of logical per-element fused traffic.
- [ ] Prototype RMS SLM-barrier reduction (`sub_group_reduce_add` chain) — see
      `rms_gpu_profile.md §3` / `mvn_analysis.md §7.2`.
- [ ] Prototype MVN Welford single-pass — see `mvn_analysis.md §7.1`.
- [x] Prototype bucketed fixed-LWS dynamic kernels for RMS/MVN and compare against the
  current `ov` and `static` harness variants.
- [ ] Consider the `USE_BLOCK_WRITE` precedence fix (I4) as a quick win.
