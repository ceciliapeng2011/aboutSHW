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
| **`ov` variant = exact OV** (args/flags/padding/fusion verified) → RMS ~2%, MVN merger −8%, vit −18% | ✅ (this pass) |
| `static` same-work baseline for optimization | ✅ |

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

## Open / next

- [x] Exact OV config (`ov` variant) — RMS within ~2%; MVN merger −8%, vit −18%.
- [ ] (optional) Reproduce OV's auto-generated MVN `FUSED_OPS` byte-for-byte (needs a
      GPU kernel-source dump) to close the vit delta — currently the only unverified item.
- [ ] Prototype RMS SLM-barrier reduction (`sub_group_reduce_add` chain) — see
      `rms_gpu_profile.md §3` / `mvn_analysis.md §7.2`.
- [ ] Prototype MVN Welford single-pass — see `mvn_analysis.md §7.1`.
- [ ] Consider the `USE_BLOCK_WRITE` precedence fix (I4) as a quick win.
