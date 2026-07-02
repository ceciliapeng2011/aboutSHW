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

## Files

| File | Purpose |
|---|---|
| `rms_gpu_bfyx_opt.cl` | **verbatim** copy of the OV kernel (`kernel_selector/cl_kernels/`) — do not edit except to optimize; re-`cp` to refresh |
| `mvn_gpu_bfyx_opt.cl` | **verbatim** copy of the OV kernel |
| `ov_norm_shim.cl` | minimal macro shim replacing the OV `#include`s (see below) |
| `test_omni_rms.py` | RMS cases: `hidden`, `q_norm`, `k_norm` |
| `test_omni_mvn.py` | MVN cases: `vit`, `merger` |

The test loader (`load_kernel`) prepends `ov_norm_shim.cl` and strips the kernel's
`#include "include/..."` lines. The shim only covers the **FP16-in / FP16-out /
FP32-accumulator** config (the Omni case). It supplies `KERNEL`, the datatype
macros, `DT_INPUT_BLOCK_READ{,2,4,8}` / `DT_OUTPUT_BLOCK_WRITE*` (via
`as_halfN(intel_sub_group_block_read_usN(...))`), `REQD_SUB_GROUP_SIZE`,
`unroll_for`, and the MVN `ACTIVATION*` hooks. Everything else in the kernels is
unmodified OV source.

> **gamma/beta:** RMS applies `gamma` inside the kernel (`ELEMENTWISE_AFFINE=1`), so
> the RMS test feeds a real weight. MVN's `gamma`/`beta` are applied by OV **fused
> ops**, not this base kernel, so the MVN test checks the core normalization only.

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

## Notes for optimization work

- Absolute `us/call` depends on the GPU (the validation box is a 32-EU iGPU; the C6
  numbers are from the PTL 4Xe profiling machine) — compare **relative** changes and
  the cost structure (`q_norm ≫ k_norm`; ViT LN dominates MVN).
- RMS gap-to-peak is the **SLM barrier chain** (see `rms_gpu_profile.md §3`); MVN gap
  is the **3-pass traffic** (see `mvn_analysis.md §7.1`, Welford). Prototype changes
  directly in the `.cl` here, keep the accuracy check green, then port back to OV.
