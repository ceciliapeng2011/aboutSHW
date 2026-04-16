# Skill: HEAD_SIZE=256 Support for PA Kernels on Xe2

LNL Xe2 reference: 64 EUs, 2 GHz, 256 GRF (16 KB), 128 KB SLM, ~30 TFLOPS FP16 peak.
CM constants: CM_GRF_WIDTH=512, REG_N=16, REG_K=16, REG_M=8, num_P_tiles=2.
CM matrix limit: `sizeof(matrix) < 16384` bytes (strict).

---

## test_kvcache_update.py (pa_kv_cache_update kernel)

**Problem:** None. HEAD_SIZE=256 compiled and passed on the first attempt. The kvcache update kernel writes incoming K/V tokens into paged block storage; its matrix sizes are independent of HEAD_SIZE (it processes per-element, not full-head accumulation).

**Fix:** Added test cases only (no kernel changes):
- `test_pa_kv_cache_update([32*1024], [0], ..., k_head_size=256, v_head_size=256, ...)` for fp16 and u8
- `test_pa_kv_cache_update([32*1024], [4*1024], ..., k_head_size=256, v_head_size=256, ...)` for fp16 and u8
- Also fixed a pre-existing bug: line 482 was missing `enable_kvcache_compress=compress_kvcache`, causing it to always default to True.

**Performance (memory-bound, 32K tokens, 8 kv_heads, past_lens=0):**

| Config | Data (MB) | BW (GB/s) |
|--------|-----------|-----------|
| hs128, fp16 | 268 | ~114 |
| hs256, fp16 | 537 | ~108 |
| hs128, u8 | ~203 | ~88 |
| hs256, u8 | ~405 | ~90 |

Near-perfect scaling: 2x data with roughly the same bandwidth. No regression.

**Suggestion:** No further work needed. This kernel is simple scatter-write with good memory access patterns.

---

## test_pa_decoding.py (pa_single_token.cm: cm_sdpa_2nd + cm_sdpa_2nd_reduce)

**Problem:** No compilation error, but severe u8 performance regression. The `#pragma unroll` on the V inner loop (`for k=0; k<HEAD_SIZE; k+=REG_N`) fully unrolls 16 iterations for hs256 (vs 8 for hs128), keeping all iterations' VmatNormal/Vmat/Vt_quant registers live simultaneously. This causes 2368-byte spill in the u8 path (256 registers saturated). Additionally, `reduce_split_step=8` causes `REDUCE_SPLIT_SIZE=32` for hs256, dispatching 2x more reduce workitems than hs128.

**Fix (two changes):**
1. **pa_single_token.cm line 469:** Conditionally remove `#pragma unroll` from V inner loop for `HEAD_SIZE > 128`:
   ```cpp
   #if HEAD_SIZE <= 128
   #pragma unroll
   #endif
   ```
   This lets the compiler use a runtime loop for hs256, reducing register live ranges from 16 iterations to 1. Eliminates the 2368-byte spill entirely. u8/by_token registers drop from 256 to 175.

2. **test_pa_decoding.py line 166:** Increase `reduce_split_step` to 16 for hs256:
   ```python
   self.reduce_split_step = 16 if head_size >= 256 else 8
   ```
   This keeps `REDUCE_SPLIT_SIZE=16` (same as hs128), halving GWS_2 dispatch from 32 to 16 workitems.

All 7 functional tests pass (4 hs128 + 3 hs256). Added perf benchmark extension in the `test_pa_perf_bandwidth_generate_single_subsequence_default_params` test (gated by `RUN_PA_PERF=1`).

**Performance (memory-bound, kv_len=32769, 32 heads, 8 kv_heads):**

*Before optimization:*

| Config | sdpa_2nd BW (GB/s) | reduce BW (GB/s) | sdpa_2nd (ms) | reduce (ms) | Spill |
|--------|--------------------|------------------|---------------|-------------|-------|
| hs128, fp16 | 105 | 40 | 1.28 | 0.054 | 0 |
| hs128, u8/by_token | 97 | 39 | 0.70 | 0.055 | 0 |
| hs128, u8/by_channel | 97 | 39 | 0.70 | 0.055 | 0 |
| hs256, fp16 | 105 | 39 | 2.58 | 0.108 | 0 |
| hs256, u8/by_token | 58 | 41 | 2.32 | 0.103 | **2368 B** |
| hs256, u8/by_channel | 54 | 41 | 2.51 | 0.104 | **2368 B** |

*After optimization:*

| Config | sdpa_2nd BW (GB/s) | reduce BW (GB/s) | sdpa_2nd (ms) | reduce (ms) | Spill |
|--------|--------------------|------------------|---------------|-------------|-------|
| hs128, fp16 | 105 | 40 | 1.28 | 0.054 | 0 |
| hs128, u8/by_token | 97 | 39 | 0.70 | 0.055 | 0 |
| hs128, u8/by_channel | 97 | 39 | 0.70 | 0.055 | 0 |
| hs256, fp16 | 105 | **62** | 2.59 | **0.069** | 0 |
| hs256, u8/by_token | **101** | **63** | **1.34** | **0.068** | **0** |
| hs256, u8/by_channel | **101** | **64** | **1.34** | **0.067** | **0** |

**End-to-end improvement (sdpa_2nd + reduce total):**
- **hs256 fp16:** 2.689 -> 2.655 ms (-1.3%, reduce-only gain)
- **hs256 u8/by_token:** 2.424 -> 1.410 ms (**-42%**)
- **hs256 u8/by_channel:** 2.614 -> 1.408 ms (**-46%**)
- **hs128:** zero regression across all configs

**Suggestion:**
- The fp16 path (no dequant) is already optimal — 105 GB/s matches hs128's bandwidth ceiling on LNL.
- For further u8 gains, consider applying the same no-unroll technique to the K dequant loop (lines 231-236 in pa_single_token.cm) which also iterates HEAD_SIZE/REG_K times with `#pragma unroll`.

---

## test_pa.py (pa_multi_token.cm -> cm_pa_xe2.hpp: pa_lsc_u8 + pa_kernel_lsc_prefetch_f16)

**Problem:** CM compilation error. The accumulator `rO` matrix hits exactly 16384 bytes:
```
matrix<float, head_size/REG_N * num_P_tiles, REG_M * REG_N> rO;
// HEAD_SIZE=256: 256/16*2=32 rows, 8*16=128 cols, 32*128*4 = 16384 bytes
// FAILS: must be strictly < 16384
```
This affects 4 code paths: 2 functions (`pa_lsc_u8`, `pa_kernel_lsc_prefetch_f16`) x 2 pipeline variants (optimized sparse, legacy per-step).

**Fix:** Split `rO` into `rO_lo` + `rO_hi`, each covering half of head_size (8192 bytes each):
```cpp
constexpr int rO_half_rows = head_size / 2 / REG_N * num_P_tiles;
matrix<float, rO_half_rows, REG_M * REG_N> rO_lo;  // [0, head_size/2)
matrix<float, rO_half_rows, REG_M * REG_N> rO_hi;  // [head_size/2, head_size)
```

All 36 access sites updated:
1. **SLM-based PV (pa_lsc_u8):** Call `ugemm_PV0`/`ugemm_PV1` twice with adjusted SLM V offset `(head_size/2)*REG_K*sizeof(half)`.
2. **Inline DPAS PV (pa_kernel_lsc_prefetch_f16):** Split `for(k=0; k<head_size; ...)` into two loops: `[0, head_size/2)` -> `rO_lo`, `[head_size/2, head_size)` -> `rO_hi`.
3. **Output store:** Two loops writing `rO_lo` then `rO_hi` via b2dO.
4. **Compiler flags:** Removed `-abortonspill` for HEAD_SIZE > 128 since rQ(128) + rO_lo(128) + rO_hi(128) = 384 registers > 256 GRF. Expected spill ~12-13 KB. HEAD_SIZE=128 retains zero-spill (no regression).

Added accuracy tests (6 cases: fp16/u8 x sparse_block_sz=1/256/128) and perf benchmarks mirroring `smoke_perf_test()`.
Parameterized the roofline formula: `roofline_ms = 293.20 * (head_size/128) * (seq_len/32768)^2 * (num_heads/32)`.

**Performance (compute-bound, seq_len=32768, 32 heads, 8 kv_heads, trunk=128 blocks):**

| Config | sb | density | MFU (GFLOPS) | Latency (ms) | Meet |
|--------|----|---------|-------------|-------------|------|
| hs128, U8 | 1 | 1.00 | 17,188 | 512 | **0.57** |
| hs128, U8 | 256 | 1.00 | 17,883 | 496 | **0.60** |
| hs128, U8 | 256 | 0.33 | 17,970 | 163 | **0.60** |
| hs256, U8 | 1 | 1.00 | 2,255 | 7,800 | **0.08** |
| hs256, U8 | 256 | 1.00 | 2,664 | 6,655 | **0.09** |
| hs256, U8 | 256 | 0.33 | 2,660 | 2,199 | **0.09** |
| hs128, FP16 | 1 | 1.00 | 18,208 | 483 | **0.61** |
| hs128, FP16 | 256 | 1.00 | 18,109 | 490 | **0.60** |
| hs128, FP16 | 256 | 0.33 | 18,719 | 156 | **0.62** |
| hs256, FP16 | 1 | 1.00 | 3,639 | 4,835 | **0.12** |
| hs256, FP16 | 256 | 1.00 | 3,531 | 5,021 | **0.12** |
| hs256, FP16 | 256 | 0.33 | 3,789 | 1,544 | **0.13** |

HEAD_SIZE=256 achieves meet 0.08-0.13 vs 0.57-0.62 for HEAD_SIZE=128. This is functionally correct but ~5-7x below ideal scaling.

**Root causes:**
1. **Register spill (~12-13 KB):** Adds scratch memory traffic every DPAS iteration. HEAD_SIZE=128 has zero spill.
2. **Split rO double-read:** V data is read from SLM (or loaded via LSC) twice per kv_step — once for rO_lo, once for rO_hi — instead of being reused in a single pass.
3. **Instruction cache pressure:** `#pragma unroll` on V loops with 2x iterations (16 for hs256 vs 8 for hs128) bloats the instruction footprint.

**Suggestions (in priority order):**
1. **Remove `#pragma unroll` from V loops for head_size >= 256.** The compiler can use runtime loops to reduce instruction footprint and register live ranges. This is the lowest-risk change with potentially the highest impact on spill reduction.
2. **Tile-based O accumulation:** Instead of splitting rO into halves that are both live simultaneously (causing 384 regs), use a single full-size rO tile and process head_size in two sequential passes — accumulate `[0, head_size/2)` into rO, store partial O to SLM or memory, then accumulate `[head_size/2, head_size)`. This eliminates the double-read penalty and halves register pressure at the cost of one extra store+load of partial O per kv_step.
3. **Large GRF mode (`-Qxcm_register_file_size=512`):** If Xe2 hardware supports 512-register mode (doubling GRF at the cost of halving thread occupancy), the original unsplit rO would fit. This trades parallelism for per-thread register budget — viable when the kernel is compute-bound rather than latency-bound.
4. **U8 SLM working set:** For pa_lsc_u8, SLM stores `4 * kv_step * head_size * sizeof(half) * 2 = 65536 bytes` for HEAD_SIZE=256. This is half the 128 KB SLM. Tiling head_size in the `load_slm_KV` lambda (load half of head_size K/V at a time) could reduce SLM pressure and improve occupancy.
5. **Reduce V re-read via SLM caching:** In the inline DPAS path (f16), each V tile is loaded via `cm_load<lsc::VNNI>` twice (once per rO half). Caching the loaded Vmat in a local variable and reusing across both halves would halve V memory traffic — but adds register pressure for the cached tile. Worth profiling.
