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

### Register usage breakdown (pa_kernel_lsc_prefetch_f16, hs256, Xe2 256-GRF mode)

**Long-lived registers (persist across all kv_step iterations):**

| Matrix | Shape | Element | Bytes | GRF regs | Lifetime |
|--------|-------|---------|-------|----------|----------|
| rQ | [16, 256] | half | 8,192 | 128 | Entire kernel (loaded once, read every KQ DPAS) |
| rO_lo | [16, 128] | float | 8,192 | 128 | Entire kernel (accumulated every kv_step) |
| rO_hi | [16, 128] | float | 8,192 | 128 | Entire kernel (accumulated every kv_step) |
| cur_max | [16] | float | 64 | 1 | Online softmax state |
| cur_sum | [16] | float | 64 | 1 | Online softmax state |
| **Total persistent** | | | **24,704** | **386** | |

**Budget = 256 regs. Oversubscription = 130 regs (~8.3 KB spill minimum).**

The fundamental constraint: `rQ(128) + rO_lo(128) + rO_hi(128) = 384` regs of persistent state, before any temporaries.

**Per-iteration temporaries (inside kv_step loop):**

| Phase | Matrix | Shape | Regs | Notes |
|-------|--------|-------|------|-------|
| KQ | Kmat | [2, 128] half | 8 | Overwritten each ri iteration |
| KQ | St | [16, 16] float | 16 | Reused in softmax |
| Softmax | max_comp | [16] float | 1 | |
| Softmax | P (Transpose output) | [16, 16] half | 8 | +8 temp inside Transpose |
| PV | Vmat | [8, 32] half | 8 | Overwritten each k iteration |
| Misc | b2dK/b2dV/addresses | — | ~14 | Descriptors, loop counters |
| **Total temporaries** | | | **~55** | |

**Peak pressure (during PV1 phase) = 386 + 55 = ~441 regs.** The compiler spills ~12-13 KB, rotating rQ and rO tiles through scratch memory every DPAS iteration.

For comparison, hs128: rQ(64) + rO(128) + temps(~55) = 247 regs < 256. **Zero spill.**

**Root causes of 5-7x MFU drop:**
1. **Register spill (~12-13 KB):** Adds scratch memory traffic every DPAS iteration. HEAD_SIZE=128 has zero spill.
2. **Split rO double-read:** V data is read from SLM (or loaded via LSC) twice per kv_step — once for rO_lo, once for rO_hi — instead of being reused in a single pass.
3. **Instruction cache pressure:** `#pragma unroll` on V loops with 2x iterations (16 for hs256 vs 8 for hs128) bloats the instruction footprint.

### Why OCL micro kernel has lower register pressure at hs256

The OCL `sdpa_micro__prefill` kernel peaks at ~180 regs for h256, vs CM PA's ~441. Three key design differences:

**1. SLM decouples KQ and VS register lifetimes (max vs sum):**
- Micro kernel: Q lives in Q_slm (not GRF). KQ result S is written to S_slm after softmax. VS reads S from S_slm. Q_tile is transient (freed before VS). Peak = `max(KQ_regs, VS_regs)` = max(110, 180) = **180**.
- CM PA: rQ lives in GRF the entire kernel. rO lives in GRF the entire kernel. Peak = `KQ_regs + VS_regs` = 128 + 256 + 55 = **441**.

**2. Head_size is tiled across 8 SG rows (not 1 thread):**
- Micro kernel VS: `wg_m_vs=8` SG rows, each accumulates `sg_tile_m=32` of 256 head dims → A_tile = 64 GRFs.
- CM PA: each thread accumulates all 256 head dims → rO = 256 GRFs.

**3. Fewer q tokens per work unit:**
- Micro kernel h256: 2 q-tokens per lane (vs CM's 16 per thread), smaller S_tile and A_tile.

### Lessons for CM PA optimization

| Approach | rQ | rO | Peak | Spill | DPAS overhead | SLM cost | Occupancy | U8+FP16? | Effort |
|----------|----|----|------|-------|---------------|----------|-----------|----------|--------|
| **Current CM PA** | 128 (GRF, permanent) | 256 (split) | ~441 | ~12 KB | None | None | 2 WG | Yes | — |
| **Q from L3 cache** | 4 (transient) | 256 (split) | ~315 | ~2-3 KB | None | None | 2 WG | **Yes** | Low |
| **Move Q to SLM** | 0 (SLM) | 256 (split) | ~311 | ~2 KB | None | 128 KB | **1 WG** | **FP16 only** | Medium |
| **num_P_tiles=1** | 128 (GRF) | 128 | ~311 | ~2 KB | **+50%** | None | 2 WG | Yes | Medium |
| **Q from L3 + num_P_tiles=1** | 4 (transient) | 128 | ~187 | 0 | +50% | None | 2 WG | Yes | Medium |
| **Q to SLM + num_P_tiles=1** | 0 (SLM) | 64 | ~119 | 0 | +50% | 128 KB | 1 WG | FP16 only | Medium-High |
| **Full micro-style redesign** | SLM | 64/SG | ~180 | 0 | None | Full | 1 WG | Yes | Very High |

### Detailed feasibility analysis of each approach

**Move Q to SLM — SLM budget problem:**

Each of the 16 threads has unique Q data (different q_start positions), so Q cannot be shared. SLM cost = 16 threads × 8 KB = **128 KB**.

| Path | K_SLM (ring×4) | V_SLM (ring×4) | Q_SLM (16 threads) | Total | Limit | Fits? |
|------|---------------|----------------|---------------------|-------|-------|-------|
| U8, hs256 | 32 KB | 32 KB | 128 KB | **192 KB** | 128 KB | **No** |
| FP16, hs256 | 0 | 0 | 128 KB | 128 KB | 128 KB | Barely |

- **U8 path: impossible** (192 KB > 128 KB SLM).
- **FP16 path: exactly fills SLM** (128 KB). But this forces 1 WG per Xe Core (128 KB SLM per core), halving occupancy from 2 WGs to 1 WG. For a compute-bound kernel, reduced occupancy hurts latency hiding.

**num_P_tiles=1 — DPAS waste problem:**

REG_N=16 is hardware-fixed. KQ DPAS always produces 16 output columns regardless of how many q tokens are consumed by PV. With num_P_tiles=1, each thread computes 16 KQ columns but PV only uses 8 → **50% KQ DPAS wasted**.

System-wide DPAS to process 16 q tokens (same work as current single thread):

| Phase | num_P_tiles=2 (1 thread) | num_P_tiles=1 (2 threads) |
|-------|--------------------------|---------------------------|
| KQ | 32 DPAS | **64 DPAS** (+100%) |
| PV | 32 DPAS | 32 DPAS (same) |
| **Total** | 64 DPAS | **96 DPAS** (+50%) |

For a kernel already at MFU 0.08-0.13, adding 50% more DPAS is costly.

**Q from L3 cache — the best standalone option:**

Instead of keeping rQ permanently in GRF (128 regs), reload Q tiles from global memory per KQ iteration, relying on L3 cache:

```cpp
// Current: rQ permanent in GRF (128 regs)
dpas(St, rQ[ri].format<int32_t>(), Kmat);

// Proposed: Qtile loaded per iteration from L3 (~4 regs transient)
cm_load(Qtile, b2dQ.set_block_x(ri*REG_K));  // Q from L3
dpas(St, Qtile.format<int32_t>(), Kmat);
```

Why L3 re-reads are near-free:
- Q is only 8 KB per thread — fits easily in 8 MB L3
- Q is read-only, same access pattern every kv_step → stays hot in L3
- L3 BW ~500 GB/s. Total Q re-reads at 32K kv_len: 2048 iters × 8 KB = 16 MB/thread, 32 threads = 512 MB. At 500 GB/s → ~1 ms vs ~500 ms kernel. **<0.2% overhead**
- L3 load latency (~100 cycles) overlaps with Kmat load (different addresses, pipelined)

Advantages: works for both U8 and FP16, zero SLM cost, zero occupancy impact, zero DPAS waste, minimal code change.

---

## Optimization Experiments (multi-token PA, hs256, seq=32K, 32h/8kvh)

### Baseline performance

| Config | Spill | MFU (GFLOPS) | Latency (ms) | Meet |
|--------|-------|-------------|-------------|------|
| hs256, FP16, sb1 | 12,736 B | 3,639 | 4,835 | 0.12 |
| hs256, U8, sb1 | 12,544 B | 2,255 | 7,800 | 0.08 |
| hs128, FP16, sb1 | 0 | 18,208 | 483 | 0.61 |
| hs128, U8, sb1 | 0 | 17,188 | 512 | 0.57 |

hs256 achieves meet 0.08-0.12 vs 0.57-0.61 for hs128 (5-7x MFU gap).

### Experiment 1: Q from L3 cache — FAILED, REVERTED

**Change:** Replaced permanent `rQ[16, 256]` half (128 regs) with per-KQ-iteration `cm_load<lsc::Transpose>` of transient `Qtile[1, REG_K*REG_N]` (~4 regs). Added `ugemm_KQ_L3` to `cm_attention_common.hpp`. Modified both `pa_lsc_u8` and `pa_kernel_lsc_prefetch_f16`.

| Config | Spill | MFU (GFLOPS) | Delta vs baseline |
|--------|-------|-------------|-------------------|
| hs256, FP16, sb1 | 4,928 B (-61%) | 2,635 | **-28%** |
| hs256, U8, sb1 | 4,352 B (-65%) | 1,894 | **-16%** |
| hs128, FP16 | 0 | ~18,100 | ~0% |
| hs128, U8 | 0 | ~17,800 | ~0% |

**Why it failed:** Each KQ iteration loads 16 Q tiles from L3 via `cm_load<lsc::Transpose>` + `cm_mul(scale_factor)`. The per-tile L3 latency (~100 cycles) × 16 tiles = ~1600 cycles overhead per kv_step, which is **not overlapped** with DPAS execution because the Q tile is consumed immediately by the next DPAS. The compiler's spill strategy (strategic eviction/reload from scratch) is more efficient because it can overlap spill traffic with DPAS pipeline stages.

**Lesson:** Reducing spill does not guarantee MFU improvement. Explicit per-iteration data movement can be worse than compiler-managed spill.

### Experiment 2: Remove V loop unroll for hs256 — FAILED, REVERTED

**Change:** Added `#if CMFLA_HEAD_SIZE <= 128` / `#pragma unroll` / `#endif` guards around 8 PV loops (4 in optimized sparse pipeline, 4 in legacy pipeline) in both `pa_lsc_u8` and `pa_kernel_lsc_prefetch_f16`.

| Config | Spill | Delta vs baseline |
|--------|-------|-------------------|
| hs256, FP16 | **20,928 B** (+64%) | Spill **increased** |
| hs256, U8 | 12,544 B (~0%) | No change |

**Why it failed:** The V loops in `cm_pa_xe2.hpp` are already split into two halves (rO_lo and rO_hi), each iterating only `head_size/2/REG_N = 8` times. With unroll, the compiler can interleave DPAS scheduling and register allocation across all 8 iterations efficiently. Without unroll, the compiler must handle a runtime loop, losing the ability to overlap DPAS latencies — actually **increasing** register pressure. This contrasts with `pa_single_token.cm` where the same technique worked because its V loop has 16 iterations (HEAD_SIZE/REG_N) with many more registers live simultaneously.

**Lesson:** Unroll removal helps when loop iteration count is large (16+) and unrolling causes excessive simultaneous register liveness. It hurts when loops are short (8 iterations) and the compiler benefits from seeing all iterations for scheduling.

### Experiment 3: Tile-based O accumulation — INFEASIBLE (not implemented)

**Concept:** Use single `rO` (128 regs) and process head_size in two sequential passes.

**Option A: Two passes over KV sequence.** Each pass iterates all kv_steps, computing KQ + PV for one half of head_size. KQ DPAS produces the same St in both passes → **100% KQ DPAS duplication**. Total DPAS per kv_step: (KQ(32) + PV(16)) × 2 = 96 vs current 64 → **+50% system DPAS**. For a compute-bound kernel, this is unacceptable.

**Option B: Per-iteration rO swap through SLM/scratch.**
- SLM: 16 threads × 16 KB (rO_lo + rO_hi) = **256 KB > 128 KB SLM limit**. Infeasible.
- Global/L3 scratch: Requires host-allocated scratch buffer (new kernel arg). Adds 4 L3 operations per kv_step (2 loads + 2 stores of 8 KB each). Given that Experiment 1 showed per-iteration L3 loads degrade MFU despite reducing spill, this approach would likely also regress.

**Not attempted** due to both variants being structurally worse than baseline.

### Remaining suggestions (not yet tested)

4. **Reduce V re-read via register caching (FP16 path).** Cache Vmat tiles across rO_lo/rO_hi to avoid double-loading. Adds ~8 regs pressure per Vmat but halves V load traffic. **Risk:** 8 extra regs may increase spill given already-tight 256 GRF budget. Medium priority.

5. **U8 SLM working set tiling.** Tile head_size in `load_slm_KV` to reduce SLM working set. **Currently unnecessary** — hs256 U8 uses 64 KB SLM (half of 128 KB limit).

6. **num_P_tiles=1 (halve q/thread).** +50% system DPAS waste. Only viable if kernel becomes memory-bound after other optimizations.

7. **Move Q to SLM (FP16-only).** Uses all 128 KB SLM, halves occupancy, U8 incompatible.

### Key findings and root cause analysis

**Spill is not the primary performance bottleneck.** Q-from-L3 reduced spill by 63% but MFU regressed 16-28%. The compiler's spill strategy is surprisingly efficient — it overlaps scratch memory traffic with DPAS pipeline stages, whereas explicit data movement (L3 loads) introduces pipeline stalls.

**The fundamental problem is register oversubscription structure:** rQ(128) + rO_lo(128) + rO_hi(128) = 384 persistent regs vs 256 available. Every approach to reduce this trades away either DPAS throughput (tile-based O, num_P_tiles=1), memory bandwidth (Q-from-L3), or occupancy (Q-to-SLM). The compiler's spill is already the least-cost option among these trade-offs.

**hs256 vs hs128 MFU gap breakdown (estimated):**
1. **Register spill traffic:** ~2-3x slowdown (12-13 KB spill adds scratch BW overhead to every DPAS iteration)
2. **rO split double V-reads:** ~1.5x (FP16 path loads each Vmat twice; U8 path reads V from SLM twice)
3. **Instruction footprint:** ~1.2x (2x V loop iterations, 2x store loops)
4. Combined: ~3.6-5.4x ≈ observed 5-7x gap

**Potential directions not yet explored:**
- **Compiler improvements:** Future CM compiler versions may better handle 384-reg kernels (smarter spill heuristics, better DPAS pipeline overlap).
- **Hybrid thread cooperation:** Like OCL micro kernel — tile head_size across multiple threads, communicate via SLM. Would require significant kernel redesign.
- **Hardware:** Xe3 adds 512-GRF mode (32 KB/thread) which would eliminate spill entirely for hs256.
