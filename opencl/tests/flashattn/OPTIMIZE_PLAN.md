# cm_sdpa_vlen optimization plan (Xe3/PTL)

Baseline: `seq=6864, sub_seq=3432, 16h/16kvh/d64` → **~9.5 ms**

Platform: PTL 4xe (Xe3, equivalent to Xe2 — `CM_HAS_LSC_UNTYPED_2D` present,
`CM_GRF_WIDTH=512`, `REG_N=16`, `REG_K=16`, `REG_M=8`).

Active kernel path: `sdpa_kernel_lsc_prefetch` (via `USE_LSC_PREFETCH=1`).

---

## How the current prefetch path works

Per kv iteration, for `d=64` (`padded_head_size/REG_K = 4` chunks):

**K phase** (inner loop `ri=0..3`):
```
ri=0: cm_prefetch K_next[row=wg_local_id, col=0 ],  cm_load<Normal> K_curr[col=0 ], DPAS
ri=1: cm_prefetch K_next[row=wg_local_id, col=16],  cm_load<Normal> K_curr[col=16], DPAS
ri=2: cm_prefetch K_next[row=wg_local_id, col=32],  cm_load<Normal> K_curr[col=32], DPAS
ri=3: cm_prefetch K_next[row=wg_local_id, col=48],  cm_load<Normal> K_curr[col=48], DPAS
```
Each thread prefetches 1 row (`kv_step/wg_local_size = 1`) of the next K tile.
K_next is spread across the full K phase — reasonable lead time.

**Softmax + Transpose** (between K and V phase)

**V phase** (inner loop `k=0,16,32,48`):
```
k=0:  cm_prefetch V_next[row=wg_local_id, col=0 ],  cm_load<VNNI> V_curr[col=0 ], scale rO, DPAS
k=16: cm_prefetch V_next[row=wg_local_id, col=16],  cm_load<VNNI> V_curr[col=16], scale rO, DPAS
...
```
V prefetch for chunk k is issued immediately before the load of that same chunk —
**zero intra-tile lead time**. Chunk 0 of the next iteration was prefetched at the tail
of the previous V loop; chunks 1–3 are cold inside the loop.

**Note on K redundancy**: `b2dK` loads the full `[kv_step × REG_K]` tile into every
thread's registers. All 16 threads load the same 512 bytes — the hardware deduplicates
this to 1× bandwidth, but 16× LSC load instruction slots are consumed.

---

## Optimization items

### 1. Replace `need_wg_mapping` with a precomputed mapping array

**Problem:** The `need_wg_mapping=1` branch runs a while-loop over `cu_seqlens` inside
every thread of every WG — O(num_sequences) work, redundantly, with SVM reads of
`cu_seqlens` per iteration.

**Fix:** Adopt the `blocked_q_starts_and_subseq_mapping` pattern from
`pa_multi_token.cm`. The host pre-computes a flat `int32[2 × wg_count]` array once:

```python
mapping = []
for i, (seq_start, seq_end) in enumerate(zip(cu_seqlens, cu_seqlens[1:])):
    seq_len = seq_end - seq_start
    for k in range((seq_len + wg_seq_len - 1) // wg_seq_len):
        mapping += [int(seq_start) + k * wg_seq_len, i]
wg_count = len(mapping) // 2
```

The kernel reads 2 scalars and derives everything in O(1) with no branch:

```cpp
int block_start_pos = mapping[wg_id * 2];
int seq_id          = mapping[wg_id * 2 + 1];
int kv_start        = cu_seqlens[seq_id];
int kv_seq_len      = cu_seqlens[seq_id + 1] - kv_start;
int q_start         = block_start_pos + wg_local_id * q_step;
```

This also eliminates the `need_wg_mapping` kernel argument, the dead-code guard
(`wg_base > wg_id` can never be true), and the `wg_count` tensor-arithmetic bug in
Python.

**Files:** `cm_sdpa_vlen.cm` (kernel dispatch block), `cmfla.py` (`__call__` method).

---

### 2. Move V prefetch into the K phase (earlier lead time)

**Problem:** `prefetch_V` for chunk k of the *current* iteration is issued right before
`cm_load<VNNI>` of that chunk — zero lead time. Intra-tile chunks 1–3 are cold.

**Fix:** During the K inner loop (`ri=0..3`), interleave the V prefetch for the
*current* `kv_pos` alongside the K prefetch for `kv_pos+kv_step`:

```cpp
for ri in 0..3:
    cm_prefetch K_next[ri*REG_K]     // already there
    cm_prefetch V_curr[ri*REG_N]     // move here from V loop
    cm_load<Normal> K_curr[ri*REG_K]
    DPAS
```

Then the V loop only does `cm_load<VNNI>` + scale + DPAS, with V already warm in L1
after the full K phase + softmax + transpose latency.

**File:** `cm_sdpa_common.hpp`, `sdpa_kernel_lsc_prefetch`.

---

### 3. Software-pipeline load→DPAS within K and V inner loops (hide L1 stall)

**Problem:** `cm_load K[ri]` is immediately followed by `DPAS(Kmat)` — back-to-back
load→use dependency stall. Same in the V loop.

**Fix:** Double-buffer `Kmat` (and `Vmat`) so the load of tile `ri+1` overlaps with
the DPAS of tile `ri`:

```cpp
matrix<half, num_K, REG_M*REG_K> Kmat_a, Kmat_b;
cm_load K[0] → Kmat_a;
for ri = 1..3:
    cm_load K[ri] → Kmat_b       // issue early, overlaps DPAS on Kmat_a
    DPAS(Kmat_a)
    swap(Kmat_a, Kmat_b)
DPAS(Kmat_a)                     // last tile
```

Same pattern for `Vmat` in the V loop (after item 2 moves V warm into L1, the
remaining L1→register latency is still ~20 cycles on Xe2/Xe3).

Costs 1 extra `Kmat`/`Vmat` register buffer per thread.

**File:** `cm_sdpa_common.hpp`, `sdpa_kernel_lsc_prefetch`.

---

### 4. Pre-loop warm-up prefetch (cold-start for kv_pos=0)

**Problem:** Before the loop no V prefetch is issued. The first iteration's V is
entirely cold.

**Fix:** Before entering the loop, prefetch all V chunks for `kv_pos=0` and all K
chunks for `kv_pos=kv_step` (next iteration):

```cpp
// warm-up before loop
for ri in 0..3:
    cm_prefetch V[kv_pos=0, chunk=ri]
    cm_prefetch K[kv_pos=kv_step, chunk=ri]
```

Combined with item 2, this gives `kv_pos=0`'s V a full K phase worth of lead time.

**File:** `cm_sdpa_common.hpp`, `sdpa_kernel_lsc_prefetch`.

---

### 5. Peel `kv_pos=0` out of the main loop (eliminate hot-path branch)

**Problem:** `if (kv_pos == 0) ugemm_PV0 / else ugemm_PV1` inside the loop adds
a branch per iteration and prevents the compiler from fully scheduling the PV1 path
(which is 100% of iterations after the first).

**Fix:** Execute `kv_pos=0` with `ugemm_PV0` before the loop; loop from
`kv_pos=kv_step` with `ugemm_PV1` unconditionally.

```cpp
// handle kv_pos=0 before loop (ugemm_PV0, no rescaling)
// loop kv_pos = kv_step .. kv_stop (always ugemm_PV1)
```

**File:** `cm_sdpa_common.hpp`, `sdpa_kernel_lsc_prefetch`.

---

## GRF occupancy analysis (corrected)

The kernel uses ~163 GRFs/thread in large GRF mode (256 GRF bank per thread).

**Xe3 has two GRF modes** (confirmed experimentally):
- `-Qxcm_register_file_size=256`: 4 thread contexts/EU, 256 GRF (16 KB) each → 64 KB total
- `-Qxcm_register_file_size=512`: 2 thread contexts/EU, 512 GRF (32 KB) each → 64 KB total

In 256-GRF mode each thread always occupies exactly one 16 KB bank regardless of
actual GRF usage — `floor(256/163)` does not apply. A thread using 163/256 GRFs
wastes 93 GRFs but does not evict other thread contexts.

**Switching to 512-GRF mode alone (q1, same work/thread) is neutral**: measured
9.390 ms vs 9.386 ms baseline. Halving context count exactly cancels any per-context
benefit — as expected since the kernel is not latency-limited by context switching.

GRF budget (informational):
- `rO`: `float[4 × 2 × 8 × 16]` = 4096 bytes = **128 GRFs** (FP32)
- `rQ`: `half[4 × 16 × 16]` = 2048 bytes = **64 GRFs** (FP16)
- Other (cur_max, cur_sum, St, P, Kmat, Vmat, descriptors): ~22 GRFs

**Conclusion:** Items 6 and 7 (head-split, reload rQ) were based on a false occupancy
model and are not expected to help. Item 6 was tried and reverted — it doubled the kv
loop cost with no occupancy benefit, causing ~71% regression.

---

### ~~6. Split head dimension~~ (obsolete — false premise)

Tried and reverted. Doubled kv-loop cost, no occupancy benefit (EU always has 4 thread
contexts in large GRF mode regardless of per-thread GRF usage).

---

### ~~7. Reload rQ on-demand~~ (obsolete — false premise)

Dropped. Same reason as item 6.

---

## Status and priority

| # | Change | Status | Result |
|---|--------|--------|--------|
| 1 | Mapping array replaces `need_wg_mapping` | Done | ~0% GPU perf, cleaner code |
| 2 | V prefetch moved to K phase | Done | −1.7% to −6.2% |
| 3 | Load/DPAS double-buffer (K) | Tried ×2, reverted | Regression both times — 4 thread contexts already hide L1 stalls |
| 4 | Pre-loop warm-up prefetch | Done (with 2) | Included in items 2+4+5 result |
| 5 | Peel kv_pos=0 / unified PV path | Done (with 2) | Included in items 2+4+5 result |
| 6 | Split head dimension (2 passes) | Tried, reverted | +71% regression — doubled kv cost, no occupancy benefit |
| 7 | Reload rQ on-demand | Dropped | False premise (occupancy not the bottleneck) |
| 8 | 2-step-ahead K prefetch + extra warm-up tile | Tried, reverted | ~0% — not prefetch-limited |
| 9 | Increase q_step to 32 (2 Q-rows per thread) | Tried ×2, reverted | +16% regression — same with 256 GRF (spill) or 512 GRF (no spill, 2 ctx/EU); exp scales linearly with Q rows, no amortization |
| 10 | kv_step=32 | Dropped | Same math ratio — exp scales with kv_step too; loop overhead saving is ALU-pipelined and invisible |
| 11 | FP16 softmax | Rejected | Accuracy loss not acceptable |
| 12 | Eliminate St→P transpose (~66 mov/iter) | Upper-bound measured | −14% potential on 2-seq; transpose IS on critical path; full elimination blocked by softmax reduction layout; faster transpose variant **Next** |

## ASM analysis (per kv iteration, d=64)

Instruction profile in main loop body:
- **12 dpas** (8 K-phase + 4 V-phase) @ ~32 cy XMX-pipe (hidden by 4 contexts)
- **17 exp** (SIMD16 serial chain in `online_softmax_update`) @ ~43 cy MATH-pipe
- **66 mov** for `Transpose2DMatrix(St→P)` @ ~66 cy ALU (pipelined with math)
- **16 mul** for rO rescaling @ ~16 cy ALU

The MATH pipe (exp) and XMX are roughly balanced at ~40–50 cy each.

**Why wg_size=32 doesn't help**: `kv_step/wg_local_size = 16/32 = 0` → invalid prefetch descriptor. Total thread count is unchanged regardless.

---

## Why "amortize softmax" doesn't work — corrected analysis

Items 9 (2 Q-rows) and the batch-2-kv idea were premised on softmax being a
*fixed* per-kv-iter cost. That is wrong.

`online_softmax_update(St[rows=kv_step, cols=q_step], ...)` runs `kv_step` exp
instructions on SIMD-`q_step` vectors. The exp count is `kv_step × q_step` elements
regardless of how the work is tiled:

| Scheme | DPAS ops/iter | exp instr/iter | ratio |
|--------|---------------|----------------|-------|
| baseline (q_step=16) | 12 × SIMD16 | 17 × SIMD16 | 1.0 |
| 2 Q-rows (q_step→32) | 24 × SIMD16 | 34 × SIMD16 | 1.0 |
| batch-2-kv | 24 × SIMD16 | 34 × SIMD16 | 1.0 |

The ratio is unchanged — there is **no amortization**. Both exp and DPAS scale
linearly together. Item 9 regressed (+14%) purely from register spill;
batch-2-kv would have the same ratio with cleaner GRF but still no gain.

**`-Qxcm_register_file_size=512` + item 9 confirmed no help** (measured):
512-GRF mode is real on Xe3 (2 contexts/EU, 32 KB/thread), so item 9 compiles
without spill. Result: 10.890 ms vs 9.386 ms baseline (+16%) — identical to the
256-GRF spill result (10.873 ms). The regression is not from spill; it is from
the math ratio being unchanged. Halving the context count from 4→2 provides no
latency-hiding benefit because the kernel is math-pipe-bound, not memory-latency-bound.

---

## True bottleneck and remaining opportunities

Per kv-iter the kernel runs:
- **MATH**: 17 exp × SIMD16 FP32 — costs 1 instruction per SIMD16 exp on the math pipe
- **XMX**: 12 dpas — hidden by 4-context overlap
- **ALU**: 66 mov (transpose) + 16 mul (rescale) — pipelined with math

To improve, the only levers that change the ratio are:

### ~~10. kv_step=32~~ (not worth trying — same math ratio)

**Idea**: double the KV tile size so the main loop runs half as many iterations,
amortizing per-iteration loop overhead over more DPAS work.

**Why it won't help**: `online_softmax_update(St[rows=kv_step, cols=q_step])` runs
`kv_step` exp calls on SIMD-`q_step` vectors. **Exp scales linearly with kv_step**:

| kv_step | exp/iter | loop iters (seq=3432) | exp total/seq |
|---------|----------|-----------------------|---------------|
| 16 | 17 | 215 | 3655 |
| 32 | 33 | 108 | 3564 |

Per-KV-token ratio unchanged — same dead end as items 9 and batch-2-kv. The only
saving is ~107 fewer iterations × ~25 ALU instructions (descriptor setup, counter,
branch), but ALU pipelines with MATH on the 4-context model so this is invisible.

**Register cost**: `St` grows from 16→32 GRF, `Kmat` doubles 4→8 GRF (~+20 GRF
total, fits within 256). Implementation is also non-trivial: `kv_step` must be
decoupled from `REG_K`, the transpose needs a new `[32×16]→[16×32]` overload, and
the V-phase P matrix `[16,32]` must be split into two `[16,16]` DPAS tiles.

**Conclusion**: dropped without experiment — the math ratio analysis is definitive.

---

### ~~11. FP16 softmax~~ (rejected — accuracy loss not acceptable)

Would halve exp instruction count (SIMD32 FP16 vs SIMD16 FP32 per instruction).
Rejected: FP16 softmax intermediate values can underflow/overflow in production
workloads; accuracy loss is not acceptable.

---

### 12. Eliminate explicit St→P transpose (remove ~66 mov/iter)

`Transpose2DMatrix(St, P)` converts `float[16,16] St` → `half[16,16] P` (transpose
+ fp32→fp16 conversion), generating ~66 SIMD16 mov instructions per kv iteration.
This sits strictly after `online_softmax_update` returns (not pipelined with exp),
adding serial ALU latency on the critical path.

**Upper-bound probe (measured)**: replacing `Transpose2DMatrix` with a direct
float→half cast (wrong results, perf only) gives:

| Config | Baseline | Skip-transpose | Δ |
|--------|----------|----------------|---|
| 2 seqs × 3432 | 9.389 ms | 8.084 ms | **−14%** |
| 16 seqs × 512 | 1.801 ms | 1.605 ms | −11% |
| 128 seqs × 64 | 0.715 ms | 0.708 ms | −1% |
| 15 seqs × 3840 | 85.403 ms | 74.134 ms | **−13%** |

The transpose IS on the critical path (not hidden by 4-context overlap — ALU and
MATH compete on the same thread when exp and movs run sequentially).

**Why it can't be trivially eliminated**: the existing `online_softmax_update`
reduces along the kv dimension as SIMD16 row-operations on `St[kv=16, q=16]`.
Reordering K-DPAS to produce `St_new[q=16, kv=16]` directly would require
*column-wise* reduction per q-row — 16 scalar chains of length 16, far worse than
16 SIMD16 row-ops. Swapping DPAS operands trades 66 ALU movs for 16× longer
softmax exp chains; net is likely a regression.

**Remaining viable approach**: find a faster transpose implementation. `Transpose_16x16`
uses 4 passes × 16 SIMD16 movs = 64 movs. The theoretical minimum for a 16×16
float→half transpose is also ~16–32 movs if the right GRF select patterns exist.
Investigate whether the XMX pipe or a different select pattern can reduce the mov
count below 32 while keeping the float→half downcast.

**File**: `cm_sdpa_common.hpp` (`Transpose2DMatrix` call), `cm_attention_common.hpp`
(`Transpose_16x16` implementation).

---

## Roofline (15 seqs × 3840, d=64, 16h, PTL 4xe)

- Compute peak: ~20 TFLOPS FP16 XMX
- Actual: ~10.6 TFLOPS (53% of peak)
- Memory: 3.77 GB at 68 GB/s → 56 ms BW-bound vs 85 ms actual
- Arithmetic intensity: 240 FLOP/byte; ridge point: ~295 FLOP/byte
- EU always has 4 thread contexts (large GRF mode, hardware-fixed)
- **Current ceiling**: math-pipe exp and XMX DPAS both at ~40–50 cy/iter;
  exp count cannot be reduced without accuracy loss (FP16 rejected) or algorithmic
  change (all batching strategies scale linearly); remaining candidate is
  removing the ~66-mov St→P transpose (item 12)
