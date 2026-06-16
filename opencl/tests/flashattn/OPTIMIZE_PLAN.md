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

### 1. WG dispatch: precomputed mapping array vs in-kernel `need_wg_mapping` while-loop

**Problem:** The `need_wg_mapping=1` branch runs a while-loop over `cu_seqlens` inside
every thread of every WG — O(num_sequences) SVM reads per thread, all redundant and
with loop-carried branch dependencies.

**Fix A — precomputed mapping (aboutSHW):** The host pre-computes a flat
`int32[2 × wg_count]` array before each inference:

```python
mapping = []
for i, (seq_start, seq_end) in enumerate(zip(cu_seqlens, cu_seqlens[1:])):
    seq_len = seq_end - seq_start
    for k in range((seq_len + wg_seq_len - 1) // wg_seq_len):
        mapping += [int(seq_start) + k * wg_seq_len, i]
wg_count = len(mapping) // 2
```

The kernel reads 2 scalars at O(1) with no branch:

```cpp
int block_start_pos = wg_mapping[wg_id * 2];
int seq_id          = wg_mapping[wg_id * 2 + 1];
int kv_start        = cu_seqlens[seq_id];
int kv_seq_len      = cu_seqlens[seq_id + 1] - kv_start;
int q_start         = block_start_pos + wg_local_id * q_step;
```

Also eliminates the dead-code guard (`wg_base > wg_id` can never be true) and the
`wg_count` tensor-arithmetic bug in Python.

**Fix B — `mem_lock` + while-loop (OV):** The original OV host code called
`stream.finish()` + blocking `copy_to` to read `cu_seqlens` onto the CPU
(in `get_mask_seqlens_from_memory`). On PC, `cu_seqlens` is a user-provided
`usm_host` tensor; `mem_lock` returns a direct CPU pointer with zero GPU sync and
zero copy. Switching `dispatch_data_func` to use `mem_lock` (same as `read_i32_input`
in `paged_attention.cpp`) eliminates the host–device sync barrier entirely. The
while-loop in the kernel then costs at most a few iterations for typical PC workloads
(≤8 sequences in VLM use), and requires no per-inference buffer upload.

**A/B kernel measurement** (`SDPA_AB=1 python cmfla.py`, PTL 4xe):

| Case | precomp (ms) | while-loop (ms) | delta |
|------|-------------|-----------------|-------|
| seq=8192 sub=8192 h=128 (1 seq)   | 58.727 | 58.585 | +0.2% (tie)    |
| seq=8192 sub=8192 h=80  (1 seq)   | 26.657 | 26.185 | +1.8% (noise)  |
| seq=8192 sub=1024 h=80  (8 seqs)  |  3.475 |  3.432 | +1.3% (noise)  |
| seq=8192 sub=64   h=80  (128 seqs)|  0.831 |  2.008 | **−58.6% win** |
| seq=6864 sub=3432 h=64  (2 seqs)  |  7.892 |  7.887 | +0.1% (tie)    |

Kernel-level winner: precomputed mapping at 128 sequences (−58.6%); tie otherwise.

**Tradeoffs:**

| Factor | Precomputed (aboutSHW) | While-loop + mem_lock (OV) |
|--------|----------------------|---------------------------|
| Kernel dispatch cost | O(1), 2 reads | O(num_seqs), ≤8 iters on PC |
| Per-inference CPU work | build mapping array | ~0 (just mem_lock ptr) |
| Per-inference GPU upload | `wg_mapping` buffer | none |
| Host–device sync barrier | none (in aboutSHW) | none (after mem_lock fix) |
| Best for | many short seqs (≥16) | few seqs (≤8), PC VLM |

**Decision: both repos use the while-loop.**

The kernel-level win of the precomputed mapping (−58.6% at 128 sequences) is real but
the target platform is PC VLM (≤8 sequences). With the host barrier eliminated via
`mem_lock`, the while-loop path has zero overhead in both repos and avoids the extra
per-inference `wg_mapping` buffer upload entirely.

- **OV (`vl_sdpa_opt.cpp`):** `stream.finish()` + `copy_to` replaced by `mem_lock`
  in `dispatch_data_func`. `VLSDPARuntimeParams`, `update_rt_params`,
  `get_mask_seqlens_from_memory` deleted. `need_wg_mapping` scalar passed as before.
- **aboutSHW (`cm_sdpa_vlen.cm` + `cmfla.py`):** aligned to OV. `wg_mapping` array
  removed; `need_wg_mapping` scalar computed host-side and passed as 5th argument.

**Files:** `cm_sdpa_vlen.cm`, `cmfla.py`, `vl_sdpa_opt.cpp`, `vl_sdpa.cpp`,
`vl_sdpa_inst.h`.

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
| 1 | Both repos: `need_wg_mapping` while-loop + `mem_lock` (no GPU sync) | Done | Kernel: while-loop costs ≤8 iters on PC VLM. OV: sync barrier removed. Precomputed mapping faster at 128+ seqs but not target workload |
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
| 12 | Eliminate St→P transpose (~66 mov/iter) | Upper-bound measured | −14% potential on 2-seq; transpose IS on critical path; full elimination blocked by softmax reduction layout |
| 13 | Fold log2e into Q pre-scale (`qscale = scale_factor * log2e`) | Done | Removes 16 mul/tile from softmax critical path; St lands in log2 domain so cm_exp needs no per-element ×log2e |
| 14 | KV blocking (`KV_BLK=2`): amortize rO rescale over 2 tiles | Done | rO rescale (64 mul) runs once per 2 kv tiles instead of once per tile; unlike Q-row or kv_step doubling, exp count is unchanged (still per token) |
| 15 | Tree reduction in `online_softmax_update_tree` | Done | Max/sum reduction depth log₂(BLK_ROWS)=5 vs linear chain depth 31 for BLK_ROWS=32; shorter loop-carried dependency chain |
| 16 | `transpose_St_to_P_half`: narrow float→half before shuffling | Done | Cast float→half first, then run 4-pass GRF shuffle on 16-bit data; halves the data-path width of each select mov (~32 effective data-movement ops vs ~64) |
| 13–16 combined | Items 13–16 (commit 0621405) | **Done — 9.25 ms → 7.654 ms (−17%)** | 2-seq × 3432 target; best improvement so far |

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

### 13. Fold log2e into Q pre-scale

**Problem:** `online_softmax_update` computes `St[r] = cm_exp((St[r] - new_max) * log2e)`.
The `* log2e` is 16 SIMD16 FP32 multiplies per kv row, serialized on the ALU after
the subtract, on the softmax critical path.

**Fix:** Pre-multiply Q at load time by `qscale = scale_factor * log2e` (a compile-time
constant). `St = K @ Q^T` then already represents log2-scaled dot products. The softmax
reduces to `cm_exp(St[r] - new_max)` with no per-element multiply. The math is
identical — only the constant folded into Q changes.

**Savings:** 16 mul/tile removed from the softmax critical path (once per kv tile, not
amortized — a true reduction in work).

**File:** `cm_sdpa_common.hpp` (`sdpa_kernel_lsc_prefetch`, Q load section).

---

### 14. KV blocking (`KV_BLK=2`): amortize rO rescale over multiple tiles

**Problem:** The rO rescale (`rO[t] *= max_comp`) runs once per kv tile — 8 (tiles) ×
8 (REG_M rows) = 64 SIMD16 multiplies per tile, serialized on the ALU.

**Why this works when items 9/10 didn't:** Items 9 and 10 tried to amortize by
increasing Q-rows or kv_step. Both cause exp count to scale linearly with the tile
size — no ratio improvement. Here, `BLK_ROWS = KV_BLK × kv_step` increases the exp
count proportionally (32 exp rows per block vs 16), but the rO rescale is done
**once per block** regardless of KV_BLK. The rO rescale is *not* a per-KV-token cost
— it is a per-online-softmax-update cost. Doubling block size doubles the KV tokens
per rescale, halving the amortized rescale cost per token.

**Savings:** 64 mul per 2 tiles → 32 amortized mul/tile. 32 mul/tile saved, off the
serial ALU path.

**File:** `cm_sdpa_common.hpp` (outer loop restructured to `kv_base += BLK_ROWS`).

---

### 15. Tree reduction in `online_softmax_update_tree`

**Problem:** With `BLK_ROWS=32`, the max/sum reduction in `online_softmax_update`
runs a linear chain of 32 `cm_max`/`cm_add` calls. Dependency depth = 31 — each
instruction waits for the previous result.

**Fix:** `online_softmax_update_tree` folds pairs into a scratch buffer and reduces
with a balanced binary tree: depth = log₂(32) = 5. The compiler can schedule
instructions within each independent pair freely.

**Constraint:** Requires `rows` to be a power of two (enforced by `static_assert`).
`BLK_ROWS = KV_BLK × kv_step` with `KV_BLK` a power of two satisfies this for
all practical values (KV_BLK=1,2,4).

**File:** `cm_sdpa_common.hpp` (new function `online_softmax_update_tree`).

---

### 16. `transpose_St_to_P_half`: narrow float→half before GRF shuffle

**Problem:** The generic `Transpose_16x16<float,half>` runs the 4-pass `select<2,1,8,2>`
shuffle on 32-bit float data. Each select moves 16 elements × 4 bytes = 512 bytes/row,
so the ALU data-path works at full 32-bit width throughout all 4 passes (64 SIMD16 movs
total), then converts to half in the final assignment.

**Fix:** `transpose_St_to_P_half` casts float→half first (16 SIMD16 narrow-converts,
~16 cycles), then calls `Transpose_16x16<half,half>`. Each select now moves
16 elements × 2 bytes = 256 bytes/row — half the data-path pressure. Total cost:
~16 (cast) + 64 (half-width shuffle) ≈ 80 half-width-equivalent ops vs 64
full-width ops — roughly neutral in instruction count but narrower data movement
may reduce ALU stall time depending on GRF bank conflicts.

**Note:** The net benefit of this item alone is uncertain without measurement; it
is included in commit 0621405 together with items 13–15.

**File:** `cm_sdpa_common.hpp` (new function `transpose_St_to_P_half`).

---

---

## Item 15 applied to PageAttention (`cm_pa_xe2.hpp`)

Item 15 (`online_softmax_update_tree`) was ported to the PA kernel and measured on
PTL 4xe (`seq_len=2558, num_heads=32, num_kv_heads=8, head_size=128, sparse_block_sz=256`).
Control: `PA_AB=1 python test_pa.py`. Results are avg latency (ms) with delta vs base (linear softmax).

| KV cache | density req/eff | base (ms) | tree (ms) | delta |
|----------|-----------------|-----------|-----------|-------|
| FP16     | 1.00 / 1.00     | 4.065     | 4.447     | **+9.4% regression** |
| FP16     | 0.33 / 0.47     | 1.793     | 1.932     | **+7.8% regression** |
| U8 by-token   | 1.00 / 1.00 | 6.142  | 6.042     | **−1.6% win** |
| U8 by-token   | 0.33 / 0.47 | 2.335  | 2.305     | **−1.3% win** |
| U8 by-channel | 1.00 / 1.00 | 6.314  | 6.009     | **−4.8% win** |
| U8 by-channel | 0.33 / 0.47 | 2.414  | 2.334     | **−3.3% win** |

**Why tree softmax hurts FP16 but helps INT8:**

- **FP16 baseline is already optimized** with `first_active`/`ugemm_PV0`: the very first
  KV tile skips the rO rescale entirely (DPAS with `acc=0`). The tree reduction adds a
  scratch `matrix<float, rows/2, cols>` into the register file, increasing register
  pressure in a kernel that is already at the GRF limit. This spills or tightens
  scheduling, causing ~9% regression.

- **INT8 baseline has dequantization overhead** (per-row `uint8→half` unpack + scale/zp
  multiply for every K and V tile). This raises total arithmetic cost, so the fixed cost
  of the tree scratch becomes a smaller fraction of per-tile work. Meanwhile, the shorter
  dependency chain (depth log₂(16)=4 vs linear depth 15) helps latency-hide the remaining
  softmax ops behind the dequant memory loads, yielding a 1–5% win.

**Default in `test_pa.py`:** linear softmax (`CMPA_USE_TREE_SOFTMAX=0`) for FP16,
tree softmax (`CMPA_USE_TREE_SOFTMAX=1`) for INT8 (by-token and by-channel).
Override via environment variable `CMPA_USE_TREE_SOFTMAX=0|1`.

---

## Item 1 A/B: precomputed `wg_mapping` vs OV-style in-kernel `while`-loop

Measured on PTL 4xe with `SDPA_AB=1 python cmfla.py`.
Control: `USE_PRECOMPUTED_MAPPING=1` (item-1, host-precomputed flat array) vs `=0` (OV-style
while-loop scanning `cu_seqlens` per thread).

| Case | precomp (ms) | while-loop (ms) | delta |
|------|-------------|-----------------|-------|
| seq=8192 sub=8192 h=128 (28h/4kvh) | 58.727 | 58.585 | +0.2% (tie) |
| seq=8192 sub=8192 h=80  (16h/16kvh)| 26.657 | 26.185 | +1.8% (noise) |
| seq=8192 sub=1024 h=80  (8 seqs)   |  3.475 |  3.432 | +1.3% (noise) |
| seq=8192 sub=64   h=80  (128 seqs) |  0.831 |  2.008 | **−58.6% win** |
| seq=6864 sub=3432 h=64  (2 seqs)   |  7.892 |  7.887 | +0.1% (tie) |

**Why:** With 128 sequences every thread scans 128 `cu_seqlens` entries (128 SVM
reads with branch-dependent loop-carried deps) before finding its WG's sequence.
With precomputed mapping it reads exactly 2 scalars at a known offset. On single-
or few-sequence workloads the while-loop terminates after 1–2 iterations and the
difference is noise.

**GPU-kernel conclusion:** precomputed mapping wins at the kernel level (−58.6% on
128-sequence workloads). `cm_sdpa_vlen.cm` in aboutSHW keeps the precomputed path;
`wg_mapping` is always present.

**Host-overhead re-analysis for OV:** The original `get_mask_seqlens_from_memory`
in OV called `stream.finish()` (full GPU sync stall) + `copy_to` (blocking D2H DMA)
before every inference to read `cu_seqlens` onto the CPU. This makes the precomputed
option worse in OV: it would add a CPU→GPU upload of the `wg_mapping` array on top
of that barrier. The while-loop avoids the extra upload.

**Root cause and fix (applied to OV):** The barrier existed because `copy_to` always
does a blocking readback regardless of allocation type. Switching to `mem_lock`
(already used by `read_i32_input` in `paged_attention.cpp`) eliminates the barrier
on PC: `cu_seqlens` is a user-provided input tensor allocated as `usm_host`, so
`mem_lock::lock()` returns `_buffer.get()` directly — zero GPU sync, zero copy.

**OV decision: keep while-loop + use `mem_lock`.**
- Target platform is PC with small num_seqs (≤8 in typical VLM use); while-loop
  terminates in ≤8 iterations — negligible cost.
- No per-inference buffer upload needed.
- `stream.finish()` + `copy_to` removed from `vl_sdpa.cpp` (replaced by `mem_lock`
  inline in `dispatch_data_func`). `VLSDPARuntimeParams`, `update_rt_params`, and
  `get_mask_seqlens_from_memory` all deleted from `vl_sdpa_opt.cpp`/`vl_sdpa_inst.h`.

**Final decision: both repos use while-loop + `mem_lock`.**
PC VLM target has ≤8 sequences; while-loop terminates in ≤8 iterations and the
`cu_seqlens` read via `mem_lock` on `usm_host` is free. Precomputed mapping wins
−58.6% at 128 sequences but that is not the target workload, and the extra
per-inference buffer upload cancels any kernel gain. `cm_sdpa_vlen.cm` in aboutSHW
now uses the same `need_wg_mapping` while-loop as OV; `wg_mapping` array removed.

---

## Roofline (15 seqs × 3840, d=64, 16h, PTL 4xe)

- Compute peak: ~20 TFLOPS FP16 XMX
- Actual: ~10.6 TFLOPS (53% of peak)
- Memory: 3.77 GB at 68 GB/s → 56 ms BW-bound vs 85 ms actual
- Arithmetic intensity: 240 FLOP/byte; ridge point: ~295 FLOP/byte
- EU always has 4 thread contexts (large GRF mode, hardware-fixed)
- **After items 13–16** (commit 0621405): 9.25 ms → **7.654 ms (−17%)** on 2-seq × 3432;
  best single-commit improvement in this optimization campaign
