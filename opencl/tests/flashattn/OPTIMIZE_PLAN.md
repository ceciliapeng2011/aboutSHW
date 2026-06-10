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

## GRF occupancy analysis (post items 1–5)

After items 1–5, the kernel uses **163 GRFs/thread** with 256 available.
Since `floor(256/163) = 1`, only **1 thread context per EU** can be live —
the EU stalls completely on any load or DPAS latency with no other work to hide it.
Reducing to ≤128 GRF would allow 2 contexts; ≤85 GRF allows 3 contexts.

GRF budget breakdown:
- `rO`: `float[padded_head_size/REG_N * num_P_tiles * REG_M * REG_N]`
  = `float[4 × 2 × 8 × 16]` = 4096 bytes = **128 GRFs** (FP32, persistent across kv loop)
- `rQ`: `half[padded_head_size/REG_K × REG_K × REG_N]`
  = `half[4 × 16 × 16]` = 2048 bytes = **64 GRFs** (FP16, loaded once, held all loop)
- Everything else (cur_max, cur_sum, St, P, Kmat, Vmat, descriptors): ~22 GRFs

---

### 6. Split head dimension — process head in 2 passes (halve rQ + rO)

**Problem:** `rO` (128 GRF) + `rQ` (64 GRF) dominate. Both scale with `head_size`.

**Fix:** Run the kv loop **twice**, each time covering only `head_size/2 = 32` output
channels (`d_start = 0` then `d_start = head_size/2`). Each pass uses:
- `rO`: 64 GRF (half the head channels)
- `rQ`: 32 GRF (half the head chunks)
- Total: ~83 GRF → **3 thread contexts per EU**

Cost: 2× kv loop body executions, but KV data accessed in pass 1 stays in L1/L2 for
pass 2 (3840 × 64 × 2 × 2 = 0.96 MB per WG-head, fits L2 on Xe3).
Net expected win: 2–3× if currently latency-bound, less if already BW-bound.

`rO` precision stays FP32 throughout.

**Implementation sketch:**
```cpp
for (int d_start = 0; d_start < head_size; d_start += head_size/2) {
    // reset rO, cur_max, cur_sum
    // load rQ for [d_start .. d_start+head_size/2]
    // kv loop over [d_start .. d_start+head_size/2] channels
    // store output for [d_start .. d_start+head_size/2]
}
```
Requires passing `d_start` offset into all block_2d_desc base addresses for K, V, Q, O.

**Files:** `cm_sdpa_common.hpp` (`sdpa_kernel_lsc_prefetch`), no Python changes needed.

---

### 7. Reload rQ on-demand, drop persistent rQ registers

**Problem:** `rQ` (64 GRF) is loaded once before the loop and held across all kv
iterations. It occupies 64 GRFs that could otherwise be used for a second thread context.

**Fix:** Eliminate the persistent `rQ` matrix. Inside each kv iteration's K inner loop,
reload the Q chunk needed for DPAS `ri` from L1 (Q was already loaded into L1 during
the initial load; re-accessing it is an L1 hit, ~20 cycles latency vs the 64 GRF saved).

Combined with no other change: 163 − 64 = **99 GRF** — still > 128, not enough alone.
Combined with item 6 (head split): already addressed.
Combined with reducing rO (not desired) — not pursued per constraints.

**Net verdict:** Only useful if combined with another reduction to cross the 128 GRF
threshold. Most likely useful as a follow-on after item 6.

**Files:** `cm_sdpa_common.hpp` (`sdpa_kernel_lsc_prefetch`).

---

## Status and priority

| # | Change | Status | Result |
|---|--------|--------|--------|
| 1 | Mapping array replaces `need_wg_mapping` | Done | ~0% GPU perf, cleaner code |
| 2 | V prefetch moved to K phase | Done | −1.7% to −6.2% |
| 3 | Load/DPAS double-buffer (K) | Tried, reverted | Regression — compiler already schedules well for d=64 |
| 4 | Pre-loop warm-up prefetch | Done (with 2) | Included in items 2+4+5 result |
| 5 | Peel kv_pos=0 / unified PV path | Done (with 2) | Included in items 2+4+5 result |
| 6 | Split head dimension (2 passes, halve rO+rQ) | **Next** | Target: 2–3× occupancy |
| 7 | Reload rQ on-demand | After 6 if needed | Only useful combined with 6 |

Roofline (15 seqs × 3840, d=64, 16h, PTL 4xe):
- Compute peak: ~20 TFLOPS FP16 XMX
- Actual: ~10.6 TFLOPS (53% of peak)
- Memory: 3.77 GB at 68 GB/s → 56 ms BW-bound vs 85 ms actual
- Arithmetic intensity: 240 FLOP/byte; ridge point: ~295 FLOP/byte
- **Conclusion: slightly memory-bound, but 47% of compute peak lost to EU stalls (low occupancy)**
