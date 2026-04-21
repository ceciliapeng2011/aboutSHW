# Skill: LNL Xe2 PA Kernel Design Reference

LNL Xe2 hardware constraints and PA kernel design decisions: register budgeting, DPAS tiling, memory hierarchy trade-offs, and roofline targets.

---

## Platform Summary

| Parameter | LNL (Xe2 iGPU) |
|-----------|-----------------|
| Xe Cores | 8 |
| VE (XMX) per Xe Core | 8 |
| EUs per core | 8 |
| Threads per VE | 8 |
| Total threads (128 GRF) | 512 |
| Total threads (256 GRF) | 256 |
| Clock | 2 GHz |
| Memory BW | 102 GB/s |
| Memory Type | LPDDR5x |
| FP16 XMX peak | ~30 TFLOPS |
| FP16 XVE peak | ~7.5 TFLOPS (1/4 of XMX) |
| Scalar (no SIMD) | 1 op/clock/VE — orders of magnitude slower |
| L3 Cache | 8 MB |
| SLM per Xe Core | 128 KB |

---

## Register File

Total register file per EU: **64 KB** (512-bit/reg * 128 regs * 8 VEs / 1024).

| Mode | Flag | Regs/thread | Bytes/thread | Threads/VE | Threads/Xe Core |
|------|------|-------------|-------------|------------|-----------------|
| Standard GRF | `-Qxcm_register_file_size=128` | 128 | 8 KB | 8 | 64 |
| Double GRF | `-Qxcm_register_file_size=256` | 256 | 16 KB | 4 | 32 |

- Xe1/Xe2 support only 128 and 256 GRF modes. Xe3 adds a 512 GRF mode.
- Register width: 256-bit on Xe1, **512-bit on Xe2/Xe3** (`CM_GRF_WIDTH=512`).
- All attention kernels require 256 GRF mode (Q tiles + O accumulator + softmax state).
- CM matrix size limit: `sizeof(matrix) < 16384` bytes (strict `<`, not `<=`).
- `-Qxcm_jit_option="-abortonspill"` makes compilation fail if spill occurs; remove for kernels that intentionally exceed 256 regs.

### Register budget rule of thumb

Each register holds 64 bytes (512 bits). Convert matrix sizes:

```
regs_needed = rows * cols * sizeof(element) / 64
```

Examples at HEAD_SIZE=128 vs 256 (num_P_tiles=2, REG_M=8, REG_N=16):

| Matrix | Shape (hs128) | Regs (hs128) | Shape (hs256) | Regs (hs256) |
|--------|---------------|-------------|---------------|-------------|
| rQ `[hs/REG_K, REG_K*REG_N]` half | [8, 256] | 64 | [16, 256] | 128 |
| rO `[hs/REG_N*nPt, REG_M*REG_N]` float | [16, 128] | 128 | [32, 128] | 256 |
| rO_lo + rO_hi (split) | 2 * [8, 128] | 128 | 2 * [16, 128] | 256 |

HEAD_SIZE=256: rQ(128) + rO_lo(128) + rO_hi(128) = **384 regs** > 256 budget. Spill is unavoidable.

---

## DPAS (Xe2)

| Parameter | Value |
|-----------|-------|
| SystolicDepth | 8 |
| RepeatCount | 8 (multi-token) or 1 (decoding) |
| VNNI_WIDTH | 2 |
| REG_K | 16 (= SystolicDepth * VNNI_WIDTH) |
| REG_M | 8 (= RepeatCount) |
| REG_N | 16 (= CM_GRF_WIDTH / 32) |
| q_step | 16 (= REG_N) |
| kv_step | 16 (= REG_K) |
| num_P_tiles | 2 (= REG_N / REG_M) |

---

## Memory Hierarchy

| Level | Size | BW | Latency | Notes |
|-------|------|-----|---------|-------|
| GRF | 16 KB/thread (256 GRF) | Register speed | 0 | All DPAS operands must be in GRF |
| SLM | 128 KB/Xe Core | ~2 TB/s (est.) | ~20 cycles | Shared within workgroup; used for K/V staging in U8 path |
| L3 | 8 MB | ~500 GB/s (est.) | ~100 cycles | Can cache ~4K tokens (nkvh=2, HD=128, fp16) |
| LPDDR5x | System | 102 GB/s | ~200+ cycles | The bottleneck for memory-bound (decode) kernels |

---

## Compute Units: XMX vs XVE vs Scalar

A kernel's compute bottleneck falls into one of three categories — **XMX-bound**, **XVE-bound**, or **scalar-bound** — depending on which execution unit dominates wall time.

**Key clarification**: TFLOPS figures count **MACCs** (multiply-accumulate). One MACC = one fused multiply-add = 2 FLOP in the traditional sense, but hardware spec sheets and roofline calculations use MACC-based TFLOPS throughout.

### XMX (Matrix Extensions)

- Matrix multiply engine (`cm_dpas`). Only unit that executes DPAS instructions.
- **~30 TFLOPS FP16** (MACC-based) on LNL.
- One DPAS instruction: RepeatCount × SystolicDepth × REG_N MACCs.
- Used for: QK^T matmul, PV matmul — the core attention DPAS operations.
- Workload is XMX-bound when the DPAS pipeline is fully utilized and adding more DPAS instructions increases wall time linearly (e.g., prefill with long Q sequences).

### XVE (Vector Engine)

- SIMD ALU on each VE. Executes vector/math instructions across SIMD lanes.
- **~7.5 TFLOPS FP16** (MACC-equivalent) — **1/4 of XMX peak**.
- Simple ops (add, mul, fma, mov, type convert): **1 MACC-equivalent cycle** per SIMD instruction.
- Expensive ops cost **multiple MACC-equivalent cycles**:

| XVE Operation | Approx. Latency (MACC-equiv cycles) | Notes |
|---------------|--------------------------------------|-------|
| `cm_exp` / `cm_log` | ~4–8 | Transcendental, multi-cycle pipeline |
| `cm_div_ieee` | ~4–8 | Iterative; `cm_inv` + mul is cheaper |
| `cm_max` (reduction) | ~1 per step | But log₂(N) steps for N-wide reduce |
| `cm_sum` (reduction) | ~1 per step | log₂(N) steps |
| sub + mul (dequant) | 2 | Two simple ops |
| VNNI repack (select + assign) | ~2 | Register shuffle |

- Used for: pack/repack, data type conversion (uint8→fp16, fp16→fp32), softmax (exp, sum, div), scale/zp dequantization, VNNI layout shuffles.
- Workload is XVE-bound when vector ALU operations between DPAS calls (dequant, format conversion, softmax) dominate. This is common in quantized kernels where every K/V tile requires multi-step dequantization before DPAS can consume it.

### Scalar Operations

- **16× slower than SIMD** — a scalar op that touches one element takes the same wall time as a SIMD op that touches 16 elements.
- On Xe2, SIMD width = 16 (fp16) or 8 (fp32). A scalar loop over N elements ≈ N SIMD-equivalent cycles vs N/16 for vectorized code.
- LUT lookup with data-dependent indexing (e.g., `centroid[packed_idx & 0x0f]`) is **inherently scalar**: each lane needs a different table index, so the hardware serializes the gathers. A 16-wide gather ≈ **16 cycles** vs 1 cycle for a uniform broadcast load.
- Other scalar-bound patterns: indirect register access, loop-carried dependencies, pointer chasing.
- In TurboQuant kernels, centroid-based 4-bit dequantization is the primary scalar bottleneck: each nibble index requires an independent lookup into a 16-entry codebook.

### Practical Bottleneck Identification

| Bottleneck | Symptom | Typical Kernels |
|------------|---------|-----------------|
| **Memory** | BW utilization near peak; DPAS and XVE idle waiting for data | Single-token decode (FP16 K/V), KV cache update |
| **XMX** | DPAS pipeline saturated; adding Q tokens increases time linearly | Prefill / multi-token PA with long sequences |
| **XVE** | Vector ALU between DPAS calls dominates; DPAS has bubbles waiting for operands | Quantized decode (uint8/int4 dequant, VNNI repack) |
| **Scalar** | LUT gathers or scalar loops dominate; both DPAS and SIMD ALU underutilized | TurboQuant 4-bit centroid dequant, in-kernel Q rotation via scalar matmul |

A kernel can shift categories depending on context length and quantization:
- FP16 single-token decode at 32K: **memory-bound** (KV read dominates).
- TurboQuant single-token decode at 32K: **memory-bound at macro level**, but scalar centroid dequant and XVE repack push wall time well above the memory floor → mixed **memory + XVE/scalar**.
- Multi-token prefill at 32K: **XMX-bound** (DPAS throughput is the ceiling).

---

## Multi-Token PA Workload Dispatch (test_pa.py)

### GWS / LWS

```python
# test_pa.py
wg_size    = 16                          # threads per workgroup (hardcoded)
q_step     = CM_GRF_WIDTH // 32 = 16    # q tokens per thread (= REG_N on Xe2)
wg_seq_len = wg_size * q_step = 256     # q tokens per workgroup
wg_count   = ceil(q_len / wg_seq_len)   # workgroups along q dimension

GWS = [1,  num_heads,  wg_count * wg_size]
LWS = [1,  1,          wg_size]
```

### Dispatch mapping (pa_multi_token.cm)

```
dim0: batch        = cm_group_id(0)                           # always 1
dim1: head         = cm_group_id(1)                           # one WG per head
dim2: q position   = cm_group_id(2) = wg_id
                     cm_local_id(2) = wg_local_id (0..15)

q_start_sg = (wg_id * 16 + wg_local_id) * q_step             # thread's q start
q_len_sg   = q_step = 16                                      # q tokens this thread
kv_stop    = (wg_id + 1) * wg_seq_len + past_lens  (causal)   # triangular KV range
```

### Summary

| Parameter | Value | Derivation |
|-----------|-------|------------|
| Threads per WG | 16 | Hardcoded `wg_size` |
| Q tokens per thread | 16 | `q_step = REG_N` |
| Q tokens per WG | 256 | `16 threads * 16 q_tokens` |
| WGs per head | `ceil(q_len / 256)` | |
| Total WGs | `num_heads * ceil(q_len / 256)` | dim1 * dim2 groups |
| KV range per WG (causal) | `(wg_id+1)*256 + past_lens` | Triangular |

### Thread cooperation within a WG

**U8 path (pa_lsc_u8) — SLM-based pipeline:**
- Threads 0-7: load K from global -> dequant -> write to SLM
- Threads 8-15: load V from global -> dequant -> write to SLM
- After barrier: all 16 threads independently compute KQ and PV using shared SLM data
- 4-slot ring buffer: load[i+2], barrier, compute[i] overlap

**FP16 path (pa_kernel_lsc_prefetch_f16) — prefetch-based pipeline:**
- 16 threads cooperatively prefetch next K/V tiles (each thread prefetches 1 row)
- Each thread independently loads its own K/V tiles via `cm_load`, computes KQ and PV
- No SLM used; relies on L3 cache hits from cooperative prefetch

**Both paths:** each thread maintains its own private `rQ`, `rO_lo`, `rO_hi`, `cur_max`, `cur_sum`. No sharing of accumulation state across threads.

### Key design difference vs OCL micro kernel

CM PA uses independent threads — each thread holds the entire rQ + rO in GRF for the full kernel lifetime. The OCL `sdpa_micro__prefill` uses cooperative SGs — 32 SGs share Q via SLM, tile head_size across SG rows, and communicate S via SLM between KQ and VS phases.

| Design choice | CM PA (current) | OCL micro kernel |
|---------------|----------------|-----------------|
| Q storage | GRF (128 regs, permanent) | Q_slm (0 GRF during VS) |
| O accumulation | 1 thread owns full head_size | 8 SGs tile head_size, each owns 32 dims |
| KQ→VS handoff | S stays in GRF → P in GRF | S written to S_slm → freed from GRF |
| Peak registers | rQ + rO + temps = **~441** | max(KQ, VS) = **~180** |
| Register peak model | **Sum** (all live simultaneously) | **Max** (phases don't overlap in GRF) |

The fundamental insight: CM PA's register peak is `rQ + rO + temps` because KQ and VS state overlap in GRF. The micro kernel's peak is `max(KQ_state, VS_state)` because SLM decouples them.

### Register pressure vs dispatch (hs256)

The q_step = 16 means `num_P_tiles = REG_N / REG_M = 16 / 8 = 2`, doubling rO size.

**Option A: Reduce num_P_tiles (dispatch change only)**

| num_P_tiles | q per thread | rQ regs | rO regs | Total persistent | Spill? |
|-------------|-------------|---------|---------|-----------------|--------|
| 2 (current) | 16 | 128 | 2 * 128 = 256 | 386 | ~12 KB |
| 1 (proposed) | 8 | 128 | 128 | 258 | ~0 |

With `num_P_tiles=1`: rO is `[head_size/REG_N, REG_M*REG_N]` = `[16, 128]` float = 8192 bytes = 128 regs. Single matrix, no split needed, under the 16384-byte CM limit. Total = rQ(128) + rO(128) + cur_max(1) + cur_sum(1) = 258 regs.

**Critical issue: 50% DPAS waste.** REG_N=16 is hardware-fixed. KQ DPAS always produces 16 output columns regardless of how many q tokens PV consumes. With num_P_tiles=1, KQ computes 16 columns but PV only uses 8 → 50% of KQ DPAS is wasted. System-wide DPAS to process 16 q tokens: current = 64 DPAS, num_P_tiles=1 (2 threads) = **96 DPAS (+50%)**. This is costly for a compute-bound kernel.

Trade-offs:
- Pro: eliminates ~12 KB spill, removes rO split complexity, removes double V reads
- Pro: single rO matrix under 16384 bytes — no split needed
- Con: **50% KQ DPAS waste**, +50% total system DPAS overhead
- Con: 2x wg_count (2x dispatch overhead), halved SLM/prefetch amortization
- Con: `wg_seq_len` drops from 256 to 128, more WGs needed, may hurt causal mask efficiency

**Option B: Move Q to SLM (FP16 only, inspired by micro kernel)**

| Layout | rQ regs | rO regs | Total persistent | Spill? |
|--------|---------|---------|-----------------|--------|
| Current (rQ in GRF) | 128 | 256 | 386 | ~12 KB |
| Q in SLM, rO split | 0 | 256 | 258 | ~0 |
| Q in SLM + num_P_tiles=1 | 0 | 128 | 130 | 0 (could use 128-GRF!) |

**Critical issue: SLM budget.** Each of the 16 threads has unique Q data (different q_start positions) — Q cannot be shared. SLM cost = 16 threads × `head_size/REG_K × REG_K × REG_N × sizeof(half)` = 16 × 8 KB = **128 KB**.

| Path | K_SLM (ring×4) | V_SLM (ring×4) | Q_SLM (16 threads) | Total | Limit | Fits? |
|------|---------------|----------------|---------------------|-------|-------|-------|
| U8, hs256 | 32 KB | 32 KB | 128 KB | **192 KB** | 128 KB | **No** |
| FP16, hs256 | 0 | 0 | 128 KB | 128 KB | 128 KB | Barely |

- **U8 path: impossible** (192 KB > 128 KB SLM).
- **FP16 path: exactly fills SLM.** But this forces 1 WG per Xe Core (128 KB per core), **halving occupancy** from 2 WGs to 1 WG.

**Option C: Reload Q from L3 cache (recommended)**

Instead of keeping rQ permanently in GRF (128 regs) or putting it in SLM, reload Q tiles from global memory each KQ iteration, relying on L3 cache hits:

```cpp
// Current: rQ permanent in GRF (128 regs)
dpas(St, rQ[ri].format<int32_t>(), Kmat);

// Proposed: Qtile loaded per iteration from L3 (~4 regs transient)
cm_load(Qtile, b2dQ.set_block_x(ri*REG_K));  // Q from L3
dpas(St, Qtile.format<int32_t>(), Kmat);
```

| Metric | Value |
|--------|-------|
| rQ regs freed | 128 → 4 (transient) = **124 regs saved** |
| Peak pressure | ~441 → ~315 (with split rO) |
| Spill | ~12 KB → ~2-3 KB |
| Q size per thread | 8 KB (trivially fits in 8 MB L3) |
| L3 re-read overhead | 2048 iters × 8 KB × 32 threads = 512 MB. At ~500 GB/s = ~1 ms vs ~500 ms kernel = **<0.2%** |
| L3 load latency | ~100 cycles, overlaps with Kmat load (pipelined) |
| SLM cost | **None** |
| Occupancy impact | **None** |
| DPAS overhead | **None** |
| U8 + FP16 | **Both** |

**Implementation changes needed (Option C):**
1. `cm_pa_xe2.hpp`: Remove `rQ` matrix declaration. Keep `b2dQ` descriptor alive beyond initial Q load. In KQ inner loop, replace `rQ[ri].format<int32_t>()` with a transient `Qtile` loaded via `cm_load<lsc::Transpose>(Qtile, b2dQ.set_block_x(ri*REG_K/2))` per iteration.
2. Same change in both `pa_lsc_u8` and `pa_kernel_lsc_prefetch_f16`.
3. For `pa_lsc_u8`: also change `ugemm_KQ` in `cm_attention_common.hpp` to accept a `b2dQ` descriptor instead of `matrix_ref<half> Qt`.
4. No host-side changes needed (same dispatch, same GWS/LWS).

---

## Roofline (Three-Way)

The classical roofline compares arithmetic intensity against memory bandwidth. For Xe2, the model must account for **three ceilings**: memory BW, XMX throughput, and XVE throughput.

All TFLOPS figures are **MACC-based** (1 MACC = 1 fused multiply-add). When counting ops for roofline, count MACCs, not individual mul/add.

### Ridge Points

```
XMX ridge  = 30 T-MACC/s / 102 GB/s  ≈ 294 MACC/byte
XVE ridge  = 7.5 T-MACC/s / 102 GB/s ≈  74 MACC/byte
```

### Classification

```
AI = MACCs / Bytes_transferred

If AI < 74:            memory-bound  (both XMX and XVE idle, waiting for data)
If 74 < AI < 294:      depends on op mix — XVE-heavy kernels hit XVE ceiling first
If AI > 294:           XMX-bound     (DPAS pipeline is the bottleneck)
```

**Important**: the classical roofline only counts MACCs that map to DPAS. Quantized kernels have substantial **non-DPAS work** (dequant, repack, softmax, LUT gather) that executes on XVE or scalar units. These must be counted separately:

- XVE simple ops (add, mul, fma): 1 MACC-equivalent cycle per SIMD instruction
- XVE transcendentals (exp, log, div): 4–8 MACC-equivalent cycles each
- Scalar ops: **16× SIMD cost** — one scalar element-wise op costs the same wall time as one full SIMD16 vector op

A kernel can appear memory-bound by DPAS AI but actually spend most wall time in XVE/scalar dequantization — the "XVE/scalar overhead gap" between the memory floor and actual execution time.

### Per-Kernel Bottleneck

| Kernel | AI (MACC/byte) | Bottleneck | Key Metric |
|--------|----------------|-----------|------------|
| Single-token decode (FP16 KV) | < 20 | Memory | BW utilization vs 102 GB/s |
| Single-token decode (TQ 4-bit) | < 20 (DPAS only) | Memory + scalar | BW util + scalar dequant overhead |
| Multi-token prefill | > 294 | XMX | DPAS utilization vs 30 TFLOPS |
| KV cache update | < 10 | Memory | BW utilization |

### Multi-token PA roofline estimate (XMX-bound regime)

```
roofline_ms = 293.20 * (head_size/128) * (seq_len/32768)^2 * (num_heads/32)
```

### Single-token decode estimate (memory + XVE/scalar regime)

```
mem_floor_ms   = total_kv_bytes / (102 GB/s)
xve_cycles     = count of SIMD vector instructions × latency_per_op
scalar_cycles  = count of scalar ops × 16  (scalar = 16× SIMD cost)
actual_ms      = max(mem_floor_ms, (xve_cycles + scalar_cycles) / clock / num_VEs)
```

For detailed TurboQuant cycle analysis, see `turboquant_single_token_analysis.md`.
