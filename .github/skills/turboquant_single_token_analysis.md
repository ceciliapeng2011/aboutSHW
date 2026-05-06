# Skill: TurboQuant Single-Token PA — Cycle Analysis & Optimization

Performance analysis of `pa_single_token_turboquant.cm` on LNL Xe2, with precise per-phase cycle budgets and optimization roadmap. Depends on platform constants from `lnl_xe2_pa_kernel_design.md`.

---

## Kernel Overview

File: `src/plugins/intel_gpu/src/graph/impls/cm/pa_single_token_turboquant.cm`

Single-token (decode) paged attention with TurboQuant 4-bit KV cache compression. Each thread processes one KV partition (256 tokens) for one group of Q heads. The kernel has five phases:

1. **Q rotation** — scalar matmul `Q_rot = Q × tq_q_t`
2. **KQ** — 4-bit centroid dequant of K + DPAS `Q × K^T`
3. **Softmax** — partition-local softmax of attention logits
4. **PV** — uint8 dequant of V + DPAS `P × V`
5. **Output** — normalize by partition sum, write partial output + LSE

---

## Reference Configuration

Qwen3-like model: HEAD_SIZE=128, 32 Q heads, 8 KV heads, batch=1, context=32K.

| Parameter | Value | Derivation |
|-----------|-------|------------|
| Q_head_chunk_size | 4 | GQA ratio 32/8 = 4, fits ≤ 8 MaxRepeatCount |
| REG_M (RepeatCount) | 1 | Single-token decode |
| REG_N | 16 | Xe2 CM_GRF_WIDTH / 32 |
| REG_K | 16 | SystolicDepth(8) × VNNI_WIDTH(2) |
| KV_PARTITION_SIZE | 256 | = KV_BLOCK_SIZE (xattn path) |
| KV_STEP | 16 | = REG_K |
| kv_steps/partition | 16 | 256 / 16 |
| ri iterations (HEAD_SIZE/REG_K) | 8 | 128 / 16 |
| Partitions (32K) | 128 | 32768 / 256 |
| Total threads | 1024 | 1 × 8 kv_heads × 128 partitions |
| HW threads (256-GRF) | 256 | 8 cores × 8 VE × 4 threads |
| Waves | 4 | 1024 / 256 |

### Per-Token KV Data Size

| Cache | Per-token/head bytes | Layout |
|-------|---------------------|--------|
| K (4-bit packed + fp16 norm) | 66 B | `HEAD_SIZE*4/8 + 2 = 66` |
| V (uint8 + fp16 scale/zp) | 132 B | `HEAD_SIZE*1 + 4 = 132` |
| **Total K+V** | **198 B** | 2.6× compression vs FP16 (256+256=512 B) |

---

## Per-Thread Cycle Budget (One Partition = 256 Tokens)

### Phase 1: Q Rotation (lines 87–98)

```cpp
for (int qi = 0; qi < Q_head_chunk_size; qi++)          // 4
    for (int j = 0; j < HEAD_SIZE; j++)                  // 128
        for (int i = 0; i < HEAD_SIZE; i++)              // 128
            acc += (float)Qmat[qi][i] * (float)tq_q_t[i * HEAD_SIZE + j];
```

- **65,536 scalar MACCs** (4 × 128 × 128).
- All MACCs are scalar: `tq_q_t[i*HEAD_SIZE+j]` is pointer-indexed, no SIMD possible.
- Scalar cost = 16× SIMD → **1,048,576 SIMD-equiv cycles**.
- This is **identical across all 128 partition threads** for the same (seq, head) — pure redundancy.

### Phase 2: KQ — K Centroid Dequant + DPAS (lines 149–191)

Per (kv_step, ri) tile — inner dequant loop (lines 164–178):

```cpp
for (int p = 0; p < REG_K/2; p += 2)           // 4 iterations
    for (int n = 0; n < REG_N; n++)             // 16 iterations
        // 4 scalar LUT lookups: centroid_f[nibble & 0x0f], centroid_f[nibble >> 4] × 2 rows
        // 4 scalar muls: deq * knorm_vec[n]
```

- Per (kv_step, ri): 4 × 16 × (4 lookups + 4 muls) = **512 scalar ops × 16 = 8,192 SIMD-equiv cycles**.
- DPAS: 1 instruction per (kv_step, ri) = RepeatCount(4) × SystolicDepth(8) × REG_N(16) = 512 MACCs.
- `rS += rS_data`: 1 SIMD add (4×16 float = 4 SIMD ops).

Total over partition (16 kv_steps × 8 ri):

| Component | Count | SIMD-equiv cycles |
|-----------|-------|-------------------|
| Centroid LUT + knorm mul (scalar) | 16 × 8 × 512 scalar ops | **1,048,576** |
| DPAS | 128 instructions | ~3,840 XMX cycles |
| rS accumulate | 128 × 4 SIMD ops | 512 |

### Phase 3: Softmax (lines 200–228)

Per Q head (×4 heads):

| Op | SIMD instructions | Latency (MACC-equiv) |
|----|-------------------|----------------------|
| `cm_mul` (scale_factor) | 16 | 16 |
| `cm_max` reduction (256 elements) | ~8 | 8 |
| sub + mul(log2e) | 2 × 16 | 32 |
| `cm_exp` | 16 | 16 × ~6 = 96 |
| fp32→fp16 convert | 16 | 16 |
| `cm_sum` (fp32) | ~8 | 8 |
| `cm_log` | 1 | ~6 |
| `cm_sum` (fp16 Pmat) | ~8 | 8 |
| **Per Q head** | | **~190** |

Total: 4 × 190 = **760 SIMD-equiv cycles**.

### Phase 4: PV — V Dequant + DPAS (lines 260–316)

Per (kv_step, ri) tile — V dequant (lines 276–291):

| Op | SIMD instructions | Notes |
|----|-------------------|-------|
| uint8→fp16 convert (`VmatNormal = Vt_quant`) | 16 | REG_K × REG_N elements / SIMD16 |
| sub zp (16 rows) | 16 | `VmatNormal[r] - temp_zp[r]` (broadcast) |
| mul scale (16 rows) | 16 | `VmatNormal[r] * temp_scale[r]` (broadcast) |
| VNNI repack (2 selects) | ~4 | Register shuffles |
| **Per (kv_step, ri)** | **52** | All SIMD — no scalar |

DPAS: 1 instruction per (kv_step, ri) = 512 MACCs.

Total over partition (16 kv_steps × 8 ri):

| Component | SIMD-equiv cycles |
|-----------|-------------------|
| V dequant + repack (SIMD) | 16 × 8 × 52 = **6,656** |
| DPAS | 128 instructions = ~3,840 XMX cycles |

### Phase 5: Output (lines 320–334)

- `cm_div_ieee`: 4 Q_heads × 8 tiles × ~6 cycles = **192 SIMD-equiv cycles**.
- SVM writes: negligible vs compute.

### Per-Thread Total

| Phase | SIMD-equiv cycles | DPAS MACCs | Dominant Unit |
|-------|-------------------|------------|---------------|
| Q rotation | **1,048,576** | 0 | Scalar |
| K centroid dequant | **1,048,576** | 0 | Scalar |
| KQ DPAS | 512 | 65,536 | XMX |
| Softmax | 760 | 0 | XVE |
| V dequant + repack | 6,656 | 0 | XVE |
| PV DPAS | 0 | 65,536 | XMX |
| Output | 192 | 0 | XVE |
| **Total** | **2,105,272** | **131,072** | **Scalar (99.6%)** |

---

## System-Wide Timing

```
256 HW threads, 4 waves.
64 VEs total (8 cores × 8 VEs), clock = 2 GHz.
4 threads per VE per wave.

XVE/Scalar time per wave:
  Per-VE: 4 threads × 2,105,272 = 8,421,088 SIMD-equiv cycles
  Wall time per wave = 8,421,088 / 2 GHz = 4.21 ms
  4 waves = ~16.8 ms

XMX time per wave:
  Per-thread: 256 DPAS instructions × ~30 cycles = 7,680 cycles
  Per-VE: 4 × 7,680 = 30,720 cycles → 15.4 μs
  4 waves = ~62 μs

Memory time:
  K: 32768 × 66 B × 8 heads = 16.5 MB
  V: 32768 × 132 B × 8 heads = 33.0 MB
  Q + tq_q_t: ~72 KB (cached)
  Output (fp32): 128 × 32 × 128 × 4 = 2.0 MB
  Total ≈ 51.5 MB / 102 GB/s = 0.505 ms
```

| Ceiling | Time | % of Wall Time |
|---------|------|----------------|
| **Memory floor** | **0.5 ms** | ~3% |
| **XMX (DPAS)** | **0.06 ms** | ~0.4% |
| **XVE/Scalar** | **~16.8 ms** | **~97%** |

**Verdict: Massively scalar-bound.** The two scalar bottlenecks (Q rotation + centroid LUT dequant) each contribute ~1M SIMD-equiv cycles per thread, dwarfing both the memory floor (0.5 ms) and DPAS time (0.06 ms) by over 30×.

---

## Optimization Roadmap

### P0: Pre-Rotate Q in Separate Kernel

**Problem**: Q rotation is 1M SIMD-equiv cycles/thread × 128 partitions, but the result is identical for all partitions of the same `(seq, head_num_idx)`. 128× pure redundancy.

**Solution**: Compute `Q_rot = Q × tq_q_t` once per head group in a dedicated DPAS kernel before the attention dispatch.

```
Q: [4, 128] × tq_q_t: [128, 128] → standard matmul, fits DPAS perfectly
DPAS cost: 4 × (128/16) × (128/16) = 256 DPAS instructions per head group
8 head groups → 2048 DPAS total → < 50 μs on XMX
```

**Implementation**:
1. New small CM kernel: `tq_q_rotation.cm` — loads Q and tq_q_t, DPAS matmul, writes Q_rot.
2. Remove lines 84–99 from `pa_single_token_turboquant.cm`.
3. Pass pre-rotated Q pointer instead of raw Q + tq_q_t.

**Impact**: Eliminates ~50% of total wall time (1M out of 2.1M SIMD-equiv cycles per thread removed, plus no tq_q_t load overhead).

### P1: Vectorize Centroid LUT via SIMD Compare+Select

**Problem**: `centroid_f[nibble_idx]` is scalar-indexed — each element requires an independent register gather. 512 scalar ops × 16 = 8,192 SIMD-equiv cycles per (kv_step, ri) tile.

**Solution**: Replace scalar indexed lookup with SIMD compare+select over 16 codebook entries:

```cpp
// Current (scalar, 256 scalar ops per p-iteration):
for (int n = 0; n < REG_N; n++) {
    float deq0_lo = centroid_f[(int)(packed_idx0 & 0x0f)];  // scalar gather
    ...
}

// Proposed (SIMD, 16 SIMD ops per nibble-vector):
vector<ushort, REG_N> nibble = packed_row & 0x0f;           // SIMD mask
vector<float, REG_N> result = centroid_f[0];                 // broadcast entry 0
#pragma unroll
for (int c = 1; c < 16; c++) {
    vector<ushort, REG_N> mask = (nibble == c);
    result = cm_sel<float>(mask, centroid_f[c], result);     // SIMD select
}
// result now has centroid_f[nibble[n]] for all n in SIMD — no scalar path
```

Per (kv_step, ri): 4 nibble-vectors × 16 compare+select = 64 SIMD ops + 4 SIMD knorm muls = **68 SIMD-equiv cycles** vs current 8,192. That's **120× speedup** for the K dequant inner loop.

Total K dequant: 16 × 8 × 68 = **8,704 SIMD-equiv cycles** (down from 1,048,576).

**Alternative (P1a)**: Precompute dequanted K in a separate kernel, write fp16 K to temp buffer. Attention kernel reads fp16 directly (like non-TQ path). Extra memory traffic: 32768 × 128 × 2 × 8 = 64 MB → 0.63 ms. Still much faster than 8.4 ms scalar cost. Simpler to implement but uses more BW.

### P2: Increase KV_PARTITION_SIZE to 512

**Problem**: 256-token partitions → 128 partitions → 4 waves. Each partition re-reads the Q matrix and re-initializes rS/Pmat/Omat. Also produces 128 partial results for the reduction kernel.

**Solution**: Double partition to 512 tokens.

Register budget check:
```
rS  = [4, 512] float = 8,192 B   (< 16,384 CM limit ✓)
Pmat = [4, 512] half = 4,096 B
Omat = [4, 128] float = 2,048 B
Qmat = [4, 128] half  = 1,024 B
Total persistent ≈ 15,360 B → fits 256-GRF (16,384 B budget)
```

Impact:
- Partitions halve: 64 → 2 waves instead of 4.
- Reduction work halves.
- After P0, also halves the number of Q loads.

### P3: Fuse K-Norm Multiply into SIMD Path

After P1 vectorizes the centroid lookup, the `knorm_vec[n]` multiply (lines 174–177) should also be vectorized:

```cpp
// After SIMD centroid lookup produces result[0..15]:
result = cm_mul<float>(result, knorm_vec);  // SIMD broadcast-multiply
Kt[row] = (half)result;                     // SIMD convert
```

This is essentially free once P1 is done — just ensuring the multiply stays in SIMD rather than falling back to scalar.

### P4: Software Prefetch for K/V

Once scalar bottlenecks are removed, the kernel becomes memory-bound at ~0.5 ms. Overlap K and V loads:

- During KQ phase: `cm_prefetch` V block for current partition
- During PV phase of block[i]: `cm_prefetch` K block[i+1] (if partition > 1 block)

### P5: fp16 Partial Output

Output writes are fp32 (line 330): 128 × 32 × 128 × 4 = 2 MB. Switch to fp16 halves this to 1 MB. The reduction kernel applies log-sum-exp correction anyway, so fp16 partials have negligible precision impact.

---

## Projected Impact

| State | Est. Wall Time | Bottleneck | Speedup |
|-------|---------------|------------|---------|
| **Current** | **~16.8 ms** | Scalar (Q rot + centroid) | 1× |
| After P0 (pre-rotate Q) | ~8.5 ms | Scalar (centroid dequant) | 2× |
| After P0 + P1 (SIMD centroid) | ~0.55 ms | Memory | 30× |
| After P0 + P1 + P2 (512 partition) | ~0.5 ms | Memory (at BW ceiling) | 34× |
| **Memory floor** | **0.505 ms** | 51.5 MB / 102 GB/s | 33× |

P0 + P1 together achieve a **~30× speedup**, bringing the kernel from scalar-bound to within touching distance of the memory bandwidth ceiling. P2 closes the remaining gap.
