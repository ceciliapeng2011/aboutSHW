# Skill: Cross-Attention GEMM QK Multi-Subsequence Design

How the GEMM QK kernel supports multiple subsequences in a single dispatch, workgroup mapping, metadata layout, and analysis of merging single-subseq path into the multi-subseq path.

---

## Context

Cross-attention GEMM QK computes `Q @ K^T` (with softmax partial reduction) for paged attention. The kernel lives in two forms:

- **Single-subseq path**: one subsequence per dispatch. Kernel receives `M`, `N`, `K`, `query_stride`, `slice_no`, `slice`, `q_start_strided` as scalar args.
- **Multi-subseq path**: all subsequences packed into one dispatch. Kernel receives `K`, `query_stride`, `num_subseqs` plus a metadata buffer.

Both paths call the **same** inner function `FUNC(id_wg_m, id_wg_n, hq, slm, key_cache, query, block_indices, block_index_begin, kq_max_wg, kq_exp_partial_sum, M, N, K, query_stride, q_start_strided, offset_partial_sum)` — the computational body is identical. Only the entry-point wrapper differs.

### Key Files

| File | Role |
|------|------|
| `opencl/tests/x_attn/xattn_gemm_qk.hpp` | Kernel entry points — `#if MULTI_SUBSEQ` selects path |
| `opencl/tests/x_attn/estimate.hpp` | Shared GEMM body (`gemm_qk` template function called by FUNC) |
| `opencl/tests/x_attn/test_gemm_qk.py` | Test driver — classes `xattn_gemmQK` (single) and `xattn_gemmQK_multi` (multi) |
| `impls/cm/xattn_gemm_qk.cm` | OV production kernel (single-subseq only today) |
| `impls/cm/paged_attention_gen.cpp` | OV host dispatch — hardcodes `slice_no=0, slice=0` |

---

## Workgroup Dispatch

### Single Subsequence

```python
N_kq_groups = ceil(N / BLOCK_WG_N)
wg_m_count  = ceil(M / BLOCK_WG_M)       # equivalently q_stride_pad // BLOCK_WG_M

GWS = [N_kq_groups * wg_m_count * SG_N * WALK_HQ,  SG_M,  num_heads // WALK_HQ]
LWS = [SG_N, SG_M, 1]
```

Total workgroups = `N_kq_groups * wg_m_count * num_heads`.

Each workgroup computes a `BLOCK_WG_M x BLOCK_WG_N` tile of the QK^T output:
- `id_wg_m` selects which M-block (query rows)
- `id_wg_n` selects which N-block (key columns / KV-cache blocks)

Mapping from flat WG id to (id_wg_m, id_wg_n) is done by `get_mn()` with `slice_no`/`slice` parameters for diagonal tiling. In practice (OV production), `slice_no=0, slice=0`, which reduces to:

```c
id_wg_m = id_wg_mn % WG_M_count;
id_wg_n = id_wg_mn / WG_M_count;
```

### Multi Subsequence

All subsequences' workgroups are flattened into a single global ID space:

```python
total_wg_count = sum(N_kq_groups_s * wg_m_count_s  for each subseq s)

GWS = [total_wg_count * SG_N * WALK_HQ,  SG_M,  num_heads // WALK_HQ]
LWS = [SG_N, SG_M, 1]
```

Each workgroup maps itself back to a subsequence and local (id_wg_m, id_wg_n) via the metadata buffer.

---

## Metadata Buffer Layout

Each subsequence occupies 16 int32 fields (64 bytes, one cacheline), defined in `xattn_gemm_qk.hpp`:

```
Index  #define                       Description
───────────────────────────────────────────────────────────────────────
 [0]   XATTN_META_SUBSEQ_Q_BEGIN    Absolute query token position in concatenated Q tensor
 [1]   XATTN_META_SUBSEQ_Q_LEN     Query length (tokens) for this subseq
 [2]   XATTN_META_M                 M = q_len // STRIDE
 [3]   XATTN_META_N                 N = kv_len // STRIDE
 [4]   XATTN_META_Q_STRIDE_PAD      ceil(M / BLOCK_WG_M) * BLOCK_WG_M
 [5]   XATTN_META_N_KQ_GROUPS       ceil(N / BLOCK_WG_N)
 [6]   XATTN_META_Q_BLOCK_PAD       ceil(q_len / xattn_block_size)
 [7]   XATTN_META_K_BLOCK_PAD       k_block_in_group * N_KQ_GROUPS
 [8]   XATTN_META_CAUSAL_START      k_block - q_block (causal mask offset)
 [9]   XATTN_META_Q_START_STRIDED   (kv_len - q_len) // STRIDE
[10]   XATTN_META_BUF_OFF_KQ_MAX    Byte offset into combined kq_max_wg buffer
[11]   XATTN_META_BUF_OFF_EXP_SUM   Byte offset into combined exp_partial_sum buffer
[12]   XATTN_META_BUF_OFF_MASK      Byte offset for mask buffer
[13]   XATTN_META_BUF_OFF_MASK_WG   Byte offset for workgroup mask buffer
[14]   XATTN_META_BLOCK_IDX_BEGIN    Start index in concatenated block_indices array
[15]   XATTN_META_WG_OFFSET          Cumulative WG count (prefix sum for WG-to-subseq mapping)
```

Built by `build_xattn_subseq_meta()` in `test_gemm_qk.py`. Buffer offsets are accumulated across subsequences so all subseqs' results pack contiguously in shared output buffers.

---

## WG-to-Subsequence Mapping (Kernel Side)

```c
uint wg_mn_flat = cm_group_id(0) / WALK_HQ;

// Linear scan through prefix-sum WG_OFFSET fields
int subseq_id = 0;
for (int s = 1; s < num_subseqs; s++) {
    int wg_off_next = meta[s * XATTN_META_STRIDE + XATTN_META_WG_OFFSET];
    if ((int)wg_mn_flat >= wg_off_next)
        subseq_id = s;
}

// Load this subseq's M, N, offsets from metadata
int meta_base = subseq_id * XATTN_META_STRIDE;
uint M = meta[meta_base + XATTN_META_M];
uint N = meta[meta_base + XATTN_META_N];
// ... (q_start_strided, block_index_begin, buf_off_*, subseq_q_begin)

// Compute local (id_wg_m, id_wg_n) within this subseq
uint wg_local = wg_mn_flat - subseq_wg_offset;
uint WG_M_count = ceil(M / BLOCK_WG_M);
id_wg_m = wg_local % WG_M_count;
id_wg_n = wg_local / WG_M_count;
```

The linear scan is O(num_subseqs) per WG, acceptable for typical num_subseqs < 10.

---

## Key Differences: Single vs Multi Entry Point

| Aspect | Single Path | Multi Path |
|--------|-------------|------------|
| **Kernel args** | M, N, K, query_stride, slice_no, slice, q_start_strided + `block_indices_begins` ptr (9 args) | K, query_stride, num_subseqs + `meta` ptr (4 args) |
| **M, N source** | Kernel scalar parameters | Loaded from metadata per WG |
| **WG mapping** | `get_mn()` with slice_no/slice params | Linear scan + simple `%`/`/` |
| **Query offset** | `hq * HEAD_SIZE` | `(subseq_q_begin * query_stride/STRIDE + hq * HEAD_SIZE)` |
| **Block index begin** | `((int*)block_indices_begins)[0]` | `meta[subseq_id * 16 + 14]` |
| **Output offsets** | `0 + hq * per_head_size` | `buf_off_xxx + hq * per_head_size` |
| **Compile flag** | `MULTI_SUBSEQ=0` (default) | `-DMULTI_SUBSEQ=1` |
| **HW requirement** | Xe1 and Xe2 (SurfaceIndex fallback for Xe1) | Xe2 only (`#error` if no `CM_HAS_LSC_UNTYPED_2D`) |

### Output Buffer Layout (Multi)

All subsequences' outputs are packed contiguously. Per-subseq byte offsets from metadata point to each region:

```
Combined kq_max_wg buffer:
  [subseq_0: head_0..head_H-1] [subseq_1: head_0..head_H-1] ...
  ^                              ^
  buf_off_kq_max[0] = 0          buf_off_kq_max[1]

Per-head offset = buf_off_kq_max + head_idx * (N_kq_groups * q_stride_pad * sizeof(SOFTMAX_TYPE))
```

Same layout applies to `kq_exp_partial_sum`.

---

## Merging Single-Path into Multi-Path

### Feasibility

Single-subseq is a special case of multi-subseq with `num_subseqs = 1`. Both paths call the same `FUNC()` with identical arguments. Only the ~30-line wrapper that derives those arguments differs.

### What happens when multi-path runs with num_subseqs = 1

1. **Subseq scan loop**: `for (int s = 1; s < 1; s++)` — **zero iterations**, body completely skipped.
2. **Metadata load**: ~6 int32 reads from a 64-byte (single cacheline) metadata buffer. L1-cached after the first WG on each Xe-core.
3. **subseq_q_begin**: 0 for single subseq → `query += (0 * stride + hq * HEAD_SIZE)` — multiply-add of zero.
4. **buf_off_kq_max / buf_off_exp_sum**: 0 for single subseq → adds zero to output pointers.
5. **WG mapping**: `wg_local = wg_mn_flat - 0`, then same `%`/`/` as single path's `get_mn(slice_no=0, slice=0)`.

### Performance Impact

| Source of overhead | Cost | Significance |
|--------------------|------|--------------|
| Metadata buffer loads (6 × int32) | ~1 cacheline read, L1-hot after first WG | **Negligible** — amortized over thousands of DPAS cycles in FUNC |
| Subseq scan loop (num_subseqs=1) | 0 iterations | **Zero** |
| Extra integer adds of zero (query, output offsets) | 3 integer add instructions | **Negligible** — compiler may optimize away |
| Loss of `get_mn()` diagonal tiling | `slice_no=0, slice=0` in OV production → `get_mn` already degenerates to same `%`/`/` | **Zero** — dead code path in practice |

**Estimated throughput regression: < 0.1%.** The wrapper overhead is ~10 scalar instructions out of thousands of DPAS + vector instructions in FUNC.

### Memory Footprint Impact

| Resource | Single Path | Multi Path (num_subseqs=1) | Delta |
|----------|-------------|---------------------------|-------|
| Metadata buffer | Not needed | 16 × int32 = 64 bytes | **+64 bytes** |
| `block_indices_begins` buffer | 1 × int32 = 4 bytes | Folded into metadata | **-4 bytes** |
| Kernel binary | Single-path variant compiled | Same variant (no second compilation) | **Smaller** (one variant vs two) |
| Output buffers | Same | Same (offsets are 0) | 0 |
| Kernel scalar args | 7 scalars + 1 ptr | 3 scalars + 1 ptr | **4 fewer args** |

Net memory: +60 bytes. But eliminates an entire kernel variant → **smaller total binary, faster compilation**.

### Register Pressure

Extra locals in multi-path wrapper (`subseq_id`, `meta_base`, `subseq_wg_offset`, `subseq_q_begin`, `buf_off_*`) are all dead before `FUNC()` is called. Compiler reuses those registers. **No register pressure increase in the hot DPAS loop.**

### Xe1 Compatibility — The Only Real Blocker

Multi-path currently has:
```c
#ifndef CM_HAS_LSC_UNTYPED_2D
#error "MULTI_SUBSEQ requires Xe2 — Xe1 SurfaceIndex path not yet supported"
#endif
```

To unify, either:
1. **Drop Xe1 support** for this kernel (if Xe1 is end-of-life for this workload).
2. **Add SurfaceIndex fallback** to the unified path — keep the `#ifdef CM_HAS_LSC_UNTYPED_2D` branches for `key_cache`, `query`, `kq_exp_partial_sum` as the single path already does, and read metadata via SVM (which Xe1 supports for scalar pointers).

### Slicing (`get_mn` with non-zero slice_no/slice)

`get_mn()` supports diagonal WG tiling for better L3 locality. However:
- OV production hardcodes `slice_no=0, slice=0` (`paged_attention_gen.cpp:666`).
- The test driver also uses `slice_no=0, slice=0` (`test_gemm_qk.py:129-130`).

If diagonal tiling is needed in the future, it can be added to the metadata-driven path (e.g., store `slice_no`/`slice` in reserved metadata fields, or implement a global slicing scheme across all subsequences). But today it is dead code.

### Summary

| Concern | Verdict |
|---------|---------|
| Throughput | **< 0.1% overhead** — cached metadata reads + zero-adds in wrapper |
| Latency | **Undetectable** — 64B metadata is one L1 cacheline |
| Memory | **+60 bytes** (negligible) |
| Binary size | **Smaller** — one kernel variant instead of two |
| Compile time | **Faster** — one compilation instead of two |
| Register pressure | **No change** in hot loop |
| Xe1 support | **Blocked** — needs SurfaceIndex fallback or Xe1 deprecation |
| Diagonal tiling | **Not used** in production; can be added to metadata later |

**Recommendation:** Merge the paths. Single-subseq becomes multi with `num_subseqs=1` and a trivial 64-byte metadata buffer. The only decision is whether to add Xe1 SurfaceIndex support to the unified path or drop Xe1.

---

## Test Validation Strategy

Multi-subseq correctness is validated by comparing against single-subseq reference runs (`test_gemm_qk.py:654-699`):

1. Run multi-subseq kernel once with all subsequences combined.
2. For each subsequence, extract its output region from the combined buffer using metadata byte offsets (`BUF_OFF_KQ_MAX`, `BUF_OFF_EXP_SUM`).
3. Run single-subseq reference (`get_gemm_ref`) independently for each subsequence.
4. Compare element-wise — must match exactly.

Test cases cover: 1 subseq baseline, 2 prefill subseqs, 3 prefill subseqs, equal-sized block subseqs.
