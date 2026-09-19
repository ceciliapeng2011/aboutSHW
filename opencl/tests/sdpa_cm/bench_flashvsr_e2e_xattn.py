"""
Variant of bench_flashvsr_e2e.py: same per-chunk timing harness and FlashVSR-config constants
(NUM_HEADS/HEAD_DIM/WINDOW_TOKENS/SPATIAL_WINDOWS/S_CTX/CHUNKS, time_kernel, print_hotspot_summary
-- all reused from bench_flashvsr_e2e.py so both scripts measure the exact same chunk sizes/
dispatch geometry), but the sparse block mask comes from XAttention's own mask-generation
pipeline (dense_xattn_adapter.py's xattn_gemm_qk -> xattn_find_block) instead of FlashVSR's
pool+score+softmax+topk kernels (flashvsr_draft_gemm.cm/flashvsr_draft_topk.cm) -- lets the two
mask-generation algorithms' overhead be compared head-to-head against the SAME
cm_sdpa_opt_prefill.cm sparse-attention consumer.

x_attn/find_block.hpp's non-causal branch previously had a pre-existing bug (sort.hpp's radix
sort was hardcoded for 16-bit `half` keys, silently reinterpreting SOFTMAX_TYPE='float' scores as
raw ushort pairs, plus a second off-by-one dropping the lowest-scoring block from selection) --
now fixed (opencl/tests/x_attn/sort.hpp, opencl/tests/x_attn/find_block.hpp; see
test_sdpa_prefill_xattn.py's module docstring and commits 05e621aa/4da00144). The non-causal mask
now matches the CPU reference at 98.8-100%, so this comparison reflects XAttention's real
mask-generation cost/quality on FlashVSR's (non-causal) self-attention, not a bug artifact.

Run with: python3 bench_flashvsr_e2e_xattn.py
"""
import gc

import numpy as np
import torch

from clops import cl

import bench_flashvsr_e2e as e2e
import dense_xattn_adapter as xattn_adapter
import test_sdpa_prefill as base
import test_sdpa_prefill_xattn as xattn_mod

NUM_HEADS = e2e.NUM_HEADS
HEAD_DIM = e2e.HEAD_DIM
WINDOW_TOKENS = e2e.WINDOW_TOKENS
SPATIAL_WINDOWS = e2e.SPATIAL_WINDOWS
S_CTX = e2e.S_CTX
CHUNKS = e2e.CHUNKS
time_kernel = e2e.time_kernel
print_hotspot_summary = e2e.print_hotspot_summary
sparse_utilization = e2e.sparse_utilization
HW_PEAK_FLOPS = e2e.HW_PEAK_FLOPS

torch.manual_seed(0)

# Representative mid-range threshold from bench_sdpa.py's own XAttention sweep (Stage 3a: 0.99/0.9/0.6/0.3).
XATTN_THRESH = 0.9


def bench_chunk_xattn(name, f, cache_groups=0, xattn_thresh=XATTN_THRESH):
    """Same per-chunk timing as bench_flashvsr_e2e.bench_chunk, but the block mask comes from
    XAttention's real mask-generation pipeline (dense_xattn_adapter.py's xattn_gemm_qk ->
    xattn_find_block) instead of FlashVSR's own pool+score+softmax+topk kernels, at the SAME
    chunk sizes / SPARSE_BLOCK_SIZE=WINDOW_TOKENS=128 (xattn_block_size=128 so gemm_qk/
    find_block's logical scoring block matches the SDPA kernel's sparse block exactly).
    """
    q_groups = f // 2
    k_groups = q_groups + cache_groups  # streaming FIFO cache prepend -- see e2e module docstring
    num_q_windows = SPATIAL_WINDOWS * q_groups
    num_k_windows = SPATIAL_WINDOWS * k_groups
    q_len = num_q_windows * WINDOW_TOKENS
    kv_len = num_k_windows * WINDOW_TOKENS

    q, k, v = base.make_qkv(batch=1, num_heads=NUM_HEADS, num_kv_heads=NUM_HEADS,
                             q_len=q_len, kv_len=kv_len, head_size=HEAD_DIM)

    gemm_cm = xattn_adapter._get_gemm_qk_kernel(NUM_HEADS, NUM_HEADS, HEAD_DIM, WINDOW_TOKENS, False)
    find_cm = xattn_adapter._get_find_block_kernel(NUM_HEADS, NUM_HEADS, HEAD_DIM, WINDOW_TOKENS, False)

    key_cache, num_blocks = xattn_adapter._build_identity_key_cache(k[0].to(torch.float16), HEAD_DIM)
    block_indices = torch.arange(num_blocks, dtype=torch.int32)
    t_key_cache = cl.tensor(key_cache.numpy())
    t_block_indices = cl.tensor(block_indices.numpy())
    t_block_indices_begins = cl.tensor(np.zeros(2, dtype=np.int32))
    t_query = cl.tensor(xattn_adapter._build_gemm_qk_query(q[0].to(torch.float16), NUM_HEADS, HEAD_DIM).numpy())

    q_start_strided = xattn_adapter._causal_q_start_strided(q_len, kv_len, False)  # 0, non-causal
    M = q_len // xattn_adapter.STRIDE
    N = kv_len // xattn_adapter.STRIDE
    K = xattn_adapter.STRIDE * HEAD_DIM
    query_stride = K * NUM_HEADS * 2

    def call_gemm():
        # n_repeats=1 (fresh t_kq_max_wg/t_kq_exp_partial_sum output tensors each call, per
        # xattn_gemmQK.__call__) -- calling with n_repeats>1 reuses the SAME output buffers
        # across enqueues; gemm_qk's max/exp-partial-sum accumulation is not idempotent against
        # its own prior output, so a real per-call latency average requires fresh buffers every
        # iteration, matching how build_sparse_mask is actually used.
        return gemm_cm(t_key_cache, t_query, t_block_indices, t_block_indices_begins,
                        q_start_strided, M, N, K, query_stride, 1)

    for _ in range(5):
        call_gemm()  # warmup
    gemm_lat = []
    for _ in range(20):
        latency, t_kq_max_wg, t_kq_exp_partial_sum = call_gemm()
        gemm_lat.append(latency[0])
    ms_gemm_qk = sum(gemm_lat) / len(gemm_lat) * 1e-6

    def call_find():
        return find_cm(t_kq_max_wg, t_kq_exp_partial_sum, q_len, M, N, xattn_thresh, 1)

    for _ in range(5):
        call_find()  # warmup
    find_lat = []
    for _ in range(20):
        latency, t_mask, _ = call_find()
        find_lat.append(latency[0])
    ms_find_block = sum(find_lat) / len(find_lat) * 1e-6

    mask = torch.from_numpy(t_mask.numpy().astype(np.int8)).to(torch.bool).unsqueeze(0)
    density = mask.float().mean().item()

    # Sparse self-attention SDPA, using the REAL mask produced above (not a synthetic density).
    ms_sdpa_sparse = time_kernel(
        lambda: xattn_mod.run_prefill_sparse(q, k, v, is_causal=False,
                                              sparse_block_size=WINDOW_TOKENS, block_mask=mask),
        n_warmup=3, n_iters=10)

    # Cross-attention, dense, kv_len=512 -- separate op, reported for scale only.
    qc, kc, vc = base.make_qkv(batch=1, num_heads=NUM_HEADS, num_kv_heads=NUM_HEADS,
                                q_len=q_len, kv_len=S_CTX, head_size=HEAD_DIM)
    ms_cross = time_kernel(lambda: base.run_prefill(qc, kc, vc, is_causal=False),
                            n_warmup=3, n_iters=10)

    kernels = [
        ("xattn_gemm_qk", ms_gemm_qk),
        ("xattn_find_block", ms_find_block),
        ("cm_sdpa_opt_prefill (self-attn, sparse)", ms_sdpa_sparse),
    ]
    total_self_attn = sum(ms for _, ms in kernels)

    print(f"\n=== [xattn mask] {name} chunk: f={f} q_len={q_len} kv_len={kv_len} "
          f"(q_groups={q_groups}, k_groups={k_groups}, thresh={xattn_thresh}, "
          f"mask_density={density:.3f}) ===")
    for label, ms in sorted(kernels, key=lambda x: -x[1]):
        pct = ms / total_self_attn * 100 if total_self_attn > 0 else 0.0
        print(f"  {label:<42} {ms:>10.3f} ms  {pct:>5.1f}%")
    print(f"  {'TOTAL self-attention e2e':<42} {total_self_attn:>10.3f} ms  100.0%")
    print(f"  {'[separate] cross-attn (kv_len=512)':<42} {ms_cross:>10.3f} ms")
    gflops, util = sparse_utilization(q_len, kv_len, density, ms_sdpa_sparse)
    print(f"  cm_sdpa_opt_prefill utilization: {gflops:,.0f} GFLOPS, {util:.1f}% of "
          f"{HW_PEAK_FLOPS/1e12:.0f} TFLOPS peak (density={density:.3f})")

    del t_key_cache, t_query, t_block_indices, t_block_indices_begins
    del t_kq_max_wg, t_kq_exp_partial_sum, t_mask
    gc.collect()
    return kernels, total_self_attn, ms_cross


if __name__ == "__main__":
    cl.profiling(True)

    results_xattn = {}
    for name, cfg in CHUNKS.items():
        results_xattn[name] = bench_chunk_xattn(name, cfg["f"], cache_groups=cfg["cache_groups"])
    print_hotspot_summary("Hotspot summary (both chunks, XAttention mask)", results_xattn)
