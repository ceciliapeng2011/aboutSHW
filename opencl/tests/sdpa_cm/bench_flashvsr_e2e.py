"""
End-to-end per-kernel timing breakdown for FlashVSR's self-attention pipeline -- the four
mask-generation kernels (flashvsr_draft_gemm.cm/flashvsr_draft_topk.cm, see
flashvsr_mask_adapter.py) PLUS the actual sparse cm_sdpa_opt_prefill.cm call that consumes the
real mask they produce -- at FlashVSR's real per-chunk configs (steady/warm-up chunks, 320x480
portrait -> 1280x1920 case, see
/home/ceciliapeng/repos/reports/flashvsr_v1.1_full_inference_pipeline.md and
/home/ceciliapeng/.claude/plans/learn-sdpa-from-home-ceciliapeng-openvin-majestic-toast.md's
"FlashVSR-config" section). Cross-attention (dense, kv_len=512) is included for scale comparison
but reported separately -- it is not on self-attention's critical path.

This differs from that plan doc's existing bench_sdpa.py FlashVSR section in one important way:
that section's sparse self-attention numbers used a SYNTHETIC density=0.3 fake mask (explicitly
flagged there as "illustrative... not derived from FlashVSR's kv_ratio/topk_ratio"). This script
uses the REAL mask produced by flashvsr_mask_adapter.build_draft_mask_gpu's actual kernel
pipeline, so the sparse-SDPA timing/density here reflects FlashVSR's real local-window+top-k
selection rather than an assumed density.

Streaming KV-cache modeling (wan_video_dit.py's SelfAttention.forward, lines 317-382; kv_ratio=3.0
default in flashvsr_full.py:326 / flashvsr_tiny_long.py:306 -> kv_len=3 window-groups cap):
pre_cache_k/v (if present) are concatenated to the FRONT of the current chunk's own k_w/v_w BEFORE
the attention call (lines 344-346), and only AFTER attention is exactly one temporal group trimmed
from the front if the total exceeds kv_len (lines 367-374). So:
  - warm-up chunk (f=6): is_stream's first call, pre_cache is None -> k_groups == q_groups == 3
    (450 windows = 57,600 tokens each side). No cache exists yet, so this is unchanged/exact, not
    an approximation.
  - steady-state chunk (f=2): every call after the first sees a stabilized cache of exactly
    kv_len=3 groups (450 windows) prepended to this chunk's own 1 group (150 windows) -- so the
    REAL self-attention call is q_groups=1 (19,200 tokens) against k_groups=4 (76,800 tokens), a 4x
    larger K/V read than a naive q_len==kv_len assumption. CHUNKS/bench_sparse_at_density below
    model this exactly (k_groups = q_groups + 3 cached groups for "steady", k_groups == q_groups
    for "warmup") instead of the previous q_groups==k_groups-for-both-chunks lower bound.

For each chunk shape, __main__ prints:
  [1] bench_dense_reference -- cm_sdpa_opt_prefill's plain DENSE kernel time + GFLOPS utilization,
      no mask-gen kernels at all (the "what if we just skip sparsity" baseline).
  [2] bench_sparse_at_density -- swept across DENSITY_SWEEP (plus FlashVSR's own real production
      density derived from TOPK_RATIO) -- a per-kernel table (pool x2, score, softmax, topk_mask,
      sparse SDPA) with time, a roofline utilization number (GFLOPS or GB/s depending on whether
      the kernel is compute- or bandwidth-bound), % of that density's total kernel time, and the
      summed total kernel time.
This is the BASELINE this module records before any optimization work lands on the mask-gen
kernels (see the plan doc's Stage A/B/C radix-select/fusion changes).

Run with: python3 bench_flashvsr_e2e.py
"""
import gc
import math
import os

import numpy as np
import torch

from clops import cl

import flashvsr_mask_adapter as gpu_mod
import flashvsr_mask_reference as ref_mod
import test_sdpa_prefill as base
import test_sdpa_prefill_xattn as xattn_mod

torch.manual_seed(0)

# Same GFLOPS/utilization convention as bench_sdpa.py's flops_for()/report() (2 GEMMs, 4 FLOPs/MAC,
# non-causal so no triangle-halving, scaled by block-mask density); HW_PEAK_FLOPS=31 matches this
# doc's own established convention for this Xe2 Arc GPU's compute roofline (vs. bench_sdpa.py's
# generic default of 20), overridable via the same env var.
HW_PEAK_FLOPS = float(os.environ.get("HW_PEAK_FLOPS", "31")) * 1e12
# Same env convention as bench_flashvsr_mask.py -- lets the mask-gen kernels' (pool/softmax/
# topk_mask) bandwidth-roofline utilization numbers be compared directly against that script's.
HW_PEAK_BW_GBS = float(os.environ.get("HW_PEAK_BW_GBS", "100"))


def flops_util(flops, ms):
    gflops = flops / (ms * 1e6) if ms > 0 else 0.0
    utilization = flops / (ms * 1e-3) / HW_PEAK_FLOPS * 100 if ms > 0 else 0.0
    return gflops, utilization


def bw_util(total_bytes, ms):
    bw = total_bytes / (ms * 1e6) if ms > 0 else 0.0
    utilization = bw / HW_PEAK_BW_GBS * 100 if HW_PEAK_BW_GBS > 0 else 0.0
    return bw, utilization


def sparse_utilization(q_len, kv_len, density, latency_ms):
    flops = NUM_HEADS * q_len * kv_len * HEAD_DIM * 4 * density
    return flops_util(flops, latency_ms)


# FlashVSR real config: 320x480 portrait source -> 1280x1920 target (4x scale), see report.
# VAE 8x downsample -> latent 160x240 -> patchify(2x2) -> post-patchify grid h*w=80*120=9600.
NUM_HEADS = 12
HEAD_DIM = 128
WINDOW_TOKENS = 128            # WindowPartition3D window = 2x8x8 = 128 tokens
SPATIAL_H, SPATIAL_W = 10, 15  # (80//8, 120//8) -- spatial windows per temporal group
SPATIAL_WINDOWS = SPATIAL_H * SPATIAL_W  # 150
LOCAL_RANGE = 11               # production value (infer_flashvsr_v1.1_full.py)
TOPK_RATIO = 2.0               # production default (wan_video_dit.py / pipeline scripts)
S_CTX = 512                    # cross-attention's fixed text-context length

# f=2 -> steady-state chunk (1 temporal group), f=6 -> warm-up chunk (3 temporal groups).
# cache_groups = extra temporal groups of K/V prepended by the streaming FIFO cache BEFORE this
# chunk's self-attention call (wan_video_dit.py:344-346) -- see module docstring. steady-state's
# stabilized cache always holds exactly kv_len=3 groups; warm-up's pre_cache is None (first call).
CHUNKS = {
    "steady": dict(f=2, cache_groups=3),
    "warmup": dict(f=6, cache_groups=0),
}


def make_windowed(num_windows, seed):
    g = torch.Generator().manual_seed(seed)
    d_full = NUM_HEADS * HEAD_DIM
    return torch.randn(num_windows, WINDOW_TOKENS, d_full, generator=g, dtype=torch.float32)


def time_kernel(fn, n_warmup=5, n_iters=20):
    cl.finish()
    for _ in range(n_warmup):
        fn()
    cl.finish()
    for _ in range(n_iters):
        fn()
    ns = cl.finish()
    return sum(ns) / len(ns) * 1e-6  # ms


def bench_dense_reference(name, f, cache_groups=0):
    """Full-density (no mask, no mask-gen kernels at all) reference point for a chunk shape:
    just cm_sdpa_opt_prefill's plain dense kernel on the same q_len/kv_len the sparse path below
    would use. This is what the sparse pipeline's total time (mask-gen + sparse SDPA) should be
    compared against to know whether sparsity is a net win at all for this shape."""
    q_groups = f // 2
    k_groups = q_groups + cache_groups  # streaming FIFO cache prepend -- see module docstring
    num_q_windows = SPATIAL_WINDOWS * q_groups
    num_k_windows = SPATIAL_WINDOWS * k_groups
    q_len = num_q_windows * WINDOW_TOKENS
    kv_len = num_k_windows * WINDOW_TOKENS

    q, k, v = base.make_qkv(batch=1, num_heads=NUM_HEADS, num_kv_heads=NUM_HEADS,
                             q_len=q_len, kv_len=kv_len, head_size=HEAD_DIM)
    ms_dense = time_kernel(lambda: base.run_prefill(q, k, v, is_causal=False),
                            n_warmup=3, n_iters=10)
    gflops, util = sparse_utilization(q_len, kv_len, 1.0, ms_dense)

    qc, kc, vc = base.make_qkv(batch=1, num_heads=NUM_HEADS, num_kv_heads=NUM_HEADS,
                                q_len=q_len, kv_len=S_CTX, head_size=HEAD_DIM)
    ms_cross = time_kernel(lambda: base.run_prefill(qc, kc, vc, is_causal=False),
                            n_warmup=3, n_iters=10)

    print(f"\n=== {name} chunk: f={f} q_len={q_len} kv_len={kv_len} -- "
          f"[1] DENSE (full density) reference ===")
    print(f"  {'cm_sdpa_opt_prefill (dense, self-attn)':<42} {ms_dense:>10.3f} ms  "
          f"{gflops:>10,.0f} GFLOPS  {util:>6.1f}% util")
    print(f"  {'[separate] cross-attn (kv_len=512, dense)':<42} {ms_cross:>10.3f} ms")
    del q, k, v, qc, kc, vc
    gc.collect()
    return dict(q_len=q_len, kv_len=kv_len, ms_dense=ms_dense, gflops=gflops, util=util,
                ms_cross=ms_cross)


def bench_sparse_at_density(name, f, cache_groups, density):
    """[2] Full mask-gen pipeline (pool x2, score, softmax, topk_mask) + sparse SDPA at a
    requested block-mask density, reporting per-kernel time/utilization/% of total, plus the
    summed total kernel time -- the baseline table Stage A/B/C's optimizations are measured
    against. `density` is inverted to flashvsr_mask_reference's `topk` GLOBAL top-k count via
    topk_ratio = density * num_k_windows (see flashvsr_mask_reference.py's docstring: density =
    topk / (SPATIAL_WINDOWS * num_k_windows) = topk_ratio / num_k_windows)."""
    q_groups = f // 2
    k_groups = q_groups + cache_groups  # streaming FIFO cache prepend -- see module docstring
    num_q_windows = SPATIAL_WINDOWS * q_groups
    num_k_windows = SPATIAL_WINDOWS * k_groups
    q_len = num_q_windows * WINDOW_TOKENS
    kv_len = num_k_windows * WINDOW_TOKENS
    topk_ratio = density * num_k_windows
    topk = max(1, int(SPATIAL_WINDOWS * topk_ratio))

    q_w = make_windowed(num_q_windows, seed=1)
    k_w = make_windowed(num_k_windows, seed=2)
    local_mask = ref_mod.build_local_block_mask_shifted_vec_normal_slide(
        SPATIAL_H, SPATIAL_W, LOCAL_RANGE, LOCAL_RANGE, include_self=True)

    gemm_kernel = gpu_mod._get_gemm_kernel(NUM_HEADS, HEAD_DIM, WINDOW_TOKENS)
    topk_kernel = gpu_mod._get_topk_kernel(NUM_HEADS)

    t_qw = cl.tensor(q_w.to(torch.float16).contiguous().numpy())
    t_kw = cl.tensor(k_w.to(torch.float16).contiguous().numpy())
    t_pooled_q = cl.tensor(np.zeros((num_q_windows, NUM_HEADS, HEAD_DIM), dtype=np.float16))
    t_pooled_k = cl.tensor(np.zeros((num_k_windows, NUM_HEADS, HEAD_DIM), dtype=np.float16))

    ms_pool_q = time_kernel(lambda: gemm_kernel.enqueue(
        "flashvsr_pool", [num_q_windows, NUM_HEADS, 1], [1, 1, 1], t_qw, t_pooled_q))
    ms_pool_k = time_kernel(lambda: gemm_kernel.enqueue(
        "flashvsr_pool", [num_k_windows, NUM_HEADS, 1], [1, 1, 1], t_kw, t_pooled_k))

    t_local_mask = cl.tensor(local_mask.to(torch.uint8).contiguous().numpy())
    t_raw_scores = cl.tensor(np.zeros((NUM_HEADS, num_q_windows, num_k_windows), dtype=np.float32))
    scale = 1.0 / math.sqrt(HEAD_DIM)
    ms_score = time_kernel(lambda: gemm_kernel.enqueue(
        "flashvsr_score", [num_q_windows, NUM_HEADS, 1], [1, 1, 1],
        t_pooled_q, t_pooled_k, t_local_mask, t_raw_scores,
        num_q_windows, num_k_windows, SPATIAL_WINDOWS, scale))

    t_probs = cl.tensor(np.zeros((NUM_HEADS, num_q_windows, num_k_windows), dtype=np.float16))
    ms_softmax = time_kernel(lambda: topk_kernel.enqueue(
        "flashvsr_softmax", [num_q_windows, NUM_HEADS, 1], [1, 1, 1],
        t_raw_scores, t_probs, num_k_windows))

    t_mask = cl.tensor(np.zeros((NUM_HEADS, num_q_windows, num_k_windows), dtype=np.uint8))
    ls = gpu_mod.TOPK_LOCAL_SIZE
    ms_topk = time_kernel(lambda: topk_kernel.enqueue(
        "flashvsr_topk_mask", [NUM_HEADS, q_groups, ls], [1, 1, ls],
        t_probs, t_mask, num_q_windows, num_k_windows, SPATIAL_WINDOWS, topk))

    mask = torch.from_numpy(t_mask.numpy()).to(torch.bool).unsqueeze(0)
    measured_density = mask.float().mean().item()

    # Sparse self-attention SDPA, using the REAL mask produced above (not a synthetic density).
    q, k, v = base.make_qkv(batch=1, num_heads=NUM_HEADS, num_kv_heads=NUM_HEADS,
                             q_len=q_len, kv_len=kv_len, head_size=HEAD_DIM)
    ms_sdpa_sparse = time_kernel(
        lambda: xattn_mod.run_prefill_sparse(q, k, v, is_causal=False,
                                              sparse_block_size=WINDOW_TOKENS, block_mask=mask),
        n_warmup=3, n_iters=10)

    # Per-kernel utilization -- formulas match bench_flashvsr_mask.py's bench_score/bench_softmax/
    # bench_topk exactly, so the two scripts' numbers are directly comparable. flashvsr_pool/
    # flashvsr_softmax/flashvsr_topk_mask are bandwidth-rooflined (elementwise/pass-count bound,
    # topk_mask's "GB/s" is an effective/equivalent number given it's really SLM-barrier-bound, not
    # a literal DRAM bandwidth measurement -- see bench_flashvsr_mask.py's module docstring);
    # flashvsr_score and the sparse SDPA call are compute-rooflined.
    pool_q_bytes = num_q_windows * WINDOW_TOKENS * NUM_HEADS * HEAD_DIM * 2
    pool_k_bytes = num_k_windows * WINDOW_TOKENS * NUM_HEADS * HEAD_DIM * 2
    score_flops = 2 * NUM_HEADS * num_q_windows * num_k_windows * HEAD_DIM
    n_probs = NUM_HEADS * num_q_windows * num_k_windows
    softmax_bytes = n_probs * 18
    topk_bytes = n_probs * (16 * 2 + 2 + 1)

    bw_pq, u_pq = bw_util(pool_q_bytes, ms_pool_q)
    bw_pk, u_pk = bw_util(pool_k_bytes, ms_pool_k)
    gf_sc, u_sc = flops_util(score_flops, ms_score)
    bw_sm, u_sm = bw_util(softmax_bytes, ms_softmax)
    bw_tk, u_tk = bw_util(topk_bytes, ms_topk)
    gf_sp, u_sp = sparse_utilization(q_len, kv_len, measured_density, ms_sdpa_sparse)

    kernels = [
        ("flashvsr_pool(Q)", ms_pool_q, f"{bw_pq:8,.1f} GB/s", u_pq),
        ("flashvsr_pool(K)", ms_pool_k, f"{bw_pk:8,.1f} GB/s", u_pk),
        ("flashvsr_score", ms_score, f"{gf_sc:8,.0f} GFLOPS", u_sc),
        ("flashvsr_softmax", ms_softmax, f"{bw_sm:8,.1f} GB/s", u_sm),
        ("flashvsr_topk_mask", ms_topk, f"{bw_tk:8,.1f} GB/s*", u_tk),
        ("cm_sdpa_opt_prefill (sparse)", ms_sdpa_sparse, f"{gf_sp:8,.0f} GFLOPS", u_sp),
    ]
    total_ms = sum(k[1] for k in kernels)

    print(f"\n--- [2] {name} chunk: density(target)={density:.4f} "
          f"density(measured)={measured_density:.4f} topk={topk} (topk_ratio={topk_ratio:.2f}) ---")
    print(f"  {'kernel':<32}{'time(ms)':>10}  {'metric':>14}  {'util%':>7}  {'% of total':>10}")
    for label, ms, metric, util in kernels:
        pct = ms / total_ms * 100 if total_ms > 0 else 0.0
        print(f"  {label:<32}{ms:>10.3f}  {metric:>14}  {util:>6.1f}%  {pct:>9.1f}%")
    print(f"  {'TOTAL kernel time':<32}{total_ms:>10.3f}  {'':>14}  {'':>7}  {100.0:>9.1f}%")

    del t_qw, t_kw, t_pooled_q, t_pooled_k, t_local_mask, t_raw_scores, t_probs, t_mask, q, k, v
    gc.collect()
    return dict(density=measured_density, topk=topk, total_ms=total_ms, kernels=kernels)


def print_hotspot_summary(title, results):
    """Kept for bench_flashvsr_e2e_xattn.py, which still returns bench_chunk_xattn's original
    (kernels, total, ms_cross) 3-tuple shape -- unrelated to bench_sparse_at_density's dict return
    above (this file's own __main__ no longer calls this; the per-density tables already show the
    same per-kernel breakdown, across MORE than 2 configs)."""
    print(f"\n=== {title} ===")
    agg = {}
    for name, (kernels, total, *_rest) in results.items():
        for label, ms in kernels:
            agg.setdefault(label, []).append(ms)
    for label, values in sorted(agg.items(), key=lambda kv: -max(kv[1])):
        print(f"  {label:<42} steady={values[0]:>10.3f} ms  warmup={values[1]:>10.3f} ms")


# Density sweep for [2]'s table -- 1.0 is the "full density" (topk covers every element, i.e. the
# mask-gen kernels run but prune nothing) reference point, down to 0.004 which matches FlashVSR's
# own real measured production density (TOPK_RATIO=2.0 at these chunk shapes -- see below, added
# per-chunk since real density = TOPK_RATIO/num_k_windows depends on chunk shape).
DENSITY_SWEEP = [1.0, 0.5, 0.1, 0.03, 0.01, 0.004]


if __name__ == "__main__":
    cl.profiling(True)

    for name, cfg in CHUNKS.items():
        bench_dense_reference(name, cfg["f"], cache_groups=cfg["cache_groups"])

        k_groups = cfg["f"] // 2 + cfg["cache_groups"]
        num_k_windows = SPATIAL_WINDOWS * k_groups
        flashvsr_real_density = TOPK_RATIO / num_k_windows  # production topk_ratio=2.0 default
        densities = sorted(set(DENSITY_SWEEP + [flashvsr_real_density]), reverse=True)

        for density in densities:
            tag = " (FlashVSR production TOPK_RATIO=2.0)" if density == flashvsr_real_density else ""
            if tag:
                print(f"\n[note] density={density:.4f}{tag}")
            bench_sparse_at_density(name, cfg["f"], cache_groups=cfg["cache_groups"],
                                     density=density)
