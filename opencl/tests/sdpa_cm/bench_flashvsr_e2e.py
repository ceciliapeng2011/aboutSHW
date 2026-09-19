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
    larger K/V read than a naive q_len==kv_len assumption. CHUNKS/bench_chunk below model this
    exactly (k_groups = q_groups + 3 cached groups for "steady", k_groups == q_groups for
    "warmup") instead of the previous q_groups==k_groups-for-both-chunks lower bound.

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


def bench_chunk(name, f, cache_groups=0, topk_ratio=TOPK_RATIO):
    q_groups = f // 2
    k_groups = q_groups + cache_groups  # streaming FIFO cache prepend -- see module docstring
    num_q_windows = SPATIAL_WINDOWS * q_groups
    num_k_windows = SPATIAL_WINDOWS * k_groups
    q_len = num_q_windows * WINDOW_TOKENS
    kv_len = num_k_windows * WINDOW_TOKENS
    # generate_draft_block_mask's `topk` is a GLOBAL top-k count over the flattened
    # (spatial_windows x num_k_windows) grid per (head, Q-temporal-group) -- see
    # flashvsr_mask_reference.py's docstring -- so resulting density = topk / (SPATIAL_WINDOWS *
    # num_k_windows) = topk_ratio / num_k_windows, independent of SPATIAL_WINDOWS itself. Callers
    # wanting a specific target density can invert that: topk_ratio = target_density * num_k_windows.
    topk = int(SPATIAL_WINDOWS * topk_ratio)

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
    density = mask.float().mean().item()

    # Sparse self-attention SDPA, using the REAL mask produced above (not a synthetic density).
    q, k, v = base.make_qkv(batch=1, num_heads=NUM_HEADS, num_kv_heads=NUM_HEADS,
                             q_len=q_len, kv_len=kv_len, head_size=HEAD_DIM)
    ms_sdpa_sparse = time_kernel(
        lambda: xattn_mod.run_prefill_sparse(q, k, v, is_causal=False,
                                              sparse_block_size=WINDOW_TOKENS, block_mask=mask),
        n_warmup=3, n_iters=10)

    # Dense (density=1.0, no mask/pruning at all) self-attention SDPA on the SAME q/k/v -- the
    # "full density" reference point: shows what the sparse path is actually buying vs. just
    # running cm_sdpa_opt_prefill's plain dense kernel and skipping all five mask-gen kernels
    # entirely. Comparable to bench_sdpa.py's bench_prefill (dense) vs.
    # bench_xattn_prefill_synthetic (sparse) split, but using the REAL per-chunk shapes here.
    ms_sdpa_dense = time_kernel(lambda: base.run_prefill(q, k, v, is_causal=False),
                                 n_warmup=3, n_iters=10)

    # Cross-attention, dense, kv_len=512 -- separate op, reported for scale only.
    qc, kc, vc = base.make_qkv(batch=1, num_heads=NUM_HEADS, num_kv_heads=NUM_HEADS,
                                q_len=q_len, kv_len=S_CTX, head_size=HEAD_DIM)
    ms_cross = time_kernel(lambda: base.run_prefill(qc, kc, vc, is_causal=False),
                            n_warmup=3, n_iters=10)

    kernels = [
        ("flashvsr_pool(Q)", ms_pool_q),
        ("flashvsr_pool(K)", ms_pool_k),
        ("flashvsr_score", ms_score),
        ("flashvsr_softmax", ms_softmax),
        ("flashvsr_topk_mask", ms_topk),
        ("cm_sdpa_opt_prefill (self-attn, sparse)", ms_sdpa_sparse),
    ]
    total_self_attn = sum(ms for _, ms in kernels)
    # "Full pipeline vs. just running dense" comparison: mask-gen overhead + sparse SDPA vs. a
    # single dense SDPA call with none of the five mask-gen kernels at all.
    total_vs_dense_speedup = ms_sdpa_dense / total_self_attn if total_self_attn > 0 else 0.0

    print(f"\n=== {name} chunk: f={f} q_len={q_len} kv_len={kv_len} "
          f"(q_groups={q_groups}, k_groups={k_groups}, mask_density={density:.3f}) ===")
    for label, ms in sorted(kernels, key=lambda x: -x[1]):
        pct = ms / total_self_attn * 100 if total_self_attn > 0 else 0.0
        print(f"  {label:<42} {ms:>10.3f} ms  {pct:>5.1f}%")
    print(f"  {'TOTAL self-attention e2e (sparse+maskgen)':<42} {total_self_attn:>10.3f} ms  100.0%")
    print(f"  {'[full density] cm_sdpa_opt_prefill (dense)':<42} {ms_sdpa_dense:>10.3f} ms  "
          f"(speedup vs. sparse+maskgen e2e: {total_vs_dense_speedup:.2f}x)")
    print(f"  {'[separate] cross-attn (kv_len=512)':<42} {ms_cross:>10.3f} ms")
    gflops, util = sparse_utilization(q_len, kv_len, density, ms_sdpa_sparse)
    print(f"  cm_sdpa_opt_prefill (sparse) utilization: {gflops:,.0f} GFLOPS, {util:.1f}% of "
          f"{HW_PEAK_FLOPS/1e12:.0f} TFLOPS peak (density={density:.3f})")
    gflops_d, util_d = sparse_utilization(q_len, kv_len, 1.0, ms_sdpa_dense)
    print(f"  cm_sdpa_opt_prefill (dense)  utilization: {gflops_d:,.0f} GFLOPS, {util_d:.1f}% of "
          f"{HW_PEAK_FLOPS/1e12:.0f} TFLOPS peak (density=1.000)")

    del t_qw, t_kw, t_pooled_q, t_pooled_k, t_local_mask, t_raw_scores, t_probs, t_mask
    gc.collect()
    return kernels, total_self_attn, ms_cross, ms_sdpa_dense


def print_hotspot_summary(title, results):
    print(f"\n=== {title} ===")
    agg = {}
    for name, (kernels, total, *_rest) in results.items():
        for label, ms in kernels:
            agg.setdefault(label, []).append(ms)
    for label, values in sorted(agg.items(), key=lambda kv: -max(kv[1])):
        print(f"  {label:<42} steady={values[0]:>10.3f} ms  warmup={values[1]:>10.3f} ms")


if __name__ == "__main__":
    cl.profiling(True)

    results = {}
    for name, cfg in CHUNKS.items():
        results[name] = bench_chunk(name, cfg["f"], cache_groups=cfg["cache_groups"])
    print_hotspot_summary("Hotspot summary (both chunks, FlashVSR draft mask)", results)

    # Density-matched comparison: drive FlashVSR's own mask kernels to XAttention's measured
    # thresh=0.9 densities (see bench_flashvsr_e2e_xattn.py) via topk_ratio = target_density *
    # num_k_windows (see bench_chunk's docstring) -- isolates "how efficiently does
    # cm_sdpa_opt_prefill run at a given density" from "which algorithm's own density is more/less
    # aggressive" (FlashVSR's production topk_ratio=2.0 targets ~0.3-0.4% density by design).
    xattn_density = {"steady": 0.713, "warmup": 0.720}
    results_matched = {}
    for name, cfg in CHUNKS.items():
        k_groups = cfg["f"] // 2 + cfg["cache_groups"]
        num_k_windows = SPATIAL_WINDOWS * k_groups
        topk_ratio = xattn_density[name] * num_k_windows
        results_matched[name] = bench_chunk(name, cfg["f"], cache_groups=cfg["cache_groups"],
                                             topk_ratio=topk_ratio)
    print_hotspot_summary(
        "Hotspot summary (both chunks, FlashVSR mask, density-matched to XAttention)",
        results_matched)
