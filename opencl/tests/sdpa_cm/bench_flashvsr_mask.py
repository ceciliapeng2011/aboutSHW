"""
Roofline benchmark for the FlashVSR draft-block-mask kernels (flashvsr_draft_gemm.cm,
flashvsr_draft_topk.cm), at FlashVSR's real per-chunk token counts (see
test_flashvsr_configs.py's module docstring for the derivation: steady-state chunk = 19200
tokens = 150 windows, warm-up chunk = 57600 tokens = 450 windows, at NUM_HEADS=12, HEAD_DIM=128,
window=128 tokens, spatial_windows=150 for the 320x480->1280x1920 case).

flashvsr_pool is expected to be memory-bandwidth-bound (reads the full windowed Q/K, same order
of bytes as the main SDPA op); roofed against DRAM bandwidth like bench_sdpa.py's bench_decode.
flashvsr_score/flashvsr_softmax operate on tiny pooled data, expected fast regardless.
flashvsr_topk_mask's cost is dominated by its own algorithm (this is the one worth watching for
regressions as it's optimized -- see the module's design-tradeoff comment in flashvsr_draft_topk.cm).
"""
import gc
import os

import numpy as np
import torch

from clops import cl
from clops.utils import Colors

import flashvsr_mask_adapter as gpu_mod

torch.manual_seed(0)

hw_peak_bw_gbs = float(os.environ.get("HW_PEAK_BW_GBS", "100"))


def report_bw(label, bytes_moved, ms):
    bw = bytes_moved / (ms * 1e6) if ms > 0 else 0.0
    utilization = bw / hw_peak_bw_gbs * 100 if hw_peak_bw_gbs > 0 else 0.0
    print(
        f"[perf] {label} | "
        f"{Colors.BOLD}{Colors.GREEN}{bw:,.1f}{Colors.END} GB/s, "
        f"latency(ms)={Colors.BOLD}{Colors.YELLOW}{ms:.3f}{Colors.END}, "
        f"{Colors.BOLD}{Colors.GREEN}{utilization:.1f}%{Colors.END} of {hw_peak_bw_gbs:.0f} GB/s roofline"
    )


def report_ms(label, ms):
    print(f"[perf] {label} | latency(ms)={Colors.BOLD}{Colors.YELLOW}{ms:.3f}{Colors.END}")


def bench_pool(num_windows, num_heads=12, head_dim=128, window_tokens=128, n_warmup=5, n_iters=20):
    kernel = gpu_mod._get_gemm_kernel(num_heads, head_dim, window_tokens)
    windowed = torch.randn(num_windows, window_tokens, num_heads * head_dim, dtype=torch.float16).numpy()
    t_in = cl.tensor(windowed)
    t_out = cl.tensor(np.zeros((num_windows, num_heads, head_dim), dtype=np.float16))

    cl.finish()
    for _ in range(n_warmup):
        kernel.enqueue("flashvsr_pool", [num_windows, num_heads, 1], [1, 1, 1], t_in, t_out)
    cl.finish()
    for _ in range(n_iters):
        kernel.enqueue("flashvsr_pool", [num_windows, num_heads, 1], [1, 1, 1], t_in, t_out)
    lat_ms = [ns * 1e-6 for ns in cl.finish()]
    avg = sum(lat_ms) / len(lat_ms)

    bytes_read = windowed.nbytes
    report_bw(f"pool  num_windows={num_windows:<4} heads={num_heads} hd={head_dim}", bytes_read, avg)
    del t_in, t_out
    gc.collect()


def bench_score(num_q_windows, num_k_windows, spatial_windows, num_heads=12, head_dim=128,
                 n_warmup=5, n_iters=20):
    kernel = gpu_mod._get_gemm_kernel(num_heads, head_dim, 128)
    pooled_q = torch.randn(num_q_windows, num_heads, head_dim, dtype=torch.float16).numpy()
    pooled_k = torch.randn(num_k_windows, num_heads, head_dim, dtype=torch.float16).numpy()
    local_mask = (torch.rand(spatial_windows, spatial_windows) > 0.5).to(torch.uint8).numpy()
    t_q = cl.tensor(pooled_q)
    t_k = cl.tensor(pooled_k)
    t_mask = cl.tensor(local_mask)
    t_scores = cl.tensor(np.zeros((num_heads, num_q_windows, num_k_windows), dtype=np.float32))
    scale = 1.0 / (head_dim ** 0.5)
    args = [t_q, t_k, t_mask, t_scores, num_q_windows, num_k_windows, spatial_windows, scale]

    cl.finish()
    for _ in range(n_warmup):
        kernel.enqueue("flashvsr_score", [num_q_windows, num_heads, 1], [1, 1, 1], *args)
    cl.finish()
    for _ in range(n_iters):
        kernel.enqueue("flashvsr_score", [num_q_windows, num_heads, 1], [1, 1, 1], *args)
    lat_ms = [ns * 1e-6 for ns in cl.finish()]
    avg = sum(lat_ms) / len(lat_ms)
    report_ms(f"score num_q={num_q_windows:<4} num_k={num_k_windows:<4} heads={num_heads}", avg)
    del t_q, t_k, t_mask, t_scores
    gc.collect()


def bench_softmax(num_q_windows, num_k_windows, num_heads=12, n_warmup=5, n_iters=20):
    kernel = gpu_mod._get_topk_kernel(num_heads)
    raw = torch.randn(num_heads, num_q_windows, num_k_windows, dtype=torch.float32).numpy()
    t_raw = cl.tensor(raw)
    t_probs = cl.tensor(np.zeros((num_heads, num_q_windows, num_k_windows), dtype=np.float16))
    args = [t_raw, t_probs, num_k_windows]

    cl.finish()
    for _ in range(n_warmup):
        kernel.enqueue("flashvsr_softmax", [num_q_windows, num_heads, 1], [1, 1, 1], *args)
    cl.finish()
    for _ in range(n_iters):
        kernel.enqueue("flashvsr_softmax", [num_q_windows, num_heads, 1], [1, 1, 1], *args)
    lat_ms = [ns * 1e-6 for ns in cl.finish()]
    avg = sum(lat_ms) / len(lat_ms)
    report_ms(f"softmax num_q={num_q_windows:<4} num_k={num_k_windows:<4} heads={num_heads}", avg)
    del t_raw, t_probs
    gc.collect()


def bench_topk(spatial_windows, num_groups, num_k_windows, topk, num_heads=12,
               n_warmup=2, n_iters=5):
    kernel = gpu_mod._get_topk_kernel(num_heads)
    num_q_windows = spatial_windows * num_groups
    probs = torch.rand(num_heads, num_q_windows, num_k_windows, dtype=torch.float32).half().numpy()
    t_probs = cl.tensor(probs)
    t_mask = cl.tensor(np.zeros((num_heads, num_q_windows, num_k_windows), dtype=np.uint8))
    args = [t_probs, t_mask, num_q_windows, num_k_windows, spatial_windows, topk]

    cl.finish()
    for _ in range(n_warmup):
        kernel.enqueue("flashvsr_topk_mask", [num_heads, num_groups, 1], [1, 1, 1], *args)
    cl.finish()
    for _ in range(n_iters):
        kernel.enqueue("flashvsr_topk_mask", [num_heads, num_groups, 1], [1, 1, 1], *args)
    lat_ms = [ns * 1e-6 for ns in cl.finish()]
    avg = sum(lat_ms) / len(lat_ms)
    report_ms(f"topk_mask spatial={spatial_windows:<4} groups={num_groups} num_k={num_k_windows:<4} "
              f"n/group={spatial_windows*num_k_windows:<7} topk={topk} heads={num_heads}", avg)
    del t_probs, t_mask
    gc.collect()


if __name__ == "__main__":
    cl.profiling(True)

    print("\n=== flashvsr_pool (bandwidth-bound) ===")
    for num_windows in [150, 450]:  # steady / warm-up chunk window counts
        bench_pool(num_windows)

    print("\n=== flashvsr_score ===")
    bench_score(150, 150, spatial_windows=150)
    bench_score(450, 450, spatial_windows=150)

    print("\n=== flashvsr_softmax ===")
    bench_softmax(150, 150)
    bench_softmax(450, 450)

    print("\n=== flashvsr_topk_mask ===")
    # steady-state chunk: 1 temporal group, 150 spatial windows, kv from cache (~450) + current
    bench_topk(spatial_windows=150, num_groups=1, num_k_windows=450, topk=300)
    # warm-up chunk: 3 temporal groups, 150 spatial windows each, no cache yet (kv==q==450)
    bench_topk(spatial_windows=150, num_groups=3, num_k_windows=450, topk=300)
