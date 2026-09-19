"""
Roofline/utilization benchmark for all four FlashVSR draft-block-mask kernels
(flashvsr_draft_gemm.cm's flashvsr_pool/flashvsr_score, flashvsr_draft_topk.cm's
flashvsr_softmax/flashvsr_topk_mask). This is the BASELINE this module records before any
optimization work lands (see the plan doc for the radix-select/fusion changes this baseline is
meant to be compared against) -- every kernel reports a roofline utilization number (GFLOPS vs.
HW_PEAK_FLOPS for compute-bound flashvsr_score, GB/s vs. HW_PEAK_BW_GBS for the
bandwidth/pass-count-bound flashvsr_pool/flashvsr_softmax/flashvsr_topk_mask), swept across:
  1. seqlen (SEQLEN_SWEEP_WINDOWS -- num_windows from well below to well above FlashVSR's own
     150/450-window chunk sizes),
  2. density/topk (DENSITY_SWEEP -- flashvsr_topk_mask only, since it's the only one of the four
     kernels whose input actually depends on the requested top-k/density; includes 1.0 as the
     "full density" / no-pruning reference point),
  3. FlashVSR's real steady-state/warm-up chunk shapes (150/450 windows, topk=300) as a distinct
     labeled section, matching bench_flashvsr_e2e.py's real end-to-end configs.

flashvsr_pool is memory-bandwidth-bound (reads the full windowed Q/K, same order of bytes as the
main SDPA op); roofed against DRAM bandwidth like bench_sdpa.py's bench_decode.
flashvsr_score is nominally compute-bound but tiny (pooled vectors only) -- its GFLOPS utilization
number is expected to be very low (a naive scalar dot-product loop, not a matrix-engine GEMM; see
the plan doc's Stage C), which this baseline makes visible directly instead of only reporting a
latency number that looks fast in isolation.
flashvsr_topk_mask's cost is dominated by its own algorithm (16 full-array binary-search passes,
independent of the requested topk/density -- expect its GB/s-equivalent number to look roughly
flat across DENSITY_SWEEP here, which is itself a baseline finding, not a bug) -- this is the one
worth watching most closely for regressions/improvements as it's optimized.
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
# Same env convention as bench_sdpa.py -- lets this script's flashvsr_score utilization be
# compared apples-to-apples against cm_sdpa_opt_prefill's own GFLOPS/utilization numbers.
hw_peak_flops = float(os.environ.get("HW_PEAK_FLOPS", "20")) * 1e12


def report_bw(label, bytes_moved, ms):
    bw = bytes_moved / (ms * 1e6) if ms > 0 else 0.0
    utilization = bw / hw_peak_bw_gbs * 100 if hw_peak_bw_gbs > 0 else 0.0
    print(
        f"[perf] {label} | "
        f"{Colors.BOLD}{Colors.GREEN}{bw:,.1f}{Colors.END} GB/s, "
        f"latency(ms)={Colors.BOLD}{Colors.YELLOW}{ms:.3f}{Colors.END}, "
        f"{Colors.BOLD}{Colors.GREEN}{utilization:.1f}%{Colors.END} of {hw_peak_bw_gbs:.0f} GB/s roofline"
    )
    return bw, utilization


def report_gflops(label, flops, ms):
    gflops = flops / (ms * 1e6) if ms > 0 else 0.0
    utilization = flops / (ms * 1e-3) / hw_peak_flops * 100 if ms > 0 and hw_peak_flops > 0 else 0.0
    print(
        f"[perf] {label} | "
        f"GFLOPS {Colors.BOLD}{Colors.GREEN}{gflops:,.0f}{Colors.END}, "
        f"latency(ms)={Colors.BOLD}{Colors.YELLOW}{ms:.3f}{Colors.END}, "
        f"{Colors.BOLD}{Colors.GREEN}{utilization:.2f}%{Colors.END} of {hw_peak_flops/1e12:.0f} TFLOPS XMX peak"
    )
    return gflops, utilization


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
    # Single QK^T-like GEMM (no PV, unlike full attention): 2 FLOPs (mul+add) per
    # (head, q_window, k_window, head_dim) MAC, same *2 convention as bench_sdpa.py's flops_for.
    flops = 2 * num_heads * num_q_windows * num_k_windows * head_dim
    result = report_gflops(
        f"score num_q={num_q_windows:<4} num_k={num_k_windows:<4} heads={num_heads}", flops, avg)
    del t_q, t_k, t_mask, t_scores
    gc.collect()
    return result


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
    # 3 full-row passes over raw_scores (fp32): pass1 read-only (max), pass2 read+write in place
    # (exp+sum), pass3 read + write half probs (normalize) -> (4+8+4+2)=18 bytes/element total.
    n = num_heads * num_q_windows * num_k_windows
    total_bytes = n * 18
    result = report_bw(
        f"softmax num_q={num_q_windows:<4} num_k={num_k_windows:<4} heads={num_heads}",
        total_bytes, avg)
    del t_raw, t_probs
    gc.collect()
    return result


def bench_topk(spatial_windows, num_groups, num_k_windows, topk, num_heads=12,
               n_warmup=2, n_iters=5):
    kernel = gpu_mod._get_topk_kernel(num_heads)
    num_q_windows = spatial_windows * num_groups
    probs = torch.rand(num_heads, num_q_windows, num_k_windows, dtype=torch.float32).half().numpy()
    t_probs = cl.tensor(probs)
    t_mask = cl.tensor(np.zeros((num_heads, num_q_windows, num_k_windows), dtype=np.uint8))
    args = [t_probs, t_mask, num_q_windows, num_k_windows, spatial_windows, topk]

    # LWS's 3rd dim MUST be TOPK_LOCAL_SIZE -- the kernel's cooperative-binary-search workgroup
    # width (flashvsr_mask_adapter.py/bench_flashvsr_e2e.py both dispatch it this way). LWS=[1,1,1]
    # (this function's own value until fixed here) leaves lid==0 as the ONLY work item while the
    # kernel body still divides work as if TOPK_LOCAL_SIZE=32 threads existed side by side (`for i
    # = lid; i < n; i += LOCAL_SIZE`), so a single thread only ever visits 1/32 of the elements and
    # the cross-lane SLM reduction reads 31 never-written slots -- silently WRONG results, not just
    # a different perf profile.
    ls = gpu_mod.TOPK_LOCAL_SIZE
    cl.finish()
    for _ in range(n_warmup):
        kernel.enqueue("flashvsr_topk_mask", [num_heads, num_groups, ls], [1, 1, ls], *args)
    cl.finish()
    for _ in range(n_iters):
        kernel.enqueue("flashvsr_topk_mask", [num_heads, num_groups, ls], [1, 1, ls], *args)
    lat_ms = [ns * 1e-6 for ns in cl.finish()]
    avg = sum(lat_ms) / len(lat_ms)
    # 16 full-array passes (binary search) reading the ushort-sized bit pattern, plus 1 final
    # pass reading (2B) + writing (1B) the mask -- expressed as an effective-bandwidth number so
    # it's directly comparable before/after an algorithm change that reduces the pass count (e.g.
    # a radix-select would cut this to 4 passes, and this formula would report ~4x more GB/s at
    # the same wall-clock improvement).
    n_total = num_heads * num_q_windows * num_k_windows
    total_bytes = n_total * (16 * 2 + 2 + 1)
    density = topk / (spatial_windows * num_k_windows) if spatial_windows * num_k_windows > 0 else 0.0
    result = report_bw(
        f"topk_mask spatial={spatial_windows:<4} groups={num_groups} num_k={num_k_windows:<4} "
        f"n/group={spatial_windows*num_k_windows:<7} topk={topk:<6} density={density:.3f} "
        f"heads={num_heads}", total_bytes, avg)
    del t_probs, t_mask
    gc.collect()
    return result


# Seqlen sweep (num_windows swept directly -- WINDOW_TOKENS=128 elsewhere converts to tokens),
# covering well below/above FlashVSR's own real chunk sizes (150/450 windows).
SEQLEN_SWEEP_WINDOWS = [64, 150, 256, 450, 600, 900]

# Density sweep for flashvsr_topk_mask -- 1.0 is the "full density" (topk covers every element,
# i.e. no pruning at all) reference point requested alongside the sparse sweep, so the
# always-scans-everything binary search's cost-vs-density-independence is visible directly (it's
# expected to be roughly flat across this list, since the algorithm always does the same 16
# full-array passes regardless of the requested topk/density -- that flatness is itself a
# baseline finding worth recording before Stage A's radix-select changes it).
DENSITY_SWEEP = [1.0, 0.5, 0.1, 0.03, 0.01, 0.004]  # 0.004 matches FlashVSR's real measured density


if __name__ == "__main__":
    cl.profiling(True)

    print("\n=== flashvsr_pool (bandwidth-bound) -- seqlen sweep ===")
    for num_windows in SEQLEN_SWEEP_WINDOWS:
        bench_pool(num_windows)

    print("\n=== flashvsr_score -- seqlen sweep (num_groups=1, spatial_windows=num_windows) ===")
    for num_windows in SEQLEN_SWEEP_WINDOWS:
        bench_score(num_windows, num_windows, spatial_windows=num_windows)

    print("\n=== flashvsr_softmax -- seqlen sweep ===")
    for num_windows in SEQLEN_SWEEP_WINDOWS:
        bench_softmax(num_windows, num_windows)

    print("\n=== flashvsr_topk_mask -- seqlen sweep (spatial=150, topk fixed at 300 i.e. TOPK_RATIO=2) ===")
    for num_k_windows in SEQLEN_SWEEP_WINDOWS:
        bench_topk(spatial_windows=150, num_groups=1, num_k_windows=num_k_windows, topk=300)

    print("\n=== flashvsr_topk_mask -- density sweep (spatial=150, num_k_windows=450, incl. full density=1.0) ===")
    n_per_group = 150 * 450
    for density in DENSITY_SWEEP:
        topk = max(1, int(n_per_group * density))
        bench_topk(spatial_windows=150, num_groups=1, num_k_windows=450, topk=topk)

    print("\n=== FlashVSR real workload (steady/warm-up chunk shapes) ===")
    print("-- flashvsr_pool --")
    for num_windows in [150, 450]:  # steady / warm-up chunk window counts
        bench_pool(num_windows)
    print("-- flashvsr_score --")
    bench_score(150, 150, spatial_windows=150)
    bench_score(450, 450, spatial_windows=150)
    print("-- flashvsr_softmax --")
    bench_softmax(150, 150)
    bench_softmax(450, 450)
    print("-- flashvsr_topk_mask --")
    # steady-state chunk: 1 temporal group, 150 spatial windows, kv from cache (~450) + current
    bench_topk(spatial_windows=150, num_groups=1, num_k_windows=450, topk=300)
    # warm-up chunk: 3 temporal groups, 150 spatial windows each, no cache yet (kv==q==450)
    bench_topk(spatial_windows=150, num_groups=3, num_k_windows=450, topk=300)
