"""
Performance vs. roofline for the SDPA-CM kernels (Stage 1a prefill, Stage 2a decode,
Stage 3a XAttention-sparse prefill).

Prefill/XAttention are compute-bound: FLOPs = 2 GEMMs (Q@K^T, P@V) each counted as 2*M*N*K
(mul+add), causal self-attention halves that (only the lower triangle is computed), and
XAttention sparsity scales it further by the GPU-measured block-mask density. MFU/utilization
formulas and the HW_PEAK_FLOPS env convention are copied verbatim from pageatten/test_pa.py.

Decode is memory-bandwidth-bound instead (a single query token reads the entire KV cache once,
negligible compute) -- pageatten/test_pa_decoding.py's _run_bandwidth_measurement() roofs it
against DRAM bandwidth (GB/s), not TFLOPS, and that convention (kv_bytes/time == GB/s, HW_PEAK_BW_GBS
env var) is mirrored here rather than reusing the FLOPs-based roofline.

Usage: HW_PEAK_FLOPS=31 HW_PEAK_BW_GBS=100 python bench_sdpa.py
"""
import gc
import os

import numpy as np
import torch

from clops import cl
from clops.utils import Colors

import test_sdpa_prefill as prefill
import test_sdpa_decode as decode
import test_sdpa_prefill_xattn as xattn_test
import dense_xattn_adapter as xa
# test_sdpa_prefill_xattn's import above already put pageatten/ on sys.path.
from generate_block_mask import generate_block_mask_with_ratio

torch.manual_seed(0)

# Same env convention/default as pageatten/test_pa.py.
hw_peak_flops = int(os.environ.get("HW_PEAK_FLOPS", "20"))
hw_peak_flops = hw_peak_flops * 1e12
print(f"{hw_peak_flops=}")

# Decode is memory-bandwidth-bound (single query token reads the whole KV cache once, negligible
# compute), not compute-bound -- pageatten/test_pa_decoding.py's _run_bandwidth_measurement()
# roofs it against DRAM bandwidth (GB/s), not TFLOPS, for exactly this reason. Mirrored here.
hw_peak_bw_gbs = float(os.environ.get("HW_PEAK_BW_GBS", "100"))
print(f"{hw_peak_bw_gbs=}")


def flops_for(num_heads, q_len, kv_len, head_size, causal_savings, density=1.0):
    # 2 GEMMs (QK^T, PV), each 2*q_len*kv_len*head_size FLOPs (mul+add) -> *4 total.
    # Causal self-attention (q_len==kv_len) only computes the lower triangle -> halve.
    flops = num_heads * q_len * kv_len * head_size * 4
    if causal_savings:
        flops *= 0.5
    return flops * density


def block_mask_density(mask, is_causal):
    """Fraction of True entries among the blocks the kernel actually VISITS.

    For non-causal, every block is visited, so this is just mask.mean() over the full square.

    For causal, only ~half the [num_q_blocks, num_k_blocks] square (k_block <= q_block) is ever
    visited -- the kernel's own kv_stop truncation skips the rest outright, it never even loads
    them. mask.mean() over the FULL square (as bench_sdpa.py originally computed it) therefore
    reports roughly HALF the density of the visited region alone, and feeding that into
    flops_for(causal_savings=True, density=...) double-discounts the causal saving: once via the
    *0.5 causal_savings factor, once again via the density itself already being ~halved by the
    all-zero future-block half of the square. Net effect: causal total_flops (and hence
    utilization) was under-reported by roughly 2x, which is what made non-causal utilization look
    implausibly higher than causal's at matched requested density instead of the expected exactly
    2x (same mask pattern, non-causal computes the full square instead of just the lower
    triangle -- see bench_xattn_prefill_synthetic's docstring/plan-doc results for the fixed
    numbers). Fix: measure density only over the causally-reachable region.
    """
    mask = mask.to(torch.bool)
    if not is_causal:
        return mask.float().mean().item()
    B, H, num_q_blocks, num_k_blocks = mask.shape
    tril = torch.tril(torch.ones(num_q_blocks, num_k_blocks, dtype=torch.bool))
    visited_true = (mask & tril).sum().item()
    visited_total = tril.sum().item() * B * H
    return visited_true / visited_total if visited_total > 0 else 0.0


def report(label, total_flops, lat_ms):
    total_avg = sum(lat_ms) / len(lat_ms)
    total_min = min(lat_ms)
    total_max = max(lat_ms)
    gflops = total_flops / (total_avg * 1e6) if total_avg > 0 else 0.0
    utilization = total_flops / (total_avg * 1e-3) / hw_peak_flops * 100 if total_avg > 0 else 0.0
    print(
        f"[perf] {label} | "
        f"GFLOPS {Colors.BOLD}{Colors.GREEN}{gflops:,.0f}{Colors.END}, "
        f"latency(ms): min={Colors.BOLD}{Colors.YELLOW}{total_min:.3f}{Colors.END} "
        f"avg={Colors.BOLD}{Colors.YELLOW}{total_avg:.3f}{Colors.END} "
        f"max={Colors.BOLD}{Colors.YELLOW}{total_max:.3f}{Colors.END}, "
        f"{Colors.BOLD}{Colors.GREEN}{utilization:.1f}%{Colors.END} of {hw_peak_flops/1e12:.0f} TFLOPS XMX peak"
    )
    return total_avg, utilization


def report_bw(label, kv_bytes, kv_ms, inter_bytes, inter_ms):
    # bytes / ms * 1e-6 == bytes/ns == GB/s (1 GB/s = 1e9 bytes/1e9 ns = 1 byte/ns), matching
    # test_pa_decoding.py's kvcache_size*num_runs/total_ns convention exactly.
    kv_bw = kv_bytes / (kv_ms * 1e6) if kv_ms > 0 else 0.0
    inter_bw = inter_bytes / (inter_ms * 1e6) if inter_ms > 0 else 0.0
    utilization = kv_bw / hw_peak_bw_gbs * 100 if hw_peak_bw_gbs > 0 else 0.0
    print(
        f"[perf] {label} | "
        f"KV-read BW {Colors.BOLD}{Colors.GREEN}{kv_bw:,.1f}{Colors.END} GB/s "
        f"(reduce {inter_bw:,.1f} GB/s), "
        f"latency(ms): decode={Colors.BOLD}{Colors.YELLOW}{kv_ms:.3f}{Colors.END} "
        f"reduce={Colors.BOLD}{Colors.YELLOW}{inter_ms:.3f}{Colors.END}, "
        f"{Colors.BOLD}{Colors.GREEN}{utilization:.1f}%{Colors.END} of {hw_peak_bw_gbs:.0f} GB/s roofline"
    )
    return kv_bw, utilization


def bench_prefill(seq_len, num_heads=32, num_kv_heads=8, head_size=128, is_causal=True,
                   batch=1, n_warmup=10, n_iters=50):
    q, k, v = prefill.make_qkv(batch, num_heads, num_kv_heads, seq_len, seq_len, head_size)
    order = prefill.TRANSPOSE_ORDER_IDENTITY
    kernel = prefill.build_prefill_kernel(num_heads, num_kv_heads, head_size, is_causal,
                                           order, order, order, order)

    t_q = cl.tensor(q.to(torch.float16).numpy())
    t_k = cl.tensor(k.to(torch.float16).numpy())
    t_v = cl.tensor(v.to(torch.float16).numpy())
    t_out = cl.tensor(np.zeros([batch, num_heads, seq_len, head_size], dtype=np.float16))

    q_step = prefill.get_cm_grf_width() // 32
    wg_size = 16
    wg_seq_len = wg_size * q_step
    wg_count = (seq_len + wg_seq_len - 1) // wg_seq_len
    GWS = [1, batch * num_heads, wg_count * wg_size]
    LWS = [1, 1, wg_size]
    args = [t_q, t_k, t_v, t_out, seq_len, seq_len]

    cl.finish()
    for _ in range(n_warmup):
        kernel.enqueue("cm_sdpa_opt_prefill", GWS, LWS, *args)
    cl.finish()
    for _ in range(n_iters):
        kernel.enqueue("cm_sdpa_opt_prefill", GWS, LWS, *args)
    lat_ms = [ns * 1e-6 for ns in cl.finish()]

    total_flops = flops_for(num_heads, seq_len, seq_len, head_size, causal_savings=is_causal) * batch
    label = f"prefill  seq_len={seq_len:<6} heads={num_heads}/{num_kv_heads} hd={head_size} causal={is_causal}"
    result = report(label, total_flops, lat_ms)
    del t_q, t_k, t_v, t_out
    gc.collect()
    return result


def bench_decode(kv_len, num_heads=32, num_kv_heads=8, head_size=128, batch=1,
                  kv_partition_size=256, n_warmup=10, n_iters=50, max_mem_bytes=8e9):
    q, k, v = decode.make_decode_qkv(batch, num_heads, num_kv_heads, kv_len, head_size)
    order = decode.TRANSPOSE_ORDER_IDENTITY
    decode_kernel = decode.build_decode_kernel(num_heads, num_kv_heads, head_size,
                                                order, order, order, order,
                                                kv_partition_size=kv_partition_size)
    reduce_kernel = decode.build_finalization_kernel(head_size)

    num_partitions = (kv_len + kv_partition_size - 1) // kv_partition_size
    bh = batch * num_heads
    GWS = [1, bh, num_partitions]
    LWS = [1, 1, 1]
    GWS2 = [1, bh, 1]
    LWS2 = [1, 1, 1]

    # Decode is memory-bandwidth-bound, so reusing one K/V buffer across every iteration is wrong:
    # after the first read it stays resident in GPU cache (the whole K+V is only a few MB at small
    # kv_len), and every later iteration reports cache bandwidth instead of DRAM bandwidth --
    # exactly why kv_len=1024/2048 originally showed an implausible 150-170 GB/s. Fix: allocate an
    # independent K/V copy per iteration (round-robin, capped by max_mem_bytes) so each kernel
    # launch touches fresh memory, mirroring test_pa_decoding.py's _run_bandwidth_measurement
    # (all_layers / "j = i % len(all_layers)") exactly.
    q_np = q.to(torch.float16).numpy()
    k_np = k.to(torch.float16).numpy()
    v_np = v.to(torch.float16).numpy()
    per_layer_bytes = q_np.nbytes + k_np.nbytes + v_np.nbytes
    total_iters = n_warmup + n_iters
    num_layers = max(1, min(total_iters, int(max_mem_bytes // per_layer_bytes)))

    layers = []
    for _ in range(num_layers):
        t_q = cl.tensor(q_np)
        t_k = cl.tensor(k_np)
        t_v = cl.tensor(v_np)
        t_exp_sums = cl.tensor(np.zeros([bh * num_partitions], dtype=np.float32))
        t_max_logits = cl.tensor(np.zeros([bh * num_partitions], dtype=np.float32))
        t_tmp_out = cl.tensor(np.zeros([bh * num_partitions * head_size], dtype=np.float16))
        t_out = cl.tensor(np.zeros([batch, num_heads, 1, head_size], dtype=np.float16))
        layers.append((t_q, t_k, t_v, t_exp_sums, t_max_logits, t_tmp_out, t_out))

    def run_once(layer):
        t_q, t_k, t_v, t_exp_sums, t_max_logits, t_tmp_out, t_out = layer
        decode_kernel.enqueue("cm_sdpa_opt_decode", GWS, LWS,
                               t_q, t_k, t_v, t_exp_sums, t_max_logits, t_tmp_out, kv_len)
        reduce_kernel.enqueue("cm_sdpa_opt_decode_reduce", GWS2, LWS2,
                               t_exp_sums, t_max_logits, t_tmp_out, t_out, num_partitions)

    cl.finish()
    for i in range(n_warmup):
        run_once(layers[i % num_layers])
    cl.finish()
    for i in range(n_iters):
        run_once(layers[(n_warmup + i) % num_layers])
    raw_ns = cl.finish()
    # [decode, reduce, decode, reduce, ...] -- same event-pairing convention as
    # test_pa_decoding.py's _run_bandwidth_measurement.
    decode_ns_total = sum(raw_ns[2 * i] for i in range(n_iters))
    reduce_ns_total = sum(raw_ns[2 * i + 1] for i in range(n_iters))
    decode_ms = decode_ns_total * 1e-6 / n_iters
    reduce_ms = reduce_ns_total * 1e-6 / n_iters

    # Bandwidth roofline (memory-bound, not compute-bound): one full K+V read per decode step,
    # dense/no padding (unlike PA's block-cache, which pads kv_len up to a block-size multiple).
    # kv_cache_quant_utils-style byte layout: fp16 K+V, 2 bytes/elem, 2 tensors.
    kv_bytes = kv_len * num_kv_heads * head_size * 2 * 2 * batch
    # Intermediate exp_sums/max_logits/tmp_out traffic the reduce kernel reads, mirroring
    # test_pa_decoding.py's intermedia_size = batch*heads*partitions*(head_size+1)*4 (fp32).
    inter_bytes = batch * num_heads * num_partitions * (head_size + 1) * 4

    label = f"decode   kv_len={kv_len:<6} heads={num_heads}/{num_kv_heads} hd={head_size} layers={num_layers}"
    result = report_bw(label, kv_bytes, decode_ms, inter_bytes, reduce_ms)
    del layers
    gc.collect()
    return result


def bench_xattn_prefill(seq_len, num_heads=32, num_kv_heads=8, head_size=128,
                         sparse_block_size=256, thresh=0.9, is_causal=True,
                         batch=1, n_warmup=10, n_iters=50):
    q, k, v = prefill.make_qkv(batch, num_heads, num_kv_heads, seq_len, seq_len, head_size)
    mask, num_q_blocks, num_k_blocks = xa.build_sparse_mask(
        q.to(torch.float16), k.to(torch.float16), num_heads, num_kv_heads, head_size,
        sparse_block_size, is_causal, thresh)
    density = block_mask_density(mask, is_causal)
    # Release gemm_qk/find_block's large intermediate device buffers (t_query alone is
    # num_heads*head_size*2*seq_len*2 bytes, e.g. 512MB at seq_len=32768) before allocating the
    # SDPA kernel's own tensors -- avoids cumulative device-memory pressure across many cases.
    gc.collect()

    kernel = xattn_test.build_sparse_prefill_kernel(num_heads, num_kv_heads, head_size,
                                                     is_causal, sparse_block_size)

    t_q = cl.tensor(q.to(torch.float16).numpy())
    t_k = cl.tensor(k.to(torch.float16).numpy())
    t_v = cl.tensor(v.to(torch.float16).numpy())
    t_out = cl.tensor(np.zeros([batch, num_heads, seq_len, head_size], dtype=np.float16))
    t_mask = cl.tensor(mask.to(torch.bool).numpy())

    q_step = prefill.get_cm_grf_width() // 32
    wg_size = 16
    wg_seq_len = wg_size * q_step
    wg_count = (seq_len + wg_seq_len - 1) // wg_seq_len
    GWS = [1, batch * num_heads, wg_count * wg_size]
    LWS = [1, 1, wg_size]
    args = [t_q, t_k, t_v, t_out, seq_len, seq_len, t_mask, num_q_blocks, num_k_blocks]

    cl.finish()
    for _ in range(n_warmup):
        kernel.enqueue("cm_sdpa_opt_prefill", GWS, LWS, *args)
    cl.finish()
    for _ in range(n_iters):
        kernel.enqueue("cm_sdpa_opt_prefill", GWS, LWS, *args)
    lat_ms = [ns * 1e-6 for ns in cl.finish()]

    total_flops = flops_for(num_heads, seq_len, seq_len, head_size,
                             causal_savings=is_causal, density=density) * batch
    label = (f"xattn    seq_len={seq_len:<6} heads={num_heads}/{num_kv_heads} hd={head_size} "
             f"block={sparse_block_size} thresh={thresh} density={density:.3f}")
    result = report(label, total_flops, lat_ms)
    del t_q, t_k, t_v, t_out, t_mask
    gc.collect()
    return result


def fake_block_mask(num_heads, seq_len, sparse_block_size, density, is_causal, batch=1, seed=42):
    """Synthetic block mask for XAttention perf sweeps -- learned from pageatten/test_pa.py's own
    perf-sweep technique (generate_block_mask_with_ratio), which "fakes" a block mask at a
    directly-requested density instead of deriving density indirectly from an xattn_thresh run
    through the real GPU scoring pipeline. Bypasses xattn_gemm_qk/find_block.hpp entirely, so it
    also sidesteps that pipeline's known reliability issues (seq_len=32768 crash, seq_len>=8192
    mask non-determinism, occasional hangs under repeated invocation -- see bench_xattn_prefill's
    __main__ comment) and lets the sweep run at any seq_len/density combination cheaply.

    Causal: reuses generate_block_mask_with_ratio directly (single trunk spanning the whole
    sequence, trunk_sz=seq_len) -- same sink-column/local-diagonal/last-row-visible structure
    PA's own sweep uses to keep the fake mask shape "XAttention-like" rather than pure noise.
    generate_block_mask_with_ratio unconditionally intersects with the lower-triangular region
    (adjust_true_ratio_tril), which is only a valid constraint for a causal mask.

    Non-causal: PA's helper can't be reused as-is (causal papers over that tril constraint), and
    XAttention's own scoring pipeline can't even produce a *correct* non-causal mask today
    (find_block.hpp's non-causal branch has a pre-existing sort-precision bug -- see
    test_sdpa_prefill_xattn.py's module docstring) -- so a synthetic mask is the only way to
    perf-test non-causal XAttention prefill at all right now. Falls back to an i.i.d.
    Bernoulli(density) draw over the full [num_heads, num_q_blocks, num_k_blocks] grid, no tril
    restriction.

    Returns (mask, effective_density): mask is [batch, num_heads, num_q_blocks, num_k_blocks]
    bool, the exact block_mask ABI run_prefill_sparse/cm_sdpa_opt_prefill.cm expects.
    effective_density is measured via block_mask_density() -- for causal, that's density among
    only the causally-visited (k_block <= q_block) region, matching how bench_xattn_prefill
    derives density from a real GPU-computed mask (see block_mask_density's docstring for why a
    plain full-square mask.mean() is wrong here) -- NOT generate_block_mask_with_ratio's own
    returned density (which uses a different, incompatible total_flops convention).
    """
    num_q_blocks = (seq_len + sparse_block_size - 1) // sparse_block_size
    num_k_blocks = num_q_blocks
    if is_causal:
        y, _ = generate_block_mask_with_ratio(
            num_heads, seq_len, seq_len, sparse_block_size, density, is_causal=True)
        mask = y[0].unsqueeze(0).expand(batch, -1, -1, -1).contiguous()
    else:
        g = torch.Generator().manual_seed(seed)
        mask = torch.rand(batch, num_heads, num_q_blocks, num_k_blocks, generator=g) < density
    mask = mask.to(torch.bool)
    effective_density = block_mask_density(mask, is_causal)
    return mask, effective_density


def bench_xattn_prefill_synthetic(seq_len, num_heads=32, num_kv_heads=8, head_size=128,
                                   sparse_block_size=256, density=0.5, is_causal=True,
                                   batch=1, n_warmup=10, n_iters=50):
    """Same kernel/dispatch as bench_xattn_prefill, but the block mask is faked at a requested
    density (see fake_block_mask) instead of GPU-scored by the real XAttention pipeline -- this is
    what makes a clean density sweep (and non-causal coverage) possible in the first place."""
    q, k, v = prefill.make_qkv(batch, num_heads, num_kv_heads, seq_len, seq_len, head_size)
    mask, effective_density = fake_block_mask(num_heads, seq_len, sparse_block_size, density,
                                               is_causal, batch=batch)

    kernel = xattn_test.build_sparse_prefill_kernel(num_heads, num_kv_heads, head_size,
                                                     is_causal, sparse_block_size)

    t_q = cl.tensor(q.to(torch.float16).numpy())
    t_k = cl.tensor(k.to(torch.float16).numpy())
    t_v = cl.tensor(v.to(torch.float16).numpy())
    t_out = cl.tensor(np.zeros([batch, num_heads, seq_len, head_size], dtype=np.float16))
    t_mask = cl.tensor(mask.numpy())

    q_step = prefill.get_cm_grf_width() // 32
    wg_size = 16
    wg_seq_len = wg_size * q_step
    wg_count = (seq_len + wg_seq_len - 1) // wg_seq_len
    GWS = [1, batch * num_heads, wg_count * wg_size]
    LWS = [1, 1, wg_size]
    num_q_blocks, num_k_blocks = mask.shape[2], mask.shape[3]
    args = [t_q, t_k, t_v, t_out, seq_len, seq_len, t_mask, num_q_blocks, num_k_blocks]

    cl.finish()
    for _ in range(n_warmup):
        kernel.enqueue("cm_sdpa_opt_prefill", GWS, LWS, *args)
    cl.finish()
    for _ in range(n_iters):
        kernel.enqueue("cm_sdpa_opt_prefill", GWS, LWS, *args)
    lat_ms = [ns * 1e-6 for ns in cl.finish()]

    total_flops = flops_for(num_heads, seq_len, seq_len, head_size,
                             causal_savings=is_causal, density=effective_density) * batch
    label = (f"xattn*   seq_len={seq_len:<6} heads={num_heads}/{num_kv_heads} hd={head_size} "
             f"block={sparse_block_size} causal={is_causal} density(req/eff)="
             f"{density:.2f}/{effective_density:.3f} (fake mask)")
    result = report(label, total_flops, lat_ms)
    del t_q, t_k, t_v, t_out, t_mask
    gc.collect()
    return result


def run_case(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except Exception as e:
        print(f"[perf] SKIPPED {fn.__name__}{args} {kwargs}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    cl.profiling(True)

    print("\n=== Stage 1a: SDPA prefill (dense, causal) ===")
    for seq_len in [512, 1024, 2048, 4096, 8192, 16384, 32768]:
        run_case(bench_prefill, seq_len)

    print("\n=== Stage 1a: SDPA prefill (dense, MHA no GQA, for comparison) ===")
    run_case(bench_prefill, 8192, num_kv_heads=32)

    print("\n=== Stage 2a: SDPA decode (dense, single-token generate) ===")
    for kv_len in [512, 1024, 2048, 4096, 8192, 16384, 32768]:
        run_case(bench_decode, kv_len)

    print("\n=== Stage 3a: SDPA prefill + XAttention (sparse, causal) ===")
    # Previously capped at seq_len=4096 and guarded against a hard GPU crash at seq_len=32768,
    # non-deterministic mask density at seq_len>=8192, and hangs under repeated invocation. Root
    # cause (fixed in x_attn/test_find_block.py + find_block.hpp): find_block's DEBUG_ACC kq_sum
    # buffer was allocated as np.float16 on the host while the kernel writes SOFTMAX_TYPE (always
    # 'float', 4 bytes) elements into it -- a 2x undersized buffer whose out-of-bounds device
    # writes scaled with num_heads * q_block_pad * k_block_pad: silent corruption of unrelated
    # buffers at smaller sizes (explaining the density non-determinism) and a hard page fault once
    # the overflow walked past mapped memory at larger sizes (explaining the crash; the hang was
    # the GPU/driver recovering from that fault). find_block.hpp's causal branch also destructively
    # overwrote its score buffer in place, made reentrant as a secondary robustness fix (harmless
    # for this loop's fresh-buffer-per-call pattern, but required for n_repeats>1 perf testing).
    # Verified: full seq_len range + all thresholds/block sizes now run clean with no crash/hang.
    for sparse_block_size in (256, 128):
        for thresh in (0.99, 0.9, 0.6, 0.3):
            for seq_len in (4096, 8192, 16384, 32768):
                run_case(bench_xattn_prefill, seq_len, sparse_block_size=sparse_block_size, thresh=thresh)

    print("\n=== Stage 3a: SDPA prefill + XAttention (fake mask, density sweep, causal) ===")
    # fake_block_mask sidesteps the real scoring pipeline's reliability issues above entirely, so
    # this sweep can cover a much wider seq_len range and an exact, repeatable density grid instead
    # of density being an indirect function of xattn_thresh.
    for seq_len in [4096, 8192, 16384, 32768]:
        for density in (0.99, 0.9, 0.6, 0.3, 0.1):
            run_case(bench_xattn_prefill_synthetic, seq_len, density=density, is_causal=True)

    print("\n=== Stage 3a: SDPA prefill + XAttention (fake mask, density sweep, non-causal) ===")
    # XAttention's own scoring pipeline cannot correctly produce a non-causal mask at all today
    # (find_block.hpp's non-causal branch has a pre-existing sort-precision bug -- see
    # test_sdpa_prefill_xattn.py's module docstring), so the fake mask is the only way to
    # perf-test this configuration right now.
    for seq_len in [4096, 8192, 16384, 32768]:
        for density in (0.99, 0.9, 0.6, 0.3, 0.1):
            run_case(bench_xattn_prefill_synthetic, seq_len, density=density, is_causal=False)
