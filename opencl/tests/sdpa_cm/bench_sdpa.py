"""
Performance vs. roofline for the SDPA-CM kernels (Stage 1a prefill, Stage 2a decode,
prefill with a generic block-sparse mask).

Prefill is compute-bound: FLOPs = 2 GEMMs (Q@K^T, P@V) each counted as 2*M*N*K (mul+add), causal
self-attention halves that (only the lower triangle is computed), and a block-sparse mask scales
it further by the GPU-measured block-mask density. MFU/utilization formulas and the HW_PEAK_FLOPS
env convention are copied verbatim from pageatten/test_pa.py.

Decode is memory-bandwidth-bound instead (a single query token reads the entire KV cache once,
negligible compute) -- pageatten/test_pa_decoding.py's _run_bandwidth_measurement() roofs it
against DRAM bandwidth (GB/s), not TFLOPS, and that convention (kv_bytes/time == GB/s, HW_PEAK_BW_GBS
env var) is mirrored here rather than reusing the FLOPs-based roofline.

Usage: HW_PEAK_FLOPS=58 HW_PEAK_BW_GBS=100 python bench_sdpa.py
"""
import gc
import os

import numpy as np
import torch

from clops import cl
from clops.utils import Colors

import test_sdpa_prefill as prefill
import test_sdpa_decode as decode
import test_sdpa_prefill_block_mask as block_mask_test
import test_sdpa_prefill_fp8 as prefill_fp8
# test_sdpa_prefill_block_mask's import above already put pageatten/ on sys.path.
from generate_block_mask import generate_block_mask_with_ratio
from hf8_e4m3_ref import encode_f16_to_hf8, build_hf8_decode_lut_f16

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
    triangle -- see bench_prefill_block_mask_synthetic's docstring/plan-doc results for the fixed
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
                   batch=1, n_warmup=10, n_iters=50, order=None):
    # order=None -> identity layout ([B,H,S,D], OCL SDPA's default). Pass
    # prefill.TRANSPOSE_ORDER_SWAPPED to benchmark the {0,2,1,3}-fused heads<->seq layout
    # ([B,S,H,D]) instead -- applied uniformly to Q/K/V/output, matching a real transpose-fused
    # subgraph end-to-end. Same total bytes moved either way; only the LSC block-load/store pitch
    # differs (see cm_sdpa_opt_common.hpp) -- this measures whether that pitch difference itself
    # costs anything.
    if order is None:
        order = prefill.TRANSPOSE_ORDER_IDENTITY
    q, k, v = prefill.make_qkv(batch, num_heads, num_kv_heads, seq_len, seq_len, head_size)
    kernel = prefill.build_prefill_kernel(num_heads, num_kv_heads, head_size, is_causal,
                                           order, order, order, order)

    q_phys = prefill.to_physical(q, order)
    k_phys = prefill.to_physical(k, order)
    v_phys = prefill.to_physical(v, order)
    t_q = cl.tensor(q_phys.to(torch.float16).numpy())
    t_k = cl.tensor(k_phys.to(torch.float16).numpy())
    t_v = cl.tensor(v_phys.to(torch.float16).numpy())
    out_shape = ([batch, seq_len, num_heads, head_size] if order == prefill.TRANSPOSE_ORDER_SWAPPED
                 else [batch, num_heads, seq_len, head_size])
    t_out = cl.tensor(np.zeros(out_shape, dtype=np.float16))

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
    order_tag = "swapped" if order == prefill.TRANSPOSE_ORDER_SWAPPED else "identity"
    label = (f"prefill  seq_len={seq_len:<6} heads={num_heads}/{num_kv_heads} hd={head_size} "
             f"causal={is_causal} order={order_tag}")
    result = report(label, total_flops, lat_ms)
    del t_q, t_k, t_v, t_out
    gc.collect()
    return result


def bench_prefill_fp8(seq_len, num_heads=32, num_kv_heads=8, head_size=128, is_causal=True,
                       batch=1, n_warmup=10, n_iters=50, order=None, sparse_block_size=1,
                       density=1.0):
    """KV_FP8E4M3=1 counterpart of bench_prefill, same shape/dispatch/timing convention -- K/V are
    uploaded as raw fp8e4m3 bytes (1 byte/elem) instead of half (2 bytes/elem), decoded to half
    once per work-group and staged through SLM (see cm_sdpa_common.hpp's sdpa_kernel_lsc_fp8 --
    measured faster than an earlier per-work-item decode variant, both bit-manipulation and
    LUT-gather, which has since been removed). Values/correctness don't matter here (this is a
    pure perf probe, see test_sdpa_prefill_fp8.py for correctness) -- only the byte pattern's
    on-the-wire size (half the K/V bytes moved vs. fp16) and the V-path's extra decode/
    VNNI-repack software overhead are what's being measured against the fp16 baseline.

    sparse_block_size>1 (with a matching density in (0,1]) additionally benchmarks a block-sparse
    mask on top of the fp8 decode -- the mask is faked at the requested density exactly like
    bench_prefill_block_mask_synthetic's fake_block_mask, so this measures the COMBINED win of fp8
    + block-sparsity (not each in isolation). SPARSE_BLOCK_SIZE must be in (1, 128, 256) -- same
    restriction cm_sdpa_opt_prefill.cm's static_assert enforces (see sdpa_kernel_lsc_fp8's doc
    comment).

    order: see bench_prefill's docstring."""
    assert sparse_block_size in (1, 128, 256), (
        "sparse_block_size must be 1 (dense), 128, or 256 -- matches "
        "cm_sdpa_opt_prefill.cm's KV_FP8E4M3 static_assert")
    if order is None:
        order = prefill.TRANSPOSE_ORDER_IDENTITY
    q, k, v = prefill.make_qkv(batch, num_heads, num_kv_heads, seq_len, seq_len, head_size)
    kernel = prefill_fp8.build_prefill_kernel_fp8(num_heads, num_kv_heads, head_size, is_causal,
                                                   order, order, order, order, sparse_block_size)

    q_phys = prefill.to_physical(q, order)
    k_hf8 = encode_f16_to_hf8(prefill.to_physical(k, order).contiguous().numpy())
    v_hf8 = encode_f16_to_hf8(prefill.to_physical(v, order).contiguous().numpy())
    t_q = cl.tensor(q_phys.to(torch.float16).numpy())
    t_k = cl.tensor(k_hf8)
    t_v = cl.tensor(v_hf8)
    out_shape = ([batch, seq_len, num_heads, head_size] if order == prefill.TRANSPOSE_ORDER_SWAPPED
                 else [batch, num_heads, seq_len, head_size])
    t_out = cl.tensor(np.zeros(out_shape, dtype=np.float16))

    q_step = prefill.get_cm_grf_width() // 32
    wg_size = 16
    wg_seq_len = wg_size * q_step
    wg_count = (seq_len + wg_seq_len - 1) // wg_seq_len
    GWS = [1, batch * num_heads, wg_count * wg_size]
    LWS = [1, 1, wg_size]
    args = [t_q, t_k, t_v, t_out, seq_len, seq_len]
    t_mask = None
    effective_density = 1.0
    if sparse_block_size > 1:
        mask, effective_density = fake_block_mask(num_heads, seq_len, sparse_block_size, density,
                                                   is_causal, batch=batch)
        num_q_blocks, num_k_blocks = mask.shape[2], mask.shape[3]
        t_mask = cl.tensor(mask.numpy())
        args += [t_mask, num_q_blocks, num_k_blocks]
    # KV_FP8E4M3 always needs the 256-entry half LUT -- sdpa_kernel_lsc_fp8 always uses the LUT
    # decode.
    t_lut = cl.tensor(build_hf8_decode_lut_f16())
    args += [t_lut]

    kernel_name = "cm_sdpa_opt_prefill_fp8"
    cl.finish()
    for _ in range(n_warmup):
        kernel.enqueue(kernel_name, GWS, LWS, *args)
    cl.finish()
    for _ in range(n_iters):
        kernel.enqueue(kernel_name, GWS, LWS, *args)
    lat_ms = [ns * 1e-6 for ns in cl.finish()]

    total_flops = flops_for(num_heads, seq_len, seq_len, head_size, causal_savings=is_causal,
                             density=effective_density) * batch
    order_tag = "swapped" if order == prefill.TRANSPOSE_ORDER_SWAPPED else "identity"
    decode_tag = "KV_FP8E4M3=1"
    sparse_tag = (f" block={sparse_block_size} density(req/eff)={density:.2f}/{effective_density:.3f}"
                  if sparse_block_size > 1 else "")
    label = (f"prefill* seq_len={seq_len:<6} heads={num_heads}/{num_kv_heads} hd={head_size} "
             f"causal={is_causal} order={order_tag} {decode_tag}{sparse_tag}")
    result = report(label, total_flops, lat_ms)
    del t_q, t_k, t_v, t_out, t_mask, t_lut
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


def bench_cross_attn(q_len, kv_len, num_heads=12, num_kv_heads=12, head_size=128,
                      batch=1, n_warmup=10, n_iters=50, order=None, op_name="cross"):
    """Dense, non-causal, q_len != kv_len -- e.g. FlashVSR's cross-attention (long video-chunk Q
    against a short fixed-length text-context K/V). bench_prefill assumes q_len==kv_len, so this
    is a thin variant calling prefill.make_qkv/build_prefill_kernel directly with independent
    lengths, same dispatch/timing structure otherwise. order: see bench_prefill's docstring.

    op_name: label prefix only (no computational difference) -- self-attention's REAL
    streaming-KV-cache shape (q_len != kv_len, see plan doc's "Streaming KV-cache mechanism"
    section) is the exact same dense/non-causal/q_len!=kv_len computation as cross-attention, so
    this function is reused as-is for that case too, just with op_name="self" to keep the printed
    label honest about which op it represents."""
    if order is None:
        order = prefill.TRANSPOSE_ORDER_IDENTITY
    q, k, v = prefill.make_qkv(batch, num_heads, num_kv_heads, q_len, kv_len, head_size)
    kernel = prefill.build_prefill_kernel(num_heads, num_kv_heads, head_size, False,
                                           order, order, order, order)

    q_phys = prefill.to_physical(q, order)
    k_phys = prefill.to_physical(k, order)
    v_phys = prefill.to_physical(v, order)
    t_q = cl.tensor(q_phys.to(torch.float16).numpy())
    t_k = cl.tensor(k_phys.to(torch.float16).numpy())
    t_v = cl.tensor(v_phys.to(torch.float16).numpy())
    out_shape = ([batch, q_len, num_heads, head_size] if order == prefill.TRANSPOSE_ORDER_SWAPPED
                 else [batch, num_heads, q_len, head_size])
    t_out = cl.tensor(np.zeros(out_shape, dtype=np.float16))

    q_step = prefill.get_cm_grf_width() // 32
    wg_size = 16
    wg_seq_len = wg_size * q_step
    wg_count = (q_len + wg_seq_len - 1) // wg_seq_len
    GWS = [1, batch * num_heads, wg_count * wg_size]
    LWS = [1, 1, wg_size]
    args = [t_q, t_k, t_v, t_out, q_len, kv_len]

    cl.finish()
    for _ in range(n_warmup):
        kernel.enqueue("cm_sdpa_opt_prefill", GWS, LWS, *args)
    cl.finish()
    for _ in range(n_iters):
        kernel.enqueue("cm_sdpa_opt_prefill", GWS, LWS, *args)
    lat_ms = [ns * 1e-6 for ns in cl.finish()]

    total_flops = flops_for(num_heads, q_len, kv_len, head_size, causal_savings=False) * batch
    order_tag = "swapped" if order == prefill.TRANSPOSE_ORDER_SWAPPED else "identity"
    label = (f"{op_name:<8} q_len={q_len:<6} kv_len={kv_len:<6} heads={num_heads}/{num_kv_heads} "
             f"hd={head_size} order={order_tag}")
    result = report(label, total_flops, lat_ms)
    del t_q, t_k, t_v, t_out
    gc.collect()
    return result


def fake_block_mask(num_heads, seq_len, sparse_block_size, density, is_causal, batch=1, seed=42):
    """Synthetic block mask for block-sparse prefill perf sweeps -- learned from pageatten/test_pa.py's own
    perf-sweep technique (generate_block_mask_with_ratio), which "fakes" a block mask at a
    directly-requested density rather than deriving density indirectly from any real GPU-scored
    mask-generation pipeline. This lets the sweep run at any seq_len/density combination cheaply,
    with no dependency on (or reliability constraints from) such a pipeline.

    Causal: reuses generate_block_mask_with_ratio directly (single trunk spanning the whole
    sequence, trunk_sz=seq_len) -- same sink-column/local-diagonal/last-row-visible structure
    PA's own sweep uses to keep the fake mask shape realistic rather than pure noise.
    generate_block_mask_with_ratio unconditionally intersects with the lower-triangular region
    (adjust_true_ratio_tril), which is only a valid constraint for a causal mask.

    Non-causal: PA's helper can't be reused as-is (causal papers over that tril constraint), so
    this falls back to an i.i.d. Bernoulli(density) draw over the full
    [num_heads, num_q_blocks, num_k_blocks] grid, no tril restriction.

    Returns (mask, effective_density): mask is [batch, num_heads, num_q_blocks, num_k_blocks]
    bool, the exact block_mask ABI run_prefill_sparse/cm_sdpa_opt_prefill.cm expects.
    effective_density is measured via block_mask_density() -- for causal, that's density among
    only the causally-visited (k_block <= q_block) region (see block_mask_density's docstring for
    why a plain full-square mask.mean() is wrong here) -- NOT generate_block_mask_with_ratio's own
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


def bench_prefill_block_mask_synthetic(seq_len, num_heads=32, num_kv_heads=8, head_size=128,
                                        sparse_block_size=256, density=0.5, is_causal=True,
                                        batch=1, n_warmup=10, n_iters=50, order=None):
    """Block-sparse prefill perf sweep with the block mask faked at a requested density (see
    fake_block_mask) -- this is what makes a clean density sweep (and non-causal coverage)
    possible without depending on any real mask-generation pipeline.
    order: see bench_prefill's docstring (applied to Q/K/V/output uniformly)."""
    if order is None:
        order = prefill.TRANSPOSE_ORDER_IDENTITY
    q, k, v = prefill.make_qkv(batch, num_heads, num_kv_heads, seq_len, seq_len, head_size)
    mask, effective_density = fake_block_mask(num_heads, seq_len, sparse_block_size, density,
                                               is_causal, batch=batch)

    kernel = block_mask_test.build_sparse_prefill_kernel(num_heads, num_kv_heads, head_size,
                                                          is_causal, sparse_block_size,
                                                          order, order, order, order)

    q_phys = prefill.to_physical(q, order)
    k_phys = prefill.to_physical(k, order)
    v_phys = prefill.to_physical(v, order)
    t_q = cl.tensor(q_phys.to(torch.float16).numpy())
    t_k = cl.tensor(k_phys.to(torch.float16).numpy())
    t_v = cl.tensor(v_phys.to(torch.float16).numpy())
    out_shape = ([batch, seq_len, num_heads, head_size] if order == prefill.TRANSPOSE_ORDER_SWAPPED
                 else [batch, num_heads, seq_len, head_size])
    t_out = cl.tensor(np.zeros(out_shape, dtype=np.float16))
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
    order_tag = "swapped" if order == prefill.TRANSPOSE_ORDER_SWAPPED else "identity"
    label = (f"blkmask* seq_len={seq_len:<6} heads={num_heads}/{num_kv_heads} hd={head_size} "
             f"block={sparse_block_size} causal={is_causal} order={order_tag} density(req/eff)="
             f"{density:.2f}/{effective_density:.3f} (fake mask)")
    result = report(label, total_flops, lat_ms)
    del t_q, t_k, t_v, t_out, t_mask
    gc.collect()
    return result


def fake_block_mask_qkv(num_heads, q_len, kv_len, sparse_block_size, density, batch=1, seed=42):
    """q_len != kv_len variant of fake_block_mask -- needed for FlashVSR's REAL streaming
    KV-cache self-attention shape (see plan doc's "Streaming KV-cache mechanism" section: steady
    -state q_len=19,200 vs. kv_len=76,800 = 3 cached + 1 current temporal groups). Non-causal only
    (fake_block_mask's causal branch pulls generate_block_mask_with_ratio, which assumes a single
    square trunk spanning one seq_len and does not generalize to independent q/k block counts;
    FlashVSR's self-attention call is always is_causal=False anyway -- see bench_sdpa.py's
    FlashVSR-config __main__ section comment -- so this omission costs nothing here)."""
    num_q_blocks = (q_len + sparse_block_size - 1) // sparse_block_size
    num_k_blocks = (kv_len + sparse_block_size - 1) // sparse_block_size
    g = torch.Generator().manual_seed(seed)
    mask = (torch.rand(batch, num_heads, num_q_blocks, num_k_blocks, generator=g) < density).to(torch.bool)
    effective_density = block_mask_density(mask, is_causal=False)
    return mask, effective_density


def bench_prefill_block_mask_synthetic_qkv(q_len, kv_len, num_heads=32, num_kv_heads=8, head_size=128,
                                            sparse_block_size=256, density=0.5,
                                            batch=1, n_warmup=10, n_iters=50, order=None):
    """q_len != kv_len variant of bench_prefill_block_mask_synthetic -- same kernel/dispatch, but Q
    and K/V get independent lengths (fake_block_mask_qkv) instead of a shared seq_len. Non-causal
    only, see fake_block_mask_qkv's docstring. Needed for FlashVSR's real streaming-KV-cache
    self-attention shape sparse benchmark (steady-state q_len=19,200/kv_len=76,800)."""
    if order is None:
        order = prefill.TRANSPOSE_ORDER_IDENTITY
    q, k, v = prefill.make_qkv(batch, num_heads, num_kv_heads, q_len, kv_len, head_size)
    mask, effective_density = fake_block_mask_qkv(num_heads, q_len, kv_len, sparse_block_size,
                                                   density, batch=batch)

    kernel = block_mask_test.build_sparse_prefill_kernel(num_heads, num_kv_heads, head_size,
                                                          False, sparse_block_size,
                                                          order, order, order, order)

    q_phys = prefill.to_physical(q, order)
    k_phys = prefill.to_physical(k, order)
    v_phys = prefill.to_physical(v, order)
    t_q = cl.tensor(q_phys.to(torch.float16).numpy())
    t_k = cl.tensor(k_phys.to(torch.float16).numpy())
    t_v = cl.tensor(v_phys.to(torch.float16).numpy())
    out_shape = ([batch, q_len, num_heads, head_size] if order == prefill.TRANSPOSE_ORDER_SWAPPED
                 else [batch, num_heads, q_len, head_size])
    t_out = cl.tensor(np.zeros(out_shape, dtype=np.float16))
    t_mask = cl.tensor(mask.numpy())

    q_step = prefill.get_cm_grf_width() // 32
    wg_size = 16
    wg_seq_len = wg_size * q_step
    wg_count = (q_len + wg_seq_len - 1) // wg_seq_len
    GWS = [1, batch * num_heads, wg_count * wg_size]
    LWS = [1, 1, wg_size]
    num_q_blocks, num_k_blocks = mask.shape[2], mask.shape[3]
    args = [t_q, t_k, t_v, t_out, q_len, kv_len, t_mask, num_q_blocks, num_k_blocks]

    cl.finish()
    for _ in range(n_warmup):
        kernel.enqueue("cm_sdpa_opt_prefill", GWS, LWS, *args)
    cl.finish()
    for _ in range(n_iters):
        kernel.enqueue("cm_sdpa_opt_prefill", GWS, LWS, *args)
    lat_ms = [ns * 1e-6 for ns in cl.finish()]

    total_flops = flops_for(num_heads, q_len, kv_len, head_size,
                             causal_savings=False, density=effective_density) * batch
    order_tag = "swapped" if order == prefill.TRANSPOSE_ORDER_SWAPPED else "identity"
    label = (f"blkmask* q_len={q_len:<6} kv_len={kv_len:<6} heads={num_heads}/{num_kv_heads} "
             f"hd={head_size} block={sparse_block_size} causal=False order={order_tag} "
             f"density(req/eff)={density:.2f}/{effective_density:.3f} (fake mask)")
    result = report(label, total_flops, lat_ms)
    del t_q, t_k, t_v, t_out, t_mask
    gc.collect()
    return result


def bench_prefill_fp8_qkv(q_len, kv_len, num_heads=32, num_kv_heads=8, head_size=128,
                           batch=1, n_warmup=10, n_iters=50, order=None, sparse_block_size=1,
                           density=1.0):
    """q_len != kv_len variant of bench_prefill_fp8 -- needed for FlashVSR's REAL streaming
    KV-cache self-attention shape (steady-state q_len=19,200/kv_len=76,800, see
    fake_block_mask_qkv's docstring). Non-causal only, same restriction as
    bench_prefill_block_mask_synthetic_qkv (FlashVSR's self-attention call is always
    is_causal=False -- see bench_sdpa.py's FlashVSR-config __main__ section comment)."""
    assert sparse_block_size in (1, 128, 256), (
        "sparse_block_size must be 1 (dense), 128, or 256 -- matches "
        "cm_sdpa_opt_prefill.cm's KV_FP8E4M3 static_assert")
    if order is None:
        order = prefill.TRANSPOSE_ORDER_IDENTITY
    q, k, v = prefill.make_qkv(batch, num_heads, num_kv_heads, q_len, kv_len, head_size)
    kernel = prefill_fp8.build_prefill_kernel_fp8(num_heads, num_kv_heads, head_size, False,
                                                   order, order, order, order, sparse_block_size)

    q_phys = prefill.to_physical(q, order)
    k_hf8 = encode_f16_to_hf8(prefill.to_physical(k, order).contiguous().numpy())
    v_hf8 = encode_f16_to_hf8(prefill.to_physical(v, order).contiguous().numpy())
    t_q = cl.tensor(q_phys.to(torch.float16).numpy())
    t_k = cl.tensor(k_hf8)
    t_v = cl.tensor(v_hf8)
    out_shape = ([batch, q_len, num_heads, head_size] if order == prefill.TRANSPOSE_ORDER_SWAPPED
                 else [batch, num_heads, q_len, head_size])
    t_out = cl.tensor(np.zeros(out_shape, dtype=np.float16))

    q_step = prefill.get_cm_grf_width() // 32
    wg_size = 16
    wg_seq_len = wg_size * q_step
    wg_count = (q_len + wg_seq_len - 1) // wg_seq_len
    GWS = [1, batch * num_heads, wg_count * wg_size]
    LWS = [1, 1, wg_size]
    args = [t_q, t_k, t_v, t_out, q_len, kv_len]
    t_mask = None
    effective_density = 1.0
    if sparse_block_size > 1:
        mask, effective_density = fake_block_mask_qkv(num_heads, q_len, kv_len, sparse_block_size,
                                                        density, batch=batch)
        num_q_blocks, num_k_blocks = mask.shape[2], mask.shape[3]
        t_mask = cl.tensor(mask.numpy())
        args += [t_mask, num_q_blocks, num_k_blocks]
    # KV_FP8E4M3 always needs the 256-entry half LUT -- sdpa_kernel_lsc_fp8 always uses the LUT
    # decode.
    t_lut = cl.tensor(build_hf8_decode_lut_f16())
    args += [t_lut]

    kernel_name = "cm_sdpa_opt_prefill_fp8"
    cl.finish()
    for _ in range(n_warmup):
        kernel.enqueue(kernel_name, GWS, LWS, *args)
    cl.finish()
    for _ in range(n_iters):
        kernel.enqueue(kernel_name, GWS, LWS, *args)
    lat_ms = [ns * 1e-6 for ns in cl.finish()]

    total_flops = flops_for(num_heads, q_len, kv_len, head_size, causal_savings=False,
                             density=effective_density) * batch
    order_tag = "swapped" if order == prefill.TRANSPOSE_ORDER_SWAPPED else "identity"
    decode_tag = "KV_FP8E4M3=1"
    sparse_tag = (f" block={sparse_block_size} density(req/eff)={density:.2f}/{effective_density:.3f}"
                  if sparse_block_size > 1 else "")
    label = (f"prefill* q_len={q_len:<6} kv_len={kv_len:<6} heads={num_heads}/{num_kv_heads} "
             f"hd={head_size} causal=False order={order_tag} {decode_tag}{sparse_tag}")
    result = report(label, total_flops, lat_ms)
    del t_q, t_k, t_v, t_out, t_mask, t_lut
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

    print("\n=== Stage 1a: SDPA prefill, fp16 vs KV_FP8E4M3=1 (decode-once-per-WG, LUT decode "
          "staged through SLM) (same shapes as above) ===")
    # Same head_size=128/num_heads=32/num_kv_heads=8 dense-causal config as the sweep above, fp16
    # and fp8 printed back-to-back per seq_len for a direct GFLOPS/latency comparison (see
    # test_sdpa_prefill_fp8.py for correctness). An earlier per-work-item decode variant (every one
    # of the wg_local_size=16 lanes in a work-group redundantly decoding the same K/V bytes, both a
    # bit-manipulation and a LUT-gather version) was measured slower than this decode-once-per-WG
    # approach and has since been removed -- see cm_sdpa_common.hpp's sdpa_kernel_lsc_fp8.
    # seq_len=32768 added: FlashVSR's real-world token count is ~32k, so this is the actually
    # relevant scale, beyond the original 16384 cap -- also useful to see whether the SLM path's
    # unexplained seq_len=16384 dip (82.9% vs 95.5-99.7% at smaller sizes) continues or recovers.
    for seq_len in [2048, 4096, 8192, 16384, 32768]:
        run_case(bench_prefill, seq_len)
        run_case(bench_prefill_fp8, seq_len)

    print("\n=== Stage 2a: SDPA decode (dense, single-token generate) ===")
    for kv_len in [512, 1024, 2048, 4096, 8192, 16384, 32768]:
        run_case(bench_decode, kv_len)

    print("\n=== Stage 3a: SDPA prefill + block mask (fake mask, density sweep, causal) ===")
    # fake_block_mask lets this sweep cover a wide seq_len range and an exact, repeatable density
    # grid without depending on any real mask-generation pipeline.
    for seq_len in [4096, 8192, 16384, 32768]:
        for density in (0.99, 0.9, 0.6, 0.3, 0.1):
            run_case(bench_prefill_block_mask_synthetic, seq_len, density=density, is_causal=True)

    print("\n=== Stage 3a: SDPA prefill + block mask (fake mask, density sweep, non-causal) ===")
    for seq_len in [4096, 8192, 16384, 32768]:
        for density in (0.99, 0.9, 0.6, 0.3, 0.1):
            run_case(bench_prefill_block_mask_synthetic, seq_len, density=density, is_causal=False)

    print("\n=== Stage 4: SDPA prefill + block mask + fp8, fp16-sparse baseline vs KV_FP8E4M3=1, "
          "same shapes, FlashVSR-realistic density (~0.35) -- measures the COMBINED win of fp8 + "
          "block-sparsity, not each in isolation. sparse_block_size=256 is the WG-uniform, "
          "simplest case (see sdpa_kernel_lsc_fp8's doc comment); 256 and 128 must both work today "
          "(both bit-exact per test_sdpa_prefill_fp8.py) but only 256 is swept at every seq_len "
          "here to bound total run time -- 128's two-level mask-merge overhead is checked "
          "separately, once, at the largest (32768) size below. ===")
    SPARSE_DENSITY = 0.35
    for seq_len in [4096, 8192, 16384, 32768]:
        run_case(bench_prefill_block_mask_synthetic, seq_len, sparse_block_size=256,
                 density=SPARSE_DENSITY, is_causal=True)
        run_case(bench_prefill_fp8, seq_len, sparse_block_size=256, density=SPARSE_DENSITY,
                 is_causal=True)

    print("\n=== Stage 4b: same as above but sparse_block_size=128 (the harder two-level "
          "skip-LOAD/skip-COMPUTE mask split -- see sdpa_kernel_lsc_fp8's doc comment), only at "
          "seq_len=32768 (FlashVSR's real scale) to see the overhead vs. 256 above without "
          "re-running the full size sweep twice ===")
    run_case(bench_prefill_block_mask_synthetic, 32768, sparse_block_size=128,
             density=SPARSE_DENSITY, is_causal=True)
    run_case(bench_prefill_fp8, 32768, sparse_block_size=128, density=SPARSE_DENSITY,
             is_causal=True)

    print("\n=== FlashVSR configs: DiT self-/cross-attention (MHA heads=12/12 hd=128) ===")
    # Shapes from /home/ceciliapeng/repos/reports/flashvsr_v1.1_full_inference_pipeline.md, for the
    # real requested case: a 320x480 (portrait) source upsampled by FlashVSR's fixed 4x scale to a
    # 1280x1920 target (compute_scaled_and_target_dims in infer_flashvsr_v1.1_full.py -- lands
    # exactly on multiples of 128 for this resolution, no flooring loss). VAE 8x spatial downsample
    # -> latent 160x240 -> patchify (2x2) -> post-patchify token grid h*w = 80*120 = 9600. Chunk
    # token count L = f*h*w: steady-state chunk (f=2) = 19200, warm-up chunk (f=6) = 57600.
    # cross-attention kv_len=512 is FlashVSR's fixed text-context length. Self-attention's own
    # kernel call is always non-causal -- temporal causality is handled entirely by a Python-side
    # FIFO KV-cache outside the attention op. The q_len==kv_len==current-chunk-only cases below
    # are KEPT AS-IS as a "no cache" reference point, but are a lower bound on real self-attention
    # cost: they don't model that FIFO cache's extra KV tokens. See plan doc's "Streaming KV-cache
    # mechanism" section / wan_video_dit.py SelfAttention.forward, lines 317-382, for the actual
    # mechanism (kv_ratio=3.0 default -> kv_len=3 window-groups cap, concat-before-attention,
    # trim-by-one-group-after) -- the CORRECTED steady-state self-attention shape it produces is
    # q_len=19,200 / kv_len=76,800 (3 cached + 1 current temporal groups), benchmarked separately
    # below via bench_cross_attn (dense, op_name="self") / bench_prefill_block_mask_synthetic_qkv
    # (sparse). Warm-up is unaffected (pre_cache is None on the first call) -- kv_len==q_len==57,600
    # already reflects the real warm-up shape exactly, no correction needed there.
    # Sparse density=0.3 below is illustrative (FlashVSR's real local-window+top-k selection
    # algorithm is not ported here -- see test_flashvsr_configs.py's module docstring), not a
    # literal derivation of FlashVSR's kv_ratio/topk_ratio/local_range geometry.
    FLASHVSR_CHUNK_TOKENS = [19200, 57600]
    # Transpose-order sweep: identity ([B,H,S,D], OCL SDPA's default) vs. swapped ([B,S,H,D],
    # the {0,2,1,3}-fused heads<->seq layout a transpose-fusable subgraph could hand the kernel
    # instead) -- see cm_sdpa_opt_common.hpp / test_sdpa_prefill.py's TRANSPOSE_ORDER_* constants.
    # Both orders move identical bytes; only the LSC block pitch differs.
    ORDERS = [prefill.TRANSPOSE_ORDER_IDENTITY, prefill.TRANSPOSE_ORDER_SWAPPED]
    for order in ORDERS:
        for seq_len in FLASHVSR_CHUNK_TOKENS:
            run_case(bench_prefill, seq_len, num_heads=12, num_kv_heads=12, head_size=128,
                      is_causal=False, order=order)
    for order in ORDERS:
        for seq_len in FLASHVSR_CHUNK_TOKENS:
            run_case(bench_prefill_block_mask_synthetic, seq_len, num_heads=12, num_kv_heads=12, head_size=128,
                      sparse_block_size=128, density=0.3, is_causal=False, order=order)
    for order in ORDERS:
        for q_len in FLASHVSR_CHUNK_TOKENS:
            run_case(bench_cross_attn, q_len, kv_len=512, num_heads=12, num_kv_heads=12,
                      head_size=128, order=order)

    print("\n=== FlashVSR configs: self-attention, CORRECTED steady-state streaming-cache shape ===")
    # Real steady-state chunk (every self-attention call after warm-up): q_len=19,200 (this
    # chunk's own 1 temporal group), kv_len=76,800 (stabilized cache of 3 groups + this chunk's own
    # 1 group = 4 groups total) -- see the comment above FLASHVSR_CHUNK_TOKENS and the plan doc's
    # "Streaming KV-cache mechanism" section. Warm-up itself needs no correction (already
    # q_len==kv_len==57,600 above, exact). op_name="self" on bench_cross_attn: identical
    # dense/non-causal/q_len!=kv_len computation to cross-attention, just relabeled.
    STEADY_Q_LEN, STEADY_KV_LEN_REAL = 19200, 76800
    for order in ORDERS:
        run_case(bench_cross_attn, STEADY_Q_LEN, kv_len=STEADY_KV_LEN_REAL, num_heads=12,
                  num_kv_heads=12, head_size=128, order=order, op_name="self")
    for order in ORDERS:
        run_case(bench_prefill_block_mask_synthetic_qkv, STEADY_Q_LEN, STEADY_KV_LEN_REAL,
                  num_heads=12, num_kv_heads=12, head_size=128, sparse_block_size=128,
                  density=0.3, order=order)
        run_case(bench_prefill_fp8_qkv, STEADY_Q_LEN, STEADY_KV_LEN_REAL,
                  num_heads=12, num_kv_heads=12, head_size=128, sparse_block_size=128,
                  density=0.3, order=order)
