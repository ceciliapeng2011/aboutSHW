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
    density = mask.float().mean().item()
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
    # Capped at seq_len=4096, the exact shape test_sdpa_prefill_xattn.py's own correctness suite
    # validates (test_mask_matches_cpu_reference/test_sparse_output_accuracy). Benchmarking at
    # larger scale, or repeatedly re-invoking the scoring pipeline within one long-running process,
    # surfaced pre-existing reliability issues in the shared, unmodified x_attn/gemm_qk.cm +
    # find_block.hpp pipeline (not touched by Stage 3a's wiring, out of scope to fix here):
    # (1) a hard GPU-side crash (CL_SUCCESS failure inside find_block) at seq_len=32768 for
    #     thresh<0.99 (reproduced standalone);
    # (2) the mask density is not run-to-run deterministic at seq_len>=8192 for identical inputs
    #     (observed 0.33 vs 0.31 across two otherwise-identical invocations) -- likely an
    #     uninitialized-buffer read in the shared scoring kernels;
    # (3) repeatedly invoking the scoring pipeline (multiple thresholds/block sizes back to back,
    #     whether after the prefill/decode sections above or in a dedicated fresh process) has been
    #     observed to hang indefinitely rather than error out cleanly, of unconfirmed root cause
    #     (accumulated device-memory pressure is one candidate).
    # See the report for details; flagged for follow-up, not fixed here. run_case() below skips
    # cases that raise, but cannot recover from a genuine hang -- if this section stalls, interrupt
    # and rely on the one validated data point already captured in earlier runs.
    for sparse_block_size in (256, 128):
        for thresh in (0.99, 0.9, 0.6, 0.3):
            run_case(bench_xattn_prefill, 4096, sparse_block_size=sparse_block_size, thresh=thresh)
