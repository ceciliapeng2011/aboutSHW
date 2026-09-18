"""
Performance vs. roofline for the SDPA-CM kernels (Stage 1a prefill, Stage 2a decode,
Stage 3a XAttention-sparse prefill), following pageatten/test_pa.py's exact methodology:
FLOPs = 2 GEMMs (Q@K^T, P@V) each counted as 2*M*N*K (mul+add), causal self-attention halves
that (only the lower triangle is computed), and XAttention sparsity scales it further by the
GPU-measured block-mask density. MFU/utilization formulas and HW_PEAK_FLOPS env convention are
copied verbatim from test_pa.py so numbers are directly comparable to PA's own roofline reports.

Usage: HW_PEAK_FLOPS=31 python bench_sdpa.py
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
                  kv_partition_size=256, n_warmup=10, n_iters=50):
    q, k, v = decode.make_decode_qkv(batch, num_heads, num_kv_heads, kv_len, head_size)
    order = decode.TRANSPOSE_ORDER_IDENTITY
    decode_kernel = decode.build_decode_kernel(num_heads, num_kv_heads, head_size,
                                                order, order, order, order,
                                                kv_partition_size=kv_partition_size)
    reduce_kernel = decode.build_finalization_kernel(head_size)

    t_q = cl.tensor(q.to(torch.float16).numpy())
    t_k = cl.tensor(k.to(torch.float16).numpy())
    t_v = cl.tensor(v.to(torch.float16).numpy())
    num_partitions = (kv_len + kv_partition_size - 1) // kv_partition_size
    bh = batch * num_heads
    t_exp_sums = cl.tensor(np.zeros([bh * num_partitions], dtype=np.float32))
    t_max_logits = cl.tensor(np.zeros([bh * num_partitions], dtype=np.float32))
    t_tmp_out = cl.tensor(np.zeros([bh * num_partitions * head_size], dtype=np.float16))
    t_out = cl.tensor(np.zeros([batch, num_heads, 1, head_size], dtype=np.float16))

    GWS = [1, bh, num_partitions]
    LWS = [1, 1, 1]
    args1 = [t_q, t_k, t_v, t_exp_sums, t_max_logits, t_tmp_out, kv_len]
    GWS2 = [1, bh, 1]
    LWS2 = [1, 1, 1]
    args2 = [t_exp_sums, t_max_logits, t_tmp_out, t_out, num_partitions]

    def run_once():
        decode_kernel.enqueue("cm_sdpa_opt_decode", GWS, LWS, *args1)
        reduce_kernel.enqueue("cm_sdpa_opt_decode_reduce", GWS2, LWS2, *args2)

    cl.finish()
    for _ in range(n_warmup):
        run_once()
    cl.finish()
    for _ in range(n_iters):
        run_once()
    raw_ns = cl.finish()
    per_iter_ms = [(raw_ns[2 * i] + raw_ns[2 * i + 1]) * 1e-6 for i in range(n_iters)]

    total_flops = flops_for(num_heads, 1, kv_len, head_size, causal_savings=False) * batch
    label = f"decode   kv_len={kv_len:<6} heads={num_heads}/{num_kv_heads} hd={head_size}"
    result = report(label, total_flops, per_iter_ms)
    del t_q, t_k, t_v, t_exp_sums, t_max_logits, t_tmp_out, t_out
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
