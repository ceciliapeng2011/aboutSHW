"""Perf sweep for the dual-source (ring+current) cm_sdpa_opt_prefill_multisource.cm kernel --
plain fp16 ring vs fp8e4m3 ring (KV_RING_FP8E4M3) x dense (all-keep mask) vs sparse
(bs128,d0.3) block mask, at FlashVSR's two real streaming-KV-cache chunk shapes (see
flashvsr/bench_flashvsr_e2e.py's module docstring for the full derivation of these numbers from
wan_video_dit.py's SelfAttention.forward):

  * warmup: pre_cache is None -> kv_len_ring=0 (sdpa_accumulate_kv_source's kv_stop<=0 no-op,
    degrades to current-only), kv_len_cur == q_len == 3 window-groups (f=6).
  * steady: stabilized cache of exactly kv_len=3 window-groups in the ring, this chunk's own
    1 window-group as current (f=2) -- kv_ratio=3.0, the real per-step shape every steady-state
    self-attention call sees.

Unlike bench_flashvsr_e2e.py's bench_dense_reference/bench_sparse_at_density (which concatenate
ring+current into ONE buffer and call the single-source cm_sdpa_opt_prefill.cm kernel -- an
approximation of the real two-buffer ABI), this sweep calls
cm_sdpa_opt_prefill_multisource.cm directly with ring and current as two INDEPENDENT device
buffers, matching the real OpenVINO BlockSparseAttention `input[1]=key_cache.<layer>` +
`input[2]=k_new_f16` ABI this kernel mirrors (see that file's own header comment).

Usage:
    HW_PEAK_FLOPS=31 python bench_sdpa_multisource.py
    MS_BENCH_CSV=my_results.csv python bench_sdpa_multisource.py
"""
import csv
import gc
import os

import numpy as np
import torch

from clops import cl
from clops.utils import Colors

import test_sdpa_prefill as base
import test_sdpa_prefill_block_mask_multisource as ms
from hf8_e4m3_ref import encode_f16_to_hf8, build_hf8_decode_lut_f16

torch.manual_seed(7)

HW_PEAK_FLOPS = float(os.environ.get("HW_PEAK_FLOPS", "31")) * 1e12

# FlashVSR real self-attention config (320x480 portrait -> 1280x1920, see
# flashvsr/bench_flashvsr_e2e.py's module docstring): self-attention has no GQA (num_kv_heads ==
# num_heads).
NUM_HEADS = 12
NUM_KV_HEADS = 12
HEAD_SIZE = 128
WINDOW_GROUP_TOKENS = 150 * 128  # SPATIAL_WINDOWS(10*15) * WINDOW_TOKENS(2x8x8) == 19,200
SPARSE_BLOCK_SIZE = 128

# (name, cache_groups): cache_groups*WINDOW_GROUP_TOKENS == kv_len_ring, 1*WINDOW_GROUP_TOKENS ==
# q_len == kv_len_cur (this chunk's own new K/V), matching wan_video_dit.py's streaming FIFO cache.
CHUNKS = (
    ("warmup", 0),
    ("steady", 3),
)
DENSITIES = (("dense", 1.0), ("sparse_d0.3", 0.3))
CSV_PATH = os.environ.get("MS_BENCH_CSV", "bench_sdpa_multisource.csv")


def flops_util(flops, ms_):
    gflops = flops / (ms_ * 1e6) if ms_ > 0 else 0.0
    util = flops / (ms_ * 1e-3) / HW_PEAK_FLOPS * 100 if ms_ > 0 else 0.0
    return gflops, util


def report(label, total_flops, lat_ms):
    total_avg = sum(lat_ms) / len(lat_ms)
    gflops, util = flops_util(total_flops, total_avg)
    print(
        f"[perf] {label} | GFLOPS {Colors.BOLD}{Colors.GREEN}{gflops:,.0f}{Colors.END}, "
        f"latency(ms) avg={Colors.BOLD}{Colors.YELLOW}{total_avg:.3f}{Colors.END}, "
        f"{Colors.BOLD}{Colors.GREEN}{util:.1f}%{Colors.END} of {HW_PEAK_FLOPS/1e12:.0f} TFLOPS XMX peak"
    )
    return total_avg, util


def _make_qkv(q_len, kv_len_ring, kv_len_cur, seed):
    g = torch.Generator().manual_seed(seed)
    q = (torch.randn(1, NUM_HEADS, q_len, HEAD_SIZE, generator=g) * 1.0).to(torch.float16)
    # cl.tensor doesn't support a genuinely zero-length dim -- upload a minimal 1-token physical
    # buffer for an empty ring (warmup chunk); the kernel never reads it (kv_len_ring arg is still
    # the true 0, see sdpa_accumulate_kv_source's kv_stop<=0 no-op).
    k_ring = (torch.randn(1, NUM_KV_HEADS, max(kv_len_ring, 1), HEAD_SIZE, generator=g) * 1.0).to(torch.float16)
    v_ring = (torch.randn(1, NUM_KV_HEADS, max(kv_len_ring, 1), HEAD_SIZE, generator=g) * 0.5).to(torch.float16)
    k_cur = (torch.randn(1, NUM_KV_HEADS, kv_len_cur, HEAD_SIZE, generator=g) * 1.0).to(torch.float16)
    v_cur = (torch.randn(1, NUM_KV_HEADS, kv_len_cur, HEAD_SIZE, generator=g) * 0.5).to(torch.float16)
    return q, k_ring, v_ring, k_cur, v_cur


def _fake_mask(num_q_blocks, num_k_blocks_ring, num_k_blocks_cur, density, seed):
    g = torch.Generator().manual_seed(seed)
    total = num_k_blocks_ring + num_k_blocks_cur
    mask = (torch.rand(1, NUM_HEADS, num_q_blocks, total, generator=g) < density).to(torch.bool)
    mask[:, :, :, 0] = True  # never an all-false row -- matches other multisource tests' convention
    return mask


def _dispatch(q_len):
    q_step = base.get_cm_grf_width() // 32
    wg_size = 16
    wg_seq_len = wg_size * q_step
    wg_count = (q_len + wg_seq_len - 1) // wg_seq_len
    return [1, NUM_HEADS, wg_count * wg_size], [1, 1, wg_size]


def bench_multisource_fp16(chunk_name, cache_groups, density, seed):
    q_len = kv_len_cur = WINDOW_GROUP_TOKENS
    kv_len_ring = cache_groups * WINDOW_GROUP_TOKENS
    q, k_ring, v_ring, k_cur, v_cur = _make_qkv(q_len, kv_len_ring, kv_len_cur, seed)

    kernel = ms.build_multisource_sparse_prefill_kernel(NUM_HEADS, NUM_KV_HEADS, HEAD_SIZE, SPARSE_BLOCK_SIZE)

    t_q = cl.tensor(q.contiguous().numpy())
    t_k_ring = cl.tensor(k_ring.contiguous().numpy())
    t_v_ring = cl.tensor(v_ring.contiguous().numpy())
    t_k_cur = cl.tensor(k_cur.contiguous().numpy())
    t_v_cur = cl.tensor(v_cur.contiguous().numpy())
    t_out = cl.tensor(np.zeros([1, NUM_HEADS, q_len, HEAD_SIZE], dtype=np.float16))

    num_q_blocks = (q_len + SPARSE_BLOCK_SIZE - 1) // SPARSE_BLOCK_SIZE
    num_k_blocks_ring = (kv_len_ring + SPARSE_BLOCK_SIZE - 1) // SPARSE_BLOCK_SIZE
    num_k_blocks_cur = (kv_len_cur + SPARSE_BLOCK_SIZE - 1) // SPARSE_BLOCK_SIZE
    mask = (_fake_mask(num_q_blocks, num_k_blocks_ring, num_k_blocks_cur, density, seed + 1)
            if density < 1.0 else
            torch.ones(1, NUM_HEADS, num_q_blocks, num_k_blocks_ring + num_k_blocks_cur, dtype=torch.bool))
    t_mask = cl.tensor(mask.numpy())

    GWS, LWS = _dispatch(q_len)
    args = [t_q, t_k_ring, t_v_ring, t_k_cur, t_v_cur, t_out, q_len, kv_len_ring, kv_len_cur,
            t_mask, num_q_blocks, num_k_blocks_ring, num_k_blocks_cur]

    cl.finish()
    for _ in range(10):
        kernel.enqueue("cm_sdpa_opt_prefill_multisource", GWS, LWS, *args)
    cl.finish()
    for _ in range(30):
        kernel.enqueue("cm_sdpa_opt_prefill_multisource", GWS, LWS, *args)
    lat_ms = [ns * 1e-6 for ns in cl.finish()]

    eff_density = mask.float().mean().item()
    total_flops = NUM_HEADS * q_len * (kv_len_ring + kv_len_cur) * HEAD_SIZE * 4 * eff_density
    label = (f"multisource-fp16 {chunk_name:<7} q={q_len:<6} kv_ring={kv_len_ring:<6} "
             f"kv_cur={kv_len_cur:<6} density(eff)={eff_density:.3f}")
    result = report(label, total_flops, lat_ms)
    del t_q, t_k_ring, t_v_ring, t_k_cur, t_v_cur, t_out, t_mask
    gc.collect()
    return result


def bench_multisource_fp8ring(chunk_name, cache_groups, density, seed):
    src = r'''#include "cm_sdpa_opt_prefill_multisource.cm"'''
    scale_factor = 1.0 / (HEAD_SIZE ** 0.5)
    opts = (
        f'-cmc -Qxcm_jit_option="-abortonspill" -Qxcm_register_file_size=256 -mCM_printregusage '
        f"-I{base.CWD} -I{base.FLASHATTN_DIR} -I{base.PAGEATTEN_DIR} "
        f"-DKERNEL_NAME=cm_sdpa_opt_prefill_multisource_fp8ring "
        f"-DCMFLA_NUM_HEADS={NUM_HEADS} -DCMFLA_NUM_KV_HEADS={NUM_KV_HEADS} "
        f"-DCMFLA_HEAD_SIZE={HEAD_SIZE} -DCMFLA_SCALE_FACTOR={scale_factor} "
        f"-DSPARSE_BLOCK_SIZE={SPARSE_BLOCK_SIZE} -DKV_RING_FP8E4M3=1"
    )
    kernel = cl.kernels(src, opts)

    q_len = kv_len_cur = WINDOW_GROUP_TOKENS
    kv_len_ring = cache_groups * WINDOW_GROUP_TOKENS
    q, k_ring, v_ring, k_cur, v_cur = _make_qkv(q_len, kv_len_ring, kv_len_cur, seed)
    k_ring_hf8 = encode_f16_to_hf8(k_ring.contiguous().numpy())
    v_ring_hf8 = encode_f16_to_hf8(v_ring.contiguous().numpy())

    t_q = cl.tensor(q.contiguous().numpy())
    t_k_ring = cl.tensor(k_ring_hf8)
    t_v_ring = cl.tensor(v_ring_hf8)
    t_k_cur = cl.tensor(k_cur.contiguous().numpy())
    t_v_cur = cl.tensor(v_cur.contiguous().numpy())
    t_out = cl.tensor(np.zeros([1, NUM_HEADS, q_len, HEAD_SIZE], dtype=np.float16))
    t_lut = cl.tensor(build_hf8_decode_lut_f16())

    num_q_blocks = (q_len + SPARSE_BLOCK_SIZE - 1) // SPARSE_BLOCK_SIZE
    num_k_blocks_ring = (kv_len_ring + SPARSE_BLOCK_SIZE - 1) // SPARSE_BLOCK_SIZE
    num_k_blocks_cur = (kv_len_cur + SPARSE_BLOCK_SIZE - 1) // SPARSE_BLOCK_SIZE
    mask = (_fake_mask(num_q_blocks, num_k_blocks_ring, num_k_blocks_cur, density, seed + 1)
            if density < 1.0 else
            torch.ones(1, NUM_HEADS, num_q_blocks, num_k_blocks_ring + num_k_blocks_cur, dtype=torch.bool))
    t_mask = cl.tensor(mask.numpy())

    GWS, LWS = _dispatch(q_len)
    args = [t_q, t_k_ring, t_v_ring, t_k_cur, t_v_cur, t_out, q_len, kv_len_ring, kv_len_cur,
            t_mask, num_q_blocks, num_k_blocks_ring, num_k_blocks_cur, t_lut]

    cl.finish()
    for _ in range(10):
        kernel.enqueue("cm_sdpa_opt_prefill_multisource_fp8ring", GWS, LWS, *args)
    cl.finish()
    for _ in range(30):
        kernel.enqueue("cm_sdpa_opt_prefill_multisource_fp8ring", GWS, LWS, *args)
    lat_ms = [ns * 1e-6 for ns in cl.finish()]

    eff_density = mask.float().mean().item()
    total_flops = NUM_HEADS * q_len * (kv_len_ring + kv_len_cur) * HEAD_SIZE * 4 * eff_density
    label = (f"multisource-fp8ring {chunk_name:<7} q={q_len:<6} kv_ring={kv_len_ring:<6} "
             f"kv_cur={kv_len_cur:<6} density(eff)={eff_density:.3f}")
    result = report(label, total_flops, lat_ms)
    del t_q, t_k_ring, t_v_ring, t_k_cur, t_v_cur, t_out, t_mask, t_lut
    gc.collect()
    return result


def run():
    results = []
    seed = 1000
    for chunk_name, cache_groups in CHUNKS:
        for sparsity_name, density in DENSITIES:
            if chunk_name == "warmup" and sparsity_name != "dense":
                # warmup has only 1 window-group of KV total (kv_len_ring==0) -- not worth a
                # separate sparse data point, same single block either way.
                continue
            for dtype_name, bench_fn in (("fp16", bench_multisource_fp16), ("fp8ring", bench_multisource_fp8ring)):
                print(f"{Colors.GREEN}[bench] chunk={chunk_name} dtype={dtype_name} sparsity={sparsity_name}{Colors.END}")
                avg_ms, util = bench_fn(chunk_name, cache_groups, density, seed)
                results.append({
                    "chunk": chunk_name, "dtype": dtype_name, "sparsity": sparsity_name,
                    "avg_ms": avg_ms, "utilization_pct": util,
                })
                seed += 1

    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["chunk", "dtype", "sparsity", "avg_ms", "utilization_pct"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\n{Colors.GREEN}[summary] heads={NUM_HEADS}/{NUM_KV_HEADS} hd={HEAD_SIZE} "
          f"window_group_tokens={WINDOW_GROUP_TOKENS} sparse_block_size={SPARSE_BLOCK_SIZE}{Colors.END}")
    print(f"{'chunk':>8} | {'dtype':>8} | {'sparsity':>12} | {'avg_ms':>10} | {'util%':>7}")
    for row in results:
        print(f"{row['chunk']:>8} | {row['dtype']:>8} | {row['sparsity']:>12} | "
              f"{row['avg_ms']:>10.3f} | {row['utilization_pct']:>6.1f}%")
    print(f"csv | {CSV_PATH}")


if __name__ == "__main__":
    cl.profiling(True)
    run()
