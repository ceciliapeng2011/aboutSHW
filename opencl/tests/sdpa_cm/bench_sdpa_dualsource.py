"""Perf sweep comparing {single-source, dual-source (cache+current)} x {fp16, int8-cache/int8} x
{dense (all-keep mask), sparse (bs128,d0.3)}, at FlashVSR's two real streaming-KV-cache chunk
shapes (see flashvsr/bench_flashvsr_e2e.py's module docstring for the full derivation of these
numbers from wan_video_dit.py's SelfAttention.forward):

  * warmup: pre_cache is None -> kv_len_cache=0 (sdpa_accumulate_kv_source_slm's kv_stop<=0 no-op,
    degrades to current-only), kv_len_cur == q_len == 3 window-groups (f=6).
  * steady: stabilized cache of exactly kv_len=3 window-groups in the cache, this chunk's own
    1 window-group as current (f=2) -- kv_ratio=3.0, the real per-step shape every steady-state
    self-attention call sees.

Dual-source benchmarks call cm_sdpa_opt_prefill_dualsource.cm directly with cache and current as
two INDEPENDENT device buffers, matching the real OpenVINO BlockSparseAttention
`input[1]=key_cache.<layer>` + `input[2]=k_new_f16` ABI this kernel mirrors (see that file's own
header comment). Single-source benchmarks concatenate cache+current into ONE buffer and call
cm_sdpa_opt_prefill.cm (the same approximation flashvsr/bench_flashvsr_e2e.py's
bench_dense_reference/bench_sparse_at_density use) -- the "equivalent concatenated-shape" baseline
this sweep uses to show how close the dual-source SLM/barrier redesign gets to single-source
performance, and how much (if any) overhead int8-cache adds over int8 on the single-source kernel.

Usage:
    HW_PEAK_FLOPS=31 python bench_sdpa_dualsource.py
    MS_BENCH_CSV=my_results.csv python bench_sdpa_dualsource.py
"""
import csv
import gc
import os

import numpy as np
import torch

from clops import cl
from clops.utils import Colors

import test_sdpa_prefill as base
import test_sdpa_prefill_block_mask as bm
import test_sdpa_prefill_block_mask_dualsource as ms
from int8_kv_ref import encode_k_by_channel, encode_by_token

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

# (name, cache_groups): cache_groups*WINDOW_GROUP_TOKENS == kv_len_cache, 1*WINDOW_GROUP_TOKENS ==
# q_len == kv_len_cur (this chunk's own new K/V), matching wan_video_dit.py's streaming FIFO cache.
CHUNKS = (
    ("warmup", 0),
    ("steady", 3),
)
DENSITIES = (("dense", 1.0), ("sparse_d0.3", 0.3))
CSV_PATH = os.environ.get("MS_BENCH_CSV", "bench_sdpa_dualsource.csv")


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


def _make_qkv(q_len, kv_len_cache, kv_len_cur, seed):
    g = torch.Generator().manual_seed(seed)
    q = (torch.randn(1, NUM_HEADS, q_len, HEAD_SIZE, generator=g) * 1.0).to(torch.float16)
    # cl.tensor doesn't support a genuinely zero-length dim -- upload a minimal physical buffer for
    # an empty cache (warmup chunk); the kernel never reads it (kv_len_cache arg is still the true 0,
    # see sdpa_accumulate_kv_source_slm's kv_stop<=0 no-op). 16 (not 1) so BY_CHANNEL int8 encode's
    # sub-block-size divisibility requirement is satisfied even for the placeholder.
    k_cache = (torch.randn(1, NUM_KV_HEADS, max(kv_len_cache, 16), HEAD_SIZE, generator=g) * 1.0).to(torch.float16)
    v_cache = (torch.randn(1, NUM_KV_HEADS, max(kv_len_cache, 16), HEAD_SIZE, generator=g) * 0.5).to(torch.float16)
    k_cur = (torch.randn(1, NUM_KV_HEADS, kv_len_cur, HEAD_SIZE, generator=g) * 1.0).to(torch.float16)
    v_cur = (torch.randn(1, NUM_KV_HEADS, kv_len_cur, HEAD_SIZE, generator=g) * 0.5).to(torch.float16)
    return q, k_cache, v_cache, k_cur, v_cur


def _fake_mask(num_q_blocks, num_k_blocks_cache, num_k_blocks_cur, density, seed):
    g = torch.Generator().manual_seed(seed)
    total = num_k_blocks_cache + num_k_blocks_cur
    mask = (torch.rand(1, NUM_HEADS, num_q_blocks, total, generator=g) < density).to(torch.bool)
    mask[:, :, :, 0] = True  # never an all-false row -- matches other dualsource tests' convention
    return mask


def _dispatch(q_len):
    q_step = base.get_cm_grf_width() // 32
    wg_size = 16
    wg_seq_len = wg_size * q_step
    wg_count = (q_len + wg_seq_len - 1) // wg_seq_len
    return [1, NUM_HEADS, wg_count * wg_size], [1, 1, wg_size]


def _mask_for(q_len, kv_len_cache, kv_len_cur, density, seed):
    num_q_blocks = (q_len + SPARSE_BLOCK_SIZE - 1) // SPARSE_BLOCK_SIZE
    num_k_blocks_cache = (kv_len_cache + SPARSE_BLOCK_SIZE - 1) // SPARSE_BLOCK_SIZE
    num_k_blocks_cur = (kv_len_cur + SPARSE_BLOCK_SIZE - 1) // SPARSE_BLOCK_SIZE
    mask = (_fake_mask(num_q_blocks, num_k_blocks_cache, num_k_blocks_cur, density, seed)
            if density < 1.0 else
            torch.ones(1, NUM_HEADS, num_q_blocks, num_k_blocks_cache + num_k_blocks_cur, dtype=torch.bool))
    return mask, num_q_blocks, num_k_blocks_cache, num_k_blocks_cur


def bench_dualsource_fp16(chunk_name, cache_groups, density, seed):
    q_len = kv_len_cur = WINDOW_GROUP_TOKENS
    kv_len_cache = cache_groups * WINDOW_GROUP_TOKENS
    q, k_cache, v_cache, k_cur, v_cur = _make_qkv(q_len, kv_len_cache, kv_len_cur, seed)

    kernel = ms.build_dualsource_sparse_prefill_kernel(NUM_HEADS, NUM_KV_HEADS, HEAD_SIZE, SPARSE_BLOCK_SIZE)

    t_q = cl.tensor(q.contiguous().numpy())
    t_k_cache = cl.tensor(k_cache.contiguous().numpy())
    t_v_cache = cl.tensor(v_cache.contiguous().numpy())
    t_k_cur = cl.tensor(k_cur.contiguous().numpy())
    t_v_cur = cl.tensor(v_cur.contiguous().numpy())
    t_out = cl.tensor(np.zeros([1, NUM_HEADS, q_len, HEAD_SIZE], dtype=np.float16))

    mask, num_q_blocks, num_k_blocks_cache, num_k_blocks_cur = _mask_for(
        q_len, kv_len_cache, kv_len_cur, density, seed + 1)
    t_mask = cl.tensor(mask.numpy())

    GWS, LWS = _dispatch(q_len)
    args = [t_q, t_k_cache, t_v_cache, t_k_cur, t_v_cur, t_out, q_len, kv_len_cache, kv_len_cur,
            t_mask, num_q_blocks, num_k_blocks_cache, num_k_blocks_cur]

    cl.finish()
    for _ in range(10):
        kernel.enqueue("cm_sdpa_opt_prefill_dualsource", GWS, LWS, *args)
    cl.finish()
    for _ in range(30):
        kernel.enqueue("cm_sdpa_opt_prefill_dualsource", GWS, LWS, *args)
    lat_ms = [ns * 1e-6 for ns in cl.finish()]

    eff_density = mask.float().mean().item()
    total_flops = NUM_HEADS * q_len * (kv_len_cache + kv_len_cur) * HEAD_SIZE * 4 * eff_density
    label = (f"dual-fp16     {chunk_name:<7} q={q_len:<6} kv_cache={kv_len_cache:<6} "
             f"kv_cur={kv_len_cur:<6} density(eff)={eff_density:.3f}")
    result = report(label, total_flops, lat_ms)
    del t_q, t_k_cache, t_v_cache, t_k_cur, t_v_cur, t_out, t_mask
    gc.collect()
    return result


def bench_dualsource_int8cache(chunk_name, cache_groups, density, seed):
    src = r'''#include "cm_sdpa_opt_prefill_dualsource.cm"'''
    scale_factor = 1.0 / (HEAD_SIZE ** 0.5)
    opts = (
        f'-cmc -Qxcm_jit_option="-abortonspill" -Qxcm_register_file_size=256 -mCM_printregusage '
        f"-I{base.CWD} -I{base.FLASHATTN_DIR} -I{base.PAGEATTEN_DIR} "
        f"-DKERNEL_NAME=cm_sdpa_opt_prefill_dualsource_int8cache "
        f"-DCMFLA_NUM_HEADS={NUM_HEADS} -DCMFLA_NUM_KV_HEADS={NUM_KV_HEADS} "
        f"-DCMFLA_HEAD_SIZE={HEAD_SIZE} -DCMFLA_SCALE_FACTOR={scale_factor} "
        f"-DSPARSE_BLOCK_SIZE={SPARSE_BLOCK_SIZE} -DKV_CACHE_COMPRESSION=2"
    )
    kernel = cl.kernels(src, opts)

    q_len = kv_len_cur = WINDOW_GROUP_TOKENS
    kv_len_cache = cache_groups * WINDOW_GROUP_TOKENS
    q, k_cache, v_cache, k_cur, v_cur = _make_qkv(q_len, kv_len_cache, kv_len_cur, seed)
    k_cache_u8, k_cache_scale, k_cache_zp = encode_k_by_channel(k_cache)
    v_cache_u8, v_cache_scale, v_cache_zp = encode_by_token(v_cache)

    t_q = cl.tensor(q.contiguous().numpy())
    t_k_cache = cl.tensor(k_cache_u8.contiguous().numpy())
    t_v_cache = cl.tensor(v_cache_u8.contiguous().numpy())
    t_k_cur = cl.tensor(k_cur.contiguous().numpy())
    t_v_cur = cl.tensor(v_cur.contiguous().numpy())
    t_out = cl.tensor(np.zeros([1, NUM_HEADS, q_len, HEAD_SIZE], dtype=np.float16))
    t_k_scale = cl.tensor(k_cache_scale.contiguous().numpy())
    t_k_zp = cl.tensor(k_cache_zp.contiguous().numpy())
    t_v_scale = cl.tensor(v_cache_scale.contiguous().numpy())
    t_v_zp = cl.tensor(v_cache_zp.contiguous().numpy())

    mask, num_q_blocks, num_k_blocks_cache, num_k_blocks_cur = _mask_for(
        q_len, kv_len_cache, kv_len_cur, density, seed + 1)
    t_mask = cl.tensor(mask.numpy())

    GWS, LWS = _dispatch(q_len)
    args = [t_q, t_k_cache, t_v_cache, t_k_cur, t_v_cur, t_out, q_len, kv_len_cache, kv_len_cur,
            t_mask, num_q_blocks, num_k_blocks_cache, num_k_blocks_cur,
            t_k_scale, t_k_zp, t_v_scale, t_v_zp]

    cl.finish()
    for _ in range(10):
        kernel.enqueue("cm_sdpa_opt_prefill_dualsource_int8cache", GWS, LWS, *args)
    cl.finish()
    for _ in range(30):
        kernel.enqueue("cm_sdpa_opt_prefill_dualsource_int8cache", GWS, LWS, *args)
    lat_ms = [ns * 1e-6 for ns in cl.finish()]

    eff_density = mask.float().mean().item()
    total_flops = NUM_HEADS * q_len * (kv_len_cache + kv_len_cur) * HEAD_SIZE * 4 * eff_density
    label = (f"dual-int8cache {chunk_name:<7} q={q_len:<6} kv_cache={kv_len_cache:<6} "
             f"kv_cur={kv_len_cur:<6} density(eff)={eff_density:.3f}")
    result = report(label, total_flops, lat_ms)
    del t_q, t_k_cache, t_v_cache, t_k_cur, t_v_cur, t_out, t_mask, t_k_scale, t_k_zp, t_v_scale, t_v_zp
    gc.collect()
    return result


def _single_source_kv(cache_groups, seed):
    """Concatenated-shape baseline: cache+current as ONE [kv_len_cache+kv_len_cur] buffer, same
    approach flashvsr/bench_flashvsr_e2e.py's bench_dense_reference uses."""
    q_len = kv_len_cur = WINDOW_GROUP_TOKENS
    kv_len_cache = cache_groups * WINDOW_GROUP_TOKENS
    q, k_cache, v_cache, k_cur, v_cur = _make_qkv(q_len, kv_len_cache, kv_len_cur, seed)
    k_cache_real = k_cache[:, :, :kv_len_cache] if kv_len_cache > 0 else k_cache[:, :, :0]
    v_cache_real = v_cache[:, :, :kv_len_cache] if kv_len_cache > 0 else v_cache[:, :, :0]
    k = torch.cat([k_cache_real, k_cur], dim=2)
    v = torch.cat([v_cache_real, v_cur], dim=2)
    return q, k, v, q_len, kv_len_cache + kv_len_cur


def bench_singlesource_fp16(chunk_name, cache_groups, density, seed):
    q, k, v, q_len, kv_len = _single_source_kv(cache_groups, seed)

    kernel = bm.build_sparse_prefill_kernel(NUM_HEADS, NUM_KV_HEADS, HEAD_SIZE, is_causal=False,
                                             sparse_block_size=SPARSE_BLOCK_SIZE)

    t_q = cl.tensor(q.contiguous().numpy())
    t_k = cl.tensor(k.contiguous().numpy())
    t_v = cl.tensor(v.contiguous().numpy())
    t_out = cl.tensor(np.zeros([1, NUM_HEADS, q_len, HEAD_SIZE], dtype=np.float16))

    num_q_blocks = (q_len + SPARSE_BLOCK_SIZE - 1) // SPARSE_BLOCK_SIZE
    num_k_blocks = (kv_len + SPARSE_BLOCK_SIZE - 1) // SPARSE_BLOCK_SIZE
    mask = (torch.rand(1, NUM_HEADS, num_q_blocks, num_k_blocks, generator=torch.Generator().manual_seed(seed + 1))
            < density).to(torch.bool) if density < 1.0 else torch.ones(1, NUM_HEADS, num_q_blocks, num_k_blocks, dtype=torch.bool)
    mask[:, :, :, 0] = True
    t_mask = cl.tensor(mask.numpy())

    GWS, LWS = _dispatch(q_len)
    args = [t_q, t_k, t_v, t_out, q_len, kv_len, t_mask, num_q_blocks, num_k_blocks]

    cl.finish()
    for _ in range(10):
        kernel.enqueue("cm_sdpa_opt_prefill", GWS, LWS, *args)
    cl.finish()
    for _ in range(30):
        kernel.enqueue("cm_sdpa_opt_prefill", GWS, LWS, *args)
    lat_ms = [ns * 1e-6 for ns in cl.finish()]

    eff_density = mask.float().mean().item()
    total_flops = NUM_HEADS * q_len * kv_len * HEAD_SIZE * 4 * eff_density
    label = f"single-fp16   {chunk_name:<7} q={q_len:<6} kv={kv_len:<6} density(eff)={eff_density:.3f}"
    result = report(label, total_flops, lat_ms)
    del t_q, t_k, t_v, t_out, t_mask
    gc.collect()
    return result


def bench_singlesource_int8(chunk_name, cache_groups, density, seed):
    q, k, v, q_len, kv_len = _single_source_kv(cache_groups, seed)
    k_u8, k_scale, k_zp = encode_k_by_channel(k)
    v_u8, v_scale, v_zp = encode_by_token(v)

    src = r'''#include "cm_sdpa_opt_prefill.cm"'''
    scale_factor = 1.0 / (HEAD_SIZE ** 0.5)
    opts = (
        f'-cmc -Qxcm_jit_option="-abortonspill" -Qxcm_register_file_size=256 -mCM_printregusage '
        f"-I{base.CWD} -I{base.FLASHATTN_DIR} -I{base.PAGEATTEN_DIR} "
        f"-DKERNEL_NAME=cm_sdpa_opt_prefill_int8_bench "
        f"-DCMFLA_NUM_HEADS={NUM_HEADS} -DCMFLA_NUM_KV_HEADS={NUM_KV_HEADS} "
        f"-DCMFLA_HEAD_SIZE={HEAD_SIZE} -DCMFLA_SCALE_FACTOR={scale_factor} "
        f"-DCMFLA_IS_CAUSAL=0 -DHAS_ATTN_MASK_INPUT=0 -DHAS_SCALE_INPUT=0 "
        f"-DSPARSE_BLOCK_SIZE={SPARSE_BLOCK_SIZE} -DKV_CACHE_COMPRESSION=2"
    )
    kernel = cl.kernels(src, opts)

    t_q = cl.tensor(q.contiguous().numpy())
    t_k = cl.tensor(k_u8.contiguous().numpy())
    t_v = cl.tensor(v_u8.contiguous().numpy())
    t_out = cl.tensor(np.zeros([1, NUM_HEADS, q_len, HEAD_SIZE], dtype=np.float16))
    t_k_scale = cl.tensor(k_scale.contiguous().numpy())
    t_k_zp = cl.tensor(k_zp.contiguous().numpy())
    t_v_scale = cl.tensor(v_scale.contiguous().numpy())
    t_v_zp = cl.tensor(v_zp.contiguous().numpy())

    num_q_blocks = (q_len + SPARSE_BLOCK_SIZE - 1) // SPARSE_BLOCK_SIZE
    num_k_blocks = (kv_len + SPARSE_BLOCK_SIZE - 1) // SPARSE_BLOCK_SIZE
    mask = (torch.rand(1, NUM_HEADS, num_q_blocks, num_k_blocks, generator=torch.Generator().manual_seed(seed + 1))
            < density).to(torch.bool) if density < 1.0 else torch.ones(1, NUM_HEADS, num_q_blocks, num_k_blocks, dtype=torch.bool)
    mask[:, :, :, 0] = True
    t_mask = cl.tensor(mask.numpy())

    GWS, LWS = _dispatch(q_len)
    args = [t_q, t_k, t_v, t_out, q_len, kv_len, t_mask, num_q_blocks, num_k_blocks,
            t_k_scale, t_k_zp, t_v_scale, t_v_zp]

    cl.finish()
    for _ in range(10):
        kernel.enqueue("cm_sdpa_opt_prefill_int8_bench", GWS, LWS, *args)
    cl.finish()
    for _ in range(30):
        kernel.enqueue("cm_sdpa_opt_prefill_int8_bench", GWS, LWS, *args)
    lat_ms = [ns * 1e-6 for ns in cl.finish()]

    eff_density = mask.float().mean().item()
    total_flops = NUM_HEADS * q_len * kv_len * HEAD_SIZE * 4 * eff_density
    label = f"single-int8   {chunk_name:<7} q={q_len:<6} kv={kv_len:<6} density(eff)={eff_density:.3f}"
    result = report(label, total_flops, lat_ms)
    del t_q, t_k, t_v, t_out, t_mask, t_k_scale, t_k_zp, t_v_scale, t_v_zp
    gc.collect()
    return result


BENCHES = (
    ("single", "fp16", bench_singlesource_fp16),
    ("single", "int8", bench_singlesource_int8),
    ("dual", "fp16", bench_dualsource_fp16),
    ("dual", "int8cache", bench_dualsource_int8cache),
)


def run():
    results = []
    seed = 1000
    for chunk_name, cache_groups in CHUNKS:
        for sparsity_name, density in DENSITIES:
            if chunk_name == "warmup" and sparsity_name != "dense":
                # warmup has only 1 window-group of KV total (kv_len_cache==0) -- not worth a
                # separate sparse data point, same single block either way.
                continue
            for topology, dtype_name, bench_fn in BENCHES:
                print(f"{Colors.GREEN}[bench] chunk={chunk_name} topology={topology} dtype={dtype_name} "
                      f"sparsity={sparsity_name}{Colors.END}")
                avg_ms, util = bench_fn(chunk_name, cache_groups, density, seed)
                results.append({
                    "chunk": chunk_name, "topology": topology, "dtype": dtype_name,
                    "sparsity": sparsity_name, "avg_ms": avg_ms, "utilization_pct": util,
                })
                seed += 1

    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["chunk", "topology", "dtype", "sparsity", "avg_ms", "utilization_pct"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\n{Colors.GREEN}[summary] heads={NUM_HEADS}/{NUM_KV_HEADS} hd={HEAD_SIZE} "
          f"window_group_tokens={WINDOW_GROUP_TOKENS} sparse_block_size={SPARSE_BLOCK_SIZE}{Colors.END}")
    print(f"{'chunk':>8} | {'topology':>8} | {'dtype':>8} | {'sparsity':>12} | {'avg_ms':>10} | {'util%':>7}")
    for row in results:
        print(f"{row['chunk']:>8} | {row['topology']:>8} | {row['dtype']:>8} | {row['sparsity']:>12} | "
              f"{row['avg_ms']:>10.3f} | {row['utilization_pct']:>6.1f}%")
    print(f"csv | {CSV_PATH}")


if __name__ == "__main__":
    cl.profiling(True)
    run()
