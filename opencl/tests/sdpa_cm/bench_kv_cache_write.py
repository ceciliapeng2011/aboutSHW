"""Perf sweep for kv_cache_write.cm: fp16 (CACHE_COMPRESSION=0) vs int8 BY_TOKEN (=1) vs int8
BY_CHANNEL (=2, K only -- benchmarked against K-shaped data since V never uses this mode), at
FlashVSR's real self-attention head config (num_kv_heads=12, head_size=128, see
bench_sdpa_dualsource.py's module docstring) and window_tokens=128 (SPARSE_BLOCK_SIZE's own
convention, also a multiple of BY_CHANNEL's GROUP_SIZE=16).

This kernel is pure data movement (scatter-copy, optionally with an on-the-fly affine quantize) --
no DPAS/compute at all -- so it's benchmarked against DRAM bandwidth (GB/s), not GFLOPS/TFLOPS
utilization, mirroring bench_sdpa.py's bench_decode convention for the other
memory-bandwidth-bound kernel in this directory.

Usage:
    HW_PEAK_BW_GBS=100 python bench_kv_cache_write.py
"""
import gc
import os

import numpy as np
import torch

from clops import cl
from clops.utils import Colors

import test_sdpa_prefill as base
from test_kv_cache_write import build_cache_write_kernel
from test_kv_cache_write_int8 import build_cache_write_kernel_int8, CACHE_COMPRESSION_BY_TOKEN, CACHE_COMPRESSION_BY_CHANNEL, GROUP_SIZE

torch.manual_seed(13)

HW_PEAK_BW_GBS = float(os.environ.get("HW_PEAK_BW_GBS", "100"))

NUM_KV_HEADS = 12
HEAD_SIZE = 128
TOKENS_PER_GROUP = 128  # a multiple of GROUP_SIZE(16), matching FlashVSR's real window_tokens.
NUM_SLOTS = 64
N_NEW_VALUES = (1, 4, 16)  # how many whole tokens_per_group-token groups land in one call.


def report(label, total_bytes, lat_ms):
    total_avg = sum(lat_ms) / len(lat_ms)
    gbs = total_bytes / (total_avg * 1e-3) / 1e9 if total_avg > 0 else 0.0
    util = gbs / HW_PEAK_BW_GBS * 100
    print(
        f"[perf] {label} | {Colors.BOLD}{Colors.GREEN}{gbs:,.1f} GB/s{Colors.END}, "
        f"latency(ms) avg={Colors.BOLD}{Colors.YELLOW}{total_avg:.4f}{Colors.END}, "
        f"{Colors.BOLD}{Colors.GREEN}{util:.1f}%{Colors.END} of {HW_PEAK_BW_GBS:.0f} GB/s DRAM peak"
    )
    return total_avg, util


def _make_updates(n_new, seed):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(n_new, TOKENS_PER_GROUP, NUM_KV_HEADS * HEAD_SIZE, generator=g,
                         dtype=torch.float32)).to(torch.float16)


def bench_fp16(n_new, seed):
    updates = _make_updates(n_new, seed)
    write_indices = torch.randperm(NUM_SLOTS, generator=torch.Generator().manual_seed(seed + 1))[:n_new].to(torch.int32)
    kernel = build_cache_write_kernel(NUM_KV_HEADS, HEAD_SIZE)

    cache_write_dim = NUM_KV_HEADS * HEAD_SIZE
    t_updates = cl.tensor(updates.contiguous().numpy())
    t_write_indices = cl.tensor(write_indices.numpy())
    t_cache = cl.tensor(np.zeros((NUM_SLOTS, TOKENS_PER_GROUP, cache_write_dim), dtype=np.float16))

    GWS = [cache_write_dim // 16, TOKENS_PER_GROUP, n_new]
    LWS = [1, 1, 1]
    lk_cache = NUM_SLOTS * TOKENS_PER_GROUP
    args = [t_updates, t_write_indices, t_cache, int(TOKENS_PER_GROUP), int(n_new), int(lk_cache)]

    cl.finish()
    for _ in range(10):
        kernel.enqueue("kv_cache_write", GWS, LWS, *args)
    cl.finish()
    for _ in range(30):
        kernel.enqueue("kv_cache_write", GWS, LWS, *args)
    lat_ms = [ns * 1e-6 for ns in cl.finish()]

    # Read `updates` once (half) + write `cache` once (half) -- the kernel's own data movement.
    total_bytes = n_new * TOKENS_PER_GROUP * cache_write_dim * 2 * 2
    label = f"fp16       n_new={n_new:<3} tokens_per_group={TOKENS_PER_GROUP}"
    result = report(label, total_bytes, lat_ms)
    del t_updates, t_write_indices, t_cache
    gc.collect()
    return result


def bench_int8(n_new, seed, compression_mode):
    updates = _make_updates(n_new, seed)
    write_indices = torch.randperm(NUM_SLOTS, generator=torch.Generator().manual_seed(seed + 1))[:n_new].to(torch.int32)
    kernel = build_cache_write_kernel_int8(NUM_KV_HEADS, HEAD_SIZE, compression_mode)

    cache_write_dim = NUM_KV_HEADS * HEAD_SIZE
    lk_cache = NUM_SLOTS * TOKENS_PER_GROUP
    region_a_size = lk_cache * NUM_KV_HEADS * HEAD_SIZE
    if compression_mode == CACHE_COMPRESSION_BY_CHANNEL:
        region_b_size = NUM_KV_HEADS * (lk_cache // GROUP_SIZE) * HEAD_SIZE * 4
        GWS = [NUM_KV_HEADS, TOKENS_PER_GROUP // GROUP_SIZE, n_new]
    else:
        region_b_size = NUM_KV_HEADS * lk_cache * 4
        GWS = [NUM_KV_HEADS, TOKENS_PER_GROUP, n_new]

    t_updates = cl.tensor(updates.contiguous().numpy())
    t_write_indices = cl.tensor(write_indices.numpy())
    t_cache = cl.tensor(np.zeros((region_a_size + region_b_size,), dtype=np.uint8))

    LWS = [1, 1, 1]
    args = [t_updates, t_write_indices, t_cache, int(TOKENS_PER_GROUP), int(n_new), int(lk_cache)]

    cl.finish()
    for _ in range(10):
        kernel.enqueue("kv_cache_write_int8", GWS, LWS, *args)
    cl.finish()
    for _ in range(30):
        kernel.enqueue("kv_cache_write_int8", GWS, LWS, *args)
    lat_ms = [ns * 1e-6 for ns in cl.finish()]

    # Read `updates` once (half) + write `cache`'s quantized bytes once (uchar, 1/4 the size).
    total_bytes = n_new * TOKENS_PER_GROUP * cache_write_dim * 2 + n_new * TOKENS_PER_GROUP * cache_write_dim
    mode_tag = "by_channel" if compression_mode == CACHE_COMPRESSION_BY_CHANNEL else "by_token"
    label = f"int8[{mode_tag}] n_new={n_new:<3} tokens_per_group={TOKENS_PER_GROUP}"
    result = report(label, total_bytes, lat_ms)
    del t_updates, t_write_indices, t_cache
    gc.collect()
    return result


def bench_int8_by_token(n_new, seed):
    return bench_int8(n_new, seed, CACHE_COMPRESSION_BY_TOKEN)


def bench_int8_by_channel(n_new, seed):
    return bench_int8(n_new, seed, CACHE_COMPRESSION_BY_CHANNEL)


BENCHES = (
    ("fp16", bench_fp16),
    ("int8_by_token", bench_int8_by_token),
    ("int8_by_channel", bench_int8_by_channel),
)


def run():
    print(f"{Colors.GREEN}[config] num_kv_heads={NUM_KV_HEADS} head_size={HEAD_SIZE} "
          f"tokens_per_group={TOKENS_PER_GROUP} num_slots={NUM_SLOTS}{Colors.END}")
    seed = 2000
    for n_new in N_NEW_VALUES:
        print(f"\n{Colors.GREEN}--- n_new={n_new} ---{Colors.END}")
        for dtype_name, bench_fn in BENCHES:
            bench_fn(n_new, seed)
            seed += 1


if __name__ == "__main__":
    cl.profiling(True)
    run()
