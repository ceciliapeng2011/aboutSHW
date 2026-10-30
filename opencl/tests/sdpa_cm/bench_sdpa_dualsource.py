"""Perf sweep comparing {single-source, dual-source (cache+current)} x {fp16, int8-cache/int8} x
{dense, sparse at several densities}, at FlashVSR's 1080p config (video_height=1152,
video_width=1920 -- flashvsr_dit.sh's default resolution) and its two real streaming-KV-cache
chunk shapes (see flashvsr/bench_flashvsr_e2e.py's module docstring for the full derivation of
these numbers from wan_video_dit.py's SelfAttention.forward):

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
    SDPA_BENCH_1080P_CSV=my_results.csv python bench_sdpa_dualsource.py
    SDPA_SPARSE_BLOCK_SIZE=256 python bench_sdpa_dualsource.py   # 128 (default) or 256
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
from int8_kv_ref import encode_cache_by_token, encode_cache_k_by_channel

TRANSPOSE_ORDER_IDENTITY = base.TRANSPOSE_ORDER_IDENTITY
TRANSPOSE_ORDER_SWAPPED = base.TRANSPOSE_ORDER_SWAPPED
KV_CACHE_COMPRESSION_BY_TOKEN = 1
KV_CACHE_COMPRESSION_BY_CHANNEL = 2

torch.manual_seed(7)

HW_PEAK_FLOPS = float(os.environ.get("HW_PEAK_FLOPS", "31")) * 1e12

# FlashVSR real self-attention config: self-attention has no GQA (num_kv_heads == num_heads).
NUM_HEADS = 12
NUM_KV_HEADS = 12
HEAD_SIZE = 128
# 1, 128, or 256. 1 means "bare dense, no block-mask infra at all" (cm_sdpa_opt_prefill.cm's own
# SPARSE_BLOCK_SIZE==1 convention) and is ONLY valid for single-source's density==1.0 point --
# cm_sdpa_opt_prefill_dualsource.cm has no such bare-dense path (static_assert SPARSE_BLOCK_SIZE>1
# always), and single-source itself needs a real block mask once density<1.0. Both cases silently
# fall back to 128 below (DUAL_SPARSE_BLOCK_SIZE / _single_block_size) rather than failing the sweep.
SPARSE_BLOCK_SIZE = int(os.environ.get("SDPA_SPARSE_BLOCK_SIZE", "128"))
assert SPARSE_BLOCK_SIZE in (1, 128, 256), f"SDPA_SPARSE_BLOCK_SIZE must be 1, 128, or 256, got {SPARSE_BLOCK_SIZE}"
DUAL_SPARSE_BLOCK_SIZE = SPARSE_BLOCK_SIZE if SPARSE_BLOCK_SIZE > 1 else 128


def _single_block_size(density):
    # SPARSE_BLOCK_SIZE==1 has no mask ABI at all -- only usable when there's truly no mask needed.
    return 128 if (SPARSE_BLOCK_SIZE == 1 and density < 1.0) else SPARSE_BLOCK_SIZE

# FlashVSR 1080p config (video_height=1152, video_width=1920 -- flashvsr_dit.sh's default
# resolution). patch_h/w = VAE 8x downsample x patchify 2x2 == /16; base_h/w = 8x8 spatial window
# (//8); one_len is this resolution's SPATIAL_WINDOWS analog.
VIDEO_HEIGHT_1080P, VIDEO_WIDTH_1080P = 1152, 1920
PATCH_H_1080P, PATCH_W_1080P = VIDEO_HEIGHT_1080P // 16, VIDEO_WIDTH_1080P // 16  # 72, 120
BASE_H_1080P, BASE_W_1080P = PATCH_H_1080P // 8, PATCH_W_1080P // 8              # 9, 15
SPATIAL_WINDOWS_1080P = BASE_H_1080P * BASE_W_1080P                              # 135
WINDOW_GROUP_TOKENS_1080P = SPATIAL_WINDOWS_1080P * 128                          # 17,280

# (name, cache_groups, cur_groups): cache_groups*WINDOW_GROUP_TOKENS_1080P == kv_len_cache,
# cur_groups*WINDOW_GROUP_TOKENS_1080P == q_len == kv_len_cur (this chunk's own new K/V), matching
# wan_video_dit.py's streaming FIFO cache. warmup (f=6, pre_cache is None): no cache yet, but its
# own current chunk is the full 3 window-groups (f=6 frames). steady (f=2): a stabilized kv_len=3
# cache plus a 1-window-group current chunk -- the normal per-step shape.
CHUNKS = (
    ("warmup", 0, 3),
    ("steady", 3, 1),
)
DENSITIES = (("dense", 1.0), ("sparse_d0.9", 0.9), ("sparse_d0.6", 0.6), ("sparse_d0.3", 0.3))
SLOT_COMPARISON_CSV_PATH = os.environ.get("SDPA_BENCH_1080P_CSV", "bench_sdpa_1080p_slot_comparison.csv")


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
    # dual-source ALWAYS needs a real block size (never 1) -- see DUAL_SPARSE_BLOCK_SIZE above.
    num_q_blocks = (q_len + DUAL_SPARSE_BLOCK_SIZE - 1) // DUAL_SPARSE_BLOCK_SIZE
    num_k_blocks_cache = (kv_len_cache + DUAL_SPARSE_BLOCK_SIZE - 1) // DUAL_SPARSE_BLOCK_SIZE
    num_k_blocks_cur = (kv_len_cur + DUAL_SPARSE_BLOCK_SIZE - 1) // DUAL_SPARSE_BLOCK_SIZE
    mask = (_fake_mask(num_q_blocks, num_k_blocks_cache, num_k_blocks_cur, density, seed)
            if density < 1.0 else
            torch.ones(1, NUM_HEADS, num_q_blocks, num_k_blocks_cache + num_k_blocks_cur, dtype=torch.bool))
    return mask, num_q_blocks, num_k_blocks_cache, num_k_blocks_cur


def bench_dualsource_fp16(chunk_name, cache_groups, density, seed, window_group_tokens=WINDOW_GROUP_TOKENS_1080P,
                           cur_groups=1):
    q_len = kv_len_cur = cur_groups * window_group_tokens
    kv_len_cache = cache_groups * window_group_tokens
    q, k_cache, v_cache, k_cur, v_cur = _make_qkv(q_len, kv_len_cache, kv_len_cur, seed)

    kernel = ms.build_dualsource_sparse_prefill_kernel(NUM_HEADS, NUM_KV_HEADS, HEAD_SIZE, DUAL_SPARSE_BLOCK_SIZE)

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


def bench_dualsource_int8cache(chunk_name, cache_groups, density, seed,
                                compression_mode=KV_CACHE_COMPRESSION_BY_CHANNEL,
                                window_group_tokens=WINDOW_GROUP_TOKENS_1080P,
                                cur_groups=1):
    # key_cache/value_cache are a single region-split packed uint8 buffer (region A: quantized
    # data, region B: scale/zp) -- see cm_sdpa_opt_prefill_dualsource.cm's / kv_cache_write.cm's
    # module docstrings. No separate key_cache_scale/key_cache_zp kernel argument. This buffer is
    # ALWAYS token-major/heads-interleaved, so (like test_sdpa_prefill_int8_cache.py)
    # K/V/Q/OUTPUT_TRANSPOSE_ORDER must be SWAPPED to match -- the real BlockSparseAttention
    # primitive's own ABI for this op.
    src = r'''#include "cm_sdpa_opt_prefill_dualsource.cm"'''
    scale_factor = 1.0 / (HEAD_SIZE ** 0.5)
    opts = (
        f'-cmc -Qxcm_jit_option="-abortonspill" -Qxcm_register_file_size=256 -mCM_printregusage '
        f"-I{base.CWD} -I{base.FLASHATTN_DIR} -I{base.PAGEATTEN_DIR} "
        f"-DKERNEL_NAME=cm_sdpa_opt_prefill_dualsource_int8cache "
        f"-DCMFLA_NUM_HEADS={NUM_HEADS} -DCMFLA_NUM_KV_HEADS={NUM_KV_HEADS} "
        f"-DCMFLA_HEAD_SIZE={HEAD_SIZE} -DCMFLA_SCALE_FACTOR={scale_factor} "
        f"-DQ_TRANSPOSE_ORDER={TRANSPOSE_ORDER_SWAPPED} -DK_TRANSPOSE_ORDER={TRANSPOSE_ORDER_SWAPPED} "
        f"-DV_TRANSPOSE_ORDER={TRANSPOSE_ORDER_SWAPPED} -DOUTPUT_TRANSPOSE_ORDER={TRANSPOSE_ORDER_SWAPPED} "
        f"-DSPARSE_BLOCK_SIZE={DUAL_SPARSE_BLOCK_SIZE} -DKV_CACHE_COMPRESSION={compression_mode}"
    )
    kernel = cl.kernels(src, opts)

    q_len = kv_len_cur = cur_groups * window_group_tokens
    kv_len_cache = cache_groups * window_group_tokens
    q, k_cache, v_cache, k_cur, v_cur = _make_qkv(q_len, kv_len_cache, kv_len_cur, seed)
    k_cache_packed = (encode_cache_k_by_channel(k_cache) if compression_mode == KV_CACHE_COMPRESSION_BY_CHANNEL
                      else encode_cache_by_token(k_cache))
    v_cache_packed = encode_cache_by_token(v_cache)

    t_q = cl.tensor(base.to_physical(q, TRANSPOSE_ORDER_SWAPPED).numpy())
    t_k_cache = cl.tensor(k_cache_packed.contiguous().numpy())
    t_v_cache = cl.tensor(v_cache_packed.contiguous().numpy())
    t_k_cur = cl.tensor(base.to_physical(k_cur, TRANSPOSE_ORDER_SWAPPED).numpy())
    t_v_cur = cl.tensor(base.to_physical(v_cur, TRANSPOSE_ORDER_SWAPPED).numpy())
    t_out = cl.tensor(np.zeros([1, q_len, NUM_HEADS, HEAD_SIZE], dtype=np.float16))

    mask, num_q_blocks, num_k_blocks_cache, num_k_blocks_cur = _mask_for(
        q_len, kv_len_cache, kv_len_cur, density, seed + 1)
    t_mask = cl.tensor(mask.numpy())

    GWS, LWS = _dispatch(q_len)
    args = [t_q, t_k_cache, t_v_cache, t_k_cur, t_v_cur, t_out, q_len, kv_len_cache, kv_len_cur,
            t_mask, num_q_blocks, num_k_blocks_cache, num_k_blocks_cur]

    cl.finish()
    for _ in range(10):
        kernel.enqueue("cm_sdpa_opt_prefill_dualsource_int8cache", GWS, LWS, *args)
    cl.finish()
    for _ in range(30):
        kernel.enqueue("cm_sdpa_opt_prefill_dualsource_int8cache", GWS, LWS, *args)
    lat_ms = [ns * 1e-6 for ns in cl.finish()]

    eff_density = mask.float().mean().item()
    total_flops = NUM_HEADS * q_len * (kv_len_cache + kv_len_cur) * HEAD_SIZE * 4 * eff_density
    mode_tag = "by_channel" if compression_mode == KV_CACHE_COMPRESSION_BY_CHANNEL else "by_token"
    label = (f"dual-int8cache[{mode_tag}] {chunk_name:<7} q={q_len:<6} kv_cache={kv_len_cache:<6} "
             f"kv_cur={kv_len_cur:<6} density(eff)={eff_density:.3f}")
    result = report(label, total_flops, lat_ms)
    del t_q, t_k_cache, t_v_cache, t_k_cur, t_v_cur, t_out, t_mask
    gc.collect()
    return result


def bench_dualsource_int8cache_by_token(chunk_name, cache_groups, density, seed,
                                         window_group_tokens=WINDOW_GROUP_TOKENS_1080P, cur_groups=1):
    return bench_dualsource_int8cache(chunk_name, cache_groups, density, seed,
                                       compression_mode=KV_CACHE_COMPRESSION_BY_TOKEN,
                                       window_group_tokens=window_group_tokens, cur_groups=cur_groups)


def _single_source_kv(cache_groups, seed, window_group_tokens=WINDOW_GROUP_TOKENS_1080P, cur_groups=1):
    """Concatenated-shape baseline: cache+current as ONE [kv_len_cache+kv_len_cur] buffer, same
    approach flashvsr/bench_flashvsr_e2e.py's bench_dense_reference uses."""
    q_len = kv_len_cur = cur_groups * window_group_tokens
    kv_len_cache = cache_groups * window_group_tokens
    q, k_cache, v_cache, k_cur, v_cur = _make_qkv(q_len, kv_len_cache, kv_len_cur, seed)
    k_cache_real = k_cache[:, :, :kv_len_cache] if kv_len_cache > 0 else k_cache[:, :, :0]
    v_cache_real = v_cache[:, :, :kv_len_cache] if kv_len_cache > 0 else v_cache[:, :, :0]
    k = torch.cat([k_cache_real, k_cur], dim=2)
    v = torch.cat([v_cache_real, v_cur], dim=2)
    return q, k, v, q_len, kv_len_cache + kv_len_cur


def bench_singlesource_fp16(chunk_name, cache_groups, density, seed, window_group_tokens=WINDOW_GROUP_TOKENS_1080P,
                             cur_groups=1):
    q, k, v, q_len, kv_len = _single_source_kv(cache_groups, seed, window_group_tokens, cur_groups)
    block_size = _single_block_size(density)

    kernel = bm.build_sparse_prefill_kernel(NUM_HEADS, NUM_KV_HEADS, HEAD_SIZE, is_causal=False,
                                             sparse_block_size=block_size)

    t_q = cl.tensor(q.contiguous().numpy())
    t_k = cl.tensor(k.contiguous().numpy())
    t_v = cl.tensor(v.contiguous().numpy())
    t_out = cl.tensor(np.zeros([1, NUM_HEADS, q_len, HEAD_SIZE], dtype=np.float16))

    GWS, LWS = _dispatch(q_len)
    if block_size == 1:
        # Bare dense -- cm_sdpa_opt_prefill.cm compiles NO mask args at all at SPARSE_BLOCK_SIZE==1.
        eff_density = 1.0
        t_mask = None
        args = [t_q, t_k, t_v, t_out, q_len, kv_len]
    else:
        num_q_blocks = (q_len + block_size - 1) // block_size
        num_k_blocks = (kv_len + block_size - 1) // block_size
        mask = (torch.rand(1, NUM_HEADS, num_q_blocks, num_k_blocks, generator=torch.Generator().manual_seed(seed + 1))
                < density).to(torch.bool) if density < 1.0 else torch.ones(1, NUM_HEADS, num_q_blocks, num_k_blocks, dtype=torch.bool)
        mask[:, :, :, 0] = True
        t_mask = cl.tensor(mask.numpy())
        eff_density = mask.float().mean().item()
        args = [t_q, t_k, t_v, t_out, q_len, kv_len, t_mask, num_q_blocks, num_k_blocks]

    cl.finish()
    for _ in range(10):
        kernel.enqueue("cm_sdpa_opt_prefill", GWS, LWS, *args)
    cl.finish()
    for _ in range(30):
        kernel.enqueue("cm_sdpa_opt_prefill", GWS, LWS, *args)
    lat_ms = [ns * 1e-6 for ns in cl.finish()]

    total_flops = NUM_HEADS * q_len * kv_len * HEAD_SIZE * 4 * eff_density
    label = f"single-fp16   {chunk_name:<7} q={q_len:<6} kv={kv_len:<6} density(eff)={eff_density:.3f}"
    result = report(label, total_flops, lat_ms)
    del t_q, t_k, t_v, t_out
    if t_mask is not None:
        del t_mask
    gc.collect()
    return result


def bench_singlesource_int8(chunk_name, cache_groups, density, seed,
                             compression_mode=KV_CACHE_COMPRESSION_BY_CHANNEL,
                             window_group_tokens=WINDOW_GROUP_TOKENS_1080P,
                             cur_groups=1):
    # key/value carry scale/zp in a region-split layout within the SAME buffer -- see
    # cm_sdpa_opt_prefill.cm's module docstring. No separate key_scale/key_zp/value_scale/
    # value_zp kernel argument. q/k/v below are IDENTITY-order ([B,H,S,D]), matching the kernel
    # build's implicit default order (no Q_TRANSPOSE_ORDER/K_TRANSPOSE_ORDER/etc override here).
    q, k, v, q_len, kv_len = _single_source_kv(cache_groups, seed, window_group_tokens, cur_groups)
    k_packed = (encode_cache_k_by_channel(k, order=TRANSPOSE_ORDER_IDENTITY)
                if compression_mode == KV_CACHE_COMPRESSION_BY_CHANNEL
                else encode_cache_by_token(k, order=TRANSPOSE_ORDER_IDENTITY))
    v_packed = encode_cache_by_token(v, order=TRANSPOSE_ORDER_IDENTITY)
    block_size = _single_block_size(density)

    src = r'''#include "cm_sdpa_opt_prefill.cm"'''
    scale_factor = 1.0 / (HEAD_SIZE ** 0.5)
    opts = (
        f'-cmc -Qxcm_jit_option="-abortonspill" -Qxcm_register_file_size=256 -mCM_printregusage '
        f"-I{base.CWD} -I{base.FLASHATTN_DIR} -I{base.PAGEATTEN_DIR} "
        f"-DKERNEL_NAME=cm_sdpa_opt_prefill_int8_bench "
        f"-DCMFLA_NUM_HEADS={NUM_HEADS} -DCMFLA_NUM_KV_HEADS={NUM_KV_HEADS} "
        f"-DCMFLA_HEAD_SIZE={HEAD_SIZE} -DCMFLA_SCALE_FACTOR={scale_factor} "
        f"-DCMFLA_IS_CAUSAL=0 -DHAS_ATTN_MASK_INPUT=0 -DHAS_SCALE_INPUT=0 "
        f"-DSPARSE_BLOCK_SIZE={block_size} -DKV_CACHE_COMPRESSION={compression_mode}"
    )
    kernel = cl.kernels(src, opts)

    t_q = cl.tensor(q.contiguous().numpy())
    t_k = cl.tensor(k_packed.contiguous().numpy())
    t_v = cl.tensor(v_packed.contiguous().numpy())
    t_out = cl.tensor(np.zeros([1, NUM_HEADS, q_len, HEAD_SIZE], dtype=np.float16))

    GWS, LWS = _dispatch(q_len)
    if block_size == 1:
        # Bare dense -- cm_sdpa_opt_prefill.cm compiles NO mask args at all at SPARSE_BLOCK_SIZE==1.
        eff_density = 1.0
        t_mask = None
        args = [t_q, t_k, t_v, t_out, q_len, kv_len]
    else:
        num_q_blocks = (q_len + block_size - 1) // block_size
        num_k_blocks = (kv_len + block_size - 1) // block_size
        mask = (torch.rand(1, NUM_HEADS, num_q_blocks, num_k_blocks, generator=torch.Generator().manual_seed(seed + 1))
                < density).to(torch.bool) if density < 1.0 else torch.ones(1, NUM_HEADS, num_q_blocks, num_k_blocks, dtype=torch.bool)
        mask[:, :, :, 0] = True
        t_mask = cl.tensor(mask.numpy())
        eff_density = mask.float().mean().item()
        args = [t_q, t_k, t_v, t_out, q_len, kv_len, t_mask, num_q_blocks, num_k_blocks]

    cl.finish()
    for _ in range(10):
        kernel.enqueue("cm_sdpa_opt_prefill_int8_bench", GWS, LWS, *args)
    cl.finish()
    for _ in range(30):
        kernel.enqueue("cm_sdpa_opt_prefill_int8_bench", GWS, LWS, *args)
    lat_ms = [ns * 1e-6 for ns in cl.finish()]

    total_flops = NUM_HEADS * q_len * kv_len * HEAD_SIZE * 4 * eff_density
    mode_tag = "by_channel" if compression_mode == KV_CACHE_COMPRESSION_BY_CHANNEL else "by_token"
    label = f"single-int8[{mode_tag}] {chunk_name:<7} q={q_len:<6} kv={kv_len:<6} density(eff)={eff_density:.3f}"
    result = report(label, total_flops, lat_ms)
    del t_q, t_k, t_v, t_out
    if t_mask is not None:
        del t_mask
    gc.collect()
    return result


def bench_singlesource_int8_by_token(chunk_name, cache_groups, density, seed,
                                      window_group_tokens=WINDOW_GROUP_TOKENS_1080P, cur_groups=1):
    return bench_singlesource_int8(chunk_name, cache_groups, density, seed,
                                    compression_mode=KV_CACHE_COMPRESSION_BY_TOKEN,
                                    window_group_tokens=window_group_tokens, cur_groups=cur_groups)


BENCHES = (
    ("single", "fp16", bench_singlesource_fp16),
    ("single", "int8_by_channel", bench_singlesource_int8),
    ("single", "int8_by_token", bench_singlesource_int8_by_token),
    ("dual", "fp16", bench_dualsource_fp16),
    ("dual", "int8cache_by_channel", bench_dualsource_int8cache),
    ("dual", "int8cache_by_token", bench_dualsource_int8cache_by_token),
)


# 1080p slot comparison: swept over CHUNKS x DENSITIES x BENCHES, same cache_groups/density
# structure run() used to sweep for the 320x480 profile, now at FlashVSR's 1080p shape
# (WINDOW_GROUP_TOKENS_1080P). "slot count" (used only in the cosmetic per-kernel [perf] label
# below) differs by WHICH kernel topology feeds the same cache_groups+current data in:
#   * single-source ("Nslot" = cache_groups+1): cache and current are concatenated into ONE
#     physical buffer before the kernel call -- the legacy/no-ring-buffer design, needing an extra
#     buffer's worth of memory to hold the concat result.
#   * dual-source ("Nslot" = cache_groups): cache (N physical ring-buffer slots) and current (a
#     separate, never concatenated buffer) are fed to the kernel directly -- the ring-buffer design
#     that avoids ever materializing that extra buffer (see
#     flashvsr/test_flashvsr_kv_ring_buffer_permutation.py).
def run_1080p_slot_comparison():
    results = []
    seed = 2000
    for chunk_name, cache_groups, cur_groups in CHUNKS:
        for sparsity_name, density in DENSITIES:
            if chunk_name == "warmup" and sparsity_name != "dense":
                # warmup is only benchmarked at dense -- the steady-state chunk (kv_ratio=3.0's real
                # per-step shape) is where the sparsity/dtype tradeoffs above actually matter.
                continue
            for topology, dtype_name, bench_fn in BENCHES:
                slot_n = cache_groups + cur_groups if topology == "single" else cache_groups
                label = f"{chunk_name}_{slot_n}slot_{dtype_name}"  # cosmetic -- [perf] print only.
                print(f"{Colors.GREEN}[bench-1080p] chunk={chunk_name} topology={topology} "
                      f"dtype={dtype_name} sparsity={sparsity_name}{Colors.END}")
                avg_ms, util = bench_fn(label, cache_groups, density, seed,
                                         window_group_tokens=WINDOW_GROUP_TOKENS_1080P, cur_groups=cur_groups)
                results.append({
                    "chunk": chunk_name, "topology": topology, "dtype": dtype_name,
                    "sparsity": sparsity_name, "avg_ms": round(avg_ms, 2), "utilization_pct": round(util, 2),
                })
                seed += 1

    with open(SLOT_COMPARISON_CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["chunk", "topology", "dtype", "sparsity", "avg_ms", "utilization_pct"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\n{Colors.GREEN}[summary-1080p] video={VIDEO_HEIGHT_1080P}x{VIDEO_WIDTH_1080P} "
          f"patch={PATCH_H_1080P}x{PATCH_W_1080P} base={BASE_H_1080P}x{BASE_W_1080P} "
          f"one_len={SPATIAL_WINDOWS_1080P} window_group_tokens={WINDOW_GROUP_TOKENS_1080P} "
          f"sparse_block_size={SPARSE_BLOCK_SIZE} hw_peak_flops={HW_PEAK_FLOPS/1e12:.0f}TFLOPS{Colors.END}")
    print(f"{'chunk':>8} | {'topology':>8} | {'dtype':>20} | {'sparsity':>12} | {'avg_ms':>10} | {'util%':>7}")
    for row in results:
        print(f"{row['chunk']:>8} | {row['topology']:>8} | {row['dtype']:>20} | {row['sparsity']:>12} | "
              f"{row['avg_ms']:>10.3f} | {row['utilization_pct']:>6.1f}%")
    print(f"csv | {SLOT_COMPARISON_CSV_PATH}")


if __name__ == "__main__":
    cl.profiling(True)
    run_1080p_slot_comparison()
