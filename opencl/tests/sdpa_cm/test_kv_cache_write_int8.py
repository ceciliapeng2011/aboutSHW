"""
int8-cache correctness for kv_cache_write.cm's CACHE_COMPRESSION=1 (BY_TOKEN) and CACHE_COMPRESSION=2
(BY_CHANNEL, K only) paths: scatters `updates` (half) into an int8-packed `cache` at
`write_indices` physical slots, quantizing on the fly (affine uint8, never spanning multiple kv
heads) with the scale/zero-point pairs packed into a dedicated region B, immediately after region
A's quantized data -- matching the real OpenVINO BlockSparseAttention primitive's on-device cache
layout -- see that file's module docstring and cm_sdpa_opt_prefill_dualsource.cm's identical
convention.

Correctness oracle (same methodology as test_sdpa_prefill_int8_cache.py -- NOT "is int8
quantization accurate enough", that's not what this isolates): dequantize the kernel's own written
bytes+scale+zp back to half, and compare against the host reference (int8_kv_ref.py's
encode_cache_by_token/encode_cache_k_by_channel, the SAME formula kv_cache_write.cm's
quantize-on-write math is adapted from, see pageatten/pa_kv_cache_update_ref.cm) applied directly
to the same `updates` slice, also round-tripped through quantize+dequantize -- isolates "does the
kernel's in-register quant formula match the reference" from ordinary quantization noise.
"""
import functools

import numpy as np
import torch

from clops import cl

import test_sdpa_prefill as base
from int8_kv_ref import (
    encode_by_token, encode_k_by_channel,
    encode_cache_by_token, dequant_cache_by_token,
    encode_cache_k_by_channel, dequant_cache_k_by_channel,
)

torch.manual_seed(51)

CACHE_COMPRESSION_BY_TOKEN = 1
CACHE_COMPRESSION_BY_CHANNEL = 2
GROUP_SIZE = 16


@functools.cache
def build_cache_write_kernel_int8(num_kv_heads, head_size, compression_mode):
    src = r'''#include "kv_cache_write.cm"'''
    opts = (
        f'-cmc -Qxcm_jit_option="-abortonspill" -Qxcm_register_file_size=256 -mCM_printregusage '
        f"-I{base.CWD} -DKERNEL_NAME=kv_cache_write_int8 "
        f"-DNUM_KV_HEADS={num_kv_heads} -DHEAD_SIZE={head_size} "
        f"-DCACHE_COMPRESSION={compression_mode} -DGROUP_SIZE={GROUP_SIZE}"
    )
    return cl.kernels(src, opts)


def run_cache_write_int8(updates, write_indices, num_slots, num_kv_heads, head_size, compression_mode):
    """updates: [n_new, tokens_per_group, num_kv_heads*head_size] half torch tensor.
    Returns a flat uint8 numpy array of length lk_cache*num_kv_heads*(head_size+4) -- region A
    (quantized data, [lk_cache, num_kv_heads, head_size]) followed by region B (packed scale/zp).
    Untouched (not-written) slots stay zero -- callers only inspect the slots they wrote.
    """
    n_new, tokens_per_group, cache_write_dim = updates.shape
    assert cache_write_dim == num_kv_heads * head_size
    kernel = build_cache_write_kernel_int8(num_kv_heads, head_size, compression_mode)

    lk_cache = num_slots * tokens_per_group
    t_updates = cl.tensor(updates.contiguous().to(torch.float16).numpy())
    t_write_indices = cl.tensor(write_indices.contiguous().to(torch.int32).numpy())
    # Region A (lk_cache*num_kv_heads*head_size bytes) is the same for both modes; region B's
    # size differs -- BY_TOKEN is num_kv_heads*lk_cache*4 (one pair/token/head), BY_CHANNEL is
    # num_kv_heads*(lk_cache/GROUP_SIZE)*head_size*4 (one pair/channel/group/head) -- these only
    # coincide when head_size == GROUP_SIZE, so the buffer must be sized per-mode, not via a
    # uniform "+4 columns" shortcut.
    region_a_size = lk_cache * num_kv_heads * head_size
    if compression_mode == CACHE_COMPRESSION_BY_CHANNEL:
        assert tokens_per_group % GROUP_SIZE == 0
        region_b_size = num_kv_heads * (lk_cache // GROUP_SIZE) * head_size * 4
        GWS = [num_kv_heads, tokens_per_group // GROUP_SIZE, n_new]
    else:
        region_b_size = num_kv_heads * lk_cache * 4
        GWS = [num_kv_heads, tokens_per_group, n_new]
    t_cache = cl.tensor(np.zeros((region_a_size + region_b_size,), dtype=np.uint8))
    LWS = [1, 1, 1]
    kernel.enqueue("kv_cache_write_int8", GWS, LWS,
                    t_updates, t_write_indices, t_cache,
                    int(tokens_per_group), int(n_new), int(lk_cache))
    return t_cache.numpy()


def _check_group(updates, write_indices, num_slots, num_kv_heads, head_size, compression_mode):
    n_new, tokens_per_group, _ = updates.shape
    lk_cache = num_slots * tokens_per_group
    cache_packed = torch.from_numpy(
        run_cache_write_int8(updates, write_indices, num_slots, num_kv_heads, head_size, compression_mode)
    ).unsqueeze(0)  # [1, lk_cache*num_kv_heads*(head_size+4)]

    if compression_mode == CACHE_COMPRESSION_BY_CHANNEL:
        kernel_roundtrip = dequant_cache_k_by_channel(cache_packed, lk_cache, head_size, num_kv_heads,
                                                       group_size=GROUP_SIZE)
    else:
        kernel_roundtrip = dequant_cache_by_token(cache_packed, lk_cache, head_size, num_kv_heads)

    for g in range(n_new):
        slot = int(write_indices[g])
        upd_bhsd = (updates[g].reshape(tokens_per_group, num_kv_heads, head_size)
                    .permute(1, 0, 2).unsqueeze(0).contiguous())  # [1, Hkv, tokens_per_group, D]

        if compression_mode == CACHE_COMPRESSION_BY_CHANNEL:
            ref_packed = encode_cache_k_by_channel(upd_bhsd, group_size=GROUP_SIZE)
            ref_roundtrip = dequant_cache_k_by_channel(ref_packed, tokens_per_group, head_size, num_kv_heads,
                                                        group_size=GROUP_SIZE)
            _, ref_scale, _ = encode_k_by_channel(upd_bhsd, sub_block_size=GROUP_SIZE)
        else:
            ref_packed = encode_cache_by_token(upd_bhsd)
            ref_roundtrip = dequant_cache_by_token(ref_packed, tokens_per_group, head_size, num_kv_heads)
            _, ref_scale, _ = encode_by_token(upd_bhsd)

        kernel_slice = kernel_roundtrip[:, :, slot * tokens_per_group:(slot + 1) * tokens_per_group, :]
        # This oracle compares two INDEPENDENT quantizations (kernel's on-device encode vs. the
        # host reference's own encode) of the same data, not a decode of identical bytes (unlike
        # test_sdpa_prefill_int8_cache.py's decode-only oracle) -- the kernel's zp is computed in
        # half precision (matching pageatten/pa_kv_cache_update_ref.cm's proven
        # process_quantization_per_* formula this is adapted from) while the host reference keeps
        # zp in float32 until the final cast, so individual elements can occasionally round to an
        # adjacent uint8 level. A fixed atol/rtol is unstable near-zero (one quantization level can
        # be a huge relative error there); bound by the reference's OWN quantization step instead --
        # at most ~1.5 steps of disagreement is a formula-precision artifact, not a correctness bug.
        step = ref_scale.abs().max().item()
        max_diff = (kernel_slice.to(torch.float32) - ref_roundtrip.to(torch.float32)).abs().max().item()
        assert max_diff <= 1.5 * step, (
            f"max diff {max_diff:.6f} exceeds 1.5x the quantization step {step:.6f}"
        )


def _make_updates(n_new, tokens_per_group, num_kv_heads, head_size, seed):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(n_new, tokens_per_group, num_kv_heads * head_size, generator=g,
                         dtype=torch.float32) * 1.0).to(torch.float16)


def test_by_token_matches_host_reference():
    num_slots, n_new, tokens_per_group, num_kv_heads, head_size = 4, 2, 128, 4, 64
    updates = _make_updates(n_new, tokens_per_group, num_kv_heads, head_size, seed=202)
    write_indices = torch.tensor([0, 2], dtype=torch.int32)
    _check_group(updates, write_indices, num_slots, num_kv_heads, head_size, CACHE_COMPRESSION_BY_TOKEN)
    print("test_by_token_matches_host_reference PASSED")


def test_by_token_head_size_128():
    num_slots, n_new, tokens_per_group, num_kv_heads, head_size = 3, 1, 128, 4, 128
    updates = _make_updates(n_new, tokens_per_group, num_kv_heads, head_size, seed=203)
    write_indices = torch.tensor([2], dtype=torch.int32)
    _check_group(updates, write_indices, num_slots, num_kv_heads, head_size, CACHE_COMPRESSION_BY_TOKEN)
    print("test_by_token_head_size_128 PASSED")


def test_single_slot_single_group():
    num_slots, n_new, tokens_per_group, num_kv_heads, head_size = 1, 1, 16, 2, 32
    updates = _make_updates(n_new, tokens_per_group, num_kv_heads, head_size, seed=204)
    write_indices = torch.tensor([0], dtype=torch.int32)
    _check_group(updates, write_indices, num_slots, num_kv_heads, head_size, CACHE_COMPRESSION_BY_TOKEN)
    print("test_single_slot_single_group PASSED")


def test_by_channel_matches_host_reference():
    num_slots, n_new, tokens_per_group, num_kv_heads, head_size = 4, 2, 128, 4, 64
    updates = _make_updates(n_new, tokens_per_group, num_kv_heads, head_size, seed=205)
    write_indices = torch.tensor([0, 2], dtype=torch.int32)
    _check_group(updates, write_indices, num_slots, num_kv_heads, head_size, CACHE_COMPRESSION_BY_CHANNEL)
    print("test_by_channel_matches_host_reference PASSED")


def test_by_channel_head_size_128():
    num_slots, n_new, tokens_per_group, num_kv_heads, head_size = 3, 1, 128, 4, 128
    updates = _make_updates(n_new, tokens_per_group, num_kv_heads, head_size, seed=206)
    write_indices = torch.tensor([2], dtype=torch.int32)
    _check_group(updates, write_indices, num_slots, num_kv_heads, head_size, CACHE_COMPRESSION_BY_CHANNEL)
    print("test_by_channel_head_size_128 PASSED")


def test_by_channel_single_group():
    # tokens_per_group == GROUP_SIZE exactly -- one BY_CHANNEL group per write call, the narrowest
    # legal case (tokens_per_group must be a multiple of GROUP_SIZE).
    num_slots, n_new, tokens_per_group, num_kv_heads, head_size = 1, 1, GROUP_SIZE, 2, 32
    updates = _make_updates(n_new, tokens_per_group, num_kv_heads, head_size, seed=207)
    write_indices = torch.tensor([0], dtype=torch.int32)
    _check_group(updates, write_indices, num_slots, num_kv_heads, head_size, CACHE_COMPRESSION_BY_CHANNEL)
    print("test_by_channel_single_group PASSED")


if __name__ == "__main__":
    cl.profiling(True)
    test_by_token_matches_host_reference()
    test_by_token_head_size_128()
    test_single_slot_single_group()
    test_by_channel_matches_host_reference()
    test_by_channel_head_size_128()
    test_by_channel_single_group()
    print("ALL PASSED")
