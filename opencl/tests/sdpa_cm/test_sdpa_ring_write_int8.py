"""
int8-ring correctness for block_sparse_attention_ring_write.cm's RING_CACHE_COMPRESSION={1,2}
path (this repo's own local extension -- no upstream counterpart, see that file's module
docstring): scatters `updates` (half) into an int8-packed `cache` ring at `write_indices` physical
slots, quantizing on the fly (affine uint8, per (kv_head, token-or-16-token-sub-block, group) --
never spanning multiple kv heads, so the written scale/zp land in exactly the layout
cm_sdpa_opt_prefill_dualsource.cm's KV_CACHE_COMPRESSION path already expects to read).

Correctness oracle (same methodology as test_sdpa_prefill_int8_cache.py -- NOT "is int8
quantization accurate enough", that's not what this isolates): dequantize the kernel's own written
bytes+scale+zp back to half, and compare against the host reference (int8_kv_ref.py's thin wrapper
around pageatten/kv_cache_quant_utils.py's quant_per_token_parts/quant_per_channel_parts -- the
SAME formula block_sparse_attention_ring_write.cm's quantize-on-write math is adapted from, see
pa_kv_cache_update_ref.cm) applied directly to the same `updates` slice, also round-tripped through
quantize+dequantize -- isolates "does the kernel's in-register quant formula match the reference"
from ordinary quantization noise.
"""
import functools

import numpy as np
import torch

from clops import cl

import test_sdpa_prefill as base
from int8_kv_ref import encode_k_by_channel, dequant_k_by_channel, encode_by_token, dequant_by_token

torch.manual_seed(51)

KV_CACHE_COMPRESSION_BY_TOKEN = 1
KV_CACHE_COMPRESSION_BY_CHANNEL = 2


@functools.cache
def build_ring_write_kernel_int8(num_kv_heads, head_size, num_slots, compression_mode):
    src = r'''#include "block_sparse_attention_ring_write.cm"'''
    opts = (
        f'-cmc -Qxcm_jit_option="-abortonspill" -Qxcm_register_file_size=256 -mCM_printregusage '
        f"-I{base.CWD} -DKERNEL_NAME=block_sparse_attention_ring_write_int8 "
        f"-DNUM_KV_HEADS={num_kv_heads} -DHEAD_SIZE={head_size} "
        f"-DRING_CACHE_COMPRESSION={int(compression_mode)} -DNUM_SLOTS={int(num_slots)}"
    )
    return cl.kernels(src, opts)


def run_ring_write_int8(updates, write_indices, num_slots, num_kv_heads, head_size,
                         compression_mode=KV_CACHE_COMPRESSION_BY_CHANNEL):
    """updates: [n_new, tokens_per_group, num_kv_heads*head_size] half torch tensor.
    Returns (cache_u8, cache_scale, cache_zp) numpy arrays:
      cache_u8    [num_slots, tokens_per_group, num_kv_heads*head_size] uint8
      cache_scale, cache_zp:
        BY_CHANNEL [num_kv_heads, num_slots*tokens_per_group//16, head_size] half
        BY_TOKEN   [num_kv_heads, num_slots*tokens_per_group] half
    Untouched (not-written) slots/positions stay zero -- callers only inspect the slots they wrote.
    """
    n_new, tokens_per_group, ring_write_dim = updates.shape
    assert ring_write_dim == num_kv_heads * head_size
    kv_len_ring = num_slots * tokens_per_group
    kernel = build_ring_write_kernel_int8(num_kv_heads, head_size, num_slots, compression_mode)

    t_updates = cl.tensor(updates.contiguous().to(torch.float16).numpy())
    t_write_indices = cl.tensor(write_indices.contiguous().to(torch.int32).numpy())
    t_cache = cl.tensor(np.zeros((num_slots, tokens_per_group, ring_write_dim), dtype=np.uint8))
    if compression_mode == KV_CACHE_COMPRESSION_BY_CHANNEL:
        meta_shape = (num_kv_heads, kv_len_ring // 16, head_size)
        GWS = [num_kv_heads, tokens_per_group // 16, n_new]
    else:
        meta_shape = (num_kv_heads, kv_len_ring)
        GWS = [num_kv_heads, tokens_per_group, n_new]
    t_scale = cl.tensor(np.zeros(meta_shape, dtype=np.float16))
    t_zp = cl.tensor(np.zeros(meta_shape, dtype=np.float16))

    LWS = [1, 1, 1]
    kernel.enqueue("block_sparse_attention_ring_write_int8", GWS, LWS,
                    t_updates, t_write_indices, t_cache, t_scale, t_zp,
                    int(tokens_per_group), int(n_new))
    return t_cache.numpy(), t_scale.numpy(), t_zp.numpy()


def _to_bhsd(cache_u8, num_slots, tokens_per_group, num_kv_heads, head_size):
    # [num_slots, tokens_per_group, num_kv_heads*head_size] -> [1, num_kv_heads, kv_len_ring, head_size]
    kv_len_ring = num_slots * tokens_per_group
    x = cache_u8.reshape(kv_len_ring, num_kv_heads, head_size)
    return torch.from_numpy(x).permute(1, 0, 2).unsqueeze(0).contiguous()


def _check_group(updates, write_indices, num_slots, num_kv_heads, head_size, compression_mode):
    n_new, tokens_per_group, _ = updates.shape
    cache_u8, cache_scale, cache_zp = run_ring_write_int8(
        updates, write_indices, num_slots, num_kv_heads, head_size, compression_mode)

    kernel_bsd_u8 = _to_bhsd(cache_u8, num_slots, tokens_per_group, num_kv_heads, head_size)

    if compression_mode == KV_CACHE_COMPRESSION_BY_CHANNEL:
        scale_t = torch.from_numpy(cache_scale).unsqueeze(0)  # [1,H,num_sub_blocks,D]
        zp_t = torch.from_numpy(cache_zp).unsqueeze(0)
        kernel_roundtrip = dequant_k_by_channel(kernel_bsd_u8, scale_t, zp_t)
    else:
        scale_t = torch.from_numpy(cache_scale).unsqueeze(0)  # [1,H,kv_len_ring]
        zp_t = torch.from_numpy(cache_zp).unsqueeze(0)
        kernel_roundtrip = dequant_by_token(kernel_bsd_u8, scale_t, zp_t)

    for g in range(n_new):
        slot = int(write_indices[g])
        upd_bhsd = (updates[g].reshape(tokens_per_group, num_kv_heads, head_size)
                    .permute(1, 0, 2).unsqueeze(0).contiguous())
        if compression_mode == KV_CACHE_COMPRESSION_BY_CHANNEL:
            ref_u8, ref_scale, ref_zp = encode_k_by_channel(upd_bhsd, sub_block_size=16)
            ref_roundtrip = dequant_k_by_channel(ref_u8, ref_scale, ref_zp, sub_block_size=16)
        else:
            ref_u8, ref_scale, ref_zp = encode_by_token(upd_bhsd)
            ref_roundtrip = dequant_by_token(ref_u8, ref_scale, ref_zp)

        kernel_slice = kernel_roundtrip[:, :, slot * tokens_per_group:(slot + 1) * tokens_per_group, :]
        # This oracle compares two INDEPENDENT quantizations (kernel's on-device encode vs. the
        # host reference's own encode) of the same data, not a decode of identical bytes (unlike
        # test_sdpa_prefill_int8_cache.py's decode-only oracle) -- the kernel's zp is computed in
        # half precision (matching pa_kv_cache_update_ref.cm's proven process_quantization_per_*
        # formula this is adapted from) while the host reference keeps zp in float32 until the
        # final cast, so individual elements can occasionally round to an adjacent uint8 level. A
        # fixed atol/rtol is unstable near-zero (one quantization level can be a huge relative
        # error there); bound by the reference's OWN quantization step (ref_scale IS the
        # dequantized value of one uint8 increment) instead -- at most ~1.5 steps of disagreement
        # is a formula-precision artifact, not a correctness bug.
        step = ref_scale.reshape(ref_scale.shape[0], ref_scale.shape[1], -1).abs().max().item()
        max_diff = (kernel_slice.to(torch.float32) - ref_roundtrip.to(torch.float32)).abs().max().item()
        assert max_diff <= 1.5 * step, (
            f"max diff {max_diff:.6f} exceeds 1.5x the quantization step {step:.6f}"
        )


def _make_updates(n_new, tokens_per_group, num_kv_heads, head_size, seed):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(n_new, tokens_per_group, num_kv_heads * head_size, generator=g,
                         dtype=torch.float32) * 1.0).to(torch.float16)


def test_by_channel_matches_host_reference():
    num_slots, n_new, tokens_per_group, num_kv_heads, head_size = 4, 2, 128, 4, 64
    updates = _make_updates(n_new, tokens_per_group, num_kv_heads, head_size, seed=201)
    write_indices = torch.tensor([1, 3], dtype=torch.int32)
    _check_group(updates, write_indices, num_slots, num_kv_heads, head_size, KV_CACHE_COMPRESSION_BY_CHANNEL)
    print("test_by_channel_matches_host_reference PASSED")


def test_by_token_matches_host_reference():
    num_slots, n_new, tokens_per_group, num_kv_heads, head_size = 4, 2, 128, 4, 64
    updates = _make_updates(n_new, tokens_per_group, num_kv_heads, head_size, seed=202)
    write_indices = torch.tensor([0, 2], dtype=torch.int32)
    _check_group(updates, write_indices, num_slots, num_kv_heads, head_size, KV_CACHE_COMPRESSION_BY_TOKEN)
    print("test_by_token_matches_host_reference PASSED")


def test_by_channel_head_size_128():
    num_slots, n_new, tokens_per_group, num_kv_heads, head_size = 3, 1, 128, 4, 128
    updates = _make_updates(n_new, tokens_per_group, num_kv_heads, head_size, seed=203)
    write_indices = torch.tensor([2], dtype=torch.int32)
    _check_group(updates, write_indices, num_slots, num_kv_heads, head_size, KV_CACHE_COMPRESSION_BY_CHANNEL)
    print("test_by_channel_head_size_128 PASSED")


def test_single_slot_single_group():
    num_slots, n_new, tokens_per_group, num_kv_heads, head_size = 1, 1, 16, 2, 32
    updates = _make_updates(n_new, tokens_per_group, num_kv_heads, head_size, seed=204)
    write_indices = torch.tensor([0], dtype=torch.int32)
    _check_group(updates, write_indices, num_slots, num_kv_heads, head_size, KV_CACHE_COMPRESSION_BY_CHANNEL)
    print("test_single_slot_single_group PASSED")


if __name__ == "__main__":
    cl.profiling(True)
    test_by_channel_matches_host_reference()
    test_by_token_matches_host_reference()
    test_by_channel_head_size_128()
    test_single_slot_single_group()
    print("ALL PASSED")
