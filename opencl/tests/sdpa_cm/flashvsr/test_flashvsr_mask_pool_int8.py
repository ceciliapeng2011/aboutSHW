"""
int8-packed-input correctness tests for draft_block_mask_pool.cm's K-pooling stage
(K_CACHE_COMPRESSION macro) -- Stage 1 of DraftBlockMask. Q is never quantized in this project
(only cache K/V ever is), so this int8 input path only ever matters for K-pooling; there is no Q
counterpart to test. Stages 2/3 (draft_block_mask_score_softmax.cm, draft_block_mask_topk_mask.cm)
only ever consume the already-pooled half output of this stage regardless of how it was produced,
so they need no int8-specific test coverage of their own.

Correctness oracle: host-encode a random windowed K tensor to int8 bytes+scale+zp (via
int8_kv_ref.py's thin wrapper around opencl/tests/pageatten/kv_cache_quant_utils.py, treating the
whole NUM_HEADS*HEAD_DIM axis as one flat channel set -- see
flashvsr_mask_adapter.encode_k_windows_int8), run the new K_CACHE_COMPRESSION!=0 kernel variant
directly on those raw bytes (flashvsr_mask_adapter.pool_k_int8), and compare against running the
EXISTING (unmodified) half kernel on the host-decoded-back-to-half version of the SAME bytes -- a
tight numeric tolerance (not a quantization-accuracy check), matching this project's established
int8 testing methodology (see test_sdpa_prefill_int8.py).
"""
import os
import sys

import numpy as np
import torch

from clops import cl

CWD = os.path.dirname(os.path.realpath(__file__))
SDPA_CM_DIR = os.path.dirname(CWD)
if SDPA_CM_DIR not in sys.path:
    sys.path.insert(0, SDPA_CM_DIR)

import flashvsr_mask_adapter as gpu_mod

torch.manual_seed(17)


def _make_windowed_k(num_windows, window_tokens, nheads, head_dim, seed):
    g = torch.Generator().manual_seed(seed)
    d_full = nheads * head_dim
    k_w = (torch.randn(num_windows, window_tokens, d_full, generator=g, dtype=torch.float32) * 2.0).to(torch.float16)
    return k_w


def _run_pool_int8_vs_half(num_windows, window_tokens, nheads, head_dim, seed, compression_mode):
    k_w = _make_windowed_k(num_windows, window_tokens, nheads, head_dim, seed)
    k_u8, k_scale, k_zp = gpu_mod.encode_k_windows_int8(k_w, compression_mode=compression_mode)

    t_int8_out = gpu_mod.pool_k_int8(k_u8, k_scale, k_zp, num_windows, nheads, head_dim, window_tokens,
                                      compression_mode=compression_mode)

    # Oracle: the EXISTING half kernel, run on the host's own dequantized-back-to-half version of
    # the SAME bytes (not the original k_w) -- isolates "does the kernel's in-register int8 decode
    # match the reference dequant formula" from ordinary quantization noise.
    d_full = nheads * head_dim
    k_w4 = k_u8.reshape(num_windows, 1, window_tokens, d_full)
    if compression_mode == gpu_mod.KV_CACHE_COMPRESSION_BY_CHANNEL:
        from int8_kv_ref import dequant_k_by_channel
        k_scale4 = k_scale.reshape(num_windows, 1, -1, d_full)
        k_zp4 = k_zp.reshape(num_windows, 1, -1, d_full)
        half_roundtrip = dequant_k_by_channel(torch.from_numpy(k_w4), torch.from_numpy(k_scale4),
                                               torch.from_numpy(k_zp4)).squeeze(1).contiguous().numpy()
    else:
        from int8_kv_ref import dequant_by_token
        k_scale4 = k_scale.reshape(num_windows, 1, window_tokens)
        k_zp4 = k_zp.reshape(num_windows, 1, window_tokens)
        half_roundtrip = dequant_by_token(torch.from_numpy(k_w4), torch.from_numpy(k_scale4),
                                           torch.from_numpy(k_zp4)).squeeze(1).contiguous().numpy()

    half_kernel = gpu_mod._get_pool_kernel(nheads, head_dim, window_tokens)
    t_half_out = gpu_mod._pool(half_kernel, half_roundtrip, num_windows, nheads, head_dim)

    return t_int8_out.numpy(), t_half_out.numpy()


def _assert_close(int8_out, half_out, label):
    diff = np.abs(int8_out.astype(np.float32) - half_out.astype(np.float32))
    max_diff = diff.max()
    assert max_diff < 2e-2, f"{label}: max abs diff {max_diff} exceeds tolerance"
    print(f"{label} PASSED (max abs diff {max_diff:.6f})")


def test_pool_int8_by_channel_matches_half_small():
    # window_tokens=16 is the minimum valid for BY_CHANNEL (one 16-token sub-block).
    int8_out, half_out = _run_pool_int8_vs_half(
        num_windows=5, window_tokens=16, nheads=3, head_dim=16, seed=21,
        compression_mode=gpu_mod.KV_CACHE_COMPRESSION_BY_CHANNEL)
    _assert_close(int8_out, half_out, "test_pool_int8_by_channel_matches_half_small")


def test_pool_int8_by_channel_matches_half_flashvsr_config():
    # FlashVSR real production values: head_dim=128, nheads=12, window_tokens=128.
    int8_out, half_out = _run_pool_int8_vs_half(
        num_windows=6, window_tokens=128, nheads=12, head_dim=128, seed=22,
        compression_mode=gpu_mod.KV_CACHE_COMPRESSION_BY_CHANNEL)
    _assert_close(int8_out, half_out, "test_pool_int8_by_channel_matches_half_flashvsr_config")


def test_pool_int8_by_token_matches_half_small():
    int8_out, half_out = _run_pool_int8_vs_half(
        num_windows=5, window_tokens=8, nheads=3, head_dim=16, seed=23,
        compression_mode=gpu_mod.KV_CACHE_COMPRESSION_BY_TOKEN)
    _assert_close(int8_out, half_out, "test_pool_int8_by_token_matches_half_small")


def test_pool_out_row_offset_matches_two_separate_dispatches():
    # HAS_OUT_ROW_OFFSET: pooling two windowed buffers (cache, cur) directly into two row-offset
    # slices of ONE shared device buffer must match pooling each into its own buffer separately
    # and concatenating on the host -- see flashvsr_mask_adapter.build_draft_mask_gpu_dualsource's
    # own use of this (cache at offset 0, cur at offset num_cache_windows).
    nheads, head_dim, window_tokens = 3, 16, 16
    num_cache, num_cur = 4, 2
    k_cache = _make_windowed_k(num_cache, window_tokens, nheads, head_dim, seed=31)
    k_cur = _make_windowed_k(num_cur, window_tokens, nheads, head_dim, seed=32)

    offset_kernel = gpu_mod._get_pool_kernel(nheads, head_dim, window_tokens, has_out_row_offset=True)
    t_shared = cl.tensor(np.zeros((num_cache + num_cur, nheads, head_dim), dtype=np.float16))
    gpu_mod._pool(offset_kernel, k_cache.numpy(), num_cache, nheads, head_dim,
                  t_out=t_shared, out_row_offset=0)
    gpu_mod._pool(offset_kernel, k_cur.numpy(), num_cur, nheads, head_dim,
                  t_out=t_shared, out_row_offset=num_cache)
    combined = t_shared.numpy()

    plain_kernel = gpu_mod._get_pool_kernel(nheads, head_dim, window_tokens)
    cache_only = gpu_mod._pool(plain_kernel, k_cache.numpy(), num_cache, nheads, head_dim).numpy()
    cur_only = gpu_mod._pool(plain_kernel, k_cur.numpy(), num_cur, nheads, head_dim).numpy()

    assert np.array_equal(combined[0:num_cache], cache_only), "cache slice (offset 0) mismatch"
    assert np.array_equal(combined[num_cache:num_cache + num_cur], cur_only), "cur slice (offset num_cache) mismatch"
    print("test_pool_out_row_offset_matches_two_separate_dispatches PASSED")


if __name__ == "__main__":
    test_pool_int8_by_channel_matches_half_small()
    test_pool_int8_by_channel_matches_half_flashvsr_config()
    test_pool_int8_by_token_matches_half_small()
    test_pool_out_row_offset_matches_two_separate_dispatches()
    print("ALL PASSED")
