"""
int8 (affine uint8, pageatten-style) cache K/V correctness for
cm_sdpa_opt_prefill_dualsource.cm's KV_CACHE_COMPRESSION path: the dual-source attention
kernel decodes int8-packed cache bytes to half once per work-group, staged through the SAME SLM
ring-buffer pipeline cm_sdpa_opt_prefill.cm's KV_CACHE_COMPRESSION path uses (see
cm_sdpa_common.hpp's sdpa_kernel_lsc_2src_slm / sdpa_accumulate_kv_source_slm) -- replacing
the retired sdpa_accumulate_kv_source's per-lane-unamortized decode.

Correctness oracle (NOT "is int8 quantization accurate enough" -- that's not what this isolates):
1. Build a random half cache K/V.
2. Encode it to int8 bytes+scale+zp and decode straight back to half on the HOST, using
   int8_kv_ref.py's thin wrapper around the SAME quant/dequant math cm_int8_kv_decode.hpp ports
   into CM. Call this k_cache_roundtripped/v_cache_roundtripped.
3. Run the KV_CACHE_COMPRESSION!=0 kernel variant with the RAW int8 bytes + scale/zp as
   key_cache/value_cache.
4. Run the already-trusted KV_CACHE_COMPRESSION=0 (default) dualsource kernel with the
   *roundtripped* half cache (not the original cache) as key_cache/value_cache.
5. Assert (3) and (4) match closely -- this isolates "does the kernel's in-register int8 decode
   match the reference dequant formula" from ordinary int8 quantization noise, which is
   deliberately NOT what is being tested here (that's why (4) uses the roundtripped cache, not the
   original).
"""
import functools

import numpy as np
import torch

from clops import cl

import test_sdpa_prefill as base
import test_sdpa_prefill_block_mask_dualsource as ms
from int8_kv_ref import (
    encode_k_by_channel, dequant_k_by_channel, encode_by_token, dequant_by_token,
)

torch.manual_seed(44)

TRANSPOSE_ORDER_IDENTITY = base.TRANSPOSE_ORDER_IDENTITY

KV_CACHE_COMPRESSION_BY_TOKEN = 1
KV_CACHE_COMPRESSION_BY_CHANNEL = 2


@functools.cache
def build_dualsource_sparse_prefill_kernel_int8cache(num_heads, num_kv_heads, head_size,
                                                       sparse_block_size, scale_factor=None,
                                                       compression_mode=KV_CACHE_COMPRESSION_BY_CHANNEL):
    if scale_factor is None:
        scale_factor = 1.0 / (head_size ** 0.5)
    src = r'''#include "cm_sdpa_opt_prefill_dualsource.cm"'''
    opts = (
        f'-cmc -Qxcm_jit_option="-abortonspill" -Qxcm_register_file_size=256 -mCM_printregusage '
        f"-I{base.CWD} -I{base.FLASHATTN_DIR} -I{base.PAGEATTEN_DIR} "
        f"-DKERNEL_NAME=cm_sdpa_opt_prefill_dualsource_int8cache "
        f"-DCMFLA_NUM_HEADS={num_heads} -DCMFLA_NUM_KV_HEADS={num_kv_heads} "
        f"-DCMFLA_HEAD_SIZE={head_size} -DCMFLA_SCALE_FACTOR={scale_factor} "
        f"-DSPARSE_BLOCK_SIZE={sparse_block_size} -DKV_CACHE_COMPRESSION={compression_mode}"
    )
    return cl.kernels(src, opts)


def _encode_cache_to_int8(k_cache, v_cache, compression_mode):
    if compression_mode == KV_CACHE_COMPRESSION_BY_CHANNEL:
        k_u8, k_scale, k_zp = encode_k_by_channel(k_cache)
        k_rt = dequant_k_by_channel(k_u8, k_scale, k_zp)
    else:
        k_u8, k_scale, k_zp = encode_by_token(k_cache)
        k_rt = dequant_by_token(k_u8, k_scale, k_zp)
    v_u8, v_scale, v_zp = encode_by_token(v_cache)
    v_rt = dequant_by_token(v_u8, v_scale, v_zp)
    return (k_u8, k_scale, k_zp, k_rt), (v_u8, v_scale, v_zp, v_rt)


def run_prefill_sparse_dualsource_int8cache(q, k_cache_u8, k_cache_scale, k_cache_zp,
                                             v_cache_u8, v_cache_scale, v_cache_zp, k_cur, v_cur,
                                             sparse_block_size, block_mask, kv_len_cache,
                                             scale=None,
                                             compression_mode=KV_CACHE_COMPRESSION_BY_CHANNEL):
    # k_cache_u8/v_cache_u8: uint8 tensors, logical [B,Hkv,kv_len_cache,D] (IDENTITY order only --
    # this prototype's int8-cache path doesn't exercise the transpose-order axis).
    batch, num_heads, q_len, head_size = q.shape
    _, num_kv_heads, _, _ = k_cache_u8.shape
    _, _, kv_len_cur, _ = k_cur.shape

    kernel = build_dualsource_sparse_prefill_kernel_int8cache(
        num_heads, num_kv_heads, head_size, sparse_block_size, scale, compression_mode)

    t_q = cl.tensor(q.contiguous().to(torch.float16).numpy())
    t_k_cache = cl.tensor(k_cache_u8.contiguous().numpy())
    t_v_cache = cl.tensor(v_cache_u8.contiguous().numpy())
    t_k_cur = cl.tensor(k_cur.contiguous().to(torch.float16).numpy())
    t_v_cur = cl.tensor(v_cur.contiguous().to(torch.float16).numpy())
    out_shape = [batch, num_heads, q_len, head_size]
    t_out = cl.tensor(np.zeros(out_shape, dtype=np.float16))

    q_step = base.get_cm_grf_width() // 32
    wg_size = 16
    wg_seq_len = wg_size * q_step
    wg_count = (q_len + wg_seq_len - 1) // wg_seq_len
    GWS = [1, batch * num_heads, wg_count * wg_size]
    LWS = [1, 1, wg_size]

    num_q_blocks = block_mask.shape[2]
    num_k_blocks_cache = (kv_len_cache + sparse_block_size - 1) // sparse_block_size
    num_k_blocks_cur = (kv_len_cur + sparse_block_size - 1) // sparse_block_size
    assert block_mask.shape[3] == num_k_blocks_cache + num_k_blocks_cur

    t_mask = cl.tensor(block_mask.to(torch.bool).numpy())

    t_k_scale = cl.tensor(k_cache_scale.contiguous().numpy())
    t_k_zp = cl.tensor(k_cache_zp.contiguous().numpy())
    t_v_scale = cl.tensor(v_cache_scale.contiguous().numpy())
    t_v_zp = cl.tensor(v_cache_zp.contiguous().numpy())

    args = [t_q, t_k_cache, t_v_cache, t_k_cur, t_v_cur, t_out, q_len, kv_len_cache, kv_len_cur,
            t_mask, num_q_blocks, num_k_blocks_cache, num_k_blocks_cur,
            t_k_scale, t_k_zp, t_v_scale, t_v_zp]
    kernel.enqueue("cm_sdpa_opt_prefill_dualsource_int8cache", GWS, LWS, *args)

    out_phys = torch.from_numpy(t_out.numpy())
    return base.from_physical(out_phys, TRANSPOSE_ORDER_IDENTITY, batch, num_heads, q_len, head_size)


def _make_qkv(batch, num_heads, num_kv_heads, q_len, kv_len_cache, kv_len_cur, head_size, seed):
    g = torch.Generator().manual_seed(seed)
    # Reasonably-scaled values (not {-2..2} integers like the fp16-only dualsource tests) so the
    # int8 cache actually exercises a spread of magnitudes, not just a few exact values.
    q = (torch.randn([batch, num_heads, q_len, head_size], generator=g) * 1.0).to(torch.float16)
    k_cache = (torch.randn([batch, num_kv_heads, kv_len_cache, head_size], generator=g) * 1.0).to(torch.float16)
    v_cache = (torch.randn([batch, num_kv_heads, kv_len_cache, head_size], generator=g) * 0.5).to(torch.float16)
    k_cur = (torch.randn([batch, num_kv_heads, kv_len_cur, head_size], generator=g) * 1.0).to(torch.float16)
    v_cur = (torch.randn([batch, num_kv_heads, kv_len_cur, head_size], generator=g) * 0.5).to(torch.float16)
    return q, k_cache, v_cache, k_cur, v_cur


def _check_against_roundtripped_oracle(q, k_cache, v_cache, k_cur, v_cur, sparse_block_size, mask,
                                        compression_mode=KV_CACHE_COMPRESSION_BY_CHANNEL):
    kv_len_cache = k_cache.shape[2]
    (k_u8, k_scale, k_zp, k_rt), (v_u8, v_scale, v_zp, v_rt) = _encode_cache_to_int8(
        k_cache, v_cache, compression_mode)

    out_int8 = run_prefill_sparse_dualsource_int8cache(
        q, k_u8, k_scale, k_zp, v_u8, v_scale, v_zp, k_cur, v_cur, sparse_block_size, mask,
        kv_len_cache, compression_mode=compression_mode)
    ref_fp16_roundtripped = ms.run_prefill_sparse_dualsource(
        q, k_rt, v_rt, k_cur, v_cur, sparse_block_size, mask)
    base.check_close(out_int8, ref_fp16_roundtripped, atol=2e-2, rtol=2e-2)
    return out_int8, ref_fp16_roundtripped


def test_int8_cache_matches_roundtripped_all_true_mask():
    sparse_block_size = 128
    q_len = kv_len_cache = kv_len_cur = 256
    q, k_cache, v_cache, k_cur, v_cur = _make_qkv(
        batch=1, num_heads=4, num_kv_heads=4, q_len=q_len, kv_len_cache=kv_len_cache,
        kv_len_cur=kv_len_cur, head_size=64, seed=101)

    num_q_blocks = (q_len + sparse_block_size - 1) // sparse_block_size
    num_k_blocks_cache = (kv_len_cache + sparse_block_size - 1) // sparse_block_size
    num_k_blocks_cur = (kv_len_cur + sparse_block_size - 1) // sparse_block_size
    mask = torch.ones(1, 4, num_q_blocks, num_k_blocks_cache + num_k_blocks_cur, dtype=torch.bool)

    _check_against_roundtripped_oracle(q, k_cache, v_cache, k_cur, v_cur, sparse_block_size, mask)
    print("test_int8_cache_matches_roundtripped_all_true_mask PASSED")


def test_int8_cache_matches_roundtripped_random_sparse_mask():
    sparse_block_size = 128
    q_len, kv_len_cache, kv_len_cur = 256, 384, 128
    q, k_cache, v_cache, k_cur, v_cur = _make_qkv(
        batch=1, num_heads=4, num_kv_heads=4, q_len=q_len, kv_len_cache=kv_len_cache,
        kv_len_cur=kv_len_cur, head_size=64, seed=102)

    num_q_blocks = (q_len + sparse_block_size - 1) // sparse_block_size
    num_k_blocks_cache = (kv_len_cache + sparse_block_size - 1) // sparse_block_size
    num_k_blocks_cur = (kv_len_cur + sparse_block_size - 1) // sparse_block_size
    g = torch.Generator().manual_seed(103)
    mask = torch.randint(0, 2, [1, 4, num_q_blocks, num_k_blocks_cache + num_k_blocks_cur],
                          generator=g).to(torch.bool)
    mask[:, :, :, 0] = True

    _check_against_roundtripped_oracle(q, k_cache, v_cache, k_cur, v_cur, sparse_block_size, mask)
    print("test_int8_cache_matches_roundtripped_random_sparse_mask PASSED")


def test_int8_cache_realistic_kv_ratio():
    # FlashVSR's real steady-state shape: current = 1 window-group, cache = 3 cached history
    # groups (kv_ratio=3.0) -- mirrors
    # test_sdpa_prefill_block_mask_dualsource.test_realistic_kv_ratio, int8-cache variant.
    sparse_block_size = 128
    q_len = kv_len_cur = sparse_block_size
    kv_len_cache = 3 * sparse_block_size
    q, k_cache, v_cache, k_cur, v_cur = _make_qkv(
        batch=1, num_heads=4, num_kv_heads=4, q_len=q_len, kv_len_cache=kv_len_cache,
        kv_len_cur=kv_len_cur, head_size=64, seed=104)

    num_q_blocks = 1
    num_k_blocks_cache, num_k_blocks_cur = 3, 1
    g = torch.Generator().manual_seed(105)
    mask = torch.randint(0, 2, [1, 4, num_q_blocks, num_k_blocks_cache + num_k_blocks_cur],
                          generator=g).to(torch.bool)
    mask[:, :, :, 0] = True

    _check_against_roundtripped_oracle(q, k_cache, v_cache, k_cur, v_cur, sparse_block_size, mask)
    print("test_int8_cache_realistic_kv_ratio PASSED")


def test_int8_cache_gqa():
    sparse_block_size = 128
    q_len, kv_len_cache, kv_len_cur = 256, 256, 128
    q, k_cache, v_cache, k_cur, v_cur = _make_qkv(
        batch=1, num_heads=8, num_kv_heads=2, q_len=q_len, kv_len_cache=kv_len_cache,
        kv_len_cur=kv_len_cur, head_size=64, seed=106)

    num_q_blocks = (q_len + sparse_block_size - 1) // sparse_block_size
    num_k_blocks_cache = (kv_len_cache + sparse_block_size - 1) // sparse_block_size
    num_k_blocks_cur = (kv_len_cur + sparse_block_size - 1) // sparse_block_size
    mask = torch.ones(1, 8, num_q_blocks, num_k_blocks_cache + num_k_blocks_cur, dtype=torch.bool)

    _check_against_roundtripped_oracle(q, k_cache, v_cache, k_cur, v_cur, sparse_block_size, mask)
    print("test_int8_cache_gqa PASSED")


def test_int8_cache_head_size_128():
    sparse_block_size = 128
    q_len, kv_len_cache, kv_len_cur = 256, 256, 128
    q, k_cache, v_cache, k_cur, v_cur = _make_qkv(
        batch=1, num_heads=4, num_kv_heads=4, q_len=q_len, kv_len_cache=kv_len_cache,
        kv_len_cur=kv_len_cur, head_size=128, seed=107)

    num_q_blocks = (q_len + sparse_block_size - 1) // sparse_block_size
    num_k_blocks_cache = (kv_len_cache + sparse_block_size - 1) // sparse_block_size
    num_k_blocks_cur = (kv_len_cur + sparse_block_size - 1) // sparse_block_size
    mask = torch.ones(1, 4, num_q_blocks, num_k_blocks_cache + num_k_blocks_cur, dtype=torch.bool)

    _check_against_roundtripped_oracle(q, k_cache, v_cache, k_cur, v_cur, sparse_block_size, mask)
    print("test_int8_cache_head_size_128 PASSED")


def test_int8_cache_by_token_mode():
    # KV_CACHE_COMPRESSION==1 (BY_TOKEN for K too, not just V) -- coverage for
    # sdpa_accumulate_kv_source_slm's k_by_channel=false path in the dual-source context.
    sparse_block_size = 128
    q_len, kv_len_cache, kv_len_cur = 256, 256, 128
    q, k_cache, v_cache, k_cur, v_cur = _make_qkv(
        batch=1, num_heads=4, num_kv_heads=4, q_len=q_len, kv_len_cache=kv_len_cache,
        kv_len_cur=kv_len_cur, head_size=64, seed=108)

    num_q_blocks = (q_len + sparse_block_size - 1) // sparse_block_size
    num_k_blocks_cache = (kv_len_cache + sparse_block_size - 1) // sparse_block_size
    num_k_blocks_cur = (kv_len_cur + sparse_block_size - 1) // sparse_block_size
    mask = torch.ones(1, 4, num_q_blocks, num_k_blocks_cache + num_k_blocks_cur, dtype=torch.bool)

    _check_against_roundtripped_oracle(q, k_cache, v_cache, k_cur, v_cur, sparse_block_size, mask,
                                        compression_mode=KV_CACHE_COMPRESSION_BY_TOKEN)
    print("test_int8_cache_by_token_mode PASSED")


def test_int8_cache_empty_cache_chunk0():
    # chunk0/warmup: cache not yet populated (kv_len_cache=0). sdpa_accumulate_kv_source_slm's
    # kv_stop<=0 guard (after the entry barrier) is checked before any b2dK/b2dV descriptor is
    # built, so this must degrade cleanly to "current only" regardless of
    # KV_CACHE_COMPRESSION -- mirrors
    # test_sdpa_prefill_block_mask_dualsource.test_empty_cache_chunk0, int8-cache variant.
    sparse_block_size = 128
    q_len = kv_len_cur = 256
    q, k_cache_placeholder, v_cache_placeholder, k_cur, v_cur = _make_qkv(
        batch=1, num_heads=4, num_kv_heads=4, q_len=q_len, kv_len_cache=16,
        kv_len_cur=kv_len_cur, head_size=64, seed=109)

    num_q_blocks = (q_len + sparse_block_size - 1) // sparse_block_size
    num_k_blocks_cache, num_k_blocks_cur = 0, (kv_len_cur + sparse_block_size - 1) // sparse_block_size
    mask = torch.ones(1, 4, num_q_blocks, num_k_blocks_cache + num_k_blocks_cur, dtype=torch.bool)

    (k_u8, k_scale, k_zp, _), (v_u8, v_scale, v_zp, _) = _encode_cache_to_int8(
        k_cache_placeholder, v_cache_placeholder, KV_CACHE_COMPRESSION_BY_CHANNEL)

    out = run_prefill_sparse_dualsource_int8cache(
        q, k_u8, k_scale, k_zp, v_u8, v_scale, v_zp, k_cur, v_cur, sparse_block_size, mask,
        kv_len_cache=0)
    ref = ms.block_mask_mod.run_prefill_sparse(q, k_cur, v_cur, is_causal=False,
                                                sparse_block_size=sparse_block_size, block_mask=mask)
    assert torch.equal(out, ref), "empty int8 cache must degrade to current-only, bit-identical to the single-source kernel"
    print("test_int8_cache_empty_cache_chunk0 PASSED")


if __name__ == "__main__":
    cl.profiling(True)
    test_int8_cache_matches_roundtripped_all_true_mask()
    test_int8_cache_matches_roundtripped_random_sparse_mask()
    test_int8_cache_realistic_kv_ratio()
    test_int8_cache_gqa()
    test_int8_cache_head_size_128()
    test_int8_cache_by_token_mode()
    test_int8_cache_empty_cache_chunk0()
    print("ALL PASSED")
