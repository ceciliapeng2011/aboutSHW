"""
fp8e4m3 ring K/V correctness for cm_sdpa_opt_prefill_multisource.cm's KV_RING_FP8E4M3 path: the
attention kernel decodes fp8e4m3-packed ring bytes to half in-register (right before the QK^T /
P@V DPAS calls, see cm_sdpa_common.hpp's hf8_e4m3_decode_to_f16 / sdpa_accumulate_kv_source<...,
KVElemT=uchar>) instead of relying on a separate reorder/dequant pass over the whole ring buffer.

Correctness oracle (NOT "is fp8 quantization accurate enough" -- that's not what this isolates):
1. Build a random half ring K/V.
2. Encode it to fp8e4m3 bytes and decode straight back to half on the HOST, using
   hf8_e4m3_ref.py's bit-exact port of the same algorithm cm_sdpa_common.hpp ports into CM. Call
   this k_ring_roundtripped_f16/v_ring_roundtripped_f16.
3. Run the KV_RING_FP8E4M3=1 kernel variant with the RAW fp8 bytes as key_ring/value_ring.
4. Run the already-trusted KV_RING_FP8E4M3=0 (default) multisource kernel with the
   *roundtripped* half ring (not the original ring) as key_ring/value_ring.
5. Assert (3) and (4) match closely -- this isolates "does the kernel's in-register fp8 decode
   match the reference bit-decode formula" from ordinary fp8 quantization noise, which is
   deliberately NOT what's being tested here (that's why (4) uses the roundtripped ring, not the
   original).

See test_hf8_e4m3_decode.py for the decode-function-only unit test (isolated from VNNI-repacking
layout bugs in the V path).
"""
import functools

import numpy as np
import torch

from clops import cl

import test_sdpa_prefill as base
import test_sdpa_prefill_block_mask_multisource as ms
from hf8_e4m3_ref import encode_f16_to_hf8, decode_hf8_to_f16, build_hf8_decode_lut_f16

torch.manual_seed(44)

TRANSPOSE_ORDER_IDENTITY = base.TRANSPOSE_ORDER_IDENTITY


@functools.cache
def build_multisource_sparse_prefill_kernel_fp8ring(num_heads, num_kv_heads, head_size,
                                                      sparse_block_size, scale_factor=None):
    if scale_factor is None:
        scale_factor = 1.0 / (head_size ** 0.5)
    src = r'''#include "cm_sdpa_opt_prefill_multisource.cm"'''
    opts = (
        f'-cmc -Qxcm_jit_option="-abortonspill" -Qxcm_register_file_size=256 -mCM_printregusage '
        f"-I{base.CWD} -I{base.FLASHATTN_DIR} -I{base.PAGEATTEN_DIR} "
        f"-DKERNEL_NAME=cm_sdpa_opt_prefill_multisource_fp8ring "
        f"-DCMFLA_NUM_HEADS={num_heads} -DCMFLA_NUM_KV_HEADS={num_kv_heads} "
        f"-DCMFLA_HEAD_SIZE={head_size} -DCMFLA_SCALE_FACTOR={scale_factor} "
        f"-DSPARSE_BLOCK_SIZE={sparse_block_size} -DKV_RING_FP8E4M3=1"
    )
    return cl.kernels(src, opts)


def _encode_ring_to_hf8_bytes(t_half: torch.Tensor):
    """t_half: logical [B,Hkv,S,D] half tensor (IDENTITY order). Returns (uint8 bytes tensor
    with the same shape, roundtripped-through-fp8 half tensor)."""
    np_half = t_half.contiguous().numpy()
    hf8_bytes = encode_f16_to_hf8(np_half)
    roundtripped = decode_hf8_to_f16(hf8_bytes)
    return torch.from_numpy(hf8_bytes.copy()), torch.from_numpy(roundtripped.copy())


def run_prefill_sparse_multisource_fp8ring(q, k_ring_hf8_bytes, v_ring_hf8_bytes, k_cur, v_cur,
                                            sparse_block_size, block_mask, kv_len_ring,
                                            scale=None):
    # k_ring_hf8_bytes/v_ring_hf8_bytes: uint8 tensors, logical [B,Hkv,kv_len_ring,D] (IDENTITY
    # order only -- this prototype's fp8 ring path doesn't exercise the transpose-order axis).
    batch, num_heads, q_len, head_size = q.shape
    _, num_kv_heads, _, _ = k_ring_hf8_bytes.shape
    _, _, kv_len_cur, _ = k_cur.shape

    kernel = build_multisource_sparse_prefill_kernel_fp8ring(num_heads, num_kv_heads, head_size,
                                                              sparse_block_size, scale)

    t_q = cl.tensor(q.contiguous().to(torch.float16).numpy())
    t_k_ring = cl.tensor(k_ring_hf8_bytes.contiguous().numpy())
    t_v_ring = cl.tensor(v_ring_hf8_bytes.contiguous().numpy())
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
    num_k_blocks_ring = (kv_len_ring + sparse_block_size - 1) // sparse_block_size
    num_k_blocks_cur = (kv_len_cur + sparse_block_size - 1) // sparse_block_size
    assert block_mask.shape[3] == num_k_blocks_ring + num_k_blocks_cur

    t_mask = cl.tensor(block_mask.to(torch.bool).numpy())

    # KV_RING_FP8E4M3 always needs the 256-entry half LUT (see
    # hf8_e4m3_ref.build_hf8_decode_lut_f16) -- matching cm_sdpa_opt_prefill_multisource.cm's
    # `#if KV_RING_FP8E4M3` hf8_lut param, which is always last in the signature.
    t_lut = cl.tensor(build_hf8_decode_lut_f16())

    args = [t_q, t_k_ring, t_v_ring, t_k_cur, t_v_cur, t_out, q_len, kv_len_ring, kv_len_cur,
            t_mask, num_q_blocks, num_k_blocks_ring, num_k_blocks_cur, t_lut]
    kernel.enqueue("cm_sdpa_opt_prefill_multisource_fp8ring", GWS, LWS, *args)

    out_phys = torch.from_numpy(t_out.numpy())
    return base.from_physical(out_phys, TRANSPOSE_ORDER_IDENTITY, batch, num_heads, q_len, head_size)


def _make_qkv(batch, num_heads, num_kv_heads, q_len, kv_len_ring, kv_len_cur, head_size, seed):
    g = torch.Generator().manual_seed(seed)
    # Reasonably-scaled values (not {-2..2} integers like the fp16-only multisource tests) so the
    # fp8e4m3 ring actually exercises a spread of exponents/denormals, not just a few exact values.
    q = (torch.randn([batch, num_heads, q_len, head_size], generator=g) * 1.0).to(torch.float16)
    k_ring = (torch.randn([batch, num_kv_heads, kv_len_ring, head_size], generator=g) * 1.0).to(torch.float16)
    v_ring = (torch.randn([batch, num_kv_heads, kv_len_ring, head_size], generator=g) * 0.5).to(torch.float16)
    k_cur = (torch.randn([batch, num_kv_heads, kv_len_cur, head_size], generator=g) * 1.0).to(torch.float16)
    v_cur = (torch.randn([batch, num_kv_heads, kv_len_cur, head_size], generator=g) * 0.5).to(torch.float16)
    return q, k_ring, v_ring, k_cur, v_cur


def _check_against_roundtripped_oracle(q, k_ring, v_ring, k_cur, v_cur, sparse_block_size, mask):
    kv_len_ring = k_ring.shape[2]
    k_ring_hf8, k_ring_rt = _encode_ring_to_hf8_bytes(k_ring)
    v_ring_hf8, v_ring_rt = _encode_ring_to_hf8_bytes(v_ring)

    out_fp8 = run_prefill_sparse_multisource_fp8ring(
        q, k_ring_hf8, v_ring_hf8, k_cur, v_cur, sparse_block_size, mask, kv_len_ring)
    ref_fp16_roundtripped = ms.run_prefill_sparse_multisource(
        q, k_ring_rt, v_ring_rt, k_cur, v_cur, sparse_block_size, mask)
    base.check_close(out_fp8, ref_fp16_roundtripped, atol=2e-2, rtol=2e-2)
    return out_fp8, ref_fp16_roundtripped


def test_fp8_ring_matches_roundtripped_all_true_mask():
    sparse_block_size = 128
    q_len = kv_len_ring = kv_len_cur = 256
    q, k_ring, v_ring, k_cur, v_cur = _make_qkv(
        batch=1, num_heads=4, num_kv_heads=4, q_len=q_len, kv_len_ring=kv_len_ring,
        kv_len_cur=kv_len_cur, head_size=64, seed=101)

    num_q_blocks = (q_len + sparse_block_size - 1) // sparse_block_size
    num_k_blocks_ring = (kv_len_ring + sparse_block_size - 1) // sparse_block_size
    num_k_blocks_cur = (kv_len_cur + sparse_block_size - 1) // sparse_block_size
    mask = torch.ones(1, 4, num_q_blocks, num_k_blocks_ring + num_k_blocks_cur, dtype=torch.bool)

    _check_against_roundtripped_oracle(q, k_ring, v_ring, k_cur, v_cur, sparse_block_size, mask)
    print("test_fp8_ring_matches_roundtripped_all_true_mask PASSED")


def test_fp8_ring_matches_roundtripped_random_sparse_mask():
    sparse_block_size = 128
    q_len, kv_len_ring, kv_len_cur = 256, 384, 128
    q, k_ring, v_ring, k_cur, v_cur = _make_qkv(
        batch=1, num_heads=4, num_kv_heads=4, q_len=q_len, kv_len_ring=kv_len_ring,
        kv_len_cur=kv_len_cur, head_size=64, seed=102)

    num_q_blocks = (q_len + sparse_block_size - 1) // sparse_block_size
    num_k_blocks_ring = (kv_len_ring + sparse_block_size - 1) // sparse_block_size
    num_k_blocks_cur = (kv_len_cur + sparse_block_size - 1) // sparse_block_size
    g = torch.Generator().manual_seed(103)
    mask = torch.randint(0, 2, [1, 4, num_q_blocks, num_k_blocks_ring + num_k_blocks_cur],
                          generator=g).to(torch.bool)
    mask[:, :, :, 0] = True

    _check_against_roundtripped_oracle(q, k_ring, v_ring, k_cur, v_cur, sparse_block_size, mask)
    print("test_fp8_ring_matches_roundtripped_random_sparse_mask PASSED")


def test_fp8_ring_realistic_kv_ratio():
    # FlashVSR's real steady-state shape: current = 1 window-group, ring = 3 cached history
    # groups (kv_ratio=3.0) -- mirrors
    # test_sdpa_prefill_block_mask_multisource.test_realistic_kv_ratio, fp8-ring variant.
    sparse_block_size = 128
    q_len = kv_len_cur = sparse_block_size
    kv_len_ring = 3 * sparse_block_size
    q, k_ring, v_ring, k_cur, v_cur = _make_qkv(
        batch=1, num_heads=4, num_kv_heads=4, q_len=q_len, kv_len_ring=kv_len_ring,
        kv_len_cur=kv_len_cur, head_size=64, seed=104)

    num_q_blocks = 1
    num_k_blocks_ring, num_k_blocks_cur = 3, 1
    g = torch.Generator().manual_seed(105)
    mask = torch.randint(0, 2, [1, 4, num_q_blocks, num_k_blocks_ring + num_k_blocks_cur],
                          generator=g).to(torch.bool)
    mask[:, :, :, 0] = True

    _check_against_roundtripped_oracle(q, k_ring, v_ring, k_cur, v_cur, sparse_block_size, mask)
    print("test_fp8_ring_realistic_kv_ratio PASSED")


def test_fp8_ring_gqa():
    sparse_block_size = 128
    q_len, kv_len_ring, kv_len_cur = 256, 256, 128
    q, k_ring, v_ring, k_cur, v_cur = _make_qkv(
        batch=1, num_heads=8, num_kv_heads=2, q_len=q_len, kv_len_ring=kv_len_ring,
        kv_len_cur=kv_len_cur, head_size=64, seed=106)

    num_q_blocks = (q_len + sparse_block_size - 1) // sparse_block_size
    num_k_blocks_ring = (kv_len_ring + sparse_block_size - 1) // sparse_block_size
    num_k_blocks_cur = (kv_len_cur + sparse_block_size - 1) // sparse_block_size
    mask = torch.ones(1, 8, num_q_blocks, num_k_blocks_ring + num_k_blocks_cur, dtype=torch.bool)

    _check_against_roundtripped_oracle(q, k_ring, v_ring, k_cur, v_cur, sparse_block_size, mask)
    print("test_fp8_ring_gqa PASSED")


def test_fp8_ring_head_size_128():
    sparse_block_size = 128
    q_len, kv_len_ring, kv_len_cur = 256, 256, 128
    q, k_ring, v_ring, k_cur, v_cur = _make_qkv(
        batch=1, num_heads=4, num_kv_heads=4, q_len=q_len, kv_len_ring=kv_len_ring,
        kv_len_cur=kv_len_cur, head_size=128, seed=107)

    num_q_blocks = (q_len + sparse_block_size - 1) // sparse_block_size
    num_k_blocks_ring = (kv_len_ring + sparse_block_size - 1) // sparse_block_size
    num_k_blocks_cur = (kv_len_cur + sparse_block_size - 1) // sparse_block_size
    mask = torch.ones(1, 4, num_q_blocks, num_k_blocks_ring + num_k_blocks_cur, dtype=torch.bool)

    _check_against_roundtripped_oracle(q, k_ring, v_ring, k_cur, v_cur, sparse_block_size, mask)
    print("test_fp8_ring_head_size_128 PASSED")


def test_fp8_ring_empty_ring_chunk0():
    # chunk0/warmup: ring not yet populated (kv_len_ring=0). sdpa_accumulate_kv_source's
    # kv_stop<=0 guard is checked before any b2dK/b2dV descriptor (or hf8 decode) is built, so
    # this must degrade cleanly to "current only" regardless of KV_RING_FP8E4M3 -- mirrors
    # test_sdpa_prefill_block_mask_multisource.test_empty_ring_chunk0, fp8-ring variant.
    sparse_block_size = 128
    q_len = kv_len_cur = 256
    q, k_ring_placeholder, v_ring_placeholder, k_cur, v_cur = _make_qkv(
        batch=1, num_heads=4, num_kv_heads=4, q_len=q_len, kv_len_ring=1,
        kv_len_cur=kv_len_cur, head_size=64, seed=108)

    num_q_blocks = (q_len + sparse_block_size - 1) // sparse_block_size
    num_k_blocks_ring, num_k_blocks_cur = 0, (kv_len_cur + sparse_block_size - 1) // sparse_block_size
    mask = torch.ones(1, 4, num_q_blocks, num_k_blocks_ring + num_k_blocks_cur, dtype=torch.bool)

    k_ring_hf8, _ = _encode_ring_to_hf8_bytes(k_ring_placeholder)
    v_ring_hf8, _ = _encode_ring_to_hf8_bytes(v_ring_placeholder)

    out = run_prefill_sparse_multisource_fp8ring(q, k_ring_hf8, v_ring_hf8, k_cur, v_cur,
                                                  sparse_block_size, mask, kv_len_ring=0)
    ref = ms.block_mask_mod.run_prefill_sparse(q, k_cur, v_cur, is_causal=False,
                                                sparse_block_size=sparse_block_size, block_mask=mask)
    assert torch.equal(out, ref), "empty fp8 ring must degrade to current-only, bit-identical to the single-source kernel"
    print("test_fp8_ring_empty_ring_chunk0 PASSED")


if __name__ == "__main__":
    cl.profiling(True)
    test_fp8_ring_matches_roundtripped_all_true_mask()
    test_fp8_ring_matches_roundtripped_random_sparse_mask()
    test_fp8_ring_realistic_kv_ratio()
    test_fp8_ring_gqa()
    test_fp8_ring_head_size_128()
    test_fp8_ring_empty_ring_chunk0()
    print("ALL PASSED")
