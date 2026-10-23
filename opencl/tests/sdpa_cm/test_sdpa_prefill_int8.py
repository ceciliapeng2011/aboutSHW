"""
int8 (affine uint8, pageatten-style) K/V correctness for cm_sdpa_opt_prefill.cm's
KV_CACHE_COMPRESSION path: the single-source, general-purpose SDPA prefill kernel (supports causal
masking, a dense additive attn_mask, and a sparse block-table mask, all in one kernel -- the "real"
OCL sdpa_opt/BlockSparseAttention v1 ABI mirror) decodes int8-packed K/V bytes to half once per
work-group, staged through SLM (see cm_sdpa_common.hpp's sdpa_kernel_lsc), via a flat affine
`(raw-zp)*scale` dequant, instead of relying on a separate reorder/dequant pass over the whole K/V
buffer. KV_CACHE_COMPRESSION is causal-only with no dense attn_mask support
-- matches sdpa_kernel_lsc's own scope.

K is quantized BY_CHANNEL (KV_CACHE_COMPRESSION=2, one scale/zp pair per 16-token sub-block per
head_size channel) or BY_TOKEN (=1, one pair per token); V is always BY_TOKEN regardless of mode --
matches opencl/tests/pageatten/kv_cache_quant_utils.py's own K/V asymmetry exactly (int8_kv_ref.py
is the thin host-side wrapper around that module's quant_per_channel_parts/quant_per_token_parts).

This is the single-source counterpart of test_sdpa_prefill_int8_cache.py's KV_CACHE_COMPRESSION
test for the narrower dual-source (cache+current) kernel -- same methodology, applied to the more
general kernel, plus two axes the cache test doesn't cover: causal masking and K/V TRANSPOSE_ORDER
(IDENTITY vs SWAPPED), since this kernel (unlike the dual-source one) already supports both.

Correctness oracle (NOT "is int8 quantization accurate enough" -- that's not what this isolates):
1. Build a random half K/V.
2. Encode it to int8 bytes+scale+zp and decode straight back to half on the HOST, using
   int8_kv_ref.py's thin wrapper around the SAME quant/dequant math cm_int8_kv_decode.hpp ports
   into CM. Call this k_roundtripped/v_roundtripped.
3. Run the KV_CACHE_COMPRESSION!=0 kernel variant with the RAW int8 bytes + scale/zp as key/value.
4. Run the already-trusted KV_CACHE_COMPRESSION=0 (default) kernel with the *roundtripped* half K/V
   (not the original K/V).
5. Assert (3) and (4) match -- this isolates "does the kernel's in-register int8 decode match the
   reference dequant formula" from ordinary int8 quantization noise, which is deliberately NOT what
   is being tested here (that's why (4) uses the roundtripped K/V, not the original).
"""
import functools

import torch

import numpy as np
from clops import cl

import test_sdpa_prefill as base
import test_sdpa_prefill_block_mask as bm
from int8_kv_ref import encode_k_by_channel, dequant_k_by_channel, encode_by_token, dequant_by_token
# test_sdpa_prefill_block_mask's import above already put pageatten/ on sys.path (see
# bench_sdpa.py's identical comment) -- generate_block_mask_with_ratio builds a realistic,
# non-trivial (not all-True/all-False) causal block-sparse pattern per head, used below by the
# sparse multi-work-group tests.
from generate_block_mask import generate_block_mask_with_ratio

torch.manual_seed(55)

TRANSPOSE_ORDER_IDENTITY = base.TRANSPOSE_ORDER_IDENTITY
TRANSPOSE_ORDER_SWAPPED = base.TRANSPOSE_ORDER_SWAPPED

KV_CACHE_COMPRESSION_BY_TOKEN = 1
KV_CACHE_COMPRESSION_BY_CHANNEL = 2


@functools.cache
def build_prefill_kernel_int8(num_heads, num_kv_heads, head_size, is_causal,
                               q_order, k_order, v_order, o_order,
                               sparse_block_size=1, scale_factor=None,
                               compression_mode=KV_CACHE_COMPRESSION_BY_CHANNEL):
    # KV_CACHE_COMPRESSION=2 (default here) calls cm_sdpa_common.hpp's sdpa_kernel_lsc with
    # k_by_channel=true -- decode-once-per-work-group, staged through SLM, affine
    # (raw-zp)*scale dequant (see cm_int8_kv_decode.hpp) -- instead of sdpa_kernel_lsc_prefetch's
    # per-work-item decode. No attn_mask support (see sdpa_kernel_lsc's doc comment).
    # Block-sparsity IS supported at sparse_block_size in (128, 256) -- see sdpa_kernel_lsc's
    # doc comment for the two-level skip-LOAD/skip-COMPUTE mask split 128 needs on top of 256's
    # simpler single-mask case. Any other value is a hard compile error in cm_sdpa_opt_prefill.cm's
    # static_assert, not silently wrong.
    if scale_factor is None:
        scale_factor = 1.0 / (head_size ** 0.5)
    assert sparse_block_size in (1, 128, 256), (
        "KV_CACHE_COMPRESSION's block-sparsity support requires sparse_block_size in (1, 128, 256)")
    src = r'''#include "cm_sdpa_opt_prefill.cm"'''
    opts = (
        f'-cmc -Qxcm_jit_option="-abortonspill" -Qxcm_register_file_size=256 -mCM_printregusage '
        f"-I{base.CWD} -I{base.FLASHATTN_DIR} -I{base.PAGEATTEN_DIR} "
        f"-DKERNEL_NAME=cm_sdpa_opt_prefill_int8 "
        f"-DCMFLA_NUM_HEADS={num_heads} -DCMFLA_NUM_KV_HEADS={num_kv_heads} "
        f"-DCMFLA_HEAD_SIZE={head_size} -DCMFLA_SCALE_FACTOR={scale_factor} "
        f"-DCMFLA_IS_CAUSAL={int(is_causal)} "
        f"-DQ_TRANSPOSE_ORDER={q_order} -DK_TRANSPOSE_ORDER={k_order} "
        f"-DV_TRANSPOSE_ORDER={v_order} -DOUTPUT_TRANSPOSE_ORDER={o_order} "
        f"-DHAS_ATTN_MASK_INPUT=0 -DHAS_SCALE_INPUT=0 "
        f"-DSPARSE_BLOCK_SIZE={sparse_block_size} -DKV_CACHE_COMPRESSION={compression_mode}"
    )
    return cl.kernels(src, opts)


def _encode_kv_to_int8(k_logical, v_logical, compression_mode):
    """k_logical/v_logical: logical [B,Hkv,S,D] half tensors. Returns a dict of device-ready
    uint8/half numpy-backing tensors plus the roundtripped-through-int8 half K/V (the oracle's
    "already-decoded" reference input)."""
    if compression_mode == KV_CACHE_COMPRESSION_BY_CHANNEL:
        k_u8, k_scale, k_zp = encode_k_by_channel(k_logical)
        k_rt = dequant_k_by_channel(k_u8, k_scale, k_zp)
    else:
        k_u8, k_scale, k_zp = encode_by_token(k_logical)
        k_rt = dequant_by_token(k_u8, k_scale, k_zp)
    v_u8, v_scale, v_zp = encode_by_token(v_logical)
    v_rt = dequant_by_token(v_u8, v_scale, v_zp)
    return (k_u8, k_scale, k_zp, k_rt), (v_u8, v_scale, v_zp, v_rt)


def run_prefill_int8(q, k_u8, k_scale, k_zp, v_u8, v_scale, v_zp, is_causal,
                      q_order=TRANSPOSE_ORDER_IDENTITY, k_order=TRANSPOSE_ORDER_IDENTITY,
                      v_order=TRANSPOSE_ORDER_IDENTITY, o_order=TRANSPOSE_ORDER_IDENTITY,
                      sparse_block_size=1, block_mask=None,
                      compression_mode=KV_CACHE_COMPRESSION_BY_CHANNEL):
    # k_u8/v_u8: uint8 tensors, logical [B,Hkv,kv_len,D] (same logical shape convention as k/v
    # half tensors elsewhere in this module). TRANSPOSE_ORDER only ever applies to the raw data --
    # scale/zp travel as their own flat, order-independent tensors (see int8_kv_ref.py's layout).
    batch, num_heads, q_len, head_size = q.shape
    _, num_kv_heads, kv_len, _ = k_u8.shape

    kernel = build_prefill_kernel_int8(num_heads, num_kv_heads, head_size, is_causal,
                                        q_order, k_order, v_order, o_order, sparse_block_size,
                                        compression_mode=compression_mode)

    q_phys = base.to_physical(q, q_order)
    k_phys = base.to_physical(k_u8, k_order)
    v_phys = base.to_physical(v_u8, v_order)

    t_q = cl.tensor(q_phys.to(torch.float16).numpy())
    t_k = cl.tensor(k_phys.contiguous().numpy())
    t_v = cl.tensor(v_phys.contiguous().numpy())
    out_shape = ([batch, q_len, num_heads, head_size] if o_order == TRANSPOSE_ORDER_SWAPPED
                 else [batch, num_heads, q_len, head_size])
    t_out = cl.tensor(np.zeros(out_shape, dtype=np.float16))

    q_step = base.get_cm_grf_width() // 32
    wg_size = 16
    wg_seq_len = wg_size * q_step
    wg_count = (q_len + wg_seq_len - 1) // wg_seq_len
    GWS = [1, batch * num_heads, wg_count * wg_size]
    LWS = [1, 1, wg_size]

    args = [t_q, t_k, t_v, t_out, q_len, kv_len]
    if sparse_block_size > 1:
        assert block_mask is not None, "sparse_block_size>1 requires a block_mask"
        num_q_blocks, num_k_blocks = block_mask.shape[2], block_mask.shape[3]
        t_mask = cl.tensor(block_mask.to(torch.bool).numpy())
        args += [t_mask, num_q_blocks, num_k_blocks]
    # KV_CACHE_COMPRESSION always needs scale/zp -- matching cm_sdpa_opt_prefill.cm's
    # `#if KV_CACHE_COMPRESSION` key_scale/key_zp/value_scale/value_zp params, always last.
    t_k_scale = cl.tensor(k_scale.contiguous().numpy())
    t_k_zp = cl.tensor(k_zp.contiguous().numpy())
    t_v_scale = cl.tensor(v_scale.contiguous().numpy())
    t_v_zp = cl.tensor(v_zp.contiguous().numpy())
    args += [t_k_scale, t_k_zp, t_v_scale, t_v_zp]

    kernel.enqueue("cm_sdpa_opt_prefill_int8", GWS, LWS, *args)

    out_phys = torch.from_numpy(t_out.numpy())
    return base.from_physical(out_phys, o_order, batch, num_heads, q_len, head_size)


def _make_qkv(batch, num_heads, num_kv_heads, q_len, kv_len, head_size, seed):
    g = torch.Generator().manual_seed(seed)
    # Reasonably-scaled values (not {-2..2} integers like the fp16-only tests in test_sdpa_prefill)
    # so the int8 K/V actually exercises a spread of magnitudes, not just a few exact values.
    q = (torch.randn([batch, num_heads, q_len, head_size], generator=g) * 1.0).to(torch.float16)
    k = (torch.randn([batch, num_kv_heads, kv_len, head_size], generator=g) * 1.0).to(torch.float16)
    v = (torch.randn([batch, num_kv_heads, kv_len, head_size], generator=g) * 0.5).to(torch.float16)
    return q, k, v


def _check_against_roundtripped_oracle(q, k, v, is_causal,
                                        q_order=TRANSPOSE_ORDER_IDENTITY, k_order=TRANSPOSE_ORDER_IDENTITY,
                                        v_order=TRANSPOSE_ORDER_IDENTITY, o_order=TRANSPOSE_ORDER_IDENTITY,
                                        sparse_block_size=1, mask=None,
                                        compression_mode=KV_CACHE_COMPRESSION_BY_CHANNEL):
    (k_u8, k_scale, k_zp, k_rt), (v_u8, v_scale, v_zp, v_rt) = _encode_kv_to_int8(k, v, compression_mode)

    out_int8 = run_prefill_int8(q, k_u8, k_scale, k_zp, v_u8, v_scale, v_zp, is_causal,
                                 q_order, k_order, v_order, o_order,
                                 sparse_block_size=sparse_block_size, block_mask=mask,
                                 compression_mode=compression_mode)
    if sparse_block_size > 1:
        ref = bm.run_prefill_sparse(q, k_rt, v_rt, is_causal, sparse_block_size, mask,
                                     q_order, k_order, v_order, o_order)
    else:
        ref = base.run_prefill(q, k_rt, v_rt, is_causal, q_order, k_order, v_order, o_order)
    base.check_close(out_int8, ref, atol=2e-2, rtol=2e-2)
    return out_int8, ref


# ---- dense (sparse_block_size=1) full-kernel tests ----
def test_int8_matches_fp16_noncausal_dense_identity():
    q, k, v = _make_qkv(batch=1, num_heads=4, num_kv_heads=4, q_len=256, kv_len=256,
                         head_size=64, seed=216)
    _check_against_roundtripped_oracle(q, k, v, is_causal=False)
    print("test_int8_matches_fp16_noncausal_dense_identity PASSED")


def test_int8_matches_fp16_causal_dense_identity():
    q, k, v = _make_qkv(batch=1, num_heads=4, num_kv_heads=4, q_len=256, kv_len=256,
                         head_size=64, seed=217)
    _check_against_roundtripped_oracle(q, k, v, is_causal=True)
    print("test_int8_matches_fp16_causal_dense_identity PASSED")


def test_int8_matches_fp16_causal_dense_swapped():
    q, k, v = _make_qkv(batch=1, num_heads=4, num_kv_heads=4, q_len=256, kv_len=256,
                         head_size=64, seed=218)
    _check_against_roundtripped_oracle(q, k, v, is_causal=True,
                                        q_order=TRANSPOSE_ORDER_SWAPPED, k_order=TRANSPOSE_ORDER_SWAPPED,
                                        v_order=TRANSPOSE_ORDER_SWAPPED, o_order=TRANSPOSE_ORDER_SWAPPED)
    print("test_int8_matches_fp16_causal_dense_swapped PASSED")


def test_int8_matches_fp16_head_size_128():
    q, k, v = _make_qkv(batch=1, num_heads=4, num_kv_heads=4, q_len=256, kv_len=256,
                         head_size=128, seed=219)
    _check_against_roundtripped_oracle(q, k, v, is_causal=True)
    print("test_int8_matches_fp16_head_size_128 PASSED")


def test_int8_matches_fp16_gqa():
    q, k, v = _make_qkv(batch=1, num_heads=8, num_kv_heads=2, q_len=256, kv_len=256,
                         head_size=64, seed=220)
    _check_against_roundtripped_oracle(q, k, v, is_causal=True)
    print("test_int8_matches_fp16_gqa PASSED")


def test_int8_matches_fp16_causal_multi_wg():
    # Regression test for a deadlock class documented in cm_sdpa_opt_prefill.cm's
    # KV_CACHE_COMPRESSION branch: with a longer sequence that needs multiple Q work-groups
    # (q_len=1024 -> wg_count=4 at q_step=wg_size=16), kv_stop genuinely differs by wg_local_id,
    # and sdpa_kernel_lsc's cm_sbarrier/cm_fence work-group barriers would deadlock the moment
    # lanes stop reaching them in lockstep -- cm_sdpa_opt_prefill.cm recomputes a
    # work-group-uniform kv_stop_wg (from wg_id, not wg_local_id) specifically for this path.
    q, k, v = _make_qkv(batch=1, num_heads=4, num_kv_heads=4, q_len=1024, kv_len=1024,
                         head_size=64, seed=221)
    _check_against_roundtripped_oracle(q, k, v, is_causal=True)
    print("test_int8_matches_fp16_causal_multi_wg PASSED")


def test_int8_matches_fp16_causal_ragged_q_len():
    # Ragged last work-group (q_len not an exact multiple of wg_seq_len==256): some lanes have
    # valid query tokens, some don't, within the SAME work-group -- exercises the
    # "no early-return before the barrier pipeline" invariant cm_sdpa_opt_prefill.cm documents.
    q, k, v = _make_qkv(batch=1, num_heads=4, num_kv_heads=4, q_len=896, kv_len=896,
                         head_size=64, seed=225)
    _check_against_roundtripped_oracle(q, k, v, is_causal=True)
    print("test_int8_matches_fp16_causal_ragged_q_len PASSED")


def test_int8_matches_fp16_by_token_mode():
    # KV_CACHE_COMPRESSION==1 (BY_TOKEN for K too, not just V) -- the less-used mode, included for
    # coverage since cm_sdpa_common.hpp's k_by_channel=false path is otherwise untested here.
    q, k, v = _make_qkv(batch=1, num_heads=4, num_kv_heads=4, q_len=256, kv_len=256,
                         head_size=64, seed=226)
    _check_against_roundtripped_oracle(q, k, v, is_causal=True,
                                        compression_mode=KV_CACHE_COMPRESSION_BY_TOKEN)
    print("test_int8_matches_fp16_by_token_mode PASSED")


# ---- sparse_block_size==256 (block-granular skip inside the barrier-pipelined SLM path) ----
def test_int8_matches_fp16_causal_sparse_multi_wg():
    sparse_block_size = 256
    q_len = kv_len = 1024
    num_heads = num_kv_heads = 4
    q, k, v = _make_qkv(batch=1, num_heads=num_heads, num_kv_heads=num_kv_heads,
                         q_len=q_len, kv_len=kv_len, head_size=64, seed=222)
    mask, density = generate_block_mask_with_ratio(
        num_heads, seq_len=q_len, trunk_sz=q_len, sparse_block_sz=sparse_block_size,
        true_ratio=0.35, is_causal=True)
    _check_against_roundtripped_oracle(q, k, v, is_causal=True,
                                        sparse_block_size=sparse_block_size, mask=mask)
    print(f"test_int8_matches_fp16_causal_sparse_multi_wg PASSED (density={density:.2f})")


def test_int8_sparse_all_masked_row_outputs_zero_not_nan():
    sparse_block_size = 256
    q_len = kv_len = 1024
    num_heads = num_kv_heads = 2
    num_q_blocks = num_k_blocks = q_len // sparse_block_size
    q, k, v = _make_qkv(batch=1, num_heads=num_heads, num_kv_heads=num_kv_heads,
                         q_len=q_len, kv_len=kv_len, head_size=64, seed=225)
    mask = torch.ones([1, num_heads, num_q_blocks, num_k_blocks], dtype=torch.bool)
    mask[:, :, 1, :] = False  # q_block 1 (tokens [256,512)): every k_block masked out

    (k_u8, k_scale, k_zp, _), (v_u8, v_scale, v_zp, _) = _encode_kv_to_int8(
        k, v, KV_CACHE_COMPRESSION_BY_CHANNEL)
    out = run_prefill_int8(q, k_u8, k_scale, k_zp, v_u8, v_scale, v_zp, is_causal=True,
                            sparse_block_size=sparse_block_size, block_mask=mask)
    masked_rows = out[:, :, 256:512, :]
    assert torch.isfinite(masked_rows).all(), "all-masked q_block produced NaN/Inf, not 0"
    assert torch.all(masked_rows == 0), "all-masked q_block should output exactly 0 (NaN-guard)"
    print("test_int8_sparse_all_masked_row_outputs_zero_not_nan PASSED")


# ---- sparse_block_size==128 (two-level skip-LOAD/skip-COMPUTE mask split) ----
def test_int8_matches_fp16_causal_sparse128_single_wg():
    sparse_block_size = 128
    q_len = kv_len = 256
    num_heads = num_kv_heads = 4
    num_q_blocks = num_k_blocks = q_len // sparse_block_size
    q, k, v = _make_qkv(batch=1, num_heads=num_heads, num_kv_heads=num_kv_heads,
                         q_len=q_len, kv_len=kv_len, head_size=64, seed=301)
    g = torch.Generator().manual_seed(302)
    mask = torch.randint(0, 2, [1, num_heads, num_q_blocks, num_k_blocks], generator=g).to(torch.bool)
    mask[:, :, :, 0] = True
    _check_against_roundtripped_oracle(q, k, v, is_causal=True,
                                        sparse_block_size=sparse_block_size, mask=mask)
    print("test_int8_matches_fp16_causal_sparse128_single_wg PASSED")


def test_int8_matches_fp16_causal_sparse128_ragged_q_len():
    sparse_block_size = 128
    q_len = kv_len = 896
    num_heads = num_kv_heads = 4
    num_q_blocks = num_k_blocks = (q_len + sparse_block_size - 1) // sparse_block_size
    q, k, v = _make_qkv(batch=1, num_heads=num_heads, num_kv_heads=num_kv_heads,
                         q_len=q_len, kv_len=kv_len, head_size=64, seed=305)
    g = torch.Generator().manual_seed(306)
    mask = torch.randint(0, 2, [1, num_heads, num_q_blocks, num_k_blocks], generator=g).to(torch.bool)
    mask[:, :, :, 0] = True
    _check_against_roundtripped_oracle(q, k, v, is_causal=True,
                                        sparse_block_size=sparse_block_size, mask=mask)
    print("test_int8_matches_fp16_causal_sparse128_ragged_q_len PASSED")


def test_int8_matches_fp16_causal_sparse128_multi_wg_gqa_swapped():
    sparse_block_size = 128
    q_len = kv_len = 1024
    num_heads, num_kv_heads = 8, 2
    q, k, v = _make_qkv(batch=1, num_heads=num_heads, num_kv_heads=num_kv_heads,
                         q_len=q_len, kv_len=kv_len, head_size=64, seed=306)
    mask, density = generate_block_mask_with_ratio(
        num_heads, seq_len=q_len, trunk_sz=q_len, sparse_block_sz=sparse_block_size,
        true_ratio=0.3, is_causal=True)
    _check_against_roundtripped_oracle(q, k, v, is_causal=True,
                                        q_order=TRANSPOSE_ORDER_SWAPPED, k_order=TRANSPOSE_ORDER_SWAPPED,
                                        v_order=TRANSPOSE_ORDER_SWAPPED, o_order=TRANSPOSE_ORDER_SWAPPED,
                                        sparse_block_size=sparse_block_size, mask=mask)
    print("test_int8_matches_fp16_causal_sparse128_multi_wg_gqa_swapped PASSED "
          f"(density={density:.2f})")


if __name__ == "__main__":
    cl.profiling(True)
    test_int8_matches_fp16_noncausal_dense_identity()
    test_int8_matches_fp16_causal_dense_identity()
    test_int8_matches_fp16_causal_dense_swapped()
    test_int8_matches_fp16_head_size_128()
    test_int8_matches_fp16_gqa()
    test_int8_matches_fp16_causal_multi_wg()
    test_int8_matches_fp16_causal_ragged_q_len()
    test_int8_matches_fp16_by_token_mode()
    test_int8_matches_fp16_causal_sparse_multi_wg()
    test_int8_sparse_all_masked_row_outputs_zero_not_nan()
    test_int8_matches_fp16_causal_sparse128_single_wg()
    test_int8_matches_fp16_causal_sparse128_ragged_q_len()
    test_int8_matches_fp16_causal_sparse128_multi_wg_gqa_swapped()
    print("ALL PASSED")
