"""
Two-source (cache + current) block-sparse SDPA prefill correctness: cm_sdpa_opt_prefill_dualsource.cm
mirrors the real BlockSparseAttention primitive's `input[1]=key_cache.<layer>` (cache) +
`input[2]=k_new_f16` (current) ABI (see
flashvsr_dit_kvcache_int8_3slot_fused_dualsource_design_zh.md Sec 3.3/10.4) -- cache and current
are passed as two INDEPENDENT device tensors (never concatenated by this harness or the kernel).

Correctness oracle: the dualsource kernel's output must match today's already-trusted
single-source cm_sdpa_opt_prefill.cm (via test_sdpa_prefill_block_mask.run_prefill_sparse) run on
the LOGICALLY CONCATENATED K/V (torch.cat([k_cache, k_cur])) with the same combined mask -- so every
test here also proves "two sources == one concatenated source" numerically, not just "close to a
torch reference".
"""
import functools
import os

import numpy as np
import torch

from clops import cl

import test_sdpa_prefill as base
import test_sdpa_prefill_block_mask as block_mask_mod

torch.manual_seed(43)

TRANSPOSE_ORDER_IDENTITY = base.TRANSPOSE_ORDER_IDENTITY
TRANSPOSE_ORDER_SWAPPED = base.TRANSPOSE_ORDER_SWAPPED


@functools.cache
def build_dualsource_sparse_prefill_kernel(num_heads, num_kv_heads, head_size, sparse_block_size,
                                             q_order=TRANSPOSE_ORDER_IDENTITY, k_order=TRANSPOSE_ORDER_IDENTITY,
                                             v_order=TRANSPOSE_ORDER_IDENTITY, o_order=TRANSPOSE_ORDER_IDENTITY,
                                             scale_factor=None, kv_blk=1):
    if scale_factor is None:
        scale_factor = 1.0 / (head_size ** 0.5)
    src = r'''#include "cm_sdpa_opt_prefill_dualsource.cm"'''
    opts = (
        f'-cmc -Qxcm_jit_option="-abortonspill" -Qxcm_register_file_size=256 -mCM_printregusage '
        f"-I{base.CWD} -I{base.FLASHATTN_DIR} -I{base.PAGEATTEN_DIR} "
        f"-DKERNEL_NAME=cm_sdpa_opt_prefill_dualsource "
        f"-DCMFLA_NUM_HEADS={num_heads} -DCMFLA_NUM_KV_HEADS={num_kv_heads} "
        f"-DCMFLA_HEAD_SIZE={head_size} -DCMFLA_SCALE_FACTOR={scale_factor} "
        f"-DQ_TRANSPOSE_ORDER={q_order} -DK_TRANSPOSE_ORDER={k_order} "
        f"-DV_TRANSPOSE_ORDER={v_order} -DOUTPUT_TRANSPOSE_ORDER={o_order} "
        f"-DSPARSE_BLOCK_SIZE={sparse_block_size} -DCMFLA_KV_BLK={kv_blk}"
    )
    return cl.kernels(src, opts)


def run_prefill_sparse_dualsource(q, k_cache, v_cache, k_cur, v_cur, sparse_block_size, block_mask,
                                    q_order=TRANSPOSE_ORDER_IDENTITY, k_order=TRANSPOSE_ORDER_IDENTITY,
                                    v_order=TRANSPOSE_ORDER_IDENTITY, o_order=TRANSPOSE_ORDER_IDENTITY,
                                    scale=None, kv_len_cache_override=None, kv_len_cur_override=None,
                                    kv_blk=1):
    # kv_len_cache_override/kv_len_cur_override let a caller pass a logical kv_len of 0 (an empty
    # source, e.g. chunk0's not-yet-populated cache) while still uploading a minimal non-empty
    # physical buffer -- cl.tensor() does not support a genuinely zero-length dimension, but the
    # kernel never reads this source at all once its kv_len is 0 (sdpa_accumulate_kv_source's
    # kv_stop<=0 no-op), so the buffer's actual contents/shape beyond kv_len don't matter.
    #
    # kv_blk (-DCMFLA_KV_BLK, default 1): overridden to 2+ by the test_*_kv_blk2 variants below to
    # exercise sdpa_accumulate_kv_source_prefetch's KV_BLK>=2 rescale-skip branch and its
    # first_write-threading fix -- at the default KV_BLK=1 that branch doesn't exist (the rescale
    # always runs unconditionally), so the fix is otherwise untested. The reference path
    # (_ref_via_concatenated_single_source) deliberately stays at its own default KV_BLK=1 -- online
    # softmax is invariant to the KV blocking factor, so a correct implementation must match
    # regardless of whether the two sides use the same KV_BLK.
    batch, num_heads, q_len, head_size = q.shape
    _, num_kv_heads, kv_len_cache, _ = k_cache.shape
    _, _, kv_len_cur, _ = k_cur.shape
    kv_len_cache_arg = kv_len_cache if kv_len_cache_override is None else kv_len_cache_override
    kv_len_cur_arg = kv_len_cur if kv_len_cur_override is None else kv_len_cur_override

    kernel = build_dualsource_sparse_prefill_kernel(num_heads, num_kv_heads, head_size, sparse_block_size,
                                                       q_order, k_order, v_order, o_order, scale, kv_blk)

    q_phys = base.to_physical(q, q_order)
    k_cache_phys = base.to_physical(k_cache, k_order)
    v_cache_phys = base.to_physical(v_cache, v_order)
    k_cur_phys = base.to_physical(k_cur, k_order)
    v_cur_phys = base.to_physical(v_cur, v_order)

    t_q = cl.tensor(q_phys.to(torch.float16).numpy())
    t_k_cache = cl.tensor(k_cache_phys.to(torch.float16).numpy())
    t_v_cache = cl.tensor(v_cache_phys.to(torch.float16).numpy())
    t_k_cur = cl.tensor(k_cur_phys.to(torch.float16).numpy())
    t_v_cur = cl.tensor(v_cur_phys.to(torch.float16).numpy())
    out_shape = ([batch, q_len, num_heads, head_size] if o_order == TRANSPOSE_ORDER_SWAPPED
                 else [batch, num_heads, q_len, head_size])
    t_out = cl.tensor(np.zeros(out_shape, dtype=np.float16))

    q_step = base.get_cm_grf_width() // 32
    wg_size = 16
    wg_seq_len = wg_size * q_step
    wg_count = (q_len + wg_seq_len - 1) // wg_seq_len
    GWS = [1, batch * num_heads, wg_count * wg_size]
    LWS = [1, 1, wg_size]

    num_q_blocks = block_mask.shape[2]
    num_k_blocks_cache = (kv_len_cache_arg + sparse_block_size - 1) // sparse_block_size
    num_k_blocks_cur = (kv_len_cur_arg + sparse_block_size - 1) // sparse_block_size
    assert block_mask.shape[3] == num_k_blocks_cache + num_k_blocks_cur, (
        f"block_mask's K-block axis ({block_mask.shape[3]}) must equal "
        f"num_k_blocks_cache+num_k_blocks_cur ({num_k_blocks_cache}+{num_k_blocks_cur})"
    )
    t_mask = cl.tensor(block_mask.to(torch.bool).numpy())

    args = [t_q, t_k_cache, t_v_cache, t_k_cur, t_v_cur, t_out, q_len, kv_len_cache_arg, kv_len_cur_arg,
            t_mask, num_q_blocks, num_k_blocks_cache, num_k_blocks_cur]
    kernel.enqueue("cm_sdpa_opt_prefill_dualsource", GWS, LWS, *args)

    out_phys = torch.from_numpy(t_out.numpy())
    return base.from_physical(out_phys, o_order, batch, num_heads, q_len, head_size)


def _ref_via_concatenated_single_source(q, k_cache, v_cache, k_cur, v_cur, sparse_block_size, block_mask):
    # The correctness oracle: today's already-trusted single-source kernel, run on the logically
    # concatenated K/V with the same combined mask.
    k_cat = torch.cat([k_cache, k_cur], dim=2)
    v_cat = torch.cat([v_cache, v_cur], dim=2)
    return block_mask_mod.run_prefill_sparse(q, k_cat, v_cat, is_causal=False,
                                              sparse_block_size=sparse_block_size, block_mask=block_mask)


def _make_qkv_dualsource(batch, num_heads, num_kv_heads, q_len, kv_len_cache, kv_len_cur, head_size, seed=0):
    g = torch.Generator().manual_seed(seed)
    low, high = -2, 3
    q = torch.randint(low, high, [batch, num_heads, q_len, head_size], generator=g).to(torch.float16)
    k_cache = torch.randint(low, high, [batch, num_kv_heads, kv_len_cache, head_size], generator=g).to(torch.float16)
    v_cache = (torch.randint(low, high, [batch, num_kv_heads, kv_len_cache, head_size], generator=g).to(torch.float16)
              / high)
    k_cur = torch.randint(low, high, [batch, num_kv_heads, kv_len_cur, head_size], generator=g).to(torch.float16)
    v_cur = (torch.randint(low, high, [batch, num_kv_heads, kv_len_cur, head_size], generator=g).to(torch.float16)
             / high)
    return q, k_cache, v_cache, k_cur, v_cur


def test_matches_concatenated_all_true_mask():
    sparse_block_size = 128
    q_len = kv_len_cache = kv_len_cur = 256
    q, k_cache, v_cache, k_cur, v_cur = _make_qkv_dualsource(
        batch=1, num_heads=4, num_kv_heads=4, q_len=q_len, kv_len_cache=kv_len_cache,
        kv_len_cur=kv_len_cur, head_size=64, seed=1)

    num_q_blocks = (q_len + sparse_block_size - 1) // sparse_block_size
    num_k_blocks_cache = (kv_len_cache + sparse_block_size - 1) // sparse_block_size
    num_k_blocks_cur = (kv_len_cur + sparse_block_size - 1) // sparse_block_size
    mask = torch.ones(1, 4, num_q_blocks, num_k_blocks_cache + num_k_blocks_cur, dtype=torch.bool)

    out = run_prefill_sparse_dualsource(q, k_cache, v_cache, k_cur, v_cur, sparse_block_size, mask)
    ref = _ref_via_concatenated_single_source(q, k_cache, v_cache, k_cur, v_cur, sparse_block_size, mask)
    assert torch.equal(out, ref), "all-True mask: dualsource must be bit-identical to the concatenated single-source kernel"
    print("test_matches_concatenated_all_true_mask PASSED")


def test_matches_concatenated_random_sparse_mask():
    sparse_block_size = 128
    q_len, kv_len_cache, kv_len_cur = 256, 384, 128
    q, k_cache, v_cache, k_cur, v_cur = _make_qkv_dualsource(
        batch=1, num_heads=4, num_kv_heads=4, q_len=q_len, kv_len_cache=kv_len_cache,
        kv_len_cur=kv_len_cur, head_size=64, seed=2)

    num_q_blocks = (q_len + sparse_block_size - 1) // sparse_block_size
    num_k_blocks_cache = (kv_len_cache + sparse_block_size - 1) // sparse_block_size
    num_k_blocks_cur = (kv_len_cur + sparse_block_size - 1) // sparse_block_size
    g = torch.Generator().manual_seed(4)
    mask = torch.randint(0, 2, [1, 4, num_q_blocks, num_k_blocks_cache + num_k_blocks_cur],
                          generator=g).to(torch.bool)
    # Avoid all-masked Q-rows (legitimate but makes softmax(all -inf)=NaN, excluded by design --
    # not what this test is isolating; see test_flashvsr_block_sparse_attention.py's identical note).
    mask[:, :, :, 0] = True

    out = run_prefill_sparse_dualsource(q, k_cache, v_cache, k_cur, v_cur, sparse_block_size, mask)
    ref = _ref_via_concatenated_single_source(q, k_cache, v_cache, k_cur, v_cur, sparse_block_size, mask)
    base.check_close(out, ref)
    print("test_matches_concatenated_random_sparse_mask PASSED")


def test_realistic_kv_ratio():
    # FlashVSR's real steady-state streaming-cache shape: current = this chunk's own new K/V
    # (1 window-group), cache = 3 cached history groups (kv_ratio=3.0), window_tokens=128 ==
    # SPARSE_BLOCK_SIZE -- same production ratio as
    # test_flashvsr_block_sparse_attention.test_end_to_end_sparse_attention_realistic_ratio, but at
    # the raw-attention level (no DraftBlockMask involved here, mask is random).
    sparse_block_size = 128
    q_len = kv_len_cur = sparse_block_size
    kv_len_cache = 3 * sparse_block_size
    q, k_cache, v_cache, k_cur, v_cur = _make_qkv_dualsource(
        batch=1, num_heads=4, num_kv_heads=4, q_len=q_len, kv_len_cache=kv_len_cache,
        kv_len_cur=kv_len_cur, head_size=64, seed=9)

    num_q_blocks = 1
    num_k_blocks_cache, num_k_blocks_cur = 3, 1
    g = torch.Generator().manual_seed(10)
    mask = torch.randint(0, 2, [1, 4, num_q_blocks, num_k_blocks_cache + num_k_blocks_cur],
                          generator=g).to(torch.bool)
    mask[:, :, :, 0] = True

    out = run_prefill_sparse_dualsource(q, k_cache, v_cache, k_cur, v_cur, sparse_block_size, mask)
    ref = _ref_via_concatenated_single_source(q, k_cache, v_cache, k_cur, v_cur, sparse_block_size, mask)
    base.check_close(out, ref)
    print("test_realistic_kv_ratio PASSED")


def test_empty_cache_chunk0():
    # chunk0/warmup: cache not yet populated (kv_len_cache=0) -- the dual-source kernel must degrade
    # cleanly to "current only" (sdpa_accumulate_kv_source's kv_stop<=0 no-op path). cl.tensor()
    # can't hold a genuinely zero-length dimension, so a minimal 1-row placeholder is uploaded for
    # k_cache/v_cache's physical buffer while kv_len_cache_override=0 tells the kernel this source is
    # empty -- the kernel never reads the placeholder's contents (see
    # run_prefill_sparse_dualsource's doc comment).
    sparse_block_size = 128
    q_len = kv_len_cur = 256
    q, k_cache_placeholder, v_cache_placeholder, k_cur, v_cur = _make_qkv_dualsource(
        batch=1, num_heads=4, num_kv_heads=4, q_len=q_len, kv_len_cache=1,
        kv_len_cur=kv_len_cur, head_size=64, seed=13)

    num_q_blocks = (q_len + sparse_block_size - 1) // sparse_block_size
    num_k_blocks_cache, num_k_blocks_cur = 0, (kv_len_cur + sparse_block_size - 1) // sparse_block_size
    mask = torch.ones(1, 4, num_q_blocks, num_k_blocks_cache + num_k_blocks_cur, dtype=torch.bool)

    out = run_prefill_sparse_dualsource(q, k_cache_placeholder, v_cache_placeholder, k_cur, v_cur,
                                          sparse_block_size, mask, kv_len_cache_override=0)
    ref = block_mask_mod.run_prefill_sparse(q, k_cur, v_cur, is_causal=False,
                                             sparse_block_size=sparse_block_size, block_mask=mask)
    assert torch.equal(out, ref), "empty cache must degrade to current-only, bit-identical to the single-source kernel"
    print("test_empty_cache_chunk0 PASSED")


def _empty_cache_vs_single_source(sparse_block_size, density, seed):
    # Same shape/spec as a real "empty cache" chunk0 call, but with an actual sparse (not all-True)
    # mask -- extends test_empty_cache_chunk0's coverage to confirm the dual-source kernel's
    # current-only degenerate path is bit-identical to single-source at EVERY sparsity this repo
    # benchmarks (bench_sdpa_dense_vs_sparse.py's dense/sparse_bs128/sparse_bs256 configs), not just
    # the all-True case. This is the correctness half of the "dual-source with an empty cache must
    # equal single-source at the same spec" property the barrier-free sdpa_kernel_lsc_prefetch_2src
    # path (cm_sdpa_common.hpp) is required to have -- see bench_sdpa_dense_vs_sparse.py's matching
    # benchmark-side parity check for the performance half of the same claim.
    q_len = kv_len_cur = 256
    q, k_cache_placeholder, v_cache_placeholder, k_cur, v_cur = _make_qkv_dualsource(
        batch=1, num_heads=4, num_kv_heads=4, q_len=q_len, kv_len_cache=1,
        kv_len_cur=kv_len_cur, head_size=64, seed=seed)

    num_q_blocks = (q_len + sparse_block_size - 1) // sparse_block_size
    num_k_blocks_cache, num_k_blocks_cur = 0, (kv_len_cur + sparse_block_size - 1) // sparse_block_size
    g = torch.Generator().manual_seed(seed + 1)
    mask = (torch.rand([1, 4, num_q_blocks, num_k_blocks_cache + num_k_blocks_cur], generator=g)
            < density).to(torch.bool)
    mask[:, :, :, 0] = True

    out = run_prefill_sparse_dualsource(q, k_cache_placeholder, v_cache_placeholder, k_cur, v_cur,
                                          sparse_block_size, mask, kv_len_cache_override=0)
    ref = block_mask_mod.run_prefill_sparse(q, k_cur, v_cur, is_causal=False,
                                             sparse_block_size=sparse_block_size, block_mask=mask)
    assert torch.equal(out, ref), "empty cache + sparse mask must still degrade to current-only, bit-identical to single-source"


def test_empty_cache_sparse_bs128():
    _empty_cache_vs_single_source(sparse_block_size=128, density=0.3, seed=16)
    print("test_empty_cache_sparse_bs128 PASSED")


def test_empty_cache_sparse_bs256():
    _empty_cache_vs_single_source(sparse_block_size=256, density=0.3, seed=17)
    print("test_empty_cache_sparse_bs256 PASSED")


def test_gqa():
    sparse_block_size = 128
    q_len, kv_len_cache, kv_len_cur = 256, 256, 128
    q, k_cache, v_cache, k_cur, v_cur = _make_qkv_dualsource(
        batch=1, num_heads=8, num_kv_heads=2, q_len=q_len, kv_len_cache=kv_len_cache,
        kv_len_cur=kv_len_cur, head_size=64, seed=14)

    num_q_blocks = (q_len + sparse_block_size - 1) // sparse_block_size
    num_k_blocks_cache = (kv_len_cache + sparse_block_size - 1) // sparse_block_size
    num_k_blocks_cur = (kv_len_cur + sparse_block_size - 1) // sparse_block_size
    mask = torch.ones(1, 8, num_q_blocks, num_k_blocks_cache + num_k_blocks_cur, dtype=torch.bool)

    out = run_prefill_sparse_dualsource(q, k_cache, v_cache, k_cur, v_cur, sparse_block_size, mask)
    ref = _ref_via_concatenated_single_source(q, k_cache, v_cache, k_cur, v_cur, sparse_block_size, mask)
    base.check_close(out, ref)
    print("test_gqa PASSED")


def test_head_size_128():
    sparse_block_size = 128
    q_len, kv_len_cache, kv_len_cur = 256, 256, 128
    q, k_cache, v_cache, k_cur, v_cur = _make_qkv_dualsource(
        batch=1, num_heads=4, num_kv_heads=4, q_len=q_len, kv_len_cache=kv_len_cache,
        kv_len_cur=kv_len_cur, head_size=128, seed=15)

    num_q_blocks = (q_len + sparse_block_size - 1) // sparse_block_size
    num_k_blocks_cache = (kv_len_cache + sparse_block_size - 1) // sparse_block_size
    num_k_blocks_cur = (kv_len_cur + sparse_block_size - 1) // sparse_block_size
    mask = torch.ones(1, 4, num_q_blocks, num_k_blocks_cache + num_k_blocks_cur, dtype=torch.bool)

    out = run_prefill_sparse_dualsource(q, k_cache, v_cache, k_cur, v_cur, sparse_block_size, mask)
    ref = _ref_via_concatenated_single_source(q, k_cache, v_cache, k_cur, v_cur, sparse_block_size, mask)
    base.check_close(out, ref)
    print("test_head_size_128 PASSED")


def test_matches_concatenated_random_sparse_mask_kv_blk2():
    # Same shape as test_matches_concatenated_random_sparse_mask, but built with
    # -DCMFLA_KV_BLK=2 -- the ONLY way to exercise sdpa_accumulate_kv_source_prefetch's
    # KV_BLK>=2 rescale-skip branch (and its first_write-threading fix for a non-empty cache
    # segment followed by a current segment) at all; at the default KV_BLK=1 this branch doesn't
    # exist.
    sparse_block_size = 128
    q_len, kv_len_cache, kv_len_cur = 256, 384, 128
    q, k_cache, v_cache, k_cur, v_cur = _make_qkv_dualsource(
        batch=1, num_heads=4, num_kv_heads=4, q_len=q_len, kv_len_cache=kv_len_cache,
        kv_len_cur=kv_len_cur, head_size=64, seed=2)

    num_q_blocks = (q_len + sparse_block_size - 1) // sparse_block_size
    num_k_blocks_cache = (kv_len_cache + sparse_block_size - 1) // sparse_block_size
    num_k_blocks_cur = (kv_len_cur + sparse_block_size - 1) // sparse_block_size
    g = torch.Generator().manual_seed(4)
    mask = torch.randint(0, 2, [1, 4, num_q_blocks, num_k_blocks_cache + num_k_blocks_cur],
                          generator=g).to(torch.bool)
    mask[:, :, :, 0] = True

    out = run_prefill_sparse_dualsource(q, k_cache, v_cache, k_cur, v_cur, sparse_block_size, mask, kv_blk=2)
    ref = _ref_via_concatenated_single_source(q, k_cache, v_cache, k_cur, v_cur, sparse_block_size, mask)
    base.check_close(out, ref)
    print("test_matches_concatenated_random_sparse_mask_kv_blk2 PASSED")


def test_realistic_kv_ratio_kv_blk2():
    # Same shape as test_realistic_kv_ratio, built with -DCMFLA_KV_BLK=2.
    sparse_block_size = 128
    q_len = kv_len_cur = sparse_block_size
    kv_len_cache = 3 * sparse_block_size
    q, k_cache, v_cache, k_cur, v_cur = _make_qkv_dualsource(
        batch=1, num_heads=4, num_kv_heads=4, q_len=q_len, kv_len_cache=kv_len_cache,
        kv_len_cur=kv_len_cur, head_size=64, seed=9)

    num_q_blocks = 1
    num_k_blocks_cache, num_k_blocks_cur = 3, 1
    g = torch.Generator().manual_seed(10)
    mask = torch.randint(0, 2, [1, 4, num_q_blocks, num_k_blocks_cache + num_k_blocks_cur],
                          generator=g).to(torch.bool)
    mask[:, :, :, 0] = True

    out = run_prefill_sparse_dualsource(q, k_cache, v_cache, k_cur, v_cur, sparse_block_size, mask, kv_blk=2)
    ref = _ref_via_concatenated_single_source(q, k_cache, v_cache, k_cur, v_cur, sparse_block_size, mask)
    base.check_close(out, ref)
    print("test_realistic_kv_ratio_kv_blk2 PASSED")


def test_fully_skipped_cache_kv_blk2():
    # Pins down the first_write/"fully block-sparse-masked-out cache" interaction described in
    # sdpa_accumulate_kv_source_prefetch's doc comment: cache's block mask is entirely False (every
    # cache block skipped), current's mask has at least one True block -- first_write must stay true
    # through all of cache's (skipped) iterations so current's first executed chunk still correctly
    # takes the rescale-skip branch (rO is genuinely still zero at that point). Built with
    # -DCMFLA_KV_BLK=2 so the rescale-skip branch actually exists to mis-fire.
    sparse_block_size = 128
    q_len, kv_len_cache, kv_len_cur = 128, 384, 256
    q, k_cache, v_cache, k_cur, v_cur = _make_qkv_dualsource(
        batch=1, num_heads=4, num_kv_heads=4, q_len=q_len, kv_len_cache=kv_len_cache,
        kv_len_cur=kv_len_cur, head_size=64, seed=21)

    num_q_blocks = (q_len + sparse_block_size - 1) // sparse_block_size
    num_k_blocks_cache = (kv_len_cache + sparse_block_size - 1) // sparse_block_size
    num_k_blocks_cur = (kv_len_cur + sparse_block_size - 1) // sparse_block_size
    mask = torch.zeros(1, 4, num_q_blocks, num_k_blocks_cache + num_k_blocks_cur, dtype=torch.bool)
    mask[:, :, :, num_k_blocks_cache:] = True  # all of cache skipped, all of current kept

    out = run_prefill_sparse_dualsource(q, k_cache, v_cache, k_cur, v_cur, sparse_block_size, mask, kv_blk=2)
    ref = _ref_via_concatenated_single_source(q, k_cache, v_cache, k_cur, v_cur, sparse_block_size, mask)
    base.check_close(out, ref)
    print("test_fully_skipped_cache_kv_blk2 PASSED")


if __name__ == "__main__":
    cl.profiling(True)
    test_matches_concatenated_all_true_mask()
    test_matches_concatenated_random_sparse_mask()
    test_realistic_kv_ratio()
    test_empty_cache_chunk0()
    test_empty_cache_sparse_bs128()
    test_empty_cache_sparse_bs256()
    test_gqa()
    test_head_size_128()
    test_matches_concatenated_random_sparse_mask_kv_blk2()
    test_realistic_kv_ratio_kv_blk2()
    test_fully_skipped_cache_kv_blk2()
    print("ALL PASSED")
