"""
Two-source (ring + current) block-sparse SDPA prefill correctness: cm_sdpa_opt_prefill_multisource.cm
mirrors the real BlockSparseAttention primitive's `input[1]=key_cache.<layer>` (ring) +
`input[2]=k_new_f16` (current) ABI (see
flashvsr_dit_kvcache_int8_3slot_fused_multisource_design_zh.md Sec 3.3/10.4) -- ring and current
are passed as two INDEPENDENT device tensors (never concatenated by this harness or the kernel).

Correctness oracle: the multisource kernel's output must match today's already-trusted
single-source cm_sdpa_opt_prefill.cm (via test_sdpa_prefill_block_mask.run_prefill_sparse) run on
the LOGICALLY CONCATENATED K/V (torch.cat([k_ring, k_cur])) with the same combined mask -- so every
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
def build_multisource_sparse_prefill_kernel(num_heads, num_kv_heads, head_size, sparse_block_size,
                                             q_order=TRANSPOSE_ORDER_IDENTITY, k_order=TRANSPOSE_ORDER_IDENTITY,
                                             v_order=TRANSPOSE_ORDER_IDENTITY, o_order=TRANSPOSE_ORDER_IDENTITY,
                                             scale_factor=None):
    if scale_factor is None:
        scale_factor = 1.0 / (head_size ** 0.5)
    src = r'''#include "cm_sdpa_opt_prefill_multisource.cm"'''
    opts = (
        f'-cmc -Qxcm_jit_option="-abortonspill" -Qxcm_register_file_size=256 -mCM_printregusage '
        f"-I{base.CWD} -I{base.FLASHATTN_DIR} -I{base.PAGEATTEN_DIR} "
        f"-DKERNEL_NAME=cm_sdpa_opt_prefill_multisource "
        f"-DCMFLA_NUM_HEADS={num_heads} -DCMFLA_NUM_KV_HEADS={num_kv_heads} "
        f"-DCMFLA_HEAD_SIZE={head_size} -DCMFLA_SCALE_FACTOR={scale_factor} "
        f"-DQ_TRANSPOSE_ORDER={q_order} -DK_TRANSPOSE_ORDER={k_order} "
        f"-DV_TRANSPOSE_ORDER={v_order} -DOUTPUT_TRANSPOSE_ORDER={o_order} "
        f"-DSPARSE_BLOCK_SIZE={sparse_block_size}"
    )
    return cl.kernels(src, opts)


def run_prefill_sparse_multisource(q, k_ring, v_ring, k_cur, v_cur, sparse_block_size, block_mask,
                                    q_order=TRANSPOSE_ORDER_IDENTITY, k_order=TRANSPOSE_ORDER_IDENTITY,
                                    v_order=TRANSPOSE_ORDER_IDENTITY, o_order=TRANSPOSE_ORDER_IDENTITY,
                                    scale=None, kv_len_ring_override=None, kv_len_cur_override=None):
    # kv_len_ring_override/kv_len_cur_override let a caller pass a logical kv_len of 0 (an empty
    # source, e.g. chunk0's not-yet-populated ring) while still uploading a minimal non-empty
    # physical buffer -- cl.tensor() does not support a genuinely zero-length dimension, but the
    # kernel never reads this source at all once its kv_len is 0 (sdpa_accumulate_kv_source's
    # kv_stop<=0 no-op), so the buffer's actual contents/shape beyond kv_len don't matter.
    batch, num_heads, q_len, head_size = q.shape
    _, num_kv_heads, kv_len_ring, _ = k_ring.shape
    _, _, kv_len_cur, _ = k_cur.shape
    kv_len_ring_arg = kv_len_ring if kv_len_ring_override is None else kv_len_ring_override
    kv_len_cur_arg = kv_len_cur if kv_len_cur_override is None else kv_len_cur_override

    kernel = build_multisource_sparse_prefill_kernel(num_heads, num_kv_heads, head_size, sparse_block_size,
                                                       q_order, k_order, v_order, o_order, scale)

    q_phys = base.to_physical(q, q_order)
    k_ring_phys = base.to_physical(k_ring, k_order)
    v_ring_phys = base.to_physical(v_ring, v_order)
    k_cur_phys = base.to_physical(k_cur, k_order)
    v_cur_phys = base.to_physical(v_cur, v_order)

    t_q = cl.tensor(q_phys.to(torch.float16).numpy())
    t_k_ring = cl.tensor(k_ring_phys.to(torch.float16).numpy())
    t_v_ring = cl.tensor(v_ring_phys.to(torch.float16).numpy())
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
    num_k_blocks_ring = (kv_len_ring_arg + sparse_block_size - 1) // sparse_block_size
    num_k_blocks_cur = (kv_len_cur_arg + sparse_block_size - 1) // sparse_block_size
    assert block_mask.shape[3] == num_k_blocks_ring + num_k_blocks_cur, (
        f"block_mask's K-block axis ({block_mask.shape[3]}) must equal "
        f"num_k_blocks_ring+num_k_blocks_cur ({num_k_blocks_ring}+{num_k_blocks_cur})"
    )
    t_mask = cl.tensor(block_mask.to(torch.bool).numpy())

    args = [t_q, t_k_ring, t_v_ring, t_k_cur, t_v_cur, t_out, q_len, kv_len_ring_arg, kv_len_cur_arg,
            t_mask, num_q_blocks, num_k_blocks_ring, num_k_blocks_cur]
    kernel.enqueue("cm_sdpa_opt_prefill_multisource", GWS, LWS, *args)

    out_phys = torch.from_numpy(t_out.numpy())
    return base.from_physical(out_phys, o_order, batch, num_heads, q_len, head_size)


def _ref_via_concatenated_single_source(q, k_ring, v_ring, k_cur, v_cur, sparse_block_size, block_mask):
    # The correctness oracle: today's already-trusted single-source kernel, run on the logically
    # concatenated K/V with the same combined mask.
    k_cat = torch.cat([k_ring, k_cur], dim=2)
    v_cat = torch.cat([v_ring, v_cur], dim=2)
    return block_mask_mod.run_prefill_sparse(q, k_cat, v_cat, is_causal=False,
                                              sparse_block_size=sparse_block_size, block_mask=block_mask)


def _make_qkv_multisource(batch, num_heads, num_kv_heads, q_len, kv_len_ring, kv_len_cur, head_size, seed=0):
    g = torch.Generator().manual_seed(seed)
    low, high = -2, 3
    q = torch.randint(low, high, [batch, num_heads, q_len, head_size], generator=g).to(torch.float16)
    k_ring = torch.randint(low, high, [batch, num_kv_heads, kv_len_ring, head_size], generator=g).to(torch.float16)
    v_ring = (torch.randint(low, high, [batch, num_kv_heads, kv_len_ring, head_size], generator=g).to(torch.float16)
              / high)
    k_cur = torch.randint(low, high, [batch, num_kv_heads, kv_len_cur, head_size], generator=g).to(torch.float16)
    v_cur = (torch.randint(low, high, [batch, num_kv_heads, kv_len_cur, head_size], generator=g).to(torch.float16)
             / high)
    return q, k_ring, v_ring, k_cur, v_cur


def test_matches_concatenated_all_true_mask():
    sparse_block_size = 128
    q_len = kv_len_ring = kv_len_cur = 256
    q, k_ring, v_ring, k_cur, v_cur = _make_qkv_multisource(
        batch=1, num_heads=4, num_kv_heads=4, q_len=q_len, kv_len_ring=kv_len_ring,
        kv_len_cur=kv_len_cur, head_size=64, seed=1)

    num_q_blocks = (q_len + sparse_block_size - 1) // sparse_block_size
    num_k_blocks_ring = (kv_len_ring + sparse_block_size - 1) // sparse_block_size
    num_k_blocks_cur = (kv_len_cur + sparse_block_size - 1) // sparse_block_size
    mask = torch.ones(1, 4, num_q_blocks, num_k_blocks_ring + num_k_blocks_cur, dtype=torch.bool)

    out = run_prefill_sparse_multisource(q, k_ring, v_ring, k_cur, v_cur, sparse_block_size, mask)
    ref = _ref_via_concatenated_single_source(q, k_ring, v_ring, k_cur, v_cur, sparse_block_size, mask)
    assert torch.equal(out, ref), "all-True mask: multisource must be bit-identical to the concatenated single-source kernel"
    print("test_matches_concatenated_all_true_mask PASSED")


def test_matches_concatenated_random_sparse_mask():
    sparse_block_size = 128
    q_len, kv_len_ring, kv_len_cur = 256, 384, 128
    q, k_ring, v_ring, k_cur, v_cur = _make_qkv_multisource(
        batch=1, num_heads=4, num_kv_heads=4, q_len=q_len, kv_len_ring=kv_len_ring,
        kv_len_cur=kv_len_cur, head_size=64, seed=2)

    num_q_blocks = (q_len + sparse_block_size - 1) // sparse_block_size
    num_k_blocks_ring = (kv_len_ring + sparse_block_size - 1) // sparse_block_size
    num_k_blocks_cur = (kv_len_cur + sparse_block_size - 1) // sparse_block_size
    g = torch.Generator().manual_seed(4)
    mask = torch.randint(0, 2, [1, 4, num_q_blocks, num_k_blocks_ring + num_k_blocks_cur],
                          generator=g).to(torch.bool)
    # Avoid all-masked Q-rows (legitimate but makes softmax(all -inf)=NaN, excluded by design --
    # not what this test is isolating; see test_flashvsr_block_sparse_attention.py's identical note).
    mask[:, :, :, 0] = True

    out = run_prefill_sparse_multisource(q, k_ring, v_ring, k_cur, v_cur, sparse_block_size, mask)
    ref = _ref_via_concatenated_single_source(q, k_ring, v_ring, k_cur, v_cur, sparse_block_size, mask)
    base.check_close(out, ref)
    print("test_matches_concatenated_random_sparse_mask PASSED")


def test_realistic_kv_ratio():
    # FlashVSR's real steady-state streaming-cache shape: current = this chunk's own new K/V
    # (1 window-group), ring = 3 cached history groups (kv_ratio=3.0), window_tokens=128 ==
    # SPARSE_BLOCK_SIZE -- same production ratio as
    # test_flashvsr_block_sparse_attention.test_end_to_end_sparse_attention_realistic_ratio, but at
    # the raw-attention level (no DraftBlockMask involved here, mask is random).
    sparse_block_size = 128
    q_len = kv_len_cur = sparse_block_size
    kv_len_ring = 3 * sparse_block_size
    q, k_ring, v_ring, k_cur, v_cur = _make_qkv_multisource(
        batch=1, num_heads=4, num_kv_heads=4, q_len=q_len, kv_len_ring=kv_len_ring,
        kv_len_cur=kv_len_cur, head_size=64, seed=9)

    num_q_blocks = 1
    num_k_blocks_ring, num_k_blocks_cur = 3, 1
    g = torch.Generator().manual_seed(10)
    mask = torch.randint(0, 2, [1, 4, num_q_blocks, num_k_blocks_ring + num_k_blocks_cur],
                          generator=g).to(torch.bool)
    mask[:, :, :, 0] = True

    out = run_prefill_sparse_multisource(q, k_ring, v_ring, k_cur, v_cur, sparse_block_size, mask)
    ref = _ref_via_concatenated_single_source(q, k_ring, v_ring, k_cur, v_cur, sparse_block_size, mask)
    base.check_close(out, ref)
    print("test_realistic_kv_ratio PASSED")


def test_empty_ring_chunk0():
    # chunk0/warmup: ring not yet populated (kv_len_ring=0) -- the dual-source kernel must degrade
    # cleanly to "current only" (sdpa_accumulate_kv_source's kv_stop<=0 no-op path). cl.tensor()
    # can't hold a genuinely zero-length dimension, so a minimal 1-row placeholder is uploaded for
    # k_ring/v_ring's physical buffer while kv_len_ring_override=0 tells the kernel this source is
    # empty -- the kernel never reads the placeholder's contents (see
    # run_prefill_sparse_multisource's doc comment).
    sparse_block_size = 128
    q_len = kv_len_cur = 256
    q, k_ring_placeholder, v_ring_placeholder, k_cur, v_cur = _make_qkv_multisource(
        batch=1, num_heads=4, num_kv_heads=4, q_len=q_len, kv_len_ring=1,
        kv_len_cur=kv_len_cur, head_size=64, seed=13)

    num_q_blocks = (q_len + sparse_block_size - 1) // sparse_block_size
    num_k_blocks_ring, num_k_blocks_cur = 0, (kv_len_cur + sparse_block_size - 1) // sparse_block_size
    mask = torch.ones(1, 4, num_q_blocks, num_k_blocks_ring + num_k_blocks_cur, dtype=torch.bool)

    out = run_prefill_sparse_multisource(q, k_ring_placeholder, v_ring_placeholder, k_cur, v_cur,
                                          sparse_block_size, mask, kv_len_ring_override=0)
    ref = block_mask_mod.run_prefill_sparse(q, k_cur, v_cur, is_causal=False,
                                             sparse_block_size=sparse_block_size, block_mask=mask)
    assert torch.equal(out, ref), "empty ring must degrade to current-only, bit-identical to the single-source kernel"
    print("test_empty_ring_chunk0 PASSED")


def test_gqa():
    sparse_block_size = 128
    q_len, kv_len_ring, kv_len_cur = 256, 256, 128
    q, k_ring, v_ring, k_cur, v_cur = _make_qkv_multisource(
        batch=1, num_heads=8, num_kv_heads=2, q_len=q_len, kv_len_ring=kv_len_ring,
        kv_len_cur=kv_len_cur, head_size=64, seed=14)

    num_q_blocks = (q_len + sparse_block_size - 1) // sparse_block_size
    num_k_blocks_ring = (kv_len_ring + sparse_block_size - 1) // sparse_block_size
    num_k_blocks_cur = (kv_len_cur + sparse_block_size - 1) // sparse_block_size
    mask = torch.ones(1, 8, num_q_blocks, num_k_blocks_ring + num_k_blocks_cur, dtype=torch.bool)

    out = run_prefill_sparse_multisource(q, k_ring, v_ring, k_cur, v_cur, sparse_block_size, mask)
    ref = _ref_via_concatenated_single_source(q, k_ring, v_ring, k_cur, v_cur, sparse_block_size, mask)
    base.check_close(out, ref)
    print("test_gqa PASSED")


def test_head_size_128():
    sparse_block_size = 128
    q_len, kv_len_ring, kv_len_cur = 256, 256, 128
    q, k_ring, v_ring, k_cur, v_cur = _make_qkv_multisource(
        batch=1, num_heads=4, num_kv_heads=4, q_len=q_len, kv_len_ring=kv_len_ring,
        kv_len_cur=kv_len_cur, head_size=128, seed=15)

    num_q_blocks = (q_len + sparse_block_size - 1) // sparse_block_size
    num_k_blocks_ring = (kv_len_ring + sparse_block_size - 1) // sparse_block_size
    num_k_blocks_cur = (kv_len_cur + sparse_block_size - 1) // sparse_block_size
    mask = torch.ones(1, 4, num_q_blocks, num_k_blocks_ring + num_k_blocks_cur, dtype=torch.bool)

    out = run_prefill_sparse_multisource(q, k_ring, v_ring, k_cur, v_cur, sparse_block_size, mask)
    ref = _ref_via_concatenated_single_source(q, k_ring, v_ring, k_cur, v_cur, sparse_block_size, mask)
    base.check_close(out, ref)
    print("test_head_size_128 PASSED")


if __name__ == "__main__":
    cl.profiling(True)
    test_matches_concatenated_all_true_mask()
    test_matches_concatenated_random_sparse_mask()
    test_realistic_kv_ratio()
    test_empty_ring_chunk0()
    test_gqa()
    test_head_size_128()
    print("ALL PASSED")
