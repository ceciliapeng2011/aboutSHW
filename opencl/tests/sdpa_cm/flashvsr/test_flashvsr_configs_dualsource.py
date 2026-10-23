"""
Dual-source (cache + current) accuracy at FlashVSR's real self-attention head config -- the
dual-vs-single apples-to-apples counterpart of test_flashvsr_configs.py, which only exercises
NUM_HEADS=12/HEAD_SIZE=128 for the single-source topology. Every existing dual-source correctness
test (test_sdpa_prefill_block_mask_dualsource.py, test_sdpa_prefill_int8_cache.py,
flashvsr/test_flashvsr_block_sparse_attention_dualsource.py) proves the kernel at toy dims
(num_heads=4, head_size=64); none of them run at FlashVSR's actual MHA head config. This file closes
that gap by replaying those same proven oracle patterns (dual-source vs. single-source run on the
logically concatenated cache+current, same seed, same mask) at NUM_HEADS=12/HEAD_SIZE=128, across
FlashVSR's two real per-step streaming-KV-cache shapes also used by bench_sdpa_dualsource.py's perf
sweep:
- warmup: kv_len_cache=0 (cache not yet populated), kv_len_cur=q_len.
- steady: kv_len_cache=3*GROUP, kv_len_cur=q_len=GROUP (kv_ratio=3.0, the real per-step shape every
  steady-state self-attention call sees).

GROUP is a proportionally smaller stand-in for FlashVSR's real WINDOW_GROUP_TOKENS=19200 -- the
oracle here is the already-trusted single-source GPU kernel (not a CPU torch reference), so there's
no CPU-feasibility constraint like test_flashvsr_configs.py's; GROUP is kept small purely so this
correctness test suite runs fast.
"""
import os
import sys

import torch

CWD = os.path.dirname(os.path.realpath(__file__))
SDPA_CM_DIR = os.path.dirname(CWD)
if SDPA_CM_DIR not in sys.path:
    sys.path.insert(0, SDPA_CM_DIR)

import test_sdpa_prefill as base    # noqa: E402
import test_sdpa_prefill_block_mask_dualsource as ms    # noqa: E402
import test_sdpa_prefill_int8_cache as int8cache_mod    # noqa: E402

torch.manual_seed(23)

NUM_HEADS = 12       # FlashVSR DiT self-attention: MHA, num_heads=12, head_dim=128.
NUM_KV_HEADS = 12
HEAD_SIZE = 128
SPARSE_BLOCK_SIZE = 128  # FlashVSR's WindowPartition3D window == 128 tokens (2x8x8).
GROUP = 2 * SPARSE_BLOCK_SIZE  # scaled-down stand-in for one WINDOW_GROUP_TOKENS group; see module docstring.


def test_dualsource_fp16_warmup_dense():
    q_len = kv_len_cur = GROUP
    q, k_cache_placeholder, v_cache_placeholder, k_cur, v_cur = ms._make_qkv_dualsource(
        batch=1, num_heads=NUM_HEADS, num_kv_heads=NUM_KV_HEADS, q_len=q_len, kv_len_cache=1,
        kv_len_cur=kv_len_cur, head_size=HEAD_SIZE, seed=101)

    num_q_blocks = (q_len + SPARSE_BLOCK_SIZE - 1) // SPARSE_BLOCK_SIZE
    num_k_blocks_cur = (kv_len_cur + SPARSE_BLOCK_SIZE - 1) // SPARSE_BLOCK_SIZE
    mask = torch.ones(1, NUM_HEADS, num_q_blocks, num_k_blocks_cur, dtype=torch.bool)

    out = ms.run_prefill_sparse_dualsource(q, k_cache_placeholder, v_cache_placeholder, k_cur, v_cur,
                                            SPARSE_BLOCK_SIZE, mask, kv_len_cache_override=0)
    ref = ms.block_mask_mod.run_prefill_sparse(q, k_cur, v_cur, is_causal=False,
                                                sparse_block_size=SPARSE_BLOCK_SIZE, block_mask=mask)
    assert torch.equal(out, ref), "warmup (empty cache) must degrade to current-only, bit-identical to single-source"
    print("test_dualsource_fp16_warmup_dense PASSED")


def test_dualsource_fp16_steady_dense():
    q_len = kv_len_cur = GROUP
    kv_len_cache = 3 * GROUP
    q, k_cache, v_cache, k_cur, v_cur = ms._make_qkv_dualsource(
        batch=1, num_heads=NUM_HEADS, num_kv_heads=NUM_KV_HEADS, q_len=q_len, kv_len_cache=kv_len_cache,
        kv_len_cur=kv_len_cur, head_size=HEAD_SIZE, seed=102)

    num_q_blocks = (q_len + SPARSE_BLOCK_SIZE - 1) // SPARSE_BLOCK_SIZE
    num_k_blocks_cache = (kv_len_cache + SPARSE_BLOCK_SIZE - 1) // SPARSE_BLOCK_SIZE
    num_k_blocks_cur = (kv_len_cur + SPARSE_BLOCK_SIZE - 1) // SPARSE_BLOCK_SIZE
    mask = torch.ones(1, NUM_HEADS, num_q_blocks, num_k_blocks_cache + num_k_blocks_cur, dtype=torch.bool)

    out = ms.run_prefill_sparse_dualsource(q, k_cache, v_cache, k_cur, v_cur, SPARSE_BLOCK_SIZE, mask)
    ref = ms._ref_via_concatenated_single_source(q, k_cache, v_cache, k_cur, v_cur, SPARSE_BLOCK_SIZE, mask)
    assert torch.equal(out, ref), "steady (kv_ratio=3.0), all-True mask: dualsource must be bit-identical to the concatenated single-source kernel"
    print("test_dualsource_fp16_steady_dense PASSED")


def test_dualsource_fp16_steady_sparse():
    q_len = kv_len_cur = GROUP
    kv_len_cache = 3 * GROUP
    q, k_cache, v_cache, k_cur, v_cur = ms._make_qkv_dualsource(
        batch=1, num_heads=NUM_HEADS, num_kv_heads=NUM_KV_HEADS, q_len=q_len, kv_len_cache=kv_len_cache,
        kv_len_cur=kv_len_cur, head_size=HEAD_SIZE, seed=103)

    num_q_blocks = (q_len + SPARSE_BLOCK_SIZE - 1) // SPARSE_BLOCK_SIZE
    num_k_blocks_cache = (kv_len_cache + SPARSE_BLOCK_SIZE - 1) // SPARSE_BLOCK_SIZE
    num_k_blocks_cur = (kv_len_cur + SPARSE_BLOCK_SIZE - 1) // SPARSE_BLOCK_SIZE
    g = torch.Generator().manual_seed(104)
    mask = (torch.rand(1, NUM_HEADS, num_q_blocks, num_k_blocks_cache + num_k_blocks_cur, generator=g)
            < 0.3).to(torch.bool)
    mask[:, :, :, 0] = True  # never an all-false row -- same convention as test_realistic_kv_ratio.

    out = ms.run_prefill_sparse_dualsource(q, k_cache, v_cache, k_cur, v_cur, SPARSE_BLOCK_SIZE, mask)
    ref = ms._ref_via_concatenated_single_source(q, k_cache, v_cache, k_cur, v_cur, SPARSE_BLOCK_SIZE, mask)
    base.check_close(out, ref)
    print("test_dualsource_fp16_steady_sparse PASSED")


def test_dualsource_int8cache_warmup_dense():
    q_len = kv_len_cur = GROUP
    q, k_cache_placeholder, v_cache_placeholder, k_cur, v_cur = int8cache_mod._make_qkv(
        batch=1, num_heads=NUM_HEADS, num_kv_heads=NUM_KV_HEADS, q_len=q_len, kv_len_cache=16,
        kv_len_cur=kv_len_cur, head_size=HEAD_SIZE, seed=105)

    num_q_blocks = (q_len + SPARSE_BLOCK_SIZE - 1) // SPARSE_BLOCK_SIZE
    num_k_blocks_cur = (kv_len_cur + SPARSE_BLOCK_SIZE - 1) // SPARSE_BLOCK_SIZE
    mask = torch.ones(1, NUM_HEADS, num_q_blocks, num_k_blocks_cur, dtype=torch.bool)

    (k_u8, k_scale, k_zp, _), (v_u8, v_scale, v_zp, _) = int8cache_mod._encode_cache_to_int8(
        k_cache_placeholder, v_cache_placeholder, int8cache_mod.KV_CACHE_COMPRESSION_BY_CHANNEL)

    out = int8cache_mod.run_prefill_sparse_dualsource_int8cache(
        q, k_u8, k_scale, k_zp, v_u8, v_scale, v_zp, k_cur, v_cur, SPARSE_BLOCK_SIZE, mask,
        kv_len_cache=0)
    ref = ms.block_mask_mod.run_prefill_sparse(q, k_cur, v_cur, is_causal=False,
                                                sparse_block_size=SPARSE_BLOCK_SIZE, block_mask=mask)
    assert torch.equal(out, ref), "warmup (empty int8 cache) must degrade to current-only, bit-identical to single-source"
    print("test_dualsource_int8cache_warmup_dense PASSED")


def test_dualsource_int8cache_steady_dense():
    q_len = kv_len_cur = GROUP
    kv_len_cache = 3 * GROUP
    q, k_cache, v_cache, k_cur, v_cur = int8cache_mod._make_qkv(
        batch=1, num_heads=NUM_HEADS, num_kv_heads=NUM_KV_HEADS, q_len=q_len, kv_len_cache=kv_len_cache,
        kv_len_cur=kv_len_cur, head_size=HEAD_SIZE, seed=106)

    num_q_blocks = (q_len + SPARSE_BLOCK_SIZE - 1) // SPARSE_BLOCK_SIZE
    num_k_blocks_cache = (kv_len_cache + SPARSE_BLOCK_SIZE - 1) // SPARSE_BLOCK_SIZE
    num_k_blocks_cur = (kv_len_cur + SPARSE_BLOCK_SIZE - 1) // SPARSE_BLOCK_SIZE
    mask = torch.ones(1, NUM_HEADS, num_q_blocks, num_k_blocks_cache + num_k_blocks_cur, dtype=torch.bool)

    int8cache_mod._check_against_roundtripped_oracle(q, k_cache, v_cache, k_cur, v_cur, SPARSE_BLOCK_SIZE, mask)
    print("test_dualsource_int8cache_steady_dense PASSED")


def test_dualsource_int8cache_steady_sparse():
    q_len = kv_len_cur = GROUP
    kv_len_cache = 3 * GROUP
    q, k_cache, v_cache, k_cur, v_cur = int8cache_mod._make_qkv(
        batch=1, num_heads=NUM_HEADS, num_kv_heads=NUM_KV_HEADS, q_len=q_len, kv_len_cache=kv_len_cache,
        kv_len_cur=kv_len_cur, head_size=HEAD_SIZE, seed=107)

    num_q_blocks = (q_len + SPARSE_BLOCK_SIZE - 1) // SPARSE_BLOCK_SIZE
    num_k_blocks_cache = (kv_len_cache + SPARSE_BLOCK_SIZE - 1) // SPARSE_BLOCK_SIZE
    num_k_blocks_cur = (kv_len_cur + SPARSE_BLOCK_SIZE - 1) // SPARSE_BLOCK_SIZE
    g = torch.Generator().manual_seed(108)
    mask = (torch.rand(1, NUM_HEADS, num_q_blocks, num_k_blocks_cache + num_k_blocks_cur, generator=g)
            < 0.3).to(torch.bool)
    mask[:, :, :, 0] = True

    int8cache_mod._check_against_roundtripped_oracle(q, k_cache, v_cache, k_cur, v_cur, SPARSE_BLOCK_SIZE, mask)
    print("test_dualsource_int8cache_steady_sparse PASSED")


if __name__ == "__main__":
    from clops import cl
    cl.profiling(True)
    test_dualsource_fp16_warmup_dense()
    test_dualsource_fp16_steady_dense()
    test_dualsource_fp16_steady_sparse()
    test_dualsource_int8cache_warmup_dense()
    test_dualsource_int8cache_steady_dense()
    test_dualsource_int8cache_steady_sparse()
    print("ALL PASSED")
