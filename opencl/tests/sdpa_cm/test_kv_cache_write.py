"""
fp16 correctness for kv_cache_write.cm's CACHE_COMPRESSION=0 (default) path: scatters
`key_cur`/`value_cur` (this chunk's own new whole temporal groups) into `key_cache`/`value_cache` at
`write_indices` physical slots, in place -- the CM-kernel mirror of the real BlockSparseAttention
primitive's `ring_write_k`/`ring_write_v` stages (see kv_cache_write.cm's module docstring).

Correctness oracle: a plain torch scatter (`cache[write_indices] = updates`) -- bit-identical,
since this is a pure fp16 copy, no numerics involved.

Also covers the read-before-write ordering invariant the real primitive's stage ordering
guarantees: `cm_sdpa_opt_prefill_dualsource` (the prefill/read stage) always runs BEFORE
`kv_cache_write` (the write-back stage) within one chunk, so a chunk's own new tokens never leak
into its own attention read.
"""
import functools

import numpy as np
import torch

from clops import cl

import test_sdpa_prefill as base
import test_sdpa_prefill_block_mask_dualsource as ms

torch.manual_seed(50)


@functools.cache
def build_cache_write_kernel(num_kv_heads, head_size):
    src = r'''#include "kv_cache_write.cm"'''
    opts = (
        f'-cmc -Qxcm_jit_option="-abortonspill" -Qxcm_register_file_size=256 -mCM_printregusage '
        f"-I{base.CWD} -DKERNEL_NAME=kv_cache_write "
        f"-DNUM_KV_HEADS={num_kv_heads} -DHEAD_SIZE={head_size}"
    )
    return cl.kernels(src, opts)


def enqueue_cache_write(t_updates_np, write_indices, t_cache, num_kv_heads, head_size, lk_cache=None):
    """Lower-level entry point operating on an ALREADY-BUILT device tensor `t_cache` (mutated in
    place, same object returned) -- lets a caller reuse one cache tensor across multiple enqueues
    (e.g. a prefill read followed by a cache_write, both against the same buffer, as
    test_read_before_write_ordering does). t_updates_np: numpy half array
    [n_new, tokens_per_group, num_kv_heads*head_size]. lk_cache: the cache's total physical
    token-row count (num_slots*tokens_per_group), used by the kernel to bounds-check
    write_indices -- defaults to t_cache's own slot/token-per-group extent when t_cache is a
    [num_slots, tokens_per_group, ...] shaped device tensor (not available for a raw flat buffer,
    in which case the caller must pass it explicitly)."""
    n_new, tokens_per_group, cache_write_dim = t_updates_np.shape
    assert cache_write_dim == num_kv_heads * head_size
    kernel = build_cache_write_kernel(num_kv_heads, head_size)

    t_updates = cl.tensor(np.ascontiguousarray(t_updates_np))
    t_write_indices = cl.tensor(write_indices.contiguous().to(torch.int32).numpy())
    if lk_cache is None:
        num_slots = t_cache.numpy().reshape(-1, tokens_per_group, cache_write_dim).shape[0]
        lk_cache = num_slots * tokens_per_group

    GWS = [cache_write_dim // 16, tokens_per_group, n_new]
    LWS = [1, 1, 1]
    kernel.enqueue("kv_cache_write", GWS, LWS,
                    t_updates, t_write_indices, t_cache, int(tokens_per_group), int(n_new), int(lk_cache))
    return t_cache


def run_cache_write(updates, write_indices, cache, num_kv_heads, head_size):
    """updates: [n_new, tokens_per_group, num_kv_heads*head_size] half torch tensor.
    write_indices: [n_new] int32 torch tensor (destination slot per group).
    cache: [num_slots, tokens_per_group, num_kv_heads*head_size] half torch tensor -- NOT modified
    (the device buffer is written in place, but this torch tensor is only the initial upload copy;
    callers read the returned tensor, never this input, for the post-write contents).
    Returns: (num_slots, tokens_per_group, num_kv_heads*head_size) half torch tensor -- `cache`
    after the scatter.
    """
    t_cache = cl.tensor(cache.contiguous().to(torch.float16).numpy())
    enqueue_cache_write(updates.contiguous().to(torch.float16).numpy(), write_indices, t_cache,
                        num_kv_heads, head_size)
    return torch.from_numpy(t_cache.numpy())


def _make_inputs(num_slots, n_new, tokens_per_group, num_kv_heads, head_size, seed):
    g = torch.Generator().manual_seed(seed)
    cache = torch.randn(num_slots, tokens_per_group, num_kv_heads * head_size, generator=g, dtype=torch.float32).to(torch.float16)
    updates = torch.randn(n_new, tokens_per_group, num_kv_heads * head_size, generator=g, dtype=torch.float32).to(torch.float16)
    write_indices = torch.randperm(num_slots, generator=g)[:n_new].to(torch.int32)
    return cache, updates, write_indices


def test_scatter_matches_torch_reference():
    num_slots, n_new, tokens_per_group, num_kv_heads, head_size = 4, 2, 128, 4, 64
    cache, updates, write_indices = _make_inputs(num_slots, n_new, tokens_per_group, num_kv_heads, head_size, seed=1)

    out = run_cache_write(updates, write_indices, cache, num_kv_heads, head_size)

    ref = cache.clone()
    ref[write_indices.long()] = updates
    assert torch.equal(out, ref), "cache-write scatter must be bit-identical to a plain torch scatter"
    print("test_scatter_matches_torch_reference PASSED")


def test_all_slots_written():
    # Every slot gets overwritten in one call (n_new == num_slots) -- exercises write_indices being
    # a full permutation, not just a subset.
    num_slots, tokens_per_group, num_kv_heads, head_size = 4, 128, 4, 64
    cache, updates, write_indices = _make_inputs(num_slots, num_slots, tokens_per_group, num_kv_heads, head_size, seed=2)

    out = run_cache_write(updates, write_indices, cache, num_kv_heads, head_size)

    ref = cache.clone()
    ref[write_indices.long()] = updates
    assert torch.equal(out, ref)
    print("test_all_slots_written PASSED")


def test_single_group_write():
    num_slots, tokens_per_group, num_kv_heads, head_size = 4, 128, 4, 128
    cache, updates, write_indices = _make_inputs(num_slots, 1, tokens_per_group, num_kv_heads, head_size, seed=3)

    out = run_cache_write(updates, write_indices, cache, num_kv_heads, head_size)

    ref = cache.clone()
    ref[write_indices.long()] = updates
    assert torch.equal(out, ref)
    print("test_single_group_write PASSED")


def test_read_before_write_ordering():
    # The real primitive's stage ordering guarantee: within one chunk, the prefill (read) stage
    # always runs BEFORE the cache_write (write) stage, against the SAME key_cache/value_cache
    # device buffer -- so attention sees the cache as it was at the START of the chunk, and only
    # afterwards does cache_write fold this chunk's own new tokens in (for the NEXT chunk's
    # benefit). Proven end-to-end here with ONE shared pair of device tensors (t_k_cache/t_v_cache),
    # reused (not rebuilt) across three enqueues: prefill(old cache) -> cache_write(scatter k_cur/
    # v_cur into it) -> prefill(same tensors, now mutated) -- matching TRANSPOSE_ORDER_SWAPPED
    # ([B,S,H,D], heads-inner-per-token), the layout both the real primitive and kv_cache_write.cm's
    # own `cache` shape assume.
    #
    # tokens_per_group=1/num_slots=kv_len_cache (one cache_write "group" per cache TOKEN, not per
    # temporal window) keeps this test's write_indices a simple token-position scatter, without
    # needing FlashVSR's own window-grouping concept -- kv_cache_write.cm's scatter mechanics don't
    # care what a "group" conceptually represents.
    sparse_block_size = 128
    q_len, kv_len_cache, kv_len_cur = 128, 128, 64
    num_heads, num_kv_heads, head_size = 4, 4, 64
    O = base.TRANSPOSE_ORDER_SWAPPED

    q, k_cache, v_cache, k_cur, v_cur = ms._make_qkv_dualsource(
        batch=1, num_heads=num_heads, num_kv_heads=num_kv_heads, q_len=q_len, kv_len_cache=kv_len_cache,
        kv_len_cur=kv_len_cur, head_size=head_size, seed=30)

    num_q_blocks = (q_len + sparse_block_size - 1) // sparse_block_size
    num_k_blocks_cache = (kv_len_cache + sparse_block_size - 1) // sparse_block_size
    num_k_blocks_cur = (kv_len_cur + sparse_block_size - 1) // sparse_block_size
    mask = torch.ones(1, num_heads, num_q_blocks, num_k_blocks_cache + num_k_blocks_cur, dtype=torch.bool)

    prefill_kernel = ms.build_dualsource_sparse_prefill_kernel(
        num_heads, num_kv_heads, head_size, sparse_block_size, O, O, O, O)
    q_step = base.get_cm_grf_width() // 32
    wg_size = 16
    wg_count = (q_len + wg_size * q_step - 1) // (wg_size * q_step)
    GWS = [1, num_heads, wg_count * wg_size]
    LWS = [1, 1, wg_size]
    t_mask = cl.tensor(mask.numpy())

    def enqueue_prefill(t_k_cache, t_v_cache):
        t_q = cl.tensor(base.to_physical(q, O).to(torch.float16).numpy())
        t_k_cur = cl.tensor(base.to_physical(k_cur, O).to(torch.float16).numpy())
        t_v_cur = cl.tensor(base.to_physical(v_cur, O).to(torch.float16).numpy())
        t_out = cl.tensor(np.zeros([1, q_len, num_heads, head_size], dtype=np.float16))
        prefill_kernel.enqueue(
            "cm_sdpa_opt_prefill_dualsource", GWS, LWS,
            t_q, t_k_cache, t_v_cache, t_k_cur, t_v_cur, t_out, q_len, kv_len_cache, kv_len_cur,
            t_mask, num_q_blocks, num_k_blocks_cache, num_k_blocks_cur)
        return base.from_physical(torch.from_numpy(t_out.numpy()), O, 1, num_heads, q_len, head_size)

    k_cache_phys = base.to_physical(k_cache, O).to(torch.float16)  # [1, kv_len_cache, Hkv, D]
    v_cache_phys = base.to_physical(v_cache, O).to(torch.float16)
    t_k_cache = cl.tensor(k_cache_phys.numpy())
    t_v_cache = cl.tensor(v_cache_phys.numpy())

    out_before = enqueue_prefill(t_k_cache, t_v_cache)

    # Scatter k_cur/v_cur (one "group" per token) into the FIRST kv_len_cur token slots of the
    # SAME cache tensors -- in place, reusing t_k_cache/t_v_cache rather than rebuilding them.
    k_cur_flat = base.to_physical(k_cur, O).to(torch.float16).reshape(kv_len_cur, 1, num_kv_heads * head_size).numpy()
    v_cur_flat = base.to_physical(v_cur, O).to(torch.float16).reshape(kv_len_cur, 1, num_kv_heads * head_size).numpy()
    write_indices = torch.arange(kv_len_cur, dtype=torch.int32)
    enqueue_cache_write(k_cur_flat, write_indices, t_k_cache, num_kv_heads, head_size)
    enqueue_cache_write(v_cur_flat, write_indices, t_v_cache, num_kv_heads, head_size)

    # Re-running prefill against the SAME (now-mutated) t_k_cache/t_v_cache tensors must reflect the
    # write: bit-identical to independently running the trusted prefill wrapper against a
    # host-side cache where the first kv_len_cur tokens were overwritten with k_cur/v_cur the same
    # way (never rebuilt/re-derived from t_k_cache/t_v_cache themselves, so this is a genuine
    # independent check, not circular).
    out_after = enqueue_prefill(t_k_cache, t_v_cache)

    k_cache_written = k_cache.clone()
    v_cache_written = v_cache.clone()
    k_cache_written[:, :, :kv_len_cur, :] = k_cur
    v_cache_written[:, :, :kv_len_cur, :] = v_cur
    ref_after = ms.run_prefill_sparse_dualsource(
        q, k_cache_written, v_cache_written, k_cur, v_cur, sparse_block_size, mask,
        q_order=O, k_order=O, v_order=O, o_order=O)

    assert not torch.equal(out_before, out_after), (
        "sanity: overwriting half the cache with different data should change the (all-True-mask) "
        "attention output -- if this fires, the write below silently didn't happen"
    )
    assert torch.equal(out_after, ref_after), (
        "prefill run AFTER cache_write must reflect the write -- read-before-write ordering means "
        "the FIRST read (out_before) must be unaffected, but any SUBSEQUENT read must see it"
    )
    print("test_read_before_write_ordering PASSED")


if __name__ == "__main__":
    cl.profiling(True)
    test_scatter_matches_torch_reference()
    test_all_slots_written()
    test_single_group_write()
    test_read_before_write_ordering()
    print("ALL PASSED")
