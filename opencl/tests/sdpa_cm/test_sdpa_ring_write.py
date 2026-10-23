"""
fp16 correctness for block_sparse_attention_ring_write.cm's RING_CACHE_COMPRESSION=0 (default)
path: scatters `key_cur`/`value_cur` (this chunk's own new whole temporal groups) into
`key_ring`/`value_ring` at `write_indices` physical slots, in place -- the CM-kernel mirror of the
real BlockSparseAttention primitive's `ring_write_k`/`ring_write_v` stages (see
block_sparse_attention_ring_write.cm's module docstring).

Correctness oracle: a plain torch scatter (`cache[write_indices] = updates`) -- bit-identical,
since this is a pure fp16 copy, no numerics involved.

Also covers the read-before-write ordering invariant the real primitive's stage ordering
guarantees: `cm_sdpa_opt_prefill_dualsource` (the prefill/read stage) always runs BEFORE
`block_sparse_attention_ring_write` (the write-back stage) within one chunk, so a chunk's own new
tokens never leak into its own attention read.
"""
import functools

import numpy as np
import torch

from clops import cl

import test_sdpa_prefill as base
import test_sdpa_prefill_block_mask_dualsource as ms

torch.manual_seed(50)


@functools.cache
def build_ring_write_kernel(num_kv_heads, head_size):
    src = r'''#include "block_sparse_attention_ring_write.cm"'''
    opts = (
        f'-cmc -Qxcm_jit_option="-abortonspill" -Qxcm_register_file_size=256 -mCM_printregusage '
        f"-I{base.CWD} -DKERNEL_NAME=block_sparse_attention_ring_write "
        f"-DNUM_KV_HEADS={num_kv_heads} -DHEAD_SIZE={head_size}"
    )
    return cl.kernels(src, opts)


def enqueue_ring_write(t_updates_np, write_indices, t_cache, num_kv_heads, head_size):
    """Lower-level entry point operating on an ALREADY-BUILT device tensor `t_cache` (mutated in
    place, same object returned) -- lets a caller reuse one ring tensor across multiple enqueues
    (e.g. a prefill read followed by a ring_write, both against the same buffer, as
    test_read_before_write_ordering does). t_updates_np: numpy half array
    [n_new, tokens_per_group, num_kv_heads*head_size]."""
    n_new, tokens_per_group, ring_write_dim = t_updates_np.shape
    assert ring_write_dim == num_kv_heads * head_size
    kernel = build_ring_write_kernel(num_kv_heads, head_size)

    t_updates = cl.tensor(np.ascontiguousarray(t_updates_np))
    t_write_indices = cl.tensor(write_indices.contiguous().to(torch.int32).numpy())

    GWS = [ring_write_dim // 16, tokens_per_group, n_new]
    LWS = [1, 1, 1]
    kernel.enqueue("block_sparse_attention_ring_write", GWS, LWS,
                    t_updates, t_write_indices, t_cache, int(tokens_per_group), int(n_new))
    return t_cache


def run_ring_write(updates, write_indices, cache, num_kv_heads, head_size):
    """updates: [n_new, tokens_per_group, num_kv_heads*head_size] half torch tensor.
    write_indices: [n_new] int32 torch tensor (destination slot per group).
    cache: [num_slots, tokens_per_group, num_kv_heads*head_size] half torch tensor -- NOT modified
    (the device buffer is written in place, but this torch tensor is only the initial upload copy;
    callers read the returned tensor, never this input, for the post-write contents).
    Returns: (num_slots, tokens_per_group, num_kv_heads*head_size) half torch tensor -- `cache`
    after the scatter.
    """
    t_cache = cl.tensor(cache.contiguous().to(torch.float16).numpy())
    enqueue_ring_write(updates.contiguous().to(torch.float16).numpy(), write_indices, t_cache,
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

    out = run_ring_write(updates, write_indices, cache, num_kv_heads, head_size)

    ref = cache.clone()
    ref[write_indices.long()] = updates
    assert torch.equal(out, ref), "ring-write scatter must be bit-identical to a plain torch scatter"
    print("test_scatter_matches_torch_reference PASSED")


def test_all_slots_written():
    # Every slot gets overwritten in one call (n_new == num_slots) -- exercises write_indices being
    # a full permutation, not just a subset.
    num_slots, tokens_per_group, num_kv_heads, head_size = 4, 128, 4, 64
    cache, updates, write_indices = _make_inputs(num_slots, num_slots, tokens_per_group, num_kv_heads, head_size, seed=2)

    out = run_ring_write(updates, write_indices, cache, num_kv_heads, head_size)

    ref = cache.clone()
    ref[write_indices.long()] = updates
    assert torch.equal(out, ref)
    print("test_all_slots_written PASSED")


def test_single_group_write():
    num_slots, tokens_per_group, num_kv_heads, head_size = 4, 128, 4, 128
    cache, updates, write_indices = _make_inputs(num_slots, 1, tokens_per_group, num_kv_heads, head_size, seed=3)

    out = run_ring_write(updates, write_indices, cache, num_kv_heads, head_size)

    ref = cache.clone()
    ref[write_indices.long()] = updates
    assert torch.equal(out, ref)
    print("test_single_group_write PASSED")


def test_read_before_write_ordering():
    # The real primitive's stage ordering guarantee: within one chunk, the prefill (read) stage
    # always runs BEFORE the ring_write (write) stage, against the SAME key_ring/value_ring device
    # buffer -- so attention sees the ring as it was at the START of the chunk, and only
    # afterwards does ring_write fold this chunk's own new tokens in (for the NEXT chunk's
    # benefit). Proven end-to-end here with ONE shared pair of device tensors (t_k_ring/t_v_ring),
    # reused (not rebuilt) across three enqueues: prefill(old ring) -> ring_write(scatter k_cur/
    # v_cur into it) -> prefill(same tensors, now mutated) -- matching TRANSPOSE_ORDER_SWAPPED
    # ([B,S,H,D], heads-inner-per-token), the layout both the real primitive and
    # block_sparse_attention_ring_write.cm's own `cache` shape assume.
    #
    # tokens_per_group=1/num_slots=kv_len_ring (one ring_write "group" per ring TOKEN, not per
    # temporal window) keeps this test's write_indices a simple token-position scatter, without
    # needing FlashVSR's own window-grouping concept -- block_sparse_attention_ring_write.cm's
    # scatter mechanics don't care what a "group" conceptually represents.
    sparse_block_size = 128
    q_len, kv_len_ring, kv_len_cur = 128, 128, 64
    num_heads, num_kv_heads, head_size = 4, 4, 64
    O = base.TRANSPOSE_ORDER_SWAPPED

    q, k_ring, v_ring, k_cur, v_cur = ms._make_qkv_dualsource(
        batch=1, num_heads=num_heads, num_kv_heads=num_kv_heads, q_len=q_len, kv_len_cache=kv_len_ring,
        kv_len_cur=kv_len_cur, head_size=head_size, seed=30)

    num_q_blocks = (q_len + sparse_block_size - 1) // sparse_block_size
    num_k_blocks_ring = (kv_len_ring + sparse_block_size - 1) // sparse_block_size
    num_k_blocks_cur = (kv_len_cur + sparse_block_size - 1) // sparse_block_size
    mask = torch.ones(1, num_heads, num_q_blocks, num_k_blocks_ring + num_k_blocks_cur, dtype=torch.bool)

    prefill_kernel = ms.build_dualsource_sparse_prefill_kernel(
        num_heads, num_kv_heads, head_size, sparse_block_size, O, O, O, O)
    q_step = base.get_cm_grf_width() // 32
    wg_size = 16
    wg_count = (q_len + wg_size * q_step - 1) // (wg_size * q_step)
    GWS = [1, num_heads, wg_count * wg_size]
    LWS = [1, 1, wg_size]
    t_mask = cl.tensor(mask.numpy())

    def enqueue_prefill(t_k_ring, t_v_ring):
        t_q = cl.tensor(base.to_physical(q, O).to(torch.float16).numpy())
        t_k_cur = cl.tensor(base.to_physical(k_cur, O).to(torch.float16).numpy())
        t_v_cur = cl.tensor(base.to_physical(v_cur, O).to(torch.float16).numpy())
        t_out = cl.tensor(np.zeros([1, q_len, num_heads, head_size], dtype=np.float16))
        prefill_kernel.enqueue(
            "cm_sdpa_opt_prefill_dualsource", GWS, LWS,
            t_q, t_k_ring, t_v_ring, t_k_cur, t_v_cur, t_out, q_len, kv_len_ring, kv_len_cur,
            t_mask, num_q_blocks, num_k_blocks_ring, num_k_blocks_cur)
        return base.from_physical(torch.from_numpy(t_out.numpy()), O, 1, num_heads, q_len, head_size)

    k_ring_phys = base.to_physical(k_ring, O).to(torch.float16)  # [1, kv_len_ring, Hkv, D]
    v_ring_phys = base.to_physical(v_ring, O).to(torch.float16)
    t_k_ring = cl.tensor(k_ring_phys.numpy())
    t_v_ring = cl.tensor(v_ring_phys.numpy())

    out_before = enqueue_prefill(t_k_ring, t_v_ring)

    # Scatter k_cur/v_cur (one "group" per token) into the FIRST kv_len_cur token slots of the
    # SAME ring tensors -- in place, reusing t_k_ring/t_v_ring rather than rebuilding them.
    k_cur_flat = base.to_physical(k_cur, O).to(torch.float16).reshape(kv_len_cur, 1, num_kv_heads * head_size).numpy()
    v_cur_flat = base.to_physical(v_cur, O).to(torch.float16).reshape(kv_len_cur, 1, num_kv_heads * head_size).numpy()
    write_indices = torch.arange(kv_len_cur, dtype=torch.int32)
    enqueue_ring_write(k_cur_flat, write_indices, t_k_ring, num_kv_heads, head_size)
    enqueue_ring_write(v_cur_flat, write_indices, t_v_ring, num_kv_heads, head_size)

    # Re-running prefill against the SAME (now-mutated) t_k_ring/t_v_ring tensors must reflect the
    # write: bit-identical to independently running the trusted prefill wrapper against a
    # host-side ring where the first kv_len_cur tokens were overwritten with k_cur/v_cur the same
    # way (never rebuilt/re-derived from t_k_ring/t_v_ring themselves, so this is a genuine
    # independent check, not circular).
    out_after = enqueue_prefill(t_k_ring, t_v_ring)

    k_ring_written = k_ring.clone()
    v_ring_written = v_ring.clone()
    k_ring_written[:, :, :kv_len_cur, :] = k_cur
    v_ring_written[:, :, :kv_len_cur, :] = v_cur
    ref_after = ms.run_prefill_sparse_dualsource(
        q, k_ring_written, v_ring_written, k_cur, v_cur, sparse_block_size, mask,
        q_order=O, k_order=O, v_order=O, o_order=O)

    assert not torch.equal(out_before, out_after), (
        "sanity: overwriting half the ring with different data should change the (all-True-mask) "
        "attention output -- if this fires, the write below silently didn't happen"
    )
    assert torch.equal(out_after, ref_after), (
        "prefill run AFTER ring_write must reflect the write -- read-before-write ordering means "
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
