"""
Offline numerical check for a proposed FlashVSR DiT self-attention optimization: replacing the
per-chunk `Slice(drop oldest) + Concat(kept, new) + Assign` windowed-KV-cache update (OpenVINO port:
modeling_flashvsr_dit.cpp's `forward_dynamic_fhw`; reference: wan_video_dit.py:344-374, see
/home/ceciliapeng/repos/reports/flashvsr_tiny_kv_cache_carry_zh.md) with a FIXED-CAPACITY RING
BUFFER (`num_slots = kv_len + 1` physical slots, each holding one `one_len`-window temporal group,
overwritten in place via `ScatterUpdate` at `write_ptr % num_slots` instead of being rebuilt by
`Concat` every step).

THE QUESTION THIS FILE ANSWERS: a ring buffer's physical slot order drifts away from chronological
(oldest..newest) order once the write pointer wraps around (see this session's chunk-by-chunk
slot-timeline table -- from the 3rd write onward, reading slots [0..num_slots) in index order no
longer yields the true age order). Self-attention only produces IDENTICAL output regardless of this
reordering if the attention computation is a permutation-invariant SET operation over the KV
sequence. This is NOT obviously true for FlashVSR's real (sparse, masked) attention path, only for
its `compatibility_mode` dense fallback -- so both are tested here, independently:

  1. `compatibility_mode` (wan_video_dit.py:204-209, `flash_attention`'s CUDA-kernel fallback --
     ALSO the path actually deployed by the OV/GPU port, see reports/
     flashvsr_dit_attention_mask_loop_handoff_1.md): plain dense SDPA, no mask. Provably
     permutation-invariant (softmax-weighted sum over KV pairs is a set operation) -- expected PASS.

  2. `sparse_block_attention` (wan_video_dit.py:317-365's real, currently-undeployed-on-GPU path:
     `generate_draft_block_mask`'s joint top-k + `build_local_block_mask_shifted_vec_normal_slide`'s
     purely-spatial local mask, then a masked attention). Only reuses this repo's
     `flashvsr_mask_reference.py` (the vendored, dependency-light -- torch+einops only, no CUDA --
     ground truth for FlashVSR's OWN mask math); deliberately does NOT invoke the CM sparse-prefill
     GPU kernel -- pulling that in would conflate "is this permutation invariant" with "does the
     GPU kernel honor the mask correctly" (the latter is covered separately by
     test_flashvsr_block_sparse_attention.py). The block-mask -> token-mask expansion and the
     masked-SDPA reference below are reimplemented standalone (a few lines; not copied) for that
     reason.

WHY GROUP-LEVEL (not row-level) permutation: `build_local_block_mask_shifted_vec_normal_slide`
takes no temporal/slot argument at all (only h//8,w//8,local_range) and is tiled IDENTICALLY across
every (Q-group,K-group) pair, and `generate_draft_block_mask`'s top-k is a value-threshold over
pooled per-window scores (permutation-EQUIVARIANT, not order-sensitive). Both properties only
support "whole temporal GROUP reordering is safe", not "reordering individual windows/tokens within
a group is safe" -- so every permutation constructed here moves whole `one_len`-window groups
around, exactly matching what the real ring buffer would do, and never splits a group.

Run: /home/ceciliapeng/OCL/aboutSHW/ocl.venv/bin/python3 flashvsr/test_flashvsr_kv_ring_buffer_permutation.py
(from the sdpa_cm/ directory)
(torch+einops only; no OpenCL/CM/GPU device needed -- this file never imports `clops`.)
"""
import torch
import torch.nn.functional as F

import flashvsr_mask_reference as ref_mod

torch.manual_seed(20260923)

# FlashVSR production defaults (wan_video_dit.py / flashvsr_tiny_kv_cache_carry_zh.md Sec 4/7):
# kv_ratio=3.0 -> kv_len=3 kept history groups -> ring buffer needs kv_len+1=4 physical slots
# (this chunk's own new group + 3 kept). topk_ratio=2.0 is the real production top-k formula
# (topk = int(square_num*topk_ratio)). Spatial extent (spatial_h/w) and head count/dim scaled down
# from the real 12-head/128-dim, ~150-window production config for CPU-feasible runtime, same
# "scaled-down but proportionate" convention as test_flashvsr_configs.py.
KV_LEN = 3
NUM_SLOTS = KV_LEN + 1
WINDOW_TOKENS = 128         # win=(2,8,8) -> 2*8*8, real production value, not scaled down.
SPATIAL_H, SPATIAL_W = 4, 4
SPATIAL_WINDOWS = SPATIAL_H * SPATIAL_W
NHEADS, HEAD_DIM = 4, 32
DIM = NHEADS * HEAD_DIM
LOCAL_RANGE = 3
TOPK = int(SPATIAL_WINDOWS * 2.0)  # TOPK_RATIO=2.0 production default.
NUM_STEPS = 6  # enough to exercise >1 full wrap of all NUM_SLOTS=4 slots (first wrap at step 2).


def random_group(seed):
    """One `one_len`-window temporal group, windowed layout: [SPATIAL_WINDOWS, WINDOW_TOKENS, DIM].
    Matches WindowPartition3D.partition's output shape for a single (wf=2) temporal slice
    (wan_video_dit.py:96-102) -- content is arbitrary (this test is about ORDER, not values)."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(SPATIAL_WINDOWS, WINDOW_TOKENS, DIM, generator=g, dtype=torch.float32)


def rope_apply_group(x, abs_pos, theta=10000.0):
    """Bakes an ABSOLUTE-position rotation into `x` (shape [..., DIM], DIM=NHEADS*HEAD_DIM), same
    pairwise-complex convention as the reference's `rope_apply` (wan_video_dit.py:265-270):
    view consecutive (real,imag) pairs within each head as complex, multiply by
    `exp(i*abs_pos*inv_freq)`. Applied ONCE per group at a fixed `abs_pos`, mirroring how the real
    model rotates Q/K exactly once (at the position they were computed), before window-partition
    and before ever touching the KV-cache state -- never re-applied on a later read, regardless of
    what array position the group later occupies. Only Q/K get this (never V, matching
    `v = self.v(x)` in the reference having no `rope_apply` call)."""
    half = HEAD_DIM // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, half, dtype=torch.float64) * 2 / HEAD_DIM))
    freqs = torch.polar(torch.ones(half, dtype=torch.float64), abs_pos * inv_freq)  # [half] complex
    lead = x.shape[:-1]
    xr = x.to(torch.float64).reshape(*lead, NHEADS, half, 2)
    xc = torch.view_as_complex(xr) * freqs  # broadcasts over the trailing `half` axis
    return torch.view_as_real(xc).reshape(*lead, DIM)


def random_rope_group(seed, abs_pos):
    """Q/K group variant: RoPE-rotated at a fixed absolute temporal position, frozen forever after
    (see `rope_apply_group`'s doc comment) -- the actual content the permutation test in
    `test_*_with_rope_*` below stores in cache/reads back, instead of position-agnostic noise."""
    return rope_apply_group(random_group(seed), abs_pos)


def windowed_to_heads(groups):
    """[groups...] each [SPATIAL_WINDOWS, WINDOW_TOKENS, DIM] -> concat -> per-head token layout
    [1, NHEADS, len(groups)*SPATIAL_WINDOWS*WINDOW_TOKENS, HEAD_DIM]. Mirrors wan_video_dit.py's
    `rearrange('(b block_n) (block_s) d -> b (block_n block_s) d', ...)` + the `b s (n d) -> b n s
    d` head-split inside `flash_attention`, done as plain reshape/permute (no einops)."""
    windowed = torch.cat(list(groups), dim=0)  # [num_windows, WINDOW_TOKENS, DIM]
    seq = windowed.reshape(-1, DIM)            # [num_windows*WINDOW_TOKENS, DIM]
    return seq.reshape(seq.shape[0], NHEADS, HEAD_DIM).permute(1, 0, 2).unsqueeze(0)


def block_mask_to_additive(mask, block_size, q_len, kv_len):
    # [B,Hq,num_q_blocks,num_k_blocks] bool -> [B,Hq,q_len,kv_len] additive (0/-inf). Generic
    # block-mask -> token-mask expansion, reimplemented standalone (see module docstring).
    full = mask.to(torch.float32)
    full = full.repeat_interleave(block_size, dim=2).repeat_interleave(block_size, dim=3)
    full = full[:, :, :q_len, :kv_len]
    return torch.where(full > 0, torch.zeros_like(full), torch.full_like(full, float("-inf")))


def dense_compatibility_attention(q, k, v):
    # wan_video_dit.py:204-209 -- no mask, no windowing.
    return F.scaled_dot_product_attention(q.double(), k.double(), v.double())


def sparse_masked_attention(q, k, v, q_groups_windowed, k_groups_windowed, seqlen, local_attn_mask):
    mask = ref_mod.generate_draft_block_mask(1, NHEADS, seqlen, q_groups_windowed, k_groups_windowed,
                                              topk=TOPK, local_attn_mask=local_attn_mask)
    q_len, kv_len = q.shape[2], k.shape[2]
    additive = block_mask_to_additive(mask, WINDOW_TOKENS, q_len, kv_len)
    out = F.scaled_dot_product_attention(q.double(), k.double(), v.double(), attn_mask=additive.double())
    row_has_any = mask[0].sum(dim=-1) > 0  # [NHEADS, num_q_windows] -- see module docstring (2).
    valid = row_has_any.repeat_interleave(WINDOW_TOKENS, dim=1).unsqueeze(0).unsqueeze(-1)
    return torch.where(valid, out, torch.zeros_like(out))


def _slot_timeline(use_rope=False):
    """Replays the real chunk-by-chunk cache update (flashvsr_tiny_kv_cache_carry_zh.md Sec 4) and
    yields, per step, (q_groups, logical_k, logical_v, ring_k, ring_v, n_new) -- `logical_*` is the
    CURRENT Concat-based implementation's chronological order; `ring_*` is what a fixed-capacity
    ring buffer's physical slots (read in index order) would contain after the same update.

    `use_rope=True`: each NEW group's Q/K gets a real RoPE rotation baked in at its own true
    absolute position (`abs_pos`, a counter that only ever increases with group CREATION order,
    never with where the group later sits in `ring_k`/`ring_v` -- exactly matching the real model,
    where `rope_apply` runs once on this chunk's own tokens before the cache is ever touched, see
    `rope_apply_group`'s doc comment). V is never rotated (matches the reference)."""
    logical_k, logical_v = [], []
    ring_k, ring_v = [None] * NUM_SLOTS, [None] * NUM_SLOTS
    write_ptr = 0
    seed = 0
    abs_pos = 0
    for step in range(NUM_STEPS):
        n_new = KV_LEN if step == 0 else 1  # chunk0 (f=6) seeds kv_len groups at once; steady +1.
        new_k, new_v, new_q = [], [], []
        for _ in range(n_new):
            if use_rope:
                new_k.append(random_rope_group(seed, abs_pos))
                new_q.append(random_rope_group(seed + 2, abs_pos))  # same abs_pos as its K.
            else:
                new_k.append(random_group(seed))
                new_q.append(random_group(seed + 2))
            new_v.append(random_group(seed + 1))  # V is never RoPE-rotated (matches reference).
            seed += 3
            abs_pos += 1

        logical_k = (logical_k + new_k)[-NUM_SLOTS:]
        logical_v = (logical_v + new_v)[-NUM_SLOTS:]
        for j in range(n_new):
            slot = (write_ptr + j) % NUM_SLOTS
            ring_k[slot] = new_k[j]
            ring_v[slot] = new_v[j]
        write_ptr = (write_ptr + n_new) % NUM_SLOTS

        yield (new_q, logical_k, logical_v,
               [g for g in ring_k if g is not None], [g for g in ring_v if g is not None], n_new)


def _run_timeline(mode, use_rope=False):
    assert mode in ("compatibility", "sparse")
    local_attn_mask = ref_mod.build_local_block_mask_shifted_vec_normal_slide(
        SPATIAL_H, SPATIAL_W, LOCAL_RANGE, LOCAL_RANGE, include_self=True)

    max_abs_diffs = []
    for step, (q_groups, lk, lv, rk, rv, n_new) in enumerate(_slot_timeline(use_rope=use_rope)):
        q = windowed_to_heads(q_groups)
        k_logical, v_logical = windowed_to_heads(lk), windowed_to_heads(lv)
        k_ring, v_ring = windowed_to_heads(rk), windowed_to_heads(rv)

        if mode == "compatibility":
            out_logical = dense_compatibility_attention(q, k_logical, v_logical)
            out_ring = dense_compatibility_attention(q, k_ring, v_ring)
        else:
            q_win = torch.cat(q_groups, dim=0)
            out_logical = sparse_masked_attention(q, k_logical, v_logical, q_win,
                                                   torch.cat(lk, dim=0), n_new, local_attn_mask)
            out_ring = sparse_masked_attention(q, k_ring, v_ring, q_win,
                                                torch.cat(rk, dim=0), n_new, local_attn_mask)

        diff = (out_logical - out_ring).abs().max().item()
        max_abs_diffs.append(diff)
        wrapped = step >= 2  # NUM_SLOTS=4, KV_LEN=3 -> physical order first diverges at step 2.
        print(f"  [{mode}] step={step} n_new={n_new} wrapped={wrapped} "
              f"num_groups_cached={len(lk)} max_abs_diff={diff:.3e}")
    return max_abs_diffs


def test_compatibility_mode_permutation_invariant():
    diffs = _run_timeline("compatibility")
    # float64 throughout -> any real logic bug shows up orders of magnitude above fp round-off.
    assert max(diffs) < 1e-9, f"compatibility_mode diverged under ring reordering: max={max(diffs):.3e}"
    print("test_compatibility_mode_permutation_invariant PASSED "
          f"(max over {NUM_STEPS} steps = {max(diffs):.3e})")


def test_sparse_block_attention_permutation_invariant():
    diffs = _run_timeline("sparse")
    assert max(diffs) < 1e-9, f"sparse_block_attention diverged under ring reordering: max={max(diffs):.3e}"
    print("test_sparse_block_attention_permutation_invariant PASSED "
          f"(max over {NUM_STEPS} steps = {max(diffs):.3e})")


def test_compatibility_mode_with_rope_permutation_invariant():
    # Same as test_compatibility_mode_permutation_invariant, but Q/K groups carry a REAL RoPE
    # rotation baked in at their own true absolute position (see rope_apply_group/_slot_timeline's
    # doc comments) -- directly answers "does the ring buffer's physical reordering interact with
    # RoPE" empirically, rather than by argument only.
    diffs = _run_timeline("compatibility", use_rope=True)
    assert max(diffs) < 1e-9, f"compatibility_mode+RoPE diverged under ring reordering: max={max(diffs):.3e}"
    print("test_compatibility_mode_with_rope_permutation_invariant PASSED "
          f"(max over {NUM_STEPS} steps = {max(diffs):.3e})")


def test_sparse_block_attention_with_rope_permutation_invariant():
    diffs = _run_timeline("sparse", use_rope=True)
    assert max(diffs) < 1e-9, f"sparse_block_attention+RoPE diverged under ring reordering: max={max(diffs):.3e}"
    print("test_sparse_block_attention_with_rope_permutation_invariant PASSED "
          f"(max over {NUM_STEPS} steps = {max(diffs):.3e})")


if __name__ == "__main__":
    test_compatibility_mode_permutation_invariant()
    test_sparse_block_attention_permutation_invariant()
    test_compatibility_mode_with_rope_permutation_invariant()
    test_sparse_block_attention_with_rope_permutation_invariant()
    print("ALL PASSED")
