"""
Correctness check for flashvsr_mask_adapter.build_draft_mask_gpu_dualsource: pooling cache and
current K windows from two INDEPENDENT device buffers must produce a bit-identical mask to
build_draft_mask_gpu's single-source path given the logically concatenated K (cache windows first,
current windows second).

See /home/ceciliapeng/repos/reports/flashvsr/flashvsr_dit_kvcache_int8_3slot_fused_dualsource_design_zh.md
Sec 10.4 for the real primitive's `input[1]=key_cache.<layer>` (cache) + `input[2]=k_new_f16`
(current) two-source ABI this mirrors -- here only the (tiny, already-pooled) mask-generation side
is exercised; test_flashvsr_block_sparse_attention_dualsource.py exercises the (large, raw K/V)
attention side.
"""
import os
import sys

import torch

CWD = os.path.dirname(os.path.realpath(__file__))
SDPA_CM_DIR = os.path.dirname(CWD)
if SDPA_CM_DIR not in sys.path:
    sys.path.insert(0, SDPA_CM_DIR)

import flashvsr_mask_reference as ref_mod
import flashvsr_mask_adapter as gpu_mod
import test_flashvsr_mask as mask_tests

torch.manual_seed(13)


def _check_dualsource_matches_concatenated(spatial_h, spatial_w, local_range,
                                             q_groups, cache_groups, cur_groups,
                                             nheads, head_dim, window_tokens, topk_ratio, seed):
    spatial_windows = spatial_h * spatial_w
    k_groups = cache_groups + cur_groups
    topk = int(spatial_windows * topk_ratio)

    q_w, k_w = mask_tests.make_windowed_qk(spatial_windows, q_groups, k_groups, nheads, head_dim,
                                            window_tokens, seed=seed)
    num_cache_windows = cache_groups * spatial_windows
    k_cache_w = k_w[:num_cache_windows]
    k_cur_w = k_w[num_cache_windows:]

    local_mask = ref_mod.build_local_block_mask_shifted_vec_normal_slide(
        spatial_h, spatial_w, local_range, local_range, include_self=True)

    mask_single = gpu_mod.build_draft_mask_gpu(1, nheads, q_groups, q_w, k_w, topk=topk,
                                                local_attn_mask=local_mask)
    mask_multi = gpu_mod.build_draft_mask_gpu_dualsource(1, nheads, q_groups, q_w, k_cache_w, k_cur_w,
                                                           topk=topk, local_attn_mask=local_mask)

    assert mask_multi.shape == mask_single.shape, (
        f"shape mismatch: multi={tuple(mask_multi.shape)} single={tuple(mask_single.shape)}"
    )
    assert torch.equal(mask_multi, mask_single), "dualsource mask must be bit-identical to single-source"


def test_small_square():
    _check_dualsource_matches_concatenated(
        spatial_h=3, spatial_w=3, local_range=3, q_groups=2, cache_groups=1, cur_groups=1,
        nheads=4, head_dim=64, window_tokens=128, topk_ratio=2.0, seed=5)
    print("test_small_square PASSED")


def test_realistic_kv_ratio():
    # Mirrors FlashVSR's real steady-state streaming-cache shape: q_groups=1 (this chunk only),
    # cache_groups=3 (kv_ratio=3.0 cached history), cur_groups=1 (this chunk's own new K) --
    # num_k_windows == 4*num_q_windows, same production ratio as
    # test_flashvsr_block_sparse_attention.test_end_to_end_sparse_attention_realistic_ratio.
    _check_dualsource_matches_concatenated(
        spatial_h=3, spatial_w=3, local_range=3, q_groups=1, cache_groups=3, cur_groups=1,
        nheads=4, head_dim=64, window_tokens=128, topk_ratio=2.0, seed=8)
    print("test_realistic_kv_ratio PASSED")


def test_empty_cache_chunk0():
    # chunk0/warmup: cache not yet populated (num_cache_windows=0) -- all K comes from "current"
    # (the chunk's own large warmup window). Must degrade cleanly to a single-source pool call.
    _check_dualsource_matches_concatenated(
        spatial_h=3, spatial_w=3, local_range=3, q_groups=3, cache_groups=0, cur_groups=3,
        nheads=4, head_dim=64, window_tokens=128, topk_ratio=2.0, seed=11)
    print("test_empty_cache_chunk0 PASSED")


def test_valid_slots_partially_warmed_cache():
    # HAS_CACHE_VALID_MASK: a FIXED-CAPACITY cache buffer (physical shape never shrinks -- see
    # draft_block_mask_score_softmax.cm's doc comment) where only the first `valid_slots` of its
    # `num_slots` physical groups are actually real data so far (the rest is garbage, never
    # written). Must be bit-identical to calling the already-trusted dualsource path with the
    # cache truncated to just its valid prefix (cache_groups=valid_slots) -- this is exactly the
    # chunk0/not-fully-warmed-cache scenario the upstream PagedAttention-style cache bugfix
    # targets (avoids ever needing a genuinely zero/short-length cache tensor).
    spatial_h, spatial_w, local_range = 3, 3, 3
    spatial_windows = spatial_h * spatial_w
    q_groups, num_slots, valid_slots, cur_groups = 1, 4, 2, 1
    nheads, head_dim, window_tokens, topk_ratio, seed = 4, 64, 128, 2.0, 19
    topk = int(spatial_windows * topk_ratio)

    q_w, k_w_full = mask_tests.make_windowed_qk(
        spatial_windows, q_groups, num_slots + cur_groups, nheads, head_dim, window_tokens, seed=seed)
    num_full_cache_windows = num_slots * spatial_windows
    k_cache_full_w = k_w_full[:num_full_cache_windows]  # full-capacity cache, only first valid_slots real
    k_cur_w = k_w_full[num_full_cache_windows:]

    local_mask = ref_mod.build_local_block_mask_shifted_vec_normal_slide(
        spatial_h, spatial_w, local_range, local_range, include_self=True)

    mask_masked = gpu_mod.build_draft_mask_gpu_dualsource(
        1, nheads, q_groups, q_w, k_cache_full_w, k_cur_w, topk=topk, local_attn_mask=local_mask,
        valid_slots=valid_slots, num_slots=num_slots)

    num_valid_cache_windows = valid_slots * spatial_windows
    k_cache_valid_w = k_cache_full_w[:num_valid_cache_windows]
    mask_ref = gpu_mod.build_draft_mask_gpu_dualsource(
        1, nheads, q_groups, q_w, k_cache_valid_w, k_cur_w, topk=topk, local_attn_mask=local_mask)

    # mask_masked carries extra (always-False) columns for the invalid cache windows that mask_ref
    # never had in the first place (full cache capacity vs. cache truncated to its valid prefix) --
    # compare only the columns both share (valid cache + cur).
    valid_cols = list(range(num_valid_cache_windows)) + list(range(num_full_cache_windows, num_full_cache_windows + k_cur_w.shape[0]))
    assert torch.equal(mask_masked[..., valid_cols], mask_ref), (
        "valid_slots-masked dualsource mask must match the cache-truncated-to-valid-prefix reference"
    )
    assert not mask_masked[..., num_valid_cache_windows:num_full_cache_windows].any(), (
        "invalid (not-yet-warmed) cache columns must never be selected by top-k"
    )
    print("test_valid_slots_partially_warmed_cache PASSED")


if __name__ == "__main__":
    test_small_square()
    test_realistic_kv_ratio()
    test_empty_cache_chunk0()
    test_valid_slots_partially_warmed_cache()
    print("ALL PASSED")
