"""
Correctness check for flashvsr_mask_adapter.build_draft_mask_gpu_multisource: pooling ring and
current K windows from two INDEPENDENT device buffers must produce a bit-identical mask to
build_draft_mask_gpu's single-source path given the logically concatenated K (ring windows first,
current windows second).

See /home/ceciliapeng/repos/reports/flashvsr/flashvsr_dit_kvcache_int8_3slot_fused_multisource_design_zh.md
Sec 10.4 for the real primitive's `input[1]=key_cache.<layer>` (ring) + `input[2]=k_new_f16`
(current) two-source ABI this mirrors -- here only the (tiny, already-pooled) mask-generation side
is exercised; test_flashvsr_block_sparse_attention_multisource.py exercises the (large, raw K/V)
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


def _check_multisource_matches_concatenated(spatial_h, spatial_w, local_range,
                                             q_groups, ring_groups, cur_groups,
                                             nheads, head_dim, window_tokens, topk_ratio, seed):
    spatial_windows = spatial_h * spatial_w
    k_groups = ring_groups + cur_groups
    topk = int(spatial_windows * topk_ratio)

    q_w, k_w = mask_tests.make_windowed_qk(spatial_windows, q_groups, k_groups, nheads, head_dim,
                                            window_tokens, seed=seed)
    num_ring_windows = ring_groups * spatial_windows
    k_ring_w = k_w[:num_ring_windows]
    k_cur_w = k_w[num_ring_windows:]

    local_mask = ref_mod.build_local_block_mask_shifted_vec_normal_slide(
        spatial_h, spatial_w, local_range, local_range, include_self=True)

    mask_single = gpu_mod.build_draft_mask_gpu(1, nheads, q_groups, q_w, k_w, topk=topk,
                                                local_attn_mask=local_mask)
    mask_multi = gpu_mod.build_draft_mask_gpu_multisource(1, nheads, q_groups, q_w, k_ring_w, k_cur_w,
                                                           topk=topk, local_attn_mask=local_mask)

    assert mask_multi.shape == mask_single.shape, (
        f"shape mismatch: multi={tuple(mask_multi.shape)} single={tuple(mask_single.shape)}"
    )
    assert torch.equal(mask_multi, mask_single), "multisource mask must be bit-identical to single-source"


def test_small_square():
    _check_multisource_matches_concatenated(
        spatial_h=3, spatial_w=3, local_range=3, q_groups=2, ring_groups=1, cur_groups=1,
        nheads=4, head_dim=64, window_tokens=128, topk_ratio=2.0, seed=5)
    print("test_small_square PASSED")


def test_realistic_kv_ratio():
    # Mirrors FlashVSR's real steady-state streaming-cache shape: q_groups=1 (this chunk only),
    # ring_groups=3 (kv_ratio=3.0 cached history), cur_groups=1 (this chunk's own new K) --
    # num_k_windows == 4*num_q_windows, same production ratio as
    # test_flashvsr_block_sparse_attention.test_end_to_end_sparse_attention_realistic_ratio.
    _check_multisource_matches_concatenated(
        spatial_h=3, spatial_w=3, local_range=3, q_groups=1, ring_groups=3, cur_groups=1,
        nheads=4, head_dim=64, window_tokens=128, topk_ratio=2.0, seed=8)
    print("test_realistic_kv_ratio PASSED")


def test_empty_ring_chunk0():
    # chunk0/warmup: ring not yet populated (num_ring_windows=0) -- all K comes from "current"
    # (the chunk's own large warmup window). Must degrade cleanly to a single-source pool call.
    _check_multisource_matches_concatenated(
        spatial_h=3, spatial_w=3, local_range=3, q_groups=3, ring_groups=0, cur_groups=3,
        nheads=4, head_dim=64, window_tokens=128, topk_ratio=2.0, seed=11)
    print("test_empty_ring_chunk0 PASSED")


if __name__ == "__main__":
    test_small_square()
    test_realistic_kv_ratio()
    test_empty_ring_chunk0()
    print("ALL PASSED")
