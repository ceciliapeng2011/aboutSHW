"""
Correctness cross-check for FlashVSR's draft-block-mask GENERATION only:
  1. GPU adapter (flashvsr_mask_adapter.build_draft_mask_gpu) vs. the vendored torch reference
     (flashvsr_mask_reference.generate_draft_block_mask), on identical random windowed Q/K.
  2. Local-window / streaming-cache invariants.

The block-sparse-ATTENTION end-to-end test (mask -> cm_sdpa_opt_prefill.cm -> compare vs. masked
SDPA reference) lives separately in test_flashvsr_block_sparse_attention.py -- kept apart from
this file so mask-generation correctness and attention-kernel correctness can be run/reasoned
about independently (a failure here means the mask is wrong; a failure there means the kernel
mishandled an otherwise-correct mask).

See /home/ceciliapeng/.claude/plans/flickering-crunching-pascal.md (Stage 3) and
/home/ceciliapeng/repos/reports/block_sparse_attention_for_flashvsr.md for the design this
implements.
"""
import torch

import flashvsr_mask_reference as ref_mod
import flashvsr_mask_adapter as gpu_mod

torch.manual_seed(11)


def _match_frac(a, b):
    return (a == b).float().mean().item()


def make_windowed_qk(spatial_windows, q_groups, k_groups, nheads, head_dim, window_tokens=128,
                      seed=0):
    g = torch.Generator().manual_seed(seed)
    d_full = nheads * head_dim
    num_q_windows = spatial_windows * q_groups
    num_k_windows = spatial_windows * k_groups
    q_w = torch.randn(num_q_windows, window_tokens, d_full, generator=g, dtype=torch.float32)
    k_w = torch.randn(num_k_windows, window_tokens, d_full, generator=g, dtype=torch.float32)
    return q_w, k_w


def _run_both(spatial_h, spatial_w, q_groups, k_groups, nheads, head_dim, local_range, topk,
              window_tokens=128, seed=0):
    spatial_windows = spatial_h * spatial_w
    q_w, k_w = make_windowed_qk(spatial_windows, q_groups, k_groups, nheads, head_dim,
                                 window_tokens, seed)
    local_mask = ref_mod.build_local_block_mask_shifted_vec_normal_slide(
        spatial_h, spatial_w, local_range, local_range, include_self=True)

    ref_mask = ref_mod.generate_draft_block_mask(1, nheads, q_groups, q_w, k_w, topk=topk,
                                                  local_attn_mask=local_mask)
    gpu_mask = gpu_mod.build_draft_mask_gpu(1, nheads, q_groups, q_w, k_w, topk=topk,
                                             local_attn_mask=local_mask)
    return ref_mask, gpu_mask, local_mask, spatial_windows


def test_gpu_matches_reference_small():
    ref_mask, gpu_mask, _, _ = _run_both(spatial_h=3, spatial_w=4, q_groups=2, k_groups=2,
                                          nheads=4, head_dim=16, local_range=3, topk=5,
                                          window_tokens=8, seed=1)
    frac = _match_frac(ref_mask, gpu_mask)
    assert frac >= 0.99, f"match_frac={frac} too low"
    print(f"test_gpu_matches_reference_small PASSED match_frac={frac:.4f}")


def test_gpu_matches_reference_flashvsr_config():
    # FlashVSR real production values: local_range=11, head_dim=128, nheads=12.
    ref_mask, gpu_mask, _, _ = _run_both(spatial_h=5, spatial_w=6, q_groups=3, k_groups=3,
                                          nheads=12, head_dim=128, local_range=11, topk=20,
                                          window_tokens=128, seed=2)
    frac = _match_frac(ref_mask, gpu_mask)
    assert frac >= 0.98, f"match_frac={frac} too low"
    print(f"test_gpu_matches_reference_flashvsr_config PASSED match_frac={frac:.4f}")


def test_streaming_cache_growth():
    # K has more temporal groups than Q (streaming FIFO cache prepended to k_w, wan_video_dit.py:344-346).
    ref_mask, gpu_mask, _, _ = _run_both(spatial_h=4, spatial_w=4, q_groups=1, k_groups=4,
                                          nheads=3, head_dim=32, local_range=3, topk=8,
                                          window_tokens=8, seed=3)
    assert ref_mask.shape == gpu_mask.shape
    frac = _match_frac(ref_mask, gpu_mask)
    assert frac >= 0.98, f"match_frac={frac} too low"
    print(f"test_streaming_cache_growth PASSED shape={tuple(ref_mask.shape)} match_frac={frac:.4f}")


def test_streaming_cache_growth_realistic_ratio():
    # Same q_groups=1/k_groups=4 shape as test_streaming_cache_growth, but at FlashVSR's actual
    # production per-kernel config (nheads=12, head_dim=128, local_range=11, window_tokens=128 --
    # matches test_gpu_matches_reference_flashvsr_config) instead of tiny synthetic values, so this
    # confirms the non-square mask holds at the real production cache depth, not just structurally.
    #
    # q_groups=1/k_groups=4 is FlashVSR's REAL steady-state streaming-cache ratio, not an arbitrary
    # growth example: wan_video_dit.py's SelfAttention.forward (lines 317-382) concatenates
    # pre_cache_k/v to the front of the current chunk's own k_w BEFORE the attention call (lines
    # 344-346), and kv_ratio=3.0's default kv_len=3 window-groups cap (flashvsr_full.py:326 /
    # flashvsr_tiny_long.py:306) means the stabilized cache always holds exactly 3 groups -- so
    # every steady-state self-attention call after warm-up sees q_groups=1 (this chunk) against
    # k_groups=3+1=4 (cache + this chunk), i.e. num_k_blocks == 4 * num_q_blocks at production
    # scale (num_q_blocks=150, num_k_blocks=600 -- non-square, but the mask ABI is already generic
    # w.r.t. num_q_blocks != num_k_blocks, see cm_sdpa_opt_prefill.cm). spatial_h/w kept small
    # (proportionate, not the full 150-window production scale) to stay CPU-feasible, per
    # test_flashvsr_configs.py's "scaled-down but proportionate" convention.
    spatial_h, spatial_w = 5, 6  # 30 windows/group -- same spatial extent as the production-config test
    spatial_windows = spatial_h * spatial_w
    topk = int(spatial_windows * 2.0)  # TOPK_RATIO=2.0 production default
    ref_mask, gpu_mask, _, _ = _run_both(spatial_h=spatial_h, spatial_w=spatial_w, q_groups=1,
                                          k_groups=4, nheads=12, head_dim=128, local_range=11,
                                          topk=topk, window_tokens=128, seed=7)
    assert ref_mask.shape == gpu_mask.shape
    assert gpu_mask.shape[-1] == 4 * gpu_mask.shape[-2], (
        f"expected num_k_blocks == 4*num_q_blocks, got shape={tuple(gpu_mask.shape)}"
    )
    frac = _match_frac(ref_mask, gpu_mask)
    assert frac >= 0.98, f"match_frac={frac} too low"
    print(f"test_streaming_cache_growth_realistic_ratio PASSED shape={tuple(ref_mask.shape)} "
          f"match_frac={frac:.4f}")


def test_local_window_invariant():
    # Every kept block must lie within the local_range spatial neighborhood -- no block outside
    # the additive local mask's support is ever kept, regardless of top-k.
    spatial_h, spatial_w, local_range = 4, 5, 3
    q_groups, k_groups = 2, 2
    _, gpu_mask, local_mask, spatial_windows = _run_both(
        spatial_h, spatial_w, q_groups, k_groups, nheads=2, head_dim=16, local_range=local_range,
        topk=1000000, window_tokens=8, seed=4)  # huge topk -> keep everything the local mask allows
    m = gpu_mask[0]  # [nheads, num_q, num_k]
    num_q = m.shape[1]
    num_k = m.shape[2]
    for l in range(num_q):
        l_sp = l % spatial_windows
        for k_idx in range(num_k):
            k_sp = k_idx % spatial_windows
            if not local_mask[l_sp, k_sp]:
                assert not m[:, l, k_idx].any(), f"block outside local window kept at l={l},k={k_idx}"
    print("test_local_window_invariant PASSED")


if __name__ == "__main__":
    test_gpu_matches_reference_small()
    test_gpu_matches_reference_flashvsr_config()
    test_streaming_cache_growth()
    test_streaming_cache_growth_realistic_ratio()
    test_local_window_invariant()
    print("ALL PASSED")
