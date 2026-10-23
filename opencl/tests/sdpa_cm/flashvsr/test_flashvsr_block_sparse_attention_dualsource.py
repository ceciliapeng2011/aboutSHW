"""
End-to-end dual-source (cache + current) block-sparse ATTENTION correctness for FlashVSR: mask
(from flashvsr_mask_adapter.build_draft_mask_gpu_dualsource, Sec 1 of
flashvsr_dit_kvcache_int8_3slot_fused_dualsource_design_zh.md's implementation) ->
cm_sdpa_opt_prefill_dualsource.cm's two-source sparse_block_mask ABI (Sec 2) -> compare against
the existing single-source end-to-end path (test_flashvsr_block_sparse_attention.py) run on the
logically concatenated cache+current K/V and mask.

Mirrors test_flashvsr_block_sparse_attention.py's test_end_to_end_sparse_attention_realistic_ratio
exactly in shape (FlashVSR's real kv_ratio=3.0 streaming-cache ratio: current = this chunk's own 1
window-group, cache = 3 cached history groups) but with cache/current never concatenated anywhere on
the GPU side of this test -- only this file's own reference computation (via
test_flashvsr_block_sparse_attention.py's already-validated helpers) concatenates them, as the
oracle.
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
import test_sdpa_prefill_block_mask_dualsource as dualsource_mod
import test_flashvsr_mask as mask_tests

torch.manual_seed(14)


def test_end_to_end_sparse_attention_dualsource_realistic_ratio():
    spatial_h, spatial_w, local_range = 3, 3, 3
    q_groups, cache_groups, cur_groups = 1, 3, 1
    nheads, head_dim, window_tokens = 4, 64, 128
    spatial_windows = spatial_h * spatial_w
    k_groups = cache_groups + cur_groups
    topk = int(spatial_windows * 2.0)  # TOPK_RATIO=2.0 production default

    q_w, k_w = mask_tests.make_windowed_qk(spatial_windows, q_groups, k_groups, nheads, head_dim,
                                            window_tokens, seed=16)
    num_cache_windows = cache_groups * spatial_windows
    k_cache_w = k_w[:num_cache_windows]
    k_cur_w = k_w[num_cache_windows:]

    local_mask = ref_mod.build_local_block_mask_shifted_vec_normal_slide(
        spatial_h, spatial_w, local_range, local_range, include_self=True)

    # ---- mask: two independent pooling sources (cache, current), never concatenated raw ----
    mask = gpu_mod.build_draft_mask_gpu_dualsource(1, nheads, q_groups, q_w, k_cache_w, k_cur_w,
                                                     topk=topk, local_attn_mask=local_mask)
    assert mask.shape[-1] == 4 * mask.shape[-2], (
        f"expected num_k_blocks (cache+cur) == 4*num_q_blocks, got shape={tuple(mask.shape)}"
    )

    num_q_windows = q_w.shape[0]
    num_cache_k_windows = k_cache_w.shape[0]
    num_cur_k_windows = k_cur_w.shape[0]
    q_len = num_q_windows * window_tokens
    kv_len_cache = num_cache_k_windows * window_tokens
    kv_len_cur = num_cur_k_windows * window_tokens

    g = torch.Generator().manual_seed(17)
    low, high = -2, 3
    q = torch.randint(low, high, [1, nheads, q_len, head_dim], generator=g).to(torch.float16)
    k_cache = torch.randint(low, high, [1, nheads, kv_len_cache, head_dim], generator=g).to(torch.float16)
    v_cache = (torch.randint(low, high, [1, nheads, kv_len_cache, head_dim], generator=g).to(torch.float16)
              / high)
    k_cur = torch.randint(low, high, [1, nheads, kv_len_cur, head_dim], generator=g).to(torch.float16)
    v_cur = (torch.randint(low, high, [1, nheads, kv_len_cur, head_dim], generator=g).to(torch.float16)
             / high)

    # ---- attention: two independent K/V sources, never concatenated raw ----
    out = dualsource_mod.run_prefill_sparse_dualsource(
        q, k_cache, v_cache, k_cur, v_cur, sparse_block_size=window_tokens, block_mask=mask)

    # ---- oracle: concatenate cache+current (this test's own reference only) and run the already-
    # trusted single-source kernel + torch reference on that ----
    k = torch.cat([k_cache, k_cur], dim=2)
    v = torch.cat([v_cache, v_cur], dim=2)
    ref = dualsource_mod.block_mask_mod.ref_sdpa_sparse(q, k, v, is_causal=False, mask=mask,
                                                          sparse_block_size=window_tokens)

    # A Q-window with zero kept K-windows is a legitimate outcome of the joint top-k on
    # uncorrelated synthetic test data (see test_flashvsr_block_sparse_attention.py's identical
    # comment) -- excluded from the numeric comparison, not papered over.
    row_has_any = mask[0].sum(dim=-1) > 0  # [nheads, num_q_windows]
    valid = row_has_any.repeat_interleave(window_tokens, dim=1).unsqueeze(0).unsqueeze(-1)
    out_valid = torch.where(valid, out.float(), torch.zeros_like(out, dtype=torch.float32))
    ref_valid = torch.where(valid, ref.float(), torch.zeros_like(ref, dtype=torch.float32))
    dualsource_mod.base.check_close(out_valid, ref_valid)
    print("test_end_to_end_sparse_attention_dualsource_realistic_ratio PASSED "
          f"shape={tuple(mask.shape)}")


if __name__ == "__main__":
    test_end_to_end_sparse_attention_dualsource_realistic_ratio()
    print("ALL PASSED")
