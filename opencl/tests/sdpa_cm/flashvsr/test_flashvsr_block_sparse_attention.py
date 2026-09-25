"""
End-to-end block-sparse-ATTENTION correctness for FlashVSR: mask (from
flashvsr_mask_adapter.build_draft_mask_gpu) -> cm_sdpa_opt_prefill.cm's sparse_block_mask ABI ->
compare against a masked torch SDPA reference using the SAME mask.

Deliberately separate from test_flashvsr_mask.py (which only checks that the mask itself is
correct, independent of any attention kernel): this file isolates "did cm_sdpa_opt_prefill.cm
honor an already-correct mask" from "is the mask itself accurate" -- the same separation of
concerns test_sdpa_prefill_block_mask.py uses for its own generic bypass-correctness test.

See /home/ceciliapeng/.claude/plans/flickering-crunching-pascal.md (Stage 3) and
/home/ceciliapeng/repos/reports/block_sparse_attention_for_flashvsr.md for the design this
implements.
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
import test_sdpa_prefill as base    # noqa: E402
import test_sdpa_prefill_block_mask as block_mask_mod    # noqa: E402
import test_flashvsr_mask as mask_tests

torch.manual_seed(12)


def test_end_to_end_sparse_attention():
    spatial_h, spatial_w, local_range = 3, 3, 3
    q_groups = k_groups = 2
    nheads, head_dim, window_tokens = 4, 64, 128
    spatial_windows = spatial_h * spatial_w
    # topk_ratio=2.0 matches wan_video_dit.py's real production formula (square_num*topk_ratio,
    # square_num=spatial_windows) -- keeps ~2 K-windows/row on average so a small synthetic test
    # doesn't degenerate into all-masked rows (NaN in softmax) the way an arbitrarily small topk
    # legitimately can under FlashVSR's own joint (not per-row) top-k semantics.
    topk = int(spatial_windows * 2.0)

    q_w, k_w = mask_tests.make_windowed_qk(spatial_windows, q_groups, k_groups, nheads, head_dim,
                                            window_tokens, seed=5)
    local_mask = ref_mod.build_local_block_mask_shifted_vec_normal_slide(
        spatial_h, spatial_w, local_range, local_range, include_self=True)
    mask = gpu_mod.build_draft_mask_gpu(1, nheads, q_groups, q_w, k_w, topk=topk,
                                         local_attn_mask=local_mask)

    num_q_windows = q_w.shape[0]
    num_k_windows = k_w.shape[0]
    q_len = num_q_windows * window_tokens
    kv_len = num_k_windows * window_tokens

    g = torch.Generator().manual_seed(6)
    low, high = -2, 3
    q = torch.randint(low, high, [1, nheads, q_len, head_dim], generator=g).to(torch.float16)
    k = torch.randint(low, high, [1, nheads, kv_len, head_dim], generator=g).to(torch.float16)
    v = (torch.randint(low, high, [1, nheads, kv_len, head_dim], generator=g).to(torch.float16)
         / high)

    out = block_mask_mod.run_prefill_sparse(q, k, v, is_causal=False, sparse_block_size=window_tokens,
                                             block_mask=mask)
    ref = block_mask_mod.ref_sdpa_sparse(q, k, v, is_causal=False, mask=mask,
                                          sparse_block_size=window_tokens)

    # A Q-window with zero kept K-windows is a legitimate outcome of the joint (not per-row)
    # top-k on uncorrelated synthetic test data (confirmed identical between the reference and
    # GPU adapter in test_flashvsr_mask.py) -- softmax(all -inf) = 0/0 = NaN there, in both the
    # kernel and the torch reference alike. Exclude those token rows from the numeric comparison;
    # the mask itself is validated separately in test_flashvsr_mask.py.
    row_has_any = mask[0].sum(dim=-1) > 0  # [nheads, num_q_windows]
    valid = row_has_any.repeat_interleave(window_tokens, dim=1).unsqueeze(0).unsqueeze(-1)
    out_valid = torch.where(valid, out.float(), torch.zeros_like(out, dtype=torch.float32))
    ref_valid = torch.where(valid, ref.float(), torch.zeros_like(ref, dtype=torch.float32))
    base.check_close(out_valid, ref_valid)
    print("test_end_to_end_sparse_attention PASSED")


def test_end_to_end_sparse_attention_realistic_ratio():
    # Mirrors test_end_to_end_sparse_attention but at FlashVSR's REAL steady-state streaming-cache
    # ratio: q_groups=1 (this chunk only) vs. k_groups=4 (kv_len=3 cached groups + this chunk's own
    # 1 group, wan_video_dit.py:317-382, kv_ratio=3.0 default). At production scale this yields a
    # NON-SQUARE mask (num_q_blocks=150 vs. num_k_blocks=600) -- confirms cm_sdpa_opt_prefill.cm's
    # sparse_block_mask ABI (already generic w.r.t. num_q_blocks != num_k_blocks) handles this
    # specific ratio end-to-end, not just the mask-generation side (that's
    # test_flashvsr_mask.test_streaming_cache_growth_realistic_ratio).
    spatial_h, spatial_w, local_range = 3, 3, 3
    q_groups, k_groups = 1, 4
    nheads, head_dim, window_tokens = 4, 64, 128
    spatial_windows = spatial_h * spatial_w
    topk = int(spatial_windows * 2.0)  # TOPK_RATIO=2.0 production default, see sibling test

    q_w, k_w = mask_tests.make_windowed_qk(spatial_windows, q_groups, k_groups, nheads, head_dim,
                                            window_tokens, seed=8)
    local_mask = ref_mod.build_local_block_mask_shifted_vec_normal_slide(
        spatial_h, spatial_w, local_range, local_range, include_self=True)
    mask = gpu_mod.build_draft_mask_gpu(1, nheads, q_groups, q_w, k_w, topk=topk,
                                         local_attn_mask=local_mask)
    assert mask.shape[-1] == 4 * mask.shape[-2], (
        f"expected num_k_blocks == 4*num_q_blocks, got shape={tuple(mask.shape)}"
    )

    num_q_windows = q_w.shape[0]
    num_k_windows = k_w.shape[0]
    q_len = num_q_windows * window_tokens
    kv_len = num_k_windows * window_tokens

    g = torch.Generator().manual_seed(9)
    low, high = -2, 3
    q = torch.randint(low, high, [1, nheads, q_len, head_dim], generator=g).to(torch.float16)
    k = torch.randint(low, high, [1, nheads, kv_len, head_dim], generator=g).to(torch.float16)
    v = (torch.randint(low, high, [1, nheads, kv_len, head_dim], generator=g).to(torch.float16)
         / high)

    out = block_mask_mod.run_prefill_sparse(q, k, v, is_causal=False, sparse_block_size=window_tokens,
                                             block_mask=mask)
    ref = block_mask_mod.ref_sdpa_sparse(q, k, v, is_causal=False, mask=mask,
                                          sparse_block_size=window_tokens)

    # See test_end_to_end_sparse_attention's comment: all-masked Q-rows are a legitimate synthetic
    # -data outcome, excluded from the numeric comparison rather than papered over.
    row_has_any = mask[0].sum(dim=-1) > 0  # [nheads, num_q_windows]
    valid = row_has_any.repeat_interleave(window_tokens, dim=1).unsqueeze(0).unsqueeze(-1)
    out_valid = torch.where(valid, out.float(), torch.zeros_like(out, dtype=torch.float32))
    ref_valid = torch.where(valid, ref.float(), torch.zeros_like(ref, dtype=torch.float32))
    base.check_close(out_valid, ref_valid)
    print("test_end_to_end_sparse_attention_realistic_ratio PASSED "
          f"shape={tuple(mask.shape)}")


if __name__ == "__main__":
    test_end_to_end_sparse_attention()
    test_end_to_end_sparse_attention_realistic_ratio()
    print("ALL PASSED")
