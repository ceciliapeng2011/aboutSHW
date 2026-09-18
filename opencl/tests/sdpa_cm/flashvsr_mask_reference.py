"""
Vendored ground truth for FlashVSR's DiT self-attention sparsity mask.

Ported near-verbatim from /home/ceciliapeng/repos/FlashVSR/diffsynth/models/wan_video_dit.py:
  - build_local_block_mask_shifted_vec_normal_slide (:66-90)
  - generate_draft_block_mask (:116-153)

Not imported directly from the FlashVSR repo to avoid pulling in its optional heavy deps
(flash_attn / flash_attn_interface / sageattention) -- same rationale as x_attn/references.py
being the standalone ground truth dense_xattn_adapter.py's tests diff against, rather than
importing X-Attention's own upstream repo.

This module is pure torch/einops (mean/einsum/softmax/topk) -- no custom kernel -- and so IS the
"host-side" mask-generation baseline; flashvsr_mask_adapter.py's GPU kernels are cross-checked
against it directly.
"""
import math

import torch
from einops import rearrange


@torch.no_grad()
def build_local_block_mask_shifted_vec_normal_slide(block_h: int,
                                                      block_w: int,
                                                      win_h: int = 6,
                                                      win_w: int = 6,
                                                      include_self: bool = True,
                                                      device=None) -> torch.Tensor:
    """Static per-resolution spatial adjacency mask: mask[i,j] True iff spatial window j lies
    within a win_h x win_w neighborhood (centered, clamped at borders) of spatial window i.
    Shape: (block_h*block_w, block_h*block_w) bool. Resolution- and local_range-dependent only,
    independent of Q/K content -- callers should cache this per (block_h,block_w,win_h,win_w)
    exactly as wan_video_dit.py's SelfAttention does (self.local_attn_mask)."""
    device = device or torch.device("cpu")
    H, W = block_h, block_w
    r = torch.arange(H, device=device)
    c = torch.arange(W, device=device)
    YY, XX = torch.meshgrid(r, c, indexing="ij")
    r_all = YY.reshape(-1)
    c_all = XX.reshape(-1)
    r_half = win_h // 2
    c_half = win_w // 2
    start_r = r_all - r_half
    end_r = start_r + win_h - 1
    start_c = c_all - c_half
    end_c = start_c + win_w - 1
    in_row = (r_all[None, :] >= start_r[:, None]) & (r_all[None, :] <= end_r[:, None])
    in_col = (c_all[None, :] >= start_c[:, None]) & (c_all[None, :] <= end_c[:, None])
    mask = in_row & in_col
    if not include_self:
        mask.fill_diagonal_(False)
    return mask


@torch.no_grad()
def generate_draft_block_mask(batch_size, nheads, seqlen,
                               q_w, k_w, topk=10, local_attn_mask=None):
    """q_w: (num_q_windows, 128, nheads*head_dim) -- WindowPartition3D-partitioned Q, window =
    (2,8,8) = 128 tokens, full model dim (not yet split per head).
    k_w: (num_k_windows, 128, nheads*head_dim) -- same, for K (may include prepended streaming
    KV-cache windows, so num_k_windows can exceed num_q_windows).
    seqlen: number of Q *temporal* window-groups (= f_q // 2, where f_q is the current chunk's
    frame count and 2 is the temporal window size) -- NOT num_q_windows itself.
    topk: number of K-windows kept, selected GLOBALLY per (head, Q-temporal-group) jointly across
    that group's spatial query windows -- not per Q-window row.
    local_attn_mask: (num_spatial_windows, num_spatial_windows) bool, from
    build_local_block_mask_shifted_vec_normal_slide -- tiled across every (Q-temporal-group,
    K-temporal-group) pair as an additive mask before softmax.

    Returns: (batch_size, nheads, num_q_windows, num_k_windows) bool mask, 1=keep/0=skip --
    directly the ABI cm_sdpa_opt_prefill.cm's SPARSE_BLOCK_SIZE=128 sparse_block_mask expects, no
    reformatting needed.
    """
    assert batch_size == 1, "Only batch_size=1 supported for now"
    assert local_attn_mask is not None, "local_attn_mask must be provided"
    avgpool_q = torch.mean(q_w, dim=1)
    avgpool_k = torch.mean(k_w, dim=1)
    avgpool_q = rearrange(avgpool_q, 's (h d) -> s h d', h=nheads)
    avgpool_k = rearrange(avgpool_k, 's (h d) -> s h d', h=nheads)
    q_heads = avgpool_q.permute(1, 0, 2)
    k_heads = avgpool_k.permute(1, 0, 2)
    D = avgpool_q.shape[-1]
    scores = torch.einsum("hld,hmd->hlm", q_heads, k_heads) / math.sqrt(D)

    repeat_head = scores.shape[0]
    repeat_len = scores.shape[1] // local_attn_mask.shape[0]
    repeat_num = scores.shape[2] // local_attn_mask.shape[1]
    local_attn_mask = local_attn_mask.unsqueeze(1).unsqueeze(0).repeat(repeat_len, 1, repeat_num, 1)
    local_attn_mask = rearrange(local_attn_mask, 'x a y b -> (x a) (y b)')
    local_attn_mask = local_attn_mask.unsqueeze(0).repeat(repeat_head, 1, 1)
    local_attn_mask = local_attn_mask.to(torch.float32)
    local_attn_mask = local_attn_mask.masked_fill(local_attn_mask == False, -float('inf'))
    local_attn_mask = local_attn_mask.masked_fill(local_attn_mask == True, 0)
    scores = scores + local_attn_mask

    attn_map = torch.softmax(scores, dim=-1)
    attn_map = rearrange(attn_map, 'h (it s1) s2 -> (h it) s1 s2', it=seqlen)
    loop_num, s1, s2 = attn_map.shape
    flat = attn_map.reshape(loop_num, -1)
    apply_topk = min(flat.shape[1] - 1, topk)
    thresholds = torch.topk(flat, k=apply_topk + 1, dim=1, largest=True).values[:, -1]
    thresholds = thresholds.unsqueeze(1)
    mask_new = (flat > thresholds).reshape(loop_num, s1, s2)
    mask_new = rearrange(mask_new, '(h it) s1 s2 -> h (it s1) s2', it=seqlen)
    mask = mask_new.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    return mask
