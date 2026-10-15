"""
On-device (CM kernel) implementation of FlashVSR's generate_draft_block_mask, driving
draft_block_mask_pool.cm, draft_block_mask_score_softmax.cm and draft_block_mask_topk_mask.cm
(ported from the generic -- no longer flashvsr_-prefixed -- impls/cm/draft_block_mask/ kernels
upstream; entry points are still named flashvsr_pool/flashvsr_score_softmax/flashvsr_topk_mask
here via -DKERNEL_NAME=... at JIT time, so this adapter's own enqueue() call sites are unchanged).
Same signature/semantics as flashvsr_mask_reference.generate_draft_block_mask -- cross-checked
against it directly in test_flashvsr_mask.py.

Pipeline (mirrors the reference function's own stages, see draft_block_mask_pool.cm's,
draft_block_mask_score_softmax.cm's and draft_block_mask_topk_mask.cm's module docstrings for the
per-kernel math/design notes):
  1. flashvsr_pool          x2  (once for q_w, once for k_w) -> pooled_q, pooled_k
  2. flashvsr_score_softmax      -> probs (float; local-mask-added score + softmax fused)
  3. flashvsr_topk_mask          -> mask (uint8, 1=keep/0=skip)
"""
import functools
import math
import os

import numpy as np
import torch

from clops import cl

CWD = os.path.dirname(os.path.realpath(__file__))

# Must match draft_block_mask_topk_mask.cm's TOPK_LOCAL_SIZE default -- number of work items per
# workgroup cooperatively binary-searching one (head, Q-temporal-group)'s top-k threshold.
TOPK_LOCAL_SIZE = 32


@functools.cache
def _get_pool_kernel(num_heads, head_dim, window_tokens):
    src = r'''#include "draft_block_mask_pool.cm"'''
    opts = (
        f'-cmc -Qxcm_jit_option="-abortonspill" -Qxcm_register_file_size=256 -mCM_printregusage '
        f"-I{CWD} -DKERNEL_NAME=flashvsr_pool "
        f"-DNUM_HEADS={num_heads} -DHEAD_DIM={head_dim} -DWINDOW_TOKENS={window_tokens}"
    )
    return cl.kernels(src, opts)


@functools.cache
def _get_score_softmax_kernel(num_heads, head_dim):
    src = r'''#include "draft_block_mask_score_softmax.cm"'''
    opts = (
        f'-cmc -Qxcm_jit_option="-abortonspill" -Qxcm_register_file_size=256 -mCM_printregusage '
        f"-I{CWD} -DKERNEL_NAME=flashvsr_score_softmax "
        f"-DNUM_HEADS={num_heads} -DHEAD_DIM={head_dim}"
    )
    return cl.kernels(src, opts)


@functools.cache
def _get_topk_kernel(num_heads):
    src = r'''#include "draft_block_mask_topk_mask.cm"'''
    opts = (
        f'-cmc -Qxcm_jit_option="-abortonspill" -Qxcm_register_file_size=256 -mCM_printregusage '
        f"-I{CWD} -DKERNEL_NAME=flashvsr_topk_mask -DNUM_HEADS={num_heads} "
        f"-DTOPK_LOCAL_SIZE={TOPK_LOCAL_SIZE}"
    )
    return cl.kernels(src, opts)


def _pool(kernel, windowed_np, num_windows, num_heads, head_dim):
    t_in = cl.tensor(windowed_np)
    t_out = cl.tensor(np.zeros((num_windows, num_heads, head_dim), dtype=np.float16))
    kernel.enqueue("flashvsr_pool", [num_windows, num_heads, 1], [1, 1, 1], t_in, t_out)
    return t_out


def build_draft_mask_gpu(batch_size, nheads, seqlen, q_w, k_w, topk=10, local_attn_mask=None):
    """Same signature/contract as flashvsr_mask_reference.generate_draft_block_mask.

    q_w: (num_q_windows, window_tokens, nheads*head_dim) fp16/fp32 torch tensor.
    k_w: (num_k_windows, window_tokens, nheads*head_dim) -- may have more windows than q_w
         (streaming KV-cache prepend).
    local_attn_mask: (spatial_windows, spatial_windows) bool torch tensor.
    topk: kept elements per (head, Q-temporal-group), selected globally (see
          draft_block_mask_topk_mask.cm).

    Returns: (batch_size, nheads, num_q_windows, num_k_windows) bool torch tensor.
    """
    assert local_attn_mask is not None, "local_attn_mask must be provided"

    num_q_windows, window_tokens, d_full = q_w.shape
    num_k_windows = k_w.shape[0]
    assert k_w.shape[1] == window_tokens and k_w.shape[2] == d_full
    assert d_full % nheads == 0
    head_dim = d_full // nheads
    spatial_windows = local_attn_mask.shape[0]
    # local_attn_mask may be [spatial_windows, spatial_windows_k] -- spatial_windows_k only
    # differs from spatial_windows for a streaming decoder whose accumulated KV cache has grown
    # past the current query window (see draft_block_mask_score_softmax.cm).
    spatial_windows_k = local_attn_mask.shape[1]
    assert num_q_windows % seqlen == 0, "num_q_windows must be seqlen * spatial_windows"
    assert num_q_windows // seqlen == spatial_windows, (
        "local_attn_mask's spatial extent must match q_w's windows-per-temporal-group"
    )

    pool_kernel = _get_pool_kernel(nheads, head_dim, window_tokens)
    score_softmax_kernel = _get_score_softmax_kernel(nheads, head_dim)
    topk_kernel = _get_topk_kernel(nheads)

    q_w_np = q_w.to(torch.float16).contiguous().numpy()
    k_w_np = k_w.to(torch.float16).contiguous().numpy()

    t_pooled_q = _pool(pool_kernel, q_w_np, num_q_windows, nheads, head_dim)
    t_pooled_k = _pool(pool_kernel, k_w_np, num_k_windows, nheads, head_dim)

    local_mask_np = local_attn_mask.to(torch.bool).to(torch.uint8).contiguous().numpy()
    t_local_mask = cl.tensor(local_mask_np)
    t_raw_scores = cl.tensor(np.zeros((nheads, num_q_windows, num_k_windows), dtype=np.float32))
    # float, not half: draft_block_mask_score_softmax.cm now emits full float32 probabilities so
    # the downstream top-k radix-select has enough precision to avoid ties (see
    # draft_block_mask_topk_mask.cm's header).
    t_probs = cl.tensor(np.zeros((nheads, num_q_windows, num_k_windows), dtype=np.float32))

    scale = 1.0 / math.sqrt(head_dim)
    score_softmax_kernel.enqueue(
        "flashvsr_score_softmax", [num_q_windows, nheads, 1], [1, 1, 1],
        t_pooled_q, t_pooled_k, t_local_mask, t_raw_scores, t_probs,
        num_q_windows, num_k_windows, spatial_windows, spatial_windows_k, scale,
    )

    num_groups = seqlen
    t_mask = cl.tensor(np.zeros((batch_size, nheads, num_q_windows, num_k_windows), dtype=np.uint8))
    topk_local_size = TOPK_LOCAL_SIZE
    topk_kernel.enqueue(
        "flashvsr_topk_mask", [nheads, num_groups, topk_local_size], [1, 1, topk_local_size],
        t_probs, t_mask, num_q_windows, num_k_windows, spatial_windows, int(topk), int(batch_size),
    )

    mask = torch.from_numpy(t_mask.numpy()).to(torch.bool)
    return mask  # -> (batch_size, nheads, num_q_windows, num_k_windows)
