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
import sys

import numpy as np
import torch

from clops import cl

CWD = os.path.dirname(os.path.realpath(__file__))
# hf8_e4m3_ref.py (the fp8e4m3<->fp16 host reference/LUT-builder) lives one level up, in sdpa_cm/
# itself, not flashvsr/ -- add it to sys.path the same way the test_flashvsr_* files in this
# directory already do for their own sdpa_cm-root imports, so this module stays importable
# regardless of which directory the importing script's own sys.path already covers.
SDPA_CM_DIR = os.path.dirname(CWD)
if SDPA_CM_DIR not in sys.path:
    sys.path.insert(0, SDPA_CM_DIR)

from hf8_e4m3_ref import build_hf8_decode_lut_f16

# Must match draft_block_mask_topk_mask.cm's TOPK_LOCAL_SIZE default -- number of work items per
# workgroup cooperatively binary-searching one (head, Q-temporal-group)'s top-k threshold.
TOPK_LOCAL_SIZE = 32


@functools.cache
def _get_pool_kernel(num_heads, head_dim, window_tokens, k_fp8e4m3=False):
    """k_fp8e4m3: compiles the K_FP8E4M3=1 variant of draft_block_mask_pool.cm (fp8e4m3-packed
    `windowed` input, decode-on-read via the LUT-gather path, extra hf8_lut kernel param). Only
    ever meaningful for a K-pooling dispatch -- Q is never quantized in this project, so Q-pooling
    always goes through the default k_fp8e4m3=False (plain half) variant; see pool_k_fp8e4m3
    below."""
    src = r'''#include "draft_block_mask_pool.cm"'''
    opts = (
        f'-cmc -Qxcm_jit_option="-abortonspill" -Qxcm_register_file_size=256 -mCM_printregusage '
        f"-I{CWD} -DKERNEL_NAME=flashvsr_pool "
        f"-DNUM_HEADS={num_heads} -DHEAD_DIM={head_dim} -DWINDOW_TOKENS={window_tokens} "
        f"-DK_FP8E4M3={int(k_fp8e4m3)}"
    )
    return cl.kernels(src, opts)


@functools.cache
def _get_hf8_lut_tensor():
    """Host-built 256-entry half decode LUT (see hf8_e4m3_ref.build_hf8_decode_lut_f16) for the
    K_FP8E4M3 pooling path's hf8_lut kernel param. A tiny (512B) constant table shared by every
    fp8 K-pooling dispatch regardless of shape, so it's built/uploaded once and cached like a
    compiled kernel rather than re-uploaded per call."""
    return cl.tensor(build_hf8_decode_lut_f16())


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


def _pool(kernel, windowed_np, num_windows, num_heads, head_dim, hf8_lut_tensor=None):
    t_in = cl.tensor(windowed_np)
    t_out = cl.tensor(np.zeros((num_windows, num_heads, head_dim), dtype=np.float16))
    if hf8_lut_tensor is not None:
        kernel.enqueue("flashvsr_pool", [num_windows, num_heads, 1], [1, 1, 1], t_in, hf8_lut_tensor, t_out)
    else:
        kernel.enqueue("flashvsr_pool", [num_windows, num_heads, 1], [1, 1, 1], t_in, t_out)
    return t_out


def pool_k_fp8e4m3(k_w_fp8_bytes, num_k_windows, nheads, head_dim, window_tokens):
    """Opt-in fp8e4m3-packed-input K-pooling dispatch: K_FP8E4M3=1 compiled variant of
    draft_block_mask_pool.cm, decoding each raw K byte on the fly (LUT-gather, see
    hf8_e4m3_decode_to_f16_lut in draft_block_mask_pool.cm) instead of mean-pooling an
    already-materialized half copy of the windowed K buffer. Q is never quantized in this project
    (only ring K/V ever is) -- there is no fp8 counterpart for Q, callers must keep routing Q
    through the existing half `_pool`/`build_draft_mask_gpu*` path.

    k_w_fp8_bytes: (num_k_windows, window_tokens, nheads*head_dim) uint8 array of fp8e4m3-encoded
    bytes (see hf8_e4m3_ref.encode_f16_to_hf8).
    Returns: (num_k_windows, nheads, head_dim) half cl.tensor -- bit-identical (see
    test_flashvsr_mask_pool_fp8.py) to running the unmodified half kernel on the
    host-decoded-back-to-half version of the same bytes.
    """
    kernel = _get_pool_kernel(nheads, head_dim, window_tokens, k_fp8e4m3=True)
    lut = _get_hf8_lut_tensor()
    return _pool(kernel, k_w_fp8_bytes, num_k_windows, nheads, head_dim, hf8_lut_tensor=lut)


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


def build_draft_mask_gpu_multisource(batch_size, nheads, seqlen, q_w, k_ring_w, k_cur_w,
                                      topk=10, local_attn_mask=None):
    """Two-source variant of build_draft_mask_gpu: mirrors the real DraftBlockMask primitive's
    `input[1]=key_cache.<layer>` (ring, persistent) + `input[2]=k_new_f16` (current, this chunk's
    own new K) ABI (see flashvsr_dit_kvcache_int8_3slot_fused_multisource_design_zh.md Sec 10.4) --
    `k_ring_w` and `k_cur_w` are two INDEPENDENT buffers, never concatenated on the host or on
    device before pooling.

    k_ring_w: (num_ring_windows, window_tokens, nheads*head_dim) -- ring/history K windows.
    k_cur_w:  (num_cur_windows, window_tokens, nheads*head_dim) -- this chunk's own new K windows.
    Combined k_w semantics (num_k_windows = num_ring_windows + num_cur_windows, ring windows first)
    are otherwise identical to build_draft_mask_gpu's single k_w -- only the pooling stage (Stage 1,
    draft_block_mask_pool.cm) ever reads the raw windowed K; that kernel already treats its input as
    an arbitrary-length windowed buffer, so pooling ring and current separately into two slices of
    one pooled_k buffer is sufficient -- score_softmax/topk_mask (Stages 2/3) are reused UNCHANGED,
    operating on the combined (but tiny, already-pooled) pooled_k exactly as build_draft_mask_gpu
    does today.

    Returns: (batch_size, nheads, num_q_windows, num_ring_windows+num_cur_windows) bool torch
    tensor -- bit-identical to build_draft_mask_gpu(..., k_w=torch.cat([k_ring_w, k_cur_w])) (see
    test_flashvsr_mask_multisource.py).
    """
    assert local_attn_mask is not None, "local_attn_mask must be provided"

    num_q_windows, window_tokens, d_full = q_w.shape
    num_ring_windows = k_ring_w.shape[0]
    num_cur_windows = k_cur_w.shape[0]
    assert k_ring_w.shape[1:] == (window_tokens, d_full)
    assert k_cur_w.shape[1:] == (window_tokens, d_full)
    assert d_full % nheads == 0
    head_dim = d_full // nheads
    spatial_windows = local_attn_mask.shape[0]
    spatial_windows_k = local_attn_mask.shape[1]
    assert num_q_windows % seqlen == 0, "num_q_windows must be seqlen * spatial_windows"
    assert num_q_windows // seqlen == spatial_windows, (
        "local_attn_mask's spatial extent must match q_w's windows-per-temporal-group"
    )

    num_k_windows = num_ring_windows + num_cur_windows

    pool_kernel = _get_pool_kernel(nheads, head_dim, window_tokens)
    score_softmax_kernel = _get_score_softmax_kernel(nheads, head_dim)
    topk_kernel = _get_topk_kernel(nheads)

    q_w_np = q_w.to(torch.float16).contiguous().numpy()
    k_ring_w_np = k_ring_w.to(torch.float16).contiguous().numpy()
    k_cur_w_np = k_cur_w.to(torch.float16).contiguous().numpy()

    t_pooled_q = _pool(pool_kernel, q_w_np, num_q_windows, nheads, head_dim)

    # Ring and current are pooled from two SEPARATE device buffers (never concatenated, raw or
    # windowed) via two independent dispatches of the unchanged draft_block_mask_pool.cm kernel.
    # cl.tensor.numpy() is a device->host copy (not a view, see clops/cl/cl.cpp's tensor::to_numpy),
    # so the two pooled results are assembled into one CONTIGUOUS HOST array first and uploaded as
    # a single device tensor -- only the tiny pooled buffer is ever concatenated, never raw K.
    pooled_k_np = np.zeros((num_k_windows, nheads, head_dim), dtype=np.float16)
    if num_ring_windows > 0:
        t_pooled_k_ring = _pool(pool_kernel, k_ring_w_np, num_ring_windows, nheads, head_dim)
        pooled_k_np[0:num_ring_windows] = t_pooled_k_ring.numpy()
    if num_cur_windows > 0:
        t_pooled_k_cur = _pool(pool_kernel, k_cur_w_np, num_cur_windows, nheads, head_dim)
        pooled_k_np[num_ring_windows:num_k_windows] = t_pooled_k_cur.numpy()
    t_pooled_k = cl.tensor(pooled_k_np)

    local_mask_np = local_attn_mask.to(torch.bool).to(torch.uint8).contiguous().numpy()
    t_local_mask = cl.tensor(local_mask_np)
    t_raw_scores = cl.tensor(np.zeros((nheads, num_q_windows, num_k_windows), dtype=np.float32))
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
    return mask  # -> (batch_size, nheads, num_q_windows, num_ring_windows+num_cur_windows)
