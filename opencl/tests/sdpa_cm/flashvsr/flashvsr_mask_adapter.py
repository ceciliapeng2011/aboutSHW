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
# int8_kv_ref.py (the int8 host reference) and cm_int8_kv_decode.hpp (the CM-side decode header)
# live one level up, in sdpa_cm/ and flashattn/ respectively -- add them to sys.path/include-path
# the same way the test_flashvsr_* files in this directory already do for their own sdpa_cm-root
# imports, so this module stays importable regardless of which directory the importing script's
# own sys.path already covers.
SDPA_CM_DIR = os.path.dirname(CWD)
FLASHATTN_DIR = os.path.join(os.path.dirname(SDPA_CM_DIR), "flashattn")
if SDPA_CM_DIR not in sys.path:
    sys.path.insert(0, SDPA_CM_DIR)

from int8_kv_ref import DEFAULT_SUB_BLOCK_SIZE, encode_k_by_channel, encode_by_token

# Must match draft_block_mask_topk_mask.cm's TOPK_LOCAL_SIZE default -- number of work items per
# workgroup cooperatively binary-searching one (head, Q-temporal-group)'s top-k threshold.
TOPK_LOCAL_SIZE = 32

KV_CACHE_COMPRESSION_NONE = 0
KV_CACHE_COMPRESSION_BY_TOKEN = 1
KV_CACHE_COMPRESSION_BY_CHANNEL = 2


@functools.cache
def _get_pool_kernel(num_heads, head_dim, window_tokens, k_compression=KV_CACHE_COMPRESSION_NONE,
                      has_out_row_offset=False):
    """k_compression: compiles the K_CACHE_COMPRESSION={1,2} variant of draft_block_mask_pool.cm
    (int8-packed `windowed` input, flat affine decode-on-read, extra k_scale/k_zp kernel params).
    Only ever meaningful for a K-pooling dispatch -- Q is never quantized in this project, so
    Q-pooling always goes through the default k_compression=NONE (plain half) variant; see
    pool_k_int8 below.

    has_out_row_offset: compiles the HAS_OUT_ROW_OFFSET=1 variant (extra trailing `out_row_offset`
    kernel arg) -- lets a dual-source caller pool cache/cur K windows directly into two slices of
    ONE shared device `pooled` buffer instead of two separate buffers stitched together on the
    host; see _pool's/pool_k_int8's own out_row_offset kwarg."""
    src = r'''#include "draft_block_mask_pool.cm"'''
    opts = (
        f'-cmc -Qxcm_jit_option="-abortonspill" -Qxcm_register_file_size=256 -mCM_printregusage '
        f"-I{CWD} -I{FLASHATTN_DIR} -DKERNEL_NAME=flashvsr_pool "
        f"-DNUM_HEADS={num_heads} -DHEAD_DIM={head_dim} -DWINDOW_TOKENS={window_tokens} "
        f"-DK_CACHE_COMPRESSION={int(k_compression)} -DHAS_OUT_ROW_OFFSET={int(has_out_row_offset)}"
    )
    return cl.kernels(src, opts)


@functools.cache
def _get_score_softmax_kernel(num_heads, head_dim, num_k_windows_cache=None):
    """num_k_windows_cache: compiles the HAS_CACHE_VALID_MASK=1 variant (extra trailing
    `valid_k_windows_cache` kernel arg) -- lets a dual-source caller mask stale/not-yet-written cache
    columns of a fixed-capacity cache buffer out of top-k selection; see
    build_draft_mask_gpu_dualsource's valid_slots/num_slots kwargs."""
    src = r'''#include "draft_block_mask_score_softmax.cm"'''
    has_cache_valid_mask = num_k_windows_cache is not None
    opts = (
        f'-cmc -Qxcm_jit_option="-abortonspill" -Qxcm_register_file_size=256 -mCM_printregusage '
        f"-I{CWD} -DKERNEL_NAME=flashvsr_score_softmax "
        f"-DNUM_HEADS={num_heads} -DHEAD_DIM={head_dim} "
        f"-DHAS_CACHE_VALID_MASK={int(has_cache_valid_mask)}"
    )
    if has_cache_valid_mask:
        opts += f" -DNUM_K_WINDOWS_CACHE={int(num_k_windows_cache)}"
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


def _pool(kernel, windowed_np, num_windows, num_heads, head_dim, t_out=None, out_row_offset=None):
    """t_out/out_row_offset: when given, writes into the caller-supplied device tensor at row
    offset `out_row_offset` instead of allocating+returning a fresh (0-offset) one -- `kernel` must
    have been built with has_out_row_offset=True whenever out_row_offset is not None (the compiled
    variant always expects the matching trailing kernel arg)."""
    t_in = cl.tensor(windowed_np)
    if t_out is None:
        t_out = cl.tensor(np.zeros((num_windows, num_heads, head_dim), dtype=np.float16))
    args = [t_in, t_out]
    if out_row_offset is not None:
        args.append(int(out_row_offset))
    kernel.enqueue("flashvsr_pool", [num_windows, num_heads, 1], [1, 1, 1], *args)
    return t_out


def encode_k_windows_int8(k_w, compression_mode=KV_CACHE_COMPRESSION_BY_CHANNEL):
    """k_w: (num_k_windows, window_tokens, nheads*head_dim) half torch tensor. Returns
    (k_w_u8, k_scale, k_zp) numpy arrays ready for pool_k_int8 below -- quantizes the WHOLE
    nheads*head_dim axis as one flat channel set (the windowed buffer has no separate per-head
    leading dim in memory, see draft_block_mask_pool.cm's K_CACHE_COMPRESSION doc comment), via a
    dummy singleton head axis so int8_kv_ref's [B,H,S,D]-shaped encoders apply directly."""
    k_w4 = k_w.unsqueeze(1).to(torch.float16)  # [num_k_windows, 1, window_tokens, d_full]
    if compression_mode == KV_CACHE_COMPRESSION_BY_CHANNEL:
        k_u8, k_scale, k_zp = encode_k_by_channel(k_w4, sub_block_size=DEFAULT_SUB_BLOCK_SIZE)
    else:
        k_u8, k_scale, k_zp = encode_by_token(k_w4)
    return (k_u8.squeeze(1).contiguous().numpy(), k_scale.squeeze(1).contiguous().numpy(),
            k_zp.squeeze(1).contiguous().numpy())


def pool_k_int8(k_w_u8, k_scale, k_zp, num_k_windows, nheads, head_dim, window_tokens,
                compression_mode=KV_CACHE_COMPRESSION_BY_CHANNEL, t_out=None, out_row_offset=None):
    """Opt-in int8-packed-input K-pooling dispatch: K_CACHE_COMPRESSION={1,2} compiled variant of
    draft_block_mask_pool.cm, decoding each raw K byte on the fly (flat affine (raw-zp)*scale, see
    cm_int8_kv_decode.hpp) instead of mean-pooling an already-materialized half copy of the
    windowed K buffer. Q is never quantized in this project (only cache K/V ever is) -- there is no
    int8 counterpart for Q, callers must keep routing Q through the existing half
    `_pool`/`build_draft_mask_gpu*` path.

    k_w_u8/k_scale/k_zp: see encode_k_windows_int8 above.
    t_out/out_row_offset: see _pool's identical kwargs.
    Returns: (num_k_windows, nheads, head_dim) half cl.tensor -- bit-identical (see
    test_flashvsr_mask_pool_int8.py) to running the unmodified half kernel on the
    host-decoded-back-to-half version of the same bytes.
    """
    kernel = _get_pool_kernel(nheads, head_dim, window_tokens, k_compression=compression_mode,
                              has_out_row_offset=out_row_offset is not None)
    t_in = cl.tensor(np.ascontiguousarray(k_w_u8))
    t_scale = cl.tensor(np.ascontiguousarray(k_scale))
    t_zp = cl.tensor(np.ascontiguousarray(k_zp))
    if t_out is None:
        t_out = cl.tensor(np.zeros((num_k_windows, nheads, head_dim), dtype=np.float16))
    args = [t_in, t_scale, t_zp, t_out]
    if out_row_offset is not None:
        args.append(int(out_row_offset))
    kernel.enqueue("flashvsr_pool", [num_k_windows, nheads, 1], [1, 1, 1], *args)
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


def build_draft_mask_gpu_dualsource(batch_size, nheads, seqlen, q_w, k_cache_w, k_cur_w,
                                      topk=10, local_attn_mask=None, k_cache_int8=None,
                                      valid_slots=None, num_slots=None):
    """Two-source variant of build_draft_mask_gpu: mirrors the real DraftBlockMask primitive's
    `input[1]=key_cache.<layer>` (cache, persistent) + `input[2]=k_new_f16` (current, this chunk's
    own new K) ABI (see flashvsr_dit_kvcache_int8_3slot_fused_dualsource_design_zh.md Sec 10.4) --
    `k_cache_w` and `k_cur_w` are two INDEPENDENT buffers, never concatenated on the host or on
    device before pooling.

    k_cache_int8: optional (k_cache_w_u8, k_cache_scale, k_cache_zp, compression_mode) tuple (see
    encode_k_windows_int8) -- when provided, the CACHE pooling dispatch is routed through
    pool_k_int8 instead of the plain-half _pool, closing the gap where this dualsource path never
    had a quantized cache-pooling entry point at all (unlike the SDPA side's
    KV_CACHE_COMPRESSION, which always has). k_cache_w is still required (its shape/dtype
    determine num_cache_windows/window_tokens/d_full) even when k_cache_int8 is given; its contents
    are only actually read when k_cache_int8 is None. Current pooling always stays half -- matches
    the SDPA side's cache-only-quantization convention exactly (current is transient, quantizing it
    buys no persistent-memory win).

    valid_slots/num_slots: for a FIXED-CAPACITY cache buffer whose physical shape never shrinks (the
    PagedAttention-style design -- see draft_block_mask_score_softmax.cm's HAS_CACHE_VALID_MASK doc
    comment), `num_slots` is the cache's total physical slot count and `valid_slots` says how many
    of those slots actually hold real (written) data so far (e.g. chunk0/warmup, before the cache is
    full) -- k_cache_w's window count must still equal the FULL (num_slots-worth) capacity; only the
    first `valid_slots * (num_cache_windows // num_slots)` of its pooled columns are ever
    top-k-selectable, the rest are masked exactly like a local_mask-excluded column (still read,
    never selected). Leave both None (default) for a cache that's always fully valid -- the
    unmodified (HAS_CACHE_VALID_MASK=0) score_softmax kernel is used in that case, bit-identical to
    before this kwarg existed.

    k_cache_w: (num_cache_windows, window_tokens, nheads*head_dim) -- cache/history K windows.
    k_cur_w:  (num_cur_windows, window_tokens, nheads*head_dim) -- this chunk's own new K windows.
    Combined k_w semantics (num_k_windows = num_cache_windows + num_cur_windows, cache windows first)
    are otherwise identical to build_draft_mask_gpu's single k_w -- only the pooling stage (Stage 1,
    draft_block_mask_pool.cm) ever reads the raw windowed K; that kernel already treats its input as
    an arbitrary-length windowed buffer, so pooling cache and current directly into two row-offset
    slices of one shared device pooled_k buffer (HAS_OUT_ROW_OFFSET, see draft_block_mask_pool.cm)
    is sufficient -- score_softmax/topk_mask (Stages 2/3) are reused UNCHANGED (aside from the
    opt-in HAS_CACHE_VALID_MASK variant above), operating on the combined (but tiny, already-pooled)
    pooled_k exactly as build_draft_mask_gpu does today.

    Returns: (batch_size, nheads, num_q_windows, num_cache_windows+num_cur_windows) bool torch
    tensor -- bit-identical to build_draft_mask_gpu(..., k_w=torch.cat([k_cache_w, k_cur_w])) (see
    test_flashvsr_mask_dualsource.py).
    """
    assert local_attn_mask is not None, "local_attn_mask must be provided"
    assert (valid_slots is None) == (num_slots is None), "valid_slots and num_slots must be given together"

    num_q_windows, window_tokens, d_full = q_w.shape
    num_cache_windows = k_cache_w.shape[0]
    num_cur_windows = k_cur_w.shape[0]
    assert k_cache_w.shape[1:] == (window_tokens, d_full)
    assert k_cur_w.shape[1:] == (window_tokens, d_full)
    assert d_full % nheads == 0
    head_dim = d_full // nheads
    spatial_windows = local_attn_mask.shape[0]
    spatial_windows_k = local_attn_mask.shape[1]
    assert num_q_windows % seqlen == 0, "num_q_windows must be seqlen * spatial_windows"
    assert num_q_windows // seqlen == spatial_windows, (
        "local_attn_mask's spatial extent must match q_w's windows-per-temporal-group"
    )

    num_k_windows = num_cache_windows + num_cur_windows

    # has_out_row_offset=True unconditionally: both the cache (offset 0) and cur (offset
    # num_cache_windows) dispatches below share ONE compiled kernel variant for symmetry, even
    # though cache's own offset is always 0 -- matches upstream pooling both sources through the
    # same HAS_OUT_ROW_OFFSET-capable kernel instance.
    pool_kernel = _get_pool_kernel(nheads, head_dim, window_tokens, has_out_row_offset=True)
    num_k_windows_cache = num_cache_windows if valid_slots is not None else None
    score_softmax_kernel = _get_score_softmax_kernel(nheads, head_dim,
                                                      num_k_windows_cache=num_k_windows_cache)
    topk_kernel = _get_topk_kernel(nheads)

    q_w_np = q_w.to(torch.float16).contiguous().numpy()
    k_cache_w_np = k_cache_w.to(torch.float16).contiguous().numpy()
    k_cur_w_np = k_cur_w.to(torch.float16).contiguous().numpy()

    t_pooled_q = _pool(_get_pool_kernel(nheads, head_dim, window_tokens), q_w_np, num_q_windows,
                       nheads, head_dim)

    # Cache and current are pooled from two SEPARATE device buffers (never concatenated, raw or
    # windowed) directly into two row-offset slices of ONE shared device `pooled_k` buffer
    # (HAS_OUT_ROW_OFFSET) -- matches upstream's actual device-side layout, no host round-trip.
    t_pooled_k = cl.tensor(np.zeros((num_k_windows, nheads, head_dim), dtype=np.float16))
    if num_cache_windows > 0:
        if k_cache_int8 is not None:
            k_cache_u8, k_cache_scale, k_cache_zp, cache_compression_mode = k_cache_int8
            pool_k_int8(k_cache_u8, k_cache_scale, k_cache_zp, num_cache_windows, nheads, head_dim,
                        window_tokens, compression_mode=cache_compression_mode,
                        t_out=t_pooled_k, out_row_offset=0)
        else:
            _pool(pool_kernel, k_cache_w_np, num_cache_windows, nheads, head_dim,
                  t_out=t_pooled_k, out_row_offset=0)
    if num_cur_windows > 0:
        _pool(pool_kernel, k_cur_w_np, num_cur_windows, nheads, head_dim,
              t_out=t_pooled_k, out_row_offset=num_cache_windows)

    local_mask_np = local_attn_mask.to(torch.bool).to(torch.uint8).contiguous().numpy()
    t_local_mask = cl.tensor(local_mask_np)
    t_raw_scores = cl.tensor(np.zeros((nheads, num_q_windows, num_k_windows), dtype=np.float32))
    t_probs = cl.tensor(np.zeros((nheads, num_q_windows, num_k_windows), dtype=np.float32))

    scale = 1.0 / math.sqrt(head_dim)
    score_softmax_args = [t_pooled_q, t_pooled_k, t_local_mask, t_raw_scores, t_probs,
                          num_q_windows, num_k_windows, spatial_windows, spatial_windows_k, scale]
    if valid_slots is not None:
        windows_per_slot = num_cache_windows // num_slots
        valid_k_windows_cache = valid_slots * windows_per_slot
        score_softmax_args.append(int(valid_k_windows_cache))
    score_softmax_kernel.enqueue(
        "flashvsr_score_softmax", [num_q_windows, nheads, 1], [1, 1, 1], *score_softmax_args,
    )

    num_groups = seqlen
    t_mask = cl.tensor(np.zeros((batch_size, nheads, num_q_windows, num_k_windows), dtype=np.uint8))
    topk_local_size = TOPK_LOCAL_SIZE
    topk_kernel.enqueue(
        "flashvsr_topk_mask", [nheads, num_groups, topk_local_size], [1, 1, topk_local_size],
        t_probs, t_mask, num_q_windows, num_k_windows, spatial_windows, int(topk), int(batch_size),
    )

    mask = torch.from_numpy(t_mask.numpy()).to(torch.bool)
    return mask  # -> (batch_size, nheads, num_q_windows, num_cache_windows+num_cur_windows)
