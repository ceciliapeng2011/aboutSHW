"""
Host-side adapter wiring XAttention's block-importance scoring pipeline
(xattn_gemm_qk -> xattn_find_block -> xattn_post_proc, under ../x_attn/) onto a dense
[B,Hkv,Sk,D] K/V tensor, for cm_sdpa_opt_prefill.cm's SPARSE_BLOCK_SIZE-gated sparse path.

Corrected design (see plan's Stage 3a "Round 1" -- the plan's literal wording doesn't work):
The plan's original idea was "one logical block per (batch,kv-head) with block_size=Sk" (i.e. no
real block boundaries at all). That is infeasible: x_attn/estimate.hpp hard-requires
`static_assert(KV_BLOCK_SIZE == 256, "block size of key(key_cache) should be 256");` -- gemm_qk's
2D LSC descriptors are built assuming exactly 256-token physical blocks, it cannot be handed one
giant Sk-sized block.

The approach actually used here: chunk the dense K/V tensor into real ceil(Sk/256) 256-token
physical blocks with an IDENTITY block table (block_indices[i] = i). Since the dense K/V memory is
already contiguous in exactly the order a paged block i would occupy, this is a degenerate-but-
valid instance of the existing paged block-table contract gemm_qk already assumes -- no new
addressing logic needed in estimate.hpp/find_block.hpp/xattn_post_proc.cm, all three are reused
unmodified. The last block may be a partial block if Sk % 256 != 0; this is zero-padded exactly
the way x_attn's own tests pad an under-full last KV block (see _build_identity_key_cache).
"""
import functools
import os
import sys

import numpy as np
import torch

from clops import cl

CWD = os.path.dirname(os.path.realpath(__file__))
XATTN_DIR = os.path.join(os.path.dirname(CWD), "x_attn")
if XATTN_DIR not in sys.path:
    sys.path.insert(0, XATTN_DIR)

import test_gemm_qk as _gemm_qk_mod          # noqa: E402  (x_attn/test_gemm_qk.py)
import test_find_block as _find_block_mod    # noqa: E402  (x_attn/test_find_block.py)
import references as _xattn_ref              # noqa: E402  (x_attn/references.py)

STRIDE = _gemm_qk_mod.STRIDE                  # 16, fixed throughout the XAttention pipeline
KV_BLOCK_SIZE = _gemm_qk_mod.KV_BLOCK_SIZE     # 256, hard-required by estimate.hpp's static_assert
BLOCK_WG_M = _gemm_qk_mod.BLOCK_WG_M
BLOCK_WG_N = _gemm_qk_mod.BLOCK_WG_N
DEFAULT_SUB_BLOCK_SIZE = _gemm_qk_mod.DEFAULT_SUB_BLOCK_SIZE
XATTN_BLOCK_SIZES = _gemm_qk_mod.XATTN_BLOCK_SIZES  # [128,256] on Xe2, [128] on Xe1 -- see estimate.hpp


def div_up(a, b):
    return (a + b - 1) // b


def _causal_q_start_strided(q_len, kv_len, is_causal):
    # Matches x_attn's own convention (test_gemm_qk.py run_func): the strided offset between
    # where K's causal diagonal starts vs. Q's own strided origin.
    return (kv_len // STRIDE - q_len // STRIDE) if is_causal else 0


def _build_identity_key_cache(k_b, head_size):
    """k_b: [Hkv, Sk, D] fp16 torch tensor (one batch item).

    Returns (key_cache, num_blocks): key_cache is [1, num_blocks, Hkv, KV_BLOCK_SIZE, D]
    contiguous fp16 -- exactly the physical layout gemm_qk's addressing expects
    (`key_cache += hk*(...); offset = block_indices_p[block_idx] * (HK*KV_BLOCK_SIZE*HEAD_SIZE*2)`)
    -- with block i occupying physical slot i (identity block table, block_indices[i] = i).
    """
    Hkv, Sk, D = k_b.shape
    num_blocks = div_up(Sk, KV_BLOCK_SIZE)
    Sk_trunc = Sk // STRIDE * STRIDE  # gemm_qk silently drops any <STRIDE remainder (test_gemm_qk.py run_gemm)
    k_pad = torch.zeros(Hkv, num_blocks * KV_BLOCK_SIZE, D, dtype=torch.float16)
    k_pad[:, :Sk_trunc, :] = k_b[:, :Sk_trunc, :]
    key_cache = k_pad.reshape(Hkv, num_blocks, KV_BLOCK_SIZE, D).permute(1, 0, 2, 3).contiguous()
    return key_cache.unsqueeze(0), num_blocks


def _build_gemm_qk_query(q_b, num_heads, head_size):
    """q_b: [Hq, Sq, D]. gemm_qk reads Q in plain heads-inner [Sq, Hq*D] layout -- its addressing
    reshapes STRIDE consecutive tokens on the fly via the block_2d_desc X-offset trick, no
    host-side Q_resh reordering needed -- row-padded to 2x width the way test_gemm_qk.py's
    run_gemm() does (LSC 2D descriptor safety margin at the last head's column range)."""
    Sq = q_b.shape[1]
    q_3d = q_b.permute(1, 0, 2).reshape(Sq, num_heads * head_size)
    q_3d_pad = torch.zeros(Sq, num_heads * head_size * 2, dtype=torch.float16)
    q_3d_pad[:, : num_heads * head_size] = q_3d
    return q_3d_pad.unsqueeze(0).contiguous()


@functools.cache
def _get_gemm_qk_kernel(num_heads, num_kv_heads, head_size, xattn_block_size, is_causal):
    return _gemm_qk_mod.xattn_gemmQK.create_instance(
        num_heads, num_kv_heads, head_size, xattn_block_size, is_causal, False, DEFAULT_SUB_BLOCK_SIZE)


@functools.cache
def _get_find_block_kernel(num_heads, num_kv_heads, head_size, xattn_block_size, is_causal):
    return _find_block_mod.xattn_find_block.create_instance(
        num_heads, num_kv_heads, head_size, xattn_block_size, is_causal)


@functools.cache
def _get_post_proc_kernel(num_heads, num_kv_heads, head_size, xattn_block_size, is_causal, merged_q_num):
    src = r'''#include "xattn_post_proc.cm"'''
    jit_option = '-abortonspill -noschedule '
    opts = f'''-cmc -Qxcm_jit_option="{jit_option}"
                -mCM_printregusage -mdump_asm -g2
                -Qxcm_register_file_size=256 -I{XATTN_DIR}
                -DKERNEL_NAME=post_proc_mask
                -DSTRIDE={STRIDE} -DHQ={num_heads} -DHK={num_kv_heads} -DHEAD_SIZE={head_size}
                -DBLOCK_SIZE={xattn_block_size} -DKV_BLOCK_SIZE={KV_BLOCK_SIZE}
                -DBLOCK_SHARE_MAX={BLOCK_WG_N} -DIS_CAUSAL={int(is_causal)}
                -DSOFTMAX_TYPE=float -DMERGED_Q_NUM={merged_q_num}
                '''
    return cl.kernels(src, opts)


def build_sparse_mask(q, k, num_heads, num_kv_heads, head_size, xattn_block_size, is_causal, xattn_thresh):
    """q: [B,Hq,Sq,D] fp16 torch tensor (logical). k: [B,Hkv,Sk,D] fp16 torch tensor (logical).

    Runs the real xattn_gemm_qk -> xattn_find_block GPU kernels once per batch item (the wrapper
    classes' own __call__ hardcodes batch=1 internally; looping in Python is the simplest correct
    way to cover batch>1) against an identity-block-table view of the dense K tensor.

    Returns (mask, num_q_blocks, num_k_blocks): mask is a torch.bool tensor
    [B, num_heads, num_q_blocks, num_k_blocks], 1=keep this KV block / 0=skip -- indexed per Q
    HEAD (not shared across a GQA group; confirmed from xattn_find_block.cm/xattn_post_proc.cm,
    both index solely by `hq = cm_group_id(1)` with stride HQ, independent of HK). num_q_blocks/
    num_k_blocks equal the mask's own dims 2/3 (returned for convenience).
    """
    assert xattn_block_size in (128, 256), "estimate.hpp/find_block.hpp only support BLOCK_SIZE 128 or 256"
    B, Hq, Sq, D = q.shape
    _, Hkv, Sk, _ = k.shape
    assert Hq == num_heads and Hkv == num_kv_heads and D == head_size

    gemm_cm = _get_gemm_qk_kernel(num_heads, num_kv_heads, head_size, xattn_block_size, is_causal)
    find_cm = _get_find_block_kernel(num_heads, num_kv_heads, head_size, xattn_block_size, is_causal)

    q_start_strided = _causal_q_start_strided(Sq, Sk, is_causal)
    M = Sq // STRIDE
    N = Sk // STRIDE
    K = STRIDE * head_size
    query_stride = K * num_heads * 2

    masks = []
    for b in range(B):
        key_cache, num_blocks = _build_identity_key_cache(k[b].to(torch.float16), head_size)
        block_indices = torch.arange(num_blocks, dtype=torch.int32)
        t_key_cache = cl.tensor(key_cache.numpy())
        t_block_indices = cl.tensor(block_indices.numpy())
        # Accepted by xattn_gemmQK.__call__'s signature but never referenced by the kernel itself
        # (dead legacy parameter) -- kept only for ABI parity with the wrapper's call convention.
        t_block_indices_begins = cl.tensor(np.zeros(2, dtype=np.int32))

        t_query = cl.tensor(_build_gemm_qk_query(q[b].to(torch.float16), num_heads, head_size).numpy())

        _, t_kq_max_wg, t_kq_exp_partial_sum = gemm_cm(
            t_key_cache, t_query, t_block_indices, t_block_indices_begins,
            q_start_strided, M, N, K, query_stride, 1)
        _, t_mask, _ = find_cm(t_kq_max_wg, t_kq_exp_partial_sum, Sq, M, N, xattn_thresh, 1)
        masks.append(torch.from_numpy(t_mask.numpy().astype(np.int8)))

    mask = torch.cat(masks, dim=0).to(torch.bool)  # [B, Hq, num_q_blocks, num_k_blocks]
    return mask, mask.shape[2], mask.shape[3]


def merge_wg_mask(mask, num_heads, num_kv_heads, head_size, xattn_block_size, is_causal, merged_q_num=1):
    """Runs the real xattn_post_proc.cm GPU kernel's Q-dimension OR-merge (fine mask -> coarser
    "wg" mask), completing the 3-kernel pipeline for API parity with PA's
    sparse_block_mask/sparse_block_mask_wg pair (pa_multi_token.cm).

    Not consumed by cm_sdpa_opt_prefill.cm today: each of its work-items already computes exactly
    one sparse_block_size-sized Q-tile on its own (no PA-style multi-block workgroup pooling that
    needs a coarser shared mask for a shared SLM load stage -- confirmed from cm_pa_xe2.hpp that
    even PA's own plain-fp16 kernel path only ever reads the fine mask, `wg_sparse_mask_base` is
    accepted but never dereferenced there). merged_q_num=1 makes this an identity map in our
    usage; included here so the full pipeline is genuinely exercised end-to-end.
    """
    B, HQ, q_block_pad, k_block_pad = mask.shape
    kernel = _get_post_proc_kernel(num_heads, num_kv_heads, head_size, xattn_block_size, is_causal, merged_q_num)
    merged_q_blocks = div_up(q_block_pad, merged_q_num)

    merged = []
    for b in range(B):
        t_mask = cl.tensor(mask[b : b + 1].to(torch.int8).numpy())
        t_merged_mask = cl.tensor(np.zeros([1, HQ, merged_q_blocks, k_block_pad], dtype=np.int8))
        meta_np = np.array([[
            0, 0, 0, 0, 0, 0, q_block_pad, k_block_pad, 0, 0, 0, 0, 0, 0, 0, 0,
        ]], dtype=np.int32)
        t_meta = cl.tensor(meta_np)
        wg_map_data = []
        for m_merged in range(merged_q_blocks):
            wg_map_data.extend([0, m_merged])
        t_wg_map = cl.tensor(np.array(wg_map_data, dtype=np.int32))

        post_wg_count = len(wg_map_data) // 2
        kernel.enqueue("post_proc_mask", [post_wg_count, HQ, 1], [1, 1, 1], t_mask, t_merged_mask, t_meta, t_wg_map)
        merged.append(torch.from_numpy(t_merged_mask.numpy().astype(np.int8)))

    return torch.cat(merged, dim=0).to(torch.bool)


def cpu_reference_mask(q, k, xattn_block_size, is_causal, xattn_thresh):
    """Pure-CPU/torch ground truth for build_sparse_mask's fine mask, independent of any GPU
    kernel -- x_attn/references.py's get_gemm_ref + find_blocks_ref (the exact functions x_attn's
    own tests cross-check the real GPU kernels against).

    Returns (mask_ref, q_block, k_block): mask_ref is [B,Hq,q_block,k_block] bool, UNPADDED (i.e.
    q_block/k_block are the true, not rounded-up-to-workgroup, block counts) -- compare against
    build_sparse_mask's output sliced to [:, :, :q_block, :k_block].
    """
    B, Hq, Sq, D = q.shape
    _, Hkv, Sk, _ = k.shape
    q_start_strided = _causal_q_start_strided(Sq, Sk, is_causal)
    _, _, _, A_sum = _xattn_ref.get_gemm_ref(
        q.to(torch.float32), k.to(torch.float32), q_start_strided=q_start_strided, block_size=xattn_block_size,
        S=STRIDE, threshold=xattn_thresh, causal=is_causal, wg_k=BLOCK_WG_N, wg_q=BLOCK_WG_M)

    q_stride = (Sq // STRIDE * STRIDE) // STRIDE  # get_gemm_ref drops the <STRIDE Q remainder too
    k_stride = Sk // STRIDE
    sum_per_token_in_block = xattn_block_size // STRIDE
    q_block = div_up(q_stride, sum_per_token_in_block)
    k_block = div_up(k_stride, sum_per_token_in_block)
    mask_ref, *_ = _xattn_ref.find_blocks_ref(
        A_sum, xattn_thresh, q_block, k_block, causal=is_causal, current_index=k_block - q_block)
    return mask_ref.to(torch.bool), q_block, k_block
