"""
Generic block-sparse 2nd stage wired onto cm_sdpa_opt_prefill.cm: a reusable test/bench harness
for the kernel's SPARSE_BLOCK_SIZE ABI (a caller-supplied [B,Hq,num_q_blocks,num_k_blocks] bool
block-table mask) independent of whatever algorithm produced the mask. FlashVSR's own
draft-block-mask tests/benches and end-to-end sparse-attention tests build on top of this module's
run_prefill_sparse/ref_sdpa_sparse rather than duplicating the kernel-dispatch plumbing.
"""
import functools
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

from clops import cl

CWD = os.path.dirname(os.path.realpath(__file__))
PAGEATTEN_DIR = os.path.join(os.path.dirname(CWD), "pageatten")
if PAGEATTEN_DIR not in sys.path:
    sys.path.insert(0, PAGEATTEN_DIR)

import test_sdpa_prefill as base    # noqa: E402  (reuse check_close/make_qkv/to_physical/...)

torch.manual_seed(42)

TRANSPOSE_ORDER_IDENTITY = base.TRANSPOSE_ORDER_IDENTITY
TRANSPOSE_ORDER_SWAPPED = base.TRANSPOSE_ORDER_SWAPPED


@functools.cache
def build_sparse_prefill_kernel(num_heads, num_kv_heads, head_size, is_causal, sparse_block_size,
                                 q_order=TRANSPOSE_ORDER_IDENTITY, k_order=TRANSPOSE_ORDER_IDENTITY,
                                 v_order=TRANSPOSE_ORDER_IDENTITY, o_order=TRANSPOSE_ORDER_IDENTITY,
                                 scale_factor=None):
    if scale_factor is None:
        scale_factor = 1.0 / (head_size ** 0.5)
    src = r'''#include "cm_sdpa_opt_prefill.cm"'''
    opts = (
        f'-cmc -Qxcm_jit_option="-abortonspill" -Qxcm_register_file_size=256 -mCM_printregusage '
        f"-I{base.CWD} -I{base.FLASHATTN_DIR} -I{base.PAGEATTEN_DIR} "
        f"-DKERNEL_NAME=cm_sdpa_opt_prefill "
        f"-DCMFLA_NUM_HEADS={num_heads} -DCMFLA_NUM_KV_HEADS={num_kv_heads} "
        f"-DCMFLA_HEAD_SIZE={head_size} -DCMFLA_SCALE_FACTOR={scale_factor} "
        f"-DCMFLA_IS_CAUSAL={int(is_causal)} "
        f"-DQ_TRANSPOSE_ORDER={q_order} -DK_TRANSPOSE_ORDER={k_order} "
        f"-DV_TRANSPOSE_ORDER={v_order} -DOUTPUT_TRANSPOSE_ORDER={o_order} "
        f"-DHAS_ATTN_MASK_INPUT=0 -DHAS_SCALE_INPUT=0 "
        f"-DSPARSE_BLOCK_SIZE={sparse_block_size}"
    )
    return cl.kernels(src, opts)


def run_prefill_sparse(q, k, v, is_causal, sparse_block_size, block_mask,
                        q_order=TRANSPOSE_ORDER_IDENTITY, k_order=TRANSPOSE_ORDER_IDENTITY,
                        v_order=TRANSPOSE_ORDER_IDENTITY, o_order=TRANSPOSE_ORDER_IDENTITY, scale=None):
    batch, num_heads, q_len, head_size = q.shape
    _, num_kv_heads, kv_len, _ = k.shape

    kernel = build_sparse_prefill_kernel(num_heads, num_kv_heads, head_size, is_causal, sparse_block_size,
                                          q_order, k_order, v_order, o_order, scale)

    q_phys = base.to_physical(q, q_order)
    k_phys = base.to_physical(k, k_order)
    v_phys = base.to_physical(v, v_order)
    t_q = cl.tensor(q_phys.to(torch.float16).numpy())
    t_k = cl.tensor(k_phys.to(torch.float16).numpy())
    t_v = cl.tensor(v_phys.to(torch.float16).numpy())
    out_shape = ([batch, q_len, num_heads, head_size] if o_order == TRANSPOSE_ORDER_SWAPPED
                 else [batch, num_heads, q_len, head_size])
    t_out = cl.tensor(np.zeros(out_shape, dtype=np.float16))

    q_step = base.get_cm_grf_width() // 32
    wg_size = 16
    wg_seq_len = wg_size * q_step
    wg_count = (q_len + wg_seq_len - 1) // wg_seq_len
    GWS = [1, batch * num_heads, wg_count * wg_size]
    LWS = [1, 1, wg_size]

    num_q_blocks, num_k_blocks = block_mask.shape[2], block_mask.shape[3]
    t_mask = cl.tensor(block_mask.to(torch.bool).numpy())

    args = [t_q, t_k, t_v, t_out, q_len, kv_len, t_mask, num_q_blocks, num_k_blocks]
    kernel.enqueue("cm_sdpa_opt_prefill", GWS, LWS, *args)

    out_phys = torch.from_numpy(t_out.numpy())
    return base.from_physical(out_phys, o_order, batch, num_heads, q_len, head_size)


def block_mask_to_additive(mask, sparse_block_size, q_len, kv_len):
    # mask: [B,Hq,num_q_blocks,num_k_blocks] bool -> additive [B,Hq,q_len,kv_len] float (0/-inf)
    full = mask.to(torch.float32)
    full = full.repeat_interleave(sparse_block_size, dim=2).repeat_interleave(sparse_block_size, dim=3)
    full = full[:, :, :q_len, :kv_len]
    return torch.where(full > 0, torch.zeros_like(full), torch.full_like(full, float("-inf")))


def ref_sdpa_sparse(q, k, v, is_causal, mask, sparse_block_size, scale=None):
    # Reference: apply the SAME block mask to a torch attention computation -- isolates "did the
    # kernel correctly skip only the intended blocks" from any question of how the mask itself
    # was generated.
    B, Hq, Sq, D = q.shape
    _, Hkv, Sk, _ = k.shape
    additive = block_mask_to_additive(mask, sparse_block_size, Sq, Sk)
    if is_causal:
        causal_add = torch.triu(torch.full((Sq, Sk), float("-inf")), diagonal=Sk - Sq + 1)
        additive = additive + causal_add.unsqueeze(0).unsqueeze(0)
    out = F.scaled_dot_product_attention(
        q.to(torch.float32), k.to(torch.float32), v.to(torch.float32),
        attn_mask=additive, is_causal=False, enable_gqa=(Hkv != Hq), scale=scale,
    )
    return out.to(torch.float16)


# ---------------------------------------------------------------------------
# Bypass correctness: an all-True mask must not change the kernel's output at all versus the
# plain (SPARSE_BLOCK_SIZE=1) Stage 1a kernel.
# ---------------------------------------------------------------------------
def test_bypass_matches_dense():
    q, k, v = base.make_qkv(batch=1, num_heads=4, num_kv_heads=4, q_len=1024, kv_len=1024, head_size=64)
    sparse_block_size = 256
    num_q_blocks = (1024 + sparse_block_size - 1) // sparse_block_size
    num_k_blocks = num_q_blocks
    all_true_mask = torch.ones(1, 4, num_q_blocks, num_k_blocks, dtype=torch.bool)

    out_sparse = run_prefill_sparse(q, k, v, is_causal=True, sparse_block_size=sparse_block_size,
                                     block_mask=all_true_mask)
    out_dense = base.run_prefill(q, k, v, is_causal=True)
    assert torch.equal(out_sparse, out_dense), "all-True sparse mask must be bit-identical to the dense kernel"
    print("test_bypass_matches_dense PASSED")


if __name__ == "__main__":
    cl.profiling(True)
    test_bypass_matches_dense()
