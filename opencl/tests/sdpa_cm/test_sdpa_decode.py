import functools

import numpy as np
import torch
import torch.nn.functional as F

from clops import cl

import test_sdpa_prefill as prefill
from test_sdpa_prefill import (
    CWD, FLASHATTN_DIR, PAGEATTEN_DIR,
    TRANSPOSE_ORDER_IDENTITY, TRANSPOSE_ORDER_SWAPPED,
    check_close, get_cm_grf_width, make_qkv, to_physical, from_physical,
)

torch.manual_seed(0)


@functools.cache
def build_decode_kernel(num_heads, num_kv_heads, head_size,
                         q_order, k_order, v_order, o_order,
                         has_attn_mask=False, has_scale_input=False,
                         scale_factor=None, kv_partition_size=256):
    if scale_factor is None:
        scale_factor = 1.0 / (head_size ** 0.5)
    src = r'''#include "cm_sdpa_opt_decode.cm"'''
    opts = (
        f'-cmc -Qxcm_jit_option="-abortonspill" -Qxcm_register_file_size=256 -mCM_printregusage '
        f"-I{CWD} -I{FLASHATTN_DIR} -I{PAGEATTEN_DIR} "
        f"-DKERNEL_NAME=cm_sdpa_opt_decode "
        f"-DCMFLA_NUM_HEADS={num_heads} -DCMFLA_NUM_KV_HEADS={num_kv_heads} "
        f"-DCMFLA_HEAD_SIZE={head_size} -DCMFLA_SCALE_FACTOR={scale_factor} "
        f"-DQ_TRANSPOSE_ORDER={q_order} -DK_TRANSPOSE_ORDER={k_order} "
        f"-DV_TRANSPOSE_ORDER={v_order} -DOUTPUT_TRANSPOSE_ORDER={o_order} "
        f"-DHAS_ATTN_MASK_INPUT={int(has_attn_mask)} -DHAS_SCALE_INPUT={int(has_scale_input)} "
        f"-DCMFLA_KV_PARTITION_SIZE={kv_partition_size}"
    )
    return cl.kernels(src, opts)


@functools.cache
def build_finalization_kernel(head_size):
    src = r'''#include "cm_sdpa_opt_decode_finalization.cm"'''
    opts = (
        f'-cmc -Qxcm_jit_option="-abortonspill" -mCM_printregusage '
        f"-I{CWD} "
        f"-DKERNEL_NAME=cm_sdpa_opt_decode_reduce "
        f"-DCMFLA_HEAD_SIZE={head_size}"
    )
    return cl.kernels(src, opts)


def run_decode(q, k, v,
               q_order=TRANSPOSE_ORDER_IDENTITY, k_order=TRANSPOSE_ORDER_IDENTITY,
               v_order=TRANSPOSE_ORDER_IDENTITY, o_order=TRANSPOSE_ORDER_IDENTITY,
               attn_mask=None, scale=None, kv_partition_size=256):
    batch, num_heads, q_len, head_size = q.shape
    _, num_kv_heads, kv_len, _ = k.shape
    assert q_len == 1

    decode_kernel = build_decode_kernel(num_heads, num_kv_heads, head_size,
                                         q_order, k_order, v_order, o_order,
                                         has_attn_mask=attn_mask is not None,
                                         has_scale_input=scale is not None,
                                         scale_factor=scale, kv_partition_size=kv_partition_size)
    reduce_kernel = build_finalization_kernel(head_size)

    q_phys = to_physical(q, q_order)
    k_phys = to_physical(k, k_order)
    v_phys = to_physical(v, v_order)

    t_q = cl.tensor(q_phys.to(torch.float16).numpy())
    t_k = cl.tensor(k_phys.to(torch.float16).numpy())
    t_v = cl.tensor(v_phys.to(torch.float16).numpy())

    num_partitions = (kv_len + kv_partition_size - 1) // kv_partition_size
    bh = batch * num_heads
    t_exp_sums = cl.tensor(np.zeros([bh * num_partitions], dtype=np.float32))
    t_max_logits = cl.tensor(np.zeros([bh * num_partitions], dtype=np.float32))
    t_tmp_out = cl.tensor(np.zeros([bh * num_partitions * head_size], dtype=np.float16))

    out_shape = ([batch, 1, num_heads, head_size] if o_order == TRANSPOSE_ORDER_SWAPPED
                 else [batch, num_heads, 1, head_size])
    t_out = cl.tensor(np.zeros(out_shape, dtype=np.float16))

    args = [t_q, t_k, t_v]
    if attn_mask is not None:
        # [1,1,1,kv_len], padded to a multiple of REG_N=16 so the kernel's fixed-width tail
        # read of the last partition never runs past the allocation.
        kv_len_padded = (kv_len + 15) // 16 * 16
        mask_np = np.zeros((kv_len_padded,), dtype=np.float16)
        mask_np[:kv_len] = attn_mask.reshape(kv_len).to(torch.float16).numpy()
        args.append(cl.tensor(mask_np))
    if scale is not None:
        args.append(float(scale))
    args += [t_exp_sums, t_max_logits, t_tmp_out, kv_len]

    GWS = [1, bh, num_partitions]
    LWS = [1, 1, 1]
    decode_kernel.enqueue("cm_sdpa_opt_decode", GWS, LWS, *args)

    GWS2 = [1, bh, 1]
    LWS2 = [1, 1, 1]
    reduce_kernel.enqueue("cm_sdpa_opt_decode_reduce", GWS2, LWS2,
                           t_exp_sums, t_max_logits, t_tmp_out, t_out, num_partitions)

    out_phys = torch.from_numpy(t_out.numpy())
    return from_physical(out_phys, o_order, batch, num_heads, 1, head_size)


def ref_sdpa(q, k, v, attn_mask=None, scale=None):
    num_heads = q.shape[1]
    num_kv_heads = k.shape[1]
    out = F.scaled_dot_product_attention(
        q.to(torch.float32), k.to(torch.float32), v.to(torch.float32),
        attn_mask=attn_mask.to(torch.float32) if attn_mask is not None else None,
        is_causal=False, enable_gqa=(num_kv_heads != num_heads), scale=scale,
    )
    return out.to(torch.float16)


def make_decode_qkv(batch, num_heads, num_kv_heads, kv_len, head_size, seed=0):
    g = torch.Generator().manual_seed(seed)
    low, high = -2, 3
    q = torch.randint(low, high, [batch, num_heads, 1, head_size], generator=g).to(torch.float16)
    k = torch.randint(low, high, [batch, num_kv_heads, kv_len, head_size], generator=g).to(torch.float16)
    v = (torch.randint(low, high, [batch, num_kv_heads, kv_len, head_size], generator=g).to(torch.float16)
         / high)
    return q, k, v


def test_single_partition():
    # kv_len <= 256 -> num_partitions == 1, exercises the finalization short-circuit path.
    q, k, v = make_decode_qkv(batch=1, num_heads=4, num_kv_heads=4, kv_len=100, head_size=64)
    out = run_decode(q, k, v)
    ref = ref_sdpa(q, k, v)
    check_close(out, ref)


def test_multi_partition():
    # kv_len=4096 -> 16 partitions; verifies the lse/max-rescale merge reproduces single-pass
    # softmax exactly (up to fp16/fp32 rounding).
    q, k, v = make_decode_qkv(batch=1, num_heads=4, num_kv_heads=4, kv_len=4096, head_size=64)
    out = run_decode(q, k, v)
    ref = ref_sdpa(q, k, v)
    check_close(out, ref)


def test_gqa_decode():
    q, k, v = make_decode_qkv(batch=1, num_heads=8, num_kv_heads=2, kv_len=1024, head_size=64)
    out = run_decode(q, k, v)
    ref = ref_sdpa(q, k, v)
    check_close(out, ref)


def test_attn_mask_decode():
    q, k, v = make_decode_qkv(batch=1, num_heads=4, num_kv_heads=4, kv_len=600, head_size=64)
    g = torch.Generator().manual_seed(5)
    mask = torch.randint(0, 2, [600], generator=g).to(torch.float16) * -10000.0
    out = run_decode(q, k, v, attn_mask=mask)
    ref = ref_sdpa(q, k, v, attn_mask=mask.reshape(1, 1, 1, 600))
    check_close(out, ref)


def test_runtime_scale_decode():
    q, k, v = make_decode_qkv(batch=1, num_heads=4, num_kv_heads=4, kv_len=600, head_size=64)
    custom_scale = 0.3
    out = run_decode(q, k, v, scale=custom_scale)
    ref = ref_sdpa(q, k, v, scale=custom_scale)
    check_close(out, ref)


def test_edge_partition_sizes():
    # kv_len not a multiple of 256 (or of kv_step=16) exercises the last-partition/last-tile
    # tail masking in both the decode kernel and the finalization's partition count.
    for kv_len in [1, 15, 17, 100, 257, 300, 4097]:
        q, k, v = make_decode_qkv(batch=1, num_heads=4, num_kv_heads=4, kv_len=kv_len, head_size=64)
        out = run_decode(q, k, v)
        ref = ref_sdpa(q, k, v)
        check_close(out, ref)


def test_batch_gt1():
    q, k, v = make_decode_qkv(batch=3, num_heads=4, num_kv_heads=4, kv_len=300, head_size=64)
    out = run_decode(q, k, v)
    ref = ref_sdpa(q, k, v)
    check_close(out, ref)


def test_head_size_128():
    q, k, v = make_decode_qkv(batch=1, num_heads=4, num_kv_heads=4, kv_len=300, head_size=128)
    out = run_decode(q, k, v)
    ref = ref_sdpa(q, k, v)
    check_close(out, ref)


def test_transpose_order_matrix():
    # Q/O orders are addressing-invariant for a single-token query (Sq=1), so only K/V orders
    # are combinatorially exercised here; Q/O orders are still passed through (both values) to
    # confirm they compile and don't change results.
    q, k, v = make_decode_qkv(batch=2, num_heads=4, num_kv_heads=4, kv_len=300, head_size=64)
    ref = ref_sdpa(q, k, v)
    orders = [TRANSPOSE_ORDER_IDENTITY, TRANSPOSE_ORDER_SWAPPED]
    for k_order in orders:
        for v_order in orders:
            for qo_order in orders:
                out = run_decode(q, k, v, q_order=qo_order, k_order=k_order,
                                  v_order=v_order, o_order=qo_order)
                check_close(out, ref)


def test_consistency_with_prefill_sq1():
    # Stage 1a (prefill) with q_len=1 and Stage 2a (decode) on the same K/V should agree.
    q, k, v = make_decode_qkv(batch=1, num_heads=4, num_kv_heads=4, kv_len=500, head_size=64)
    out_decode = run_decode(q, k, v)
    out_prefill = prefill.run_prefill(q, k, v, is_causal=False)
    check_close(out_decode, out_prefill)


def test_combined_prefill_then_decode():
    # Simulated prefill -> generate transition: run Stage 1a over a prompt, then Stage 2a for
    # one generated token appended to the same K/V buffer, on the same underlying sequence.
    batch, num_heads, num_kv_heads, head_size = 1, 4, 4, 64
    prompt_len = 200
    q_p, k, v = make_qkv(batch=batch, num_heads=num_heads, num_kv_heads=num_kv_heads,
                          q_len=prompt_len, kv_len=prompt_len, head_size=head_size)
    out_prefill = prefill.run_prefill(q_p, k, v, is_causal=True)
    ref_prefill = prefill.ref_sdpa(q_p, k, v, is_causal=True)
    check_close(out_prefill, ref_prefill)

    # one new token attends to the whole prompt (kv unchanged, as if this new token's own
    # key/value were already appended and kv_len==prompt_len for this check).
    g = torch.Generator().manual_seed(7)
    q_new = torch.randint(-2, 3, [batch, num_heads, 1, head_size], generator=g).to(torch.float16)
    out_decode = run_decode(q_new, k, v)
    ref_decode = ref_sdpa(q_new, k, v)
    check_close(out_decode, ref_decode)


if __name__ == "__main__":
    cl.profiling(True)
    test_single_partition()
    print("test_single_partition PASSED")
    test_multi_partition()
    print("test_multi_partition PASSED")
    test_gqa_decode()
    print("test_gqa_decode PASSED")
    test_attn_mask_decode()
    print("test_attn_mask_decode PASSED")
    test_runtime_scale_decode()
    print("test_runtime_scale_decode PASSED")
    test_edge_partition_sizes()
    print("test_edge_partition_sizes PASSED")
    test_batch_gt1()
    print("test_batch_gt1 PASSED")
    test_head_size_128()
    print("test_head_size_128 PASSED")
    test_transpose_order_matrix()
    print("test_transpose_order_matrix PASSED")
    test_consistency_with_prefill_sq1()
    print("test_consistency_with_prefill_sq1 PASSED")
    test_combined_prefill_then_decode()
    print("test_combined_prefill_then_decode PASSED")
