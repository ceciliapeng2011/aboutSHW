import functools
import os

import numpy as np
import torch
import torch.nn.functional as F

from clops import cl

torch.manual_seed(0)

TRANSPOSE_ORDER_IDENTITY = 0  # [B, H, S, D]  -- OCL SDPA default order
TRANSPOSE_ORDER_SWAPPED = 1   # [B, S, H, D]  -- {0,2,1,3} heads<->seq swap fused in

CWD = os.path.dirname(os.path.realpath(__file__))
FLASHATTN_DIR = os.path.join(os.path.dirname(CWD), "flashattn")
PAGEATTEN_DIR = os.path.join(os.path.dirname(CWD), "pageatten")


def check_close(actual, expected, atol=1e-2, rtol=1e-2):
    actual = actual.to(torch.float32)
    expected = expected.to(torch.float32)
    diff = (actual - expected).abs()
    scale = torch.maximum(actual.abs(), expected.abs()).clamp(min=atol)
    rtol_max = (diff / scale).max().item()
    atol_max = diff.max().item()
    print(f"[check_close] atol_max={atol_max:.6f} rtol_max={rtol_max:.6f}")
    assert torch.allclose(actual, expected, atol=atol, rtol=rtol), (
        f"mismatch: atol_max={atol_max}, rtol_max={rtol_max}"
    )


@functools.cache
def get_cm_grf_width():
    k = cl.kernels(
        r'''extern "C" _GENX_MAIN_ void cm_get_grf_width(int* info [[type("svmptr_t")]]) {
        info[0] = CM_GRF_WIDTH;
    }''',
        "-cmc",
    )
    t = cl.tensor([2], np.dtype(np.int32))
    k.enqueue("cm_get_grf_width", [1], [1], t)
    return int(t.numpy()[0])


@functools.cache
def build_prefill_kernel(num_heads, num_kv_heads, head_size, is_causal,
                          q_order, k_order, v_order, o_order,
                          has_attn_mask=False, has_scale_input=False,
                          scale_factor=None):
    if scale_factor is None:
        scale_factor = 1.0 / (head_size ** 0.5)
    src = r'''#include "cm_sdpa_opt_prefill.cm"'''
    opts = (
        f'-cmc -Qxcm_jit_option="-abortonspill" -Qxcm_register_file_size=256 -mCM_printregusage '
        f"-I{CWD} -I{FLASHATTN_DIR} -I{PAGEATTEN_DIR} "
        f"-DKERNEL_NAME=cm_sdpa_opt_prefill "
        f"-DCMFLA_NUM_HEADS={num_heads} -DCMFLA_NUM_KV_HEADS={num_kv_heads} "
        f"-DCMFLA_HEAD_SIZE={head_size} -DCMFLA_SCALE_FACTOR={scale_factor} "
        f"-DCMFLA_IS_CAUSAL={int(is_causal)} "
        f"-DQ_TRANSPOSE_ORDER={q_order} -DK_TRANSPOSE_ORDER={k_order} "
        f"-DV_TRANSPOSE_ORDER={v_order} -DOUTPUT_TRANSPOSE_ORDER={o_order} "
        f"-DHAS_ATTN_MASK_INPUT={int(has_attn_mask)} -DHAS_SCALE_INPUT={int(has_scale_input)}"
    )
    return cl.kernels(src, opts)


def to_physical(t, order):
    # t: logical [B, H, S, D]. order SWAPPED -> physical [B, S, H, D] contiguous buffer.
    if order == TRANSPOSE_ORDER_SWAPPED:
        return t.transpose(1, 2).contiguous()
    return t.contiguous()


def from_physical(t_phys, order, batch, num_heads, seq_len, head_size):
    # Inverse of to_physical: reconstruct logical [B, H, S, D] from the physical buffer.
    if order == TRANSPOSE_ORDER_SWAPPED:
        t = t_phys.view(batch, seq_len, num_heads, head_size)
        return t.transpose(1, 2).contiguous()
    return t_phys.view(batch, num_heads, seq_len, head_size)


def run_prefill(q, k, v, is_causal,
                 q_order=TRANSPOSE_ORDER_IDENTITY, k_order=TRANSPOSE_ORDER_IDENTITY,
                 v_order=TRANSPOSE_ORDER_IDENTITY, o_order=TRANSPOSE_ORDER_IDENTITY,
                 attn_mask=None, scale=None):
    batch, num_heads, q_len, head_size = q.shape
    _, num_kv_heads, kv_len, _ = k.shape

    kernel = build_prefill_kernel(num_heads, num_kv_heads, head_size, is_causal,
                                   q_order, k_order, v_order, o_order,
                                   has_attn_mask=attn_mask is not None,
                                   has_scale_input=scale is not None,
                                   scale_factor=scale)

    q_phys = to_physical(q, q_order)
    k_phys = to_physical(k, k_order)
    v_phys = to_physical(v, v_order)

    t_q = cl.tensor(q_phys.to(torch.float16).numpy())
    t_k = cl.tensor(k_phys.to(torch.float16).numpy())
    t_v = cl.tensor(v_phys.to(torch.float16).numpy())
    out_shape = ([batch, q_len, num_heads, head_size] if o_order == TRANSPOSE_ORDER_SWAPPED
                 else [batch, num_heads, q_len, head_size])
    t_out = cl.tensor(np.zeros(out_shape, dtype=np.float16))

    q_step = get_cm_grf_width() // 32
    wg_size = 16
    wg_seq_len = wg_size * q_step
    wg_count = (q_len + wg_seq_len - 1) // wg_seq_len
    GWS = [1, batch * num_heads, wg_count * wg_size]
    LWS = [1, 1, wg_size]

    args = [t_q, t_k, t_v]
    if attn_mask is not None:
        # [1,1,q_len,kv_len], additive, broadcast over batch/heads -- same ABI as OCL sdpa_opt.
        # Row stride is padded to a multiple of 8 half-elements (16 bytes), matching the kernel's
        # mask_pitch_bytes -- the LSC 2D block descriptor needs a 16-byte-aligned pitch, which a
        # bare kv_len*sizeof(half) stride isn't guaranteed to give.
        kv_len_padded = (kv_len + 7) // 8 * 8
        mask_np = np.zeros((q_len, kv_len_padded), dtype=np.float16)
        mask_np[:, :kv_len] = attn_mask.reshape(q_len, kv_len).to(torch.float16).numpy()
        t_mask = cl.tensor(mask_np)
        args.append(t_mask)
    if scale is not None:
        args.append(float(scale))
    args.append(t_out)
    args += [q_len, kv_len]

    kernel.enqueue("cm_sdpa_opt_prefill", GWS, LWS, *args)

    out_phys = torch.from_numpy(t_out.numpy())
    return from_physical(out_phys, o_order, batch, num_heads, q_len, head_size)


def make_qkv(batch, num_heads, num_kv_heads, q_len, kv_len, head_size, seed=0):
    g = torch.Generator().manual_seed(seed)
    low, high = -2, 3
    q = torch.randint(low, high, [batch, num_heads, q_len, head_size], generator=g).to(torch.float16)
    k = torch.randint(low, high, [batch, num_kv_heads, kv_len, head_size], generator=g).to(torch.float16)
    v = (torch.randint(low, high, [batch, num_kv_heads, kv_len, head_size], generator=g).to(torch.float16)
         / high)
    return q, k, v


def ref_sdpa(q, k, v, is_causal, attn_mask=None, scale=None):
    num_heads = q.shape[1]
    num_kv_heads = k.shape[1]
    out = F.scaled_dot_product_attention(
        q.to(torch.float32), k.to(torch.float32), v.to(torch.float32),
        attn_mask=attn_mask.to(torch.float32) if attn_mask is not None else None,
        is_causal=is_causal, enable_gqa=(num_kv_heads != num_heads), scale=scale,
    )
    return out.to(torch.float16)


def test_basic_noncausal():
    q, k, v = make_qkv(batch=1, num_heads=4, num_kv_heads=4, q_len=256, kv_len=256, head_size=64)
    out = run_prefill(q, k, v, is_causal=False)
    ref = ref_sdpa(q, k, v, is_causal=False)
    check_close(out, ref)


def test_causal():
    q, k, v = make_qkv(batch=1, num_heads=4, num_kv_heads=4, q_len=256, kv_len=256, head_size=64)
    out = run_prefill(q, k, v, is_causal=True)
    ref = ref_sdpa(q, k, v, is_causal=True)
    check_close(out, ref)


def test_gqa():
    q, k, v = make_qkv(batch=1, num_heads=8, num_kv_heads=2, q_len=256, kv_len=256, head_size=64)
    out = run_prefill(q, k, v, is_causal=False)
    ref = ref_sdpa(q, k, v, is_causal=False)
    check_close(out, ref)


def test_head_size_128():
    q, k, v = make_qkv(batch=1, num_heads=4, num_kv_heads=4, q_len=256, kv_len=256, head_size=128)
    out = run_prefill(q, k, v, is_causal=False)
    ref = ref_sdpa(q, k, v, is_causal=False)
    check_close(out, ref)


def test_tail_shapes():
    q, k, v = make_qkv(batch=1, num_heads=4, num_kv_heads=4, q_len=37, kv_len=100, head_size=64)
    out = run_prefill(q, k, v, is_causal=False)
    ref = ref_sdpa(q, k, v, is_causal=False)
    check_close(out, ref)


def test_causal_tail_shapes():
    q, k, v = make_qkv(batch=1, num_heads=4, num_kv_heads=4, q_len=200, kv_len=200, head_size=64)
    out = run_prefill(q, k, v, is_causal=True)
    ref = ref_sdpa(q, k, v, is_causal=True)
    check_close(out, ref)


def test_batch_gt1():
    q, k, v = make_qkv(batch=3, num_heads=4, num_kv_heads=4, q_len=128, kv_len=128, head_size=64)
    out = run_prefill(q, k, v, is_causal=False)
    ref = ref_sdpa(q, k, v, is_causal=False)
    check_close(out, ref)


def test_transpose_order_matrix():
    q, k, v = make_qkv(batch=2, num_heads=4, num_kv_heads=4, q_len=128, kv_len=128, head_size=64)
    ref = ref_sdpa(q, k, v, is_causal=False)
    orders = [TRANSPOSE_ORDER_IDENTITY, TRANSPOSE_ORDER_SWAPPED]
    for q_order in orders:
        for k_order in orders:
            for v_order in orders:
                for o_order in orders:
                    out = run_prefill(q, k, v, is_causal=False,
                                       q_order=q_order, k_order=k_order,
                                       v_order=v_order, o_order=o_order)
                    check_close(out, ref)


def test_attn_mask():
    q, k, v = make_qkv(batch=1, num_heads=4, num_kv_heads=4, q_len=128, kv_len=128, head_size=64)
    g = torch.Generator().manual_seed(1)
    mask = torch.randint(0, 2, [128, 128], generator=g).to(torch.float16) * -10000.0
    out = run_prefill(q, k, v, is_causal=False, attn_mask=mask)
    ref = ref_sdpa(q, k, v, is_causal=False, attn_mask=mask)
    check_close(out, ref)


def test_attn_mask_causal():
    q, k, v = make_qkv(batch=1, num_heads=4, num_kv_heads=4, q_len=128, kv_len=128, head_size=64)
    g = torch.Generator().manual_seed(2)
    mask = torch.randint(0, 2, [128, 128], generator=g).to(torch.float16) * -10000.0
    out = run_prefill(q, k, v, is_causal=True, attn_mask=mask)
    ref = ref_sdpa(q, k, v, is_causal=True, attn_mask=mask)
    check_close(out, ref)


def test_runtime_scale():
    q, k, v = make_qkv(batch=1, num_heads=4, num_kv_heads=4, q_len=128, kv_len=128, head_size=64)
    custom_scale = 0.35
    out = run_prefill(q, k, v, is_causal=False, scale=custom_scale)
    ref = ref_sdpa(q, k, v, is_causal=False, scale=custom_scale)
    check_close(out, ref)


def test_attn_mask_and_runtime_scale():
    q, k, v = make_qkv(batch=2, num_heads=8, num_kv_heads=2, q_len=100, kv_len=100, head_size=64)
    g = torch.Generator().manual_seed(3)
    mask = torch.randint(0, 2, [100, 100], generator=g).to(torch.float16) * -10000.0
    custom_scale = 0.2
    out = run_prefill(q, k, v, is_causal=False, attn_mask=mask, scale=custom_scale)
    ref = ref_sdpa(q, k, v, is_causal=False, attn_mask=mask, scale=custom_scale)
    check_close(out, ref)


if __name__ == "__main__":
    cl.profiling(True)
    test_basic_noncausal()
    print("test_basic_noncausal PASSED")
    test_causal()
    print("test_causal PASSED")
    test_gqa()
    print("test_gqa PASSED")
    test_head_size_128()
    print("test_head_size_128 PASSED")
    test_tail_shapes()
    print("test_tail_shapes PASSED")
    test_causal_tail_shapes()
    print("test_causal_tail_shapes PASSED")
    test_batch_gt1()
    print("test_batch_gt1 PASSED")
    test_transpose_order_matrix()
    print("test_transpose_order_matrix PASSED")
    test_attn_mask()
    print("test_attn_mask PASSED")
    test_attn_mask_causal()
    print("test_attn_mask_causal PASSED")
    test_runtime_scale()
    print("test_runtime_scale PASSED")
    test_attn_mask_and_runtime_scale()
    print("test_attn_mask_and_runtime_scale PASSED")
