import torch
import torch.nn as nn
import torch.nn.functional as F

import functools

from clops import cl
from clops.utils import Colors
import os

import numpy as np

def get_cm_grf_width():
    cm_kernels = cl.kernels(r'''
    extern "C" _GENX_MAIN_ void cm_get_grf_width(int * info [[type("svmptr_t")]]) {
        info[0] = CM_GRF_WIDTH;
    }''', f"-cmc")
    t_info = cl.tensor([2], np.dtype(np.int32))
    cm_kernels.enqueue("cm_get_grf_width", [1], [1], t_info)
    return t_info.numpy()[0]

CM_GRF_WIDTH = get_cm_grf_width()

def round_up_to_tile(value, tile_size):
    return (value + tile_size - 1) // tile_size * tile_size

def get_default_kv_blk(head_size):
    padded_head_size = round_up_to_tile(head_size, 16)
    tail_size = padded_head_size - head_size
    # KV_BLK=2 amortizes softmax/rescale overhead but increases live tiled state.
    # Use the CM kernel's padded head shape, not only the model-visible head size:
    #   - padded<=64 and exact/near-exact 16-aligned heads benchmark well with KV_BLK=2;
    #   - tail-heavy padded heads, including Omni HD=72 -> padded HD=80, prefer KV_BLK=1.
    if padded_head_size <= 64 or tail_size <= 1:
        return 2
    return 1

class flash_attn_cm:
    def __init__(self, num_heads, num_kv_heads, head_size, is_causal = False):
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.is_causal = is_causal
        src1 = r'''#include "cm_sdpa_vlen.cm"'''
        cwd = os.path.dirname(os.path.realpath(__file__))
        print(f"compiling {cwd} {num_heads=} {head_size=} ...")

        scale_factor = 1.0/(head_size**0.5)
        default_kv_blk = get_default_kv_blk(head_size)
        kv_blk = int(os.environ.get("CMFLA_KV_BLK", str(default_kv_blk)))
        self.kernels = cl.kernels(src1,
                     (f'-cmc -Qxcm_jit_option="-abortonspill" -Qxcm_register_file_size=256  -mCM_printregusage -I{cwd}'
                      f' -I{os.path.dirname(os.path.dirname(cwd))}/tests/pageatten'
                      f" -DKERNEL_NAME=cm_sdpa_vlen"
                      f" -DCMFLA_NUM_HEADS={num_heads}"
                      f" -DCMFLA_NUM_KV_HEADS={num_kv_heads}"
                      f" -DCMFLA_HEAD_SIZE={head_size}"
                      f" -DCMFLA_SCALE_FACTOR={scale_factor}"
                      f" -DCMFLA_IS_CAUSAL={int(is_causal)}"
                      f" -DCMFLA_KV_BLK={kv_blk}"
                      f" -mdump_asm -g2")
                     )

    def qkv_fused(self, qkv, n_repeats = 1):
        seq_len, total_heads, head_size = qkv.shape
        old_dtype = qkv.dtype
        assert total_heads == (self.num_heads + self.num_kv_heads * 2)
        assert head_size == self.head_size
        t_qkv = cl.tensor(qkv.to(torch.float16).detach().numpy())
        t_out = cl.tensor([seq_len, self.num_heads, self.head_size], np.dtype(np.float16))
        wg_size = 16
        q_step = CM_GRF_WIDTH//32 # or 8 on Xe1
        wg_seq_len = wg_size * q_step
        wg_count = (seq_len + wg_seq_len - 1) // wg_seq_len
        GWS = [1, self.num_heads, wg_count * wg_size]
        LWS = [1, 1, wg_size]
        print(f"calling qkv_fused {GWS=} {LWS=} x {n_repeats} times")
        for _ in range(n_repeats):
            self.kernels.enqueue("cm_sdpa_qkv_fused", GWS, LWS, seq_len, t_qkv, t_out)
        attn_output = torch.from_numpy(t_out.numpy()).to(old_dtype)
        return attn_output

    def __call__(self, q, k, v, cu_seqlens, n_repeats = 1):
        q_len = q.shape[0]
        kv_len = k.shape[0]
        old_dtype = q.dtype
        assert q_len == kv_len
        t_q = cl.tensor(q.to(torch.float16).detach().numpy())
        t_k = cl.tensor(k.to(torch.float16).detach().numpy())
        t_v = cl.tensor(v.to(torch.float16).detach().numpy())
        t_cu_seqlens = cl.tensor(cu_seqlens.numpy().astype(np.int32))
        t_out = cl.tensor([q.shape[0], self.num_heads, self.head_size], np.dtype(np.float16))

        q_step = CM_GRF_WIDTH // 32
        max_seq_len = int((cu_seqlens[1:] - cu_seqlens[:-1]).max())
        wg_size = min(16, (max_seq_len + q_step - 1) // q_step)
        wg_seq_len = wg_size * q_step
        need_wg_mapping = 0
        if wg_size == 16 and max_seq_len > wg_seq_len:
            need_wg_mapping = 1

        if need_wg_mapping:
            wg_count = 0
            for i in range(len(cu_seqlens) - 1):
                seq_len = int(cu_seqlens[i + 1]) - int(cu_seqlens[i])
                wg_count += (seq_len + wg_seq_len - 1) // wg_seq_len
        else:
            wg_count = len(cu_seqlens) - 1

        GWS = [self.num_heads, wg_count * wg_size]
        LWS = [1, wg_size]
        print(f"calling {q_step=} {need_wg_mapping=} {wg_count=} {GWS=} {LWS=}")
        for _ in range(n_repeats):
            self.kernels.enqueue("cm_sdpa_vlen", GWS, LWS, t_q, t_k, t_v, t_out, t_cu_seqlens, need_wg_mapping, 0, 0)
        attn_output = torch.from_numpy(t_out.numpy()).to(old_dtype)
        return attn_output

    @staticmethod
    @functools.cache
    def create_instance(num_heads, num_kv_heads, head_size, is_causal):
        return flash_attn_cm(num_heads, num_kv_heads, head_size, is_causal)

def flash_attn_vlen_ref(q, k, v, cu_seqlens, is_causal = False):
    seq_length, num_heads, head_size = q.shape
    kv_seq_length, num_kv_heads, head_size = k.shape
    old_dtype = q.dtype
    q = q.transpose(0, 1)
    k = k.transpose(0, 1)
    v = v.transpose(0, 1)

    # print(f"============2 {cu_seqlens=} {seq_length=} {num_heads=}")
    # print(f"============2 {q.shape=} {q.is_contiguous()=} {k.shape=} {k.is_contiguous()=} {v.shape=} {v.is_contiguous()=}")
    if is_causal:
        attn_output = F.scaled_dot_product_attention(
            q.unsqueeze(0).to(torch.float16), k.unsqueeze(0).to(torch.float16), v.unsqueeze(0).to(torch.float16),
            is_causal = True,
            dropout_p=0.0,
            enable_gqa = (num_kv_heads != num_heads)
        )
    else:
        attention_mask = torch.zeros([1, seq_length, seq_length], device=q.device, dtype=torch.bool)
        if len(cu_seqlens):
            for i in range(1, len(cu_seqlens)):
                attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = True        
        else:
            attention_mask[...] = True

        attn_output = F.scaled_dot_product_attention(
            q.unsqueeze(0).to(torch.float16), k.unsqueeze(0).to(torch.float16), v.unsqueeze(0).to(torch.float16),
            attn_mask = attention_mask,
            dropout_p=0.0,
            enable_gqa = (num_kv_heads != num_heads)
        )
    attn_output = attn_output.squeeze(0).transpose(0, 1)
    # print(f"============2 {attn_output.shape=} ")    
    print(".")
    return attn_output.to(old_dtype)


def flash_attn(q, k, v, cu_seqlens):
    _, num_heads, head_size = q.shape
    func = flash_attn_cm.create_instance(num_heads, head_size)
    # return flash_attn_vlen_ref(q, k, v, cu_seqlens)
    return func(q, k, v, cu_seqlens)


def check_close(input, other, atol=1e-3, rtol=1e-3):
    print(f"[check_close] {input.shape}{input.dtype} vs {other.shape}{other.dtype}")
    diff = (input - other).abs()
    # Clamp denominator at atol so near-zero values don't inflate the relative metric.
    # When both tensors are tiny (< atol), absolute error is the meaningful measure anyway.
    scale = torch.maximum(input.abs(), other.abs()).clamp(min=atol)
    rtol_max = (diff / scale).max()
    atol_max = diff.max()
    print(f"[check_close] rtol_max: {rtol_max}")
    print(f"[check_close] atol_max: {atol_max}")
    if not torch.allclose(input, other, atol=atol, rtol=rtol):
        close_check = torch.isclose(input, other, atol=atol, rtol=rtol)
        not_close_indices = torch.where(~close_check) # Invert the close check to find failures
        print(f"Not close indices: {not_close_indices}")
        print(f"    input_tensor: {input[not_close_indices]}")
        print(f"    other_tensor: {other[not_close_indices]}")
        assert 0

def test_flash_attn_cm(seq_len, sub_seq_len, num_heads = 16, num_kv_heads = 16, head_size = 80, acc_check = True):
    cl.profiling(True)
    torch.manual_seed(0)
    torch.set_printoptions(linewidth=1024)

    import numpy as np
    q_len = kv_len = seq_len
    cu_seqlens = torch.tensor([i for i in range(0, seq_len, sub_seq_len)] + [seq_len], dtype=torch.int32)
    # print(f'{cu_seqlens=}')

    low = -1
    high = 2
    act_dtype = torch.float16
    q = torch.randint(low, high, [q_len, num_heads, head_size]).to(dtype=act_dtype)
    k = torch.randint(low, high, [kv_len, num_kv_heads, head_size]).to(dtype=act_dtype)
    v = torch.randint(low, high, [kv_len, num_kv_heads, head_size]).to(dtype=act_dtype)/high

    func = flash_attn_cm.create_instance(num_heads, num_kv_heads, head_size, False)
    out = func(q, k, v, cu_seqlens) # warmup
    if acc_check:
        ref = flash_attn_vlen_ref(q, k, v, cu_seqlens)
        check_close(ref, out)
    cl.finish()

    out = func(q, k, v, cu_seqlens, 100)
    latency = cl.finish()
    # for i,ns in enumerate(latency): print(f"[{i}]  {ns*1e-6:.3f} ms")
    avg_ms = sum(latency[10:]) / len(latency[10:]) * 1e-6
    num_seqs = len(cu_seqlens) - 1
    real_flops = num_seqs * sub_seq_len * sub_seq_len * 4 * num_heads * head_size
    hw_peak_flops = 20e12  # PTL 4xe XMX FP16 peak
    utilization = real_flops / (avg_ms * 1e-3) / hw_peak_flops * 100
    print(f" {seq_len=} {sub_seq_len=} average latency: {Colors.BOLD}{Colors.YELLOW}{avg_ms:.3f} ms{Colors.END}"
          f"  |  {real_flops/1e9:.1f} GFLOP"
          f"  |  {Colors.BOLD}{Colors.GREEN}{utilization:.1f}%{Colors.END} of {hw_peak_flops/1e12:.0f} TFLOPS XMX peak")
    return avg_ms, utilization


def test_flash_attn_causal_batch1(seq_len, num_heads = 16, num_kv_heads = 16, head_size = 80):
    cl.profiling(True)
    torch.manual_seed(0)
    torch.set_printoptions(linewidth=1024)
    
    import numpy as np

    low = -1
    high = 2
    act_dtype = torch.float16
    q = torch.randint(low, high, [seq_len, num_heads, head_size]).to(dtype=act_dtype)
    k = torch.randint(low, high, [seq_len, num_kv_heads, head_size]).to(dtype=act_dtype)
    v = torch.randint(low, high, [seq_len, num_kv_heads, head_size]).to(dtype=act_dtype)/high

    is_causal = True
    ref = flash_attn_vlen_ref(q, k, v, [], is_causal=is_causal)

    func = flash_attn_cm.create_instance(num_heads, num_kv_heads, head_size, is_causal)
    
    qkv = torch.cat((q,k,v), 1)
    
    out = func.qkv_fused(qkv)
    out = func.qkv_fused(qkv, n_repeats=20)
    latency = cl.finish()
    # for i,ns in enumerate(latency): print(f"[{i}]  {ns*1e-6:.3f} ms")
    print(f" qkv_fused_causal {seq_len=} average latency: {sum(latency[10:])/len(latency[10:])*1e-6:.3f} ms")
    check_close(ref, out)
    #assert 0

if __name__ == "__main__":
    # test_flash_attn_causal_batch1(seq_len=8192, num_heads = 28, num_kv_heads = 4, head_size = 128)
    # for seqlen in range(1025, 1055, 1):
    #     test_flash_attn_causal_batch1(seqlen, num_heads = 28, num_kv_heads = 4, head_size = 128)
    # test_flash_attn_causal_batch1(113, num_heads = 28, num_kv_heads = 4, head_size = 128)

    # test_flash_attn_cm(8192, 8192, num_heads = 28, num_kv_heads = 4, head_size = 128, acc_check=False)
    # test_flash_attn_cm(8192, 8192, acc_check=False)
    # test_flash_attn_cm(8192, 1024)
    # test_flash_attn_cm(8192, 64)
    # test_flash_attn_cm(8190, 64)
    # test_flash_attn_cm(seq_len=32, sub_seq_len=14, num_heads = 28, num_kv_heads = 4, head_size = 128)

    # for seqlen in range(1, 1055, 1):
    #     for sub_seq_len in range(1, 64, 1):
    #         test_flash_attn_cm(seqlen, sub_seq_len, num_heads = 1, num_kv_heads = 1, head_size = 128)

    # Qwen3-Omni-4B vs Qwen3-VL-4B VLSDPA utilization sweep (PTL 4Xe)
    # seq_len/sub_seq_len = M_v(total)/M-per-image from perf_analysis_omni_vs_vl.md §4.2b
    CASES = [
        ("C1: 448×448×2",   1568,  784),
        ("C2: 512×384×2",   1536,  768),
        ("C3: 1024×512×2",  4096,  2048),
        ("C4: 1260×700×2",  6864,  3432),
        ("C5: 1280×768×15", 57600, 3840),
    ]
    TARGET_EFF = 55.0  # % XMX utilization target

    omni_results = []
    for label, seq_len, sub_seq_len in CASES:
        acc_check = seq_len < 50000
        avg_ms, eff = test_flash_attn_cm(seq_len=seq_len, sub_seq_len=sub_seq_len,
                                         num_heads=16, num_kv_heads=16, head_size=72,
                                         acc_check=acc_check)
        omni_results.append((label, sub_seq_len, avg_ms, eff))

    vl_results = []
    for label, seq_len, sub_seq_len in CASES:
        acc_check = seq_len < 50000
        avg_ms, eff = test_flash_attn_cm(seq_len=seq_len, sub_seq_len=sub_seq_len,
                                         num_heads=16, num_kv_heads=16, head_size=64,
                                         acc_check=acc_check)
        vl_results.append((label, sub_seq_len, avg_ms, eff))

    def print_summary(title, omni_head_size_for_eff):
        print()
        print("=" * 112)
        print(f"  {title}")
        print(f"  Target: {TARGET_EFF}% XMX FP16 efficiency  |  Peak: 20.07 TFLOPS")
        print("=" * 112)
        hdr = (f"  {'Test':<22}  {'M/img':>6}  {'Omni ms':>9}  {'Omni Eff%':>10}  {'Omni vs55%':>11}"
               f"  {'VL ms':>9}  {'VL Eff%':>8}  {'VL vs55%':>9}  {'Delta Eff':>10}")
        print(hdr)
        print("-" * 112)
        for (label, seq_len, sub_seq_len), (_, m_img, omni_ms, _), (_, _, vl_ms, vl_eff) in zip(
                CASES, omni_results, vl_results):
            num_seqs = seq_len // sub_seq_len
            real_flops = num_seqs * m_img * m_img * 4 * 16 * omni_head_size_for_eff
            hw_peak_flops = 20e12
            omni_eff = real_flops / (omni_ms * 1e-3) / hw_peak_flops * 100
            omni_vs55 = omni_eff / TARGET_EFF * 100
            vl_vs55   = vl_eff   / TARGET_EFF * 100
            delta     = vl_eff - omni_eff
            omni_pass = "✅" if omni_vs55 >= 100 else "❌"
            vl_pass   = "✅" if vl_vs55   >= 100 else "❌"
            print(f"  {label:<22}  {m_img:>6}  {omni_ms:>9.2f}  {omni_eff:>9.1f}%  {omni_vs55:>9.1f}% {omni_pass}  {vl_ms:>9.2f}  {vl_eff:>7.1f}%  {vl_vs55:>8.1f}% {vl_pass}  VL+{delta:>5.1f}pp")
        print("=" * 112)

    # Group 1: model-visible useful work. Omni uses HD=72, VL uses HD=64.
    print_summary(
        "Group 1: useful-HD utilization — Omni HD=72 vs VL HD=64",
        omni_head_size_for_eff=72,
    )

    # Group 2: executed tile work. Omni HD=72 is rounded to padded HD=80 by the CM kernel.
    print_summary(
        "Group 2: padded-HD utilization — Omni executed HD=80 vs VL HD=64",
        omni_head_size_for_eff=80,
    )