"""
Benchmark item 9 (2 Q-rows/thread) with -Qxcm_register_file_size=512 (2 contexts/EU)
vs baseline (256 GRF, 4 contexts/EU).
"""
import torch, numpy as np, sys
sys.path.insert(0, '.')
from clops import cl
import os

cl.profiling(True)
torch.manual_seed(0)

cwd = os.path.dirname(os.path.realpath('/ceciliapeng/VM/aboutSHW/opencl/tests/flashattn/cmfla.py'))
cwd = '/ceciliapeng/VM/aboutSHW/opencl/tests/flashattn'
scale_factor = 1.0 / (64 ** 0.5)
src1 = '#include "cm_sdpa_vlen.cm"'

def make_kernel(use_q2, reg_size):
    flags = (f'-cmc -Qxcm_register_file_size={reg_size} -mCM_printregusage -I{cwd}'
             f' -I/ceciliapeng/VM/aboutSHW/opencl/tests/pageatten'
             f' -DKERNEL_NAME=cm_sdpa_vlen'
             f' -DCMFLA_NUM_HEADS=16'
             f' -DCMFLA_NUM_KV_HEADS=16'
             f' -DCMFLA_HEAD_SIZE=64'
             f' -DCMFLA_SCALE_FACTOR={scale_factor}'
             f' -DCMFLA_IS_CAUSAL=0'
             f' -DUSE_Q2={int(use_q2)}'
             f' -mdump_asm -g2')
    print(f'Compiling use_q2={use_q2} reg_size={reg_size}...')
    return cl.kernels(src1, flags)

configs = [
    (2,   3432, '2 seqs x 3432'),
    (16,  512,  '16 seqs x 512'),
    (128, 64,   '128 seqs x 64'),
    (15,  3840, '15 seqs x 3840'),
]
num_heads = 16; num_kv_heads = 16; head_size = 64
act_dtype = torch.float16
q_step = 16  # CM_GRF_WIDTH//32

def run_bench(kernels, use_q2, configs):
    tokens_per_thread = 2 * q_step if use_q2 else q_step
    for num_seqs, sub_len, label in configs:
        seq_len = num_seqs * sub_len
        cu_seqlens = torch.arange(0, seq_len + 1, sub_len, dtype=torch.int32)
        q = torch.randn(seq_len, num_heads,    head_size, dtype=act_dtype)
        k = torch.randn(seq_len, num_kv_heads, head_size, dtype=act_dtype)
        v = torch.randn(seq_len, num_kv_heads, head_size, dtype=act_dtype)

        import numpy as np2
        max_seq_len = int((cu_seqlens[1:] - cu_seqlens[:-1]).max())
        wg_size = min(16, (max_seq_len + tokens_per_thread - 1) // tokens_per_thread)
        wg_seq_len = wg_size * tokens_per_thread
        mapping = []
        for i in range(len(cu_seqlens) - 1):
            seq_start = int(cu_seqlens[i])
            seq_end   = int(cu_seqlens[i + 1])
            slen      = seq_end - seq_start
            for k_blk in range((slen + wg_seq_len - 1) // wg_seq_len):
                mapping.append(seq_start + k_blk * wg_seq_len)
                mapping.append(i)
        wg_count = len(mapping) // 2

        t_q = cl.tensor(q.numpy())
        t_k = cl.tensor(k.numpy())
        t_v = cl.tensor(v.numpy())
        t_cu = cl.tensor(np2.array(cu_seqlens.numpy(), dtype=np2.int32))
        t_map = cl.tensor(np2.array(mapping, dtype=np2.int32))
        t_out = cl.tensor([seq_len, num_heads, head_size], np2.dtype(np2.float16))

        GWS = [num_heads, wg_count * wg_size]
        LWS = [1, wg_size]

        # warm-up
        kernels.enqueue("cm_sdpa_vlen", GWS, LWS, t_q, t_k, t_v, t_out, t_cu, t_map, 0, 0)
        cl.finish()
        for _ in range(100):
            kernels.enqueue("cm_sdpa_vlen", GWS, LWS, t_q, t_k, t_v, t_out, t_cu, t_map, 0, 0)
        lat = cl.finish()
        avg = sum(lat[10:]) / len(lat[10:]) * 1e-6
        print(f'  {label}: {avg:.3f} ms')

# Build all variants first, then benchmark
print('=== Building kernels ===')
k_base  = make_kernel(use_q2=False, reg_size=256)   # baseline
k_q2_256 = make_kernel(use_q2=True,  reg_size=256)  # item 9 with spill (reference)
k_q2_512 = make_kernel(use_q2=True,  reg_size=512)  # item 9 without spill (2 ctx/EU)
k_base_512 = make_kernel(use_q2=False, reg_size=512) # baseline in 512-GRF mode (2 ctx)

print('\n=== Baseline (256 GRF, 4 ctx/EU) ===')
run_bench(k_base, use_q2=False, configs=configs)

print('\n=== 512 GRF only, q1 (2 ctx/EU, same work/thread) ===')
run_bench(k_base_512, use_q2=False, configs=configs)

print('\n=== Item 9 + 256 GRF (spill, 4 ctx/EU) ===')
run_bench(k_q2_256, use_q2=True, configs=configs)

print('\n=== Item 9 + 512 GRF (no spill, 2 ctx/EU) ===')
run_bench(k_q2_512, use_q2=True, configs=configs)
