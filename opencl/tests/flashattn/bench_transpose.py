"""
Item 12 upper-bound: measure perf with transpose removed (SKIP_TRANSPOSE=1).
Results will be wrong but cycle count is valid for bounding the win.
"""
import torch, numpy as np, sys
sys.path.insert(0, '.')
from clops import cl

cl.profiling(True)
torch.manual_seed(0)

cwd = '/ceciliapeng/VM/aboutSHW/opencl/tests/flashattn'
scale_factor = 1.0 / (64 ** 0.5)
src1 = '#include "cm_sdpa_vlen.cm"'

def make_kernel(skip_transpose):
    flags = (f'-cmc -Qxcm_jit_option="-abortonspill" -Qxcm_register_file_size=256 -I{cwd}'
             f' -I/ceciliapeng/VM/aboutSHW/opencl/tests/pageatten'
             f' -DKERNEL_NAME=cm_sdpa_vlen'
             f' -DCMFLA_NUM_HEADS=16 -DCMFLA_NUM_KV_HEADS=16 -DCMFLA_HEAD_SIZE=64'
             f' -DCMFLA_SCALE_FACTOR={scale_factor} -DCMFLA_IS_CAUSAL=0 -DUSE_Q2=0'
             + (' -DSKIP_TRANSPOSE=1' if skip_transpose else ''))
    print(f'Compiling skip_transpose={skip_transpose}...')
    return cl.kernels(src1, flags)

configs = [
    (2,   3432, '2 seqs x 3432'),
    (16,  512,  '16 seqs x 512'),
    (128, 64,   '128 seqs x 64'),
    (15,  3840, '15 seqs x 3840'),
]
num_heads = 16; num_kv_heads = 16; head_size = 64
act_dtype = torch.float16; q_step = 16

def run_bench(kernels, configs):
    for num_seqs, sub_len, label in configs:
        seq_len = num_seqs * sub_len
        cu_seqlens = torch.arange(0, seq_len + 1, sub_len, dtype=torch.int32)
        q = torch.randn(seq_len, num_heads,    head_size, dtype=act_dtype)
        k = torch.randn(seq_len, num_kv_heads, head_size, dtype=act_dtype)
        v = torch.randn(seq_len, num_kv_heads, head_size, dtype=act_dtype)
        max_seq_len = int((cu_seqlens[1:] - cu_seqlens[:-1]).max())
        wg_size = min(16, (max_seq_len + q_step - 1) // q_step)
        wg_seq_len = wg_size * q_step
        mapping = []
        for i in range(len(cu_seqlens) - 1):
            s, e = int(cu_seqlens[i]), int(cu_seqlens[i+1])
            for blk in range((e - s + wg_seq_len - 1) // wg_seq_len):
                mapping += [s + blk * wg_seq_len, i]
        wg_count = len(mapping) // 2
        t_q = cl.tensor(q.numpy()); t_k = cl.tensor(k.numpy()); t_v = cl.tensor(v.numpy())
        t_cu = cl.tensor(np.array(cu_seqlens.numpy(), dtype=np.int32))
        t_map = cl.tensor(np.array(mapping, dtype=np.int32))
        t_out = cl.tensor([seq_len, num_heads, head_size], np.dtype(np.float16))
        GWS = [num_heads, wg_count * wg_size]; LWS = [1, wg_size]
        kernels.enqueue("cm_sdpa_vlen", GWS, LWS, t_q, t_k, t_v, t_out, t_cu, t_map, 0, 0)
        cl.finish()
        for _ in range(100):
            kernels.enqueue("cm_sdpa_vlen", GWS, LWS, t_q, t_k, t_v, t_out, t_cu, t_map, 0, 0)
        lat = cl.finish()
        avg = sum(lat[10:]) / len(lat[10:]) * 1e-6
        print(f'  {label}: {avg:.3f} ms')

k_base = make_kernel(skip_transpose=False)
k_skip = make_kernel(skip_transpose=True)

print('\n--- Baseline (with transpose) ---')
run_bench(k_base, configs)
print('\n--- Item 12 upper-bound (transpose skipped) ---')
run_bench(k_skip, configs)
