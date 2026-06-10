import torch, numpy as np, sys
sys.path.insert(0, '.')
from clops import cl
from cmfla import flash_attn_cm
cl.profiling(True)
torch.manual_seed(0)

configs = [
    (2,   3432, '2 seqs x 3432'),
    (16,  512,  '16 seqs x 512'),
    (128, 64,   '128 seqs x 64'),
    (15,  3840, '15 seqs x 3840 (57600 total)'),
]
num_heads = 16; num_kv_heads = 16; head_size = 64
act_dtype = torch.float16
func = flash_attn_cm.create_instance(num_heads, num_kv_heads, head_size, False)

for num_seqs, sub_len, label in configs:
    seq_len = num_seqs * sub_len
    cu_seqlens = torch.arange(0, seq_len+1, sub_len, dtype=torch.int32)
    q = torch.randn(seq_len, num_heads,    head_size, dtype=act_dtype)
    k = torch.randn(seq_len, num_kv_heads, head_size, dtype=act_dtype)
    v = torch.randn(seq_len, num_kv_heads, head_size, dtype=act_dtype)
    func(q, k, v, cu_seqlens); cl.finish()
    func(q, k, v, cu_seqlens, 100)
    lat = cl.finish()
    avg = sum(lat[10:])/len(lat[10:])*1e-6
    print(f'{label}: {avg:.3f} ms')
