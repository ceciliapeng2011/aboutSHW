import sys; sys.path.insert(0,'.')
from clops import cl
cwd='/ceciliapeng/VM/aboutSHW/opencl/tests/flashattn'
src='#include "cm_sdpa_vlen.cm"'
flags=(f'-cmc -Qxcm_register_file_size=256 -I{cwd}'
       f' -I/ceciliapeng/VM/aboutSHW/opencl/tests/pageatten'
       f' -DKERNEL_NAME=cm_sdpa_vlen -DCMFLA_NUM_HEADS=16 -DCMFLA_NUM_KV_HEADS=16'
       f' -DCMFLA_HEAD_SIZE=64 -DCMFLA_SCALE_FACTOR=0.125 -DCMFLA_IS_CAUSAL=0'
       f' -DUSE_Q2=0 -DSKIP_TRANSPOSE=1')
print('Compiling...')
k=cl.kernels(src,flags)
print('OK')
