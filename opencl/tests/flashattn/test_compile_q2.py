import sys
sys.path.insert(0, '.')
from clops import cl
import numpy as np

cl.profiling(True)
cwd = '/ceciliapeng/VM/aboutSHW/opencl/tests/flashattn'
scale_factor = 1.0/(64**0.5)
src1 = '#include "cm_sdpa_vlen.cm"'

for use_q2 in [0, 1]:
    print(f'Compiling USE_Q2={use_q2}...')
    try:
        k = cl.kernels(src1,
             f'-cmc -Qxcm_register_file_size=256 -mCM_printregusage -I{cwd}'
             f' -I/ceciliapeng/VM/aboutSHW/opencl/tests/pageatten'
             f' -DKERNEL_NAME=cm_sdpa_vlen'
             f' -DCMFLA_NUM_HEADS=16'
             f' -DCMFLA_NUM_KV_HEADS=16'
             f' -DCMFLA_HEAD_SIZE=64'
             f' -DCMFLA_SCALE_FACTOR={scale_factor}'
             f' -DCMFLA_IS_CAUSAL=0'
             f' -DUSE_Q2={use_q2}'
             f' -mdump_asm -g2')
        print(f'USE_Q2={use_q2} OK')
    except Exception as e:
        print(f'USE_Q2={use_q2} FAILED: {e}')
