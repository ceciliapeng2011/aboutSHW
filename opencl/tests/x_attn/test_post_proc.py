import os
import torch
import numpy as np

from clops import cl
from clops.utils import Colors
from clops import compare

def div_up(a, b):
    return (a + b - 1) // b
def rnd_up(a, b):
    return (a + b - 1) // b * b

#xe_arch: 1: xe, 2: xe2
def get_cm_grf_width():
    cm_kernels = cl.kernels(r'''
    extern "C" _GENX_MAIN_ void cm_get_grf_width(int * info [[type("svmptr_t")]]) {
        info[0] = CM_GRF_WIDTH;
    }''', f"-cmc")
    t_info = cl.tensor([2], np.dtype(np.int32))
    cm_kernels.enqueue("cm_get_grf_width", [1], [1], t_info)
    return t_info.numpy()[0]
CM_GRF_WIDTH = get_cm_grf_width()
PA_WG_SIZE = 256 if CM_GRF_WIDTH == 512 else 128

def test_post_proc(HQ = 1, BLOCK_SIZE = 128):
    assert PA_WG_SIZE % BLOCK_SIZE == 0, "PA WG size must be divisible by XATTN block size."
    MERGED_Q_NUM = PA_WG_SIZE // BLOCK_SIZE
    assert MERGED_Q_NUM == 1 or MERGED_Q_NUM == 2, "PA WG size must be no less than XATTN block size."

    STRIDE = 16
    def create_kernels():
        # kernel
        src = r'''#include "xattn_post_proc.hpp"'''
        cwd = os.path.dirname(os.path.realpath(__file__))
        print(f"compiling {cwd} ...")

        jit_option = '-abortonspill -noschedule '
        kernels = cl.kernels(src, f'''-cmc -Qxcm_jit_option="{jit_option}"
                            -mCM_printregusage -mdump_asm -g2
                            -Qxcm_register_file_size=256 -I{cwd}
                            -DSTRIDE={STRIDE} -DHQ={HQ}
                            -DBLOCK_SIZE={BLOCK_SIZE}
                            -DMERGED_Q_NUM={MERGED_Q_NUM}''')
        return kernels

    kernels = create_kernels()
    sum_per_token_in_block = BLOCK_SIZE // STRIDE
    qk = [
        (4096, 4096),
        (4096+1, 4096),
        (4096, 32768),
        (4096+1, 32768)
        # (612, 612)
    ]
    for q, k in qk:
        print(f"{Colors.GREEN}Test {q=} {k=} {BLOCK_SIZE=} ... {Colors.END}")
        q_stride_pad = q // STRIDE
        q_block_valid = q_stride_pad // sum_per_token_in_block
        q_block_pad = div_up(q, BLOCK_SIZE)
        k_block_pad = div_up(k, BLOCK_SIZE)
        org = np.random.randint(low=0, high=2, size=[1, HQ, q_block_pad, k_block_pad], dtype=np.int8)
        if q_block_valid != q_block_pad:
            org[:,:,-1,:] = 1
        # print(f'{org=}')

        t_mask = cl.tensor(org)
        t_merged_mask = cl.tensor(np.ones([1, HQ, div_up(q_block_pad, MERGED_Q_NUM), k_block_pad], dtype=np.int8) * 100)
        # block_mask, merged_block_mask, q_stride_pad, q_block_pad, k_block_pad
        params = [t_mask, t_merged_mask, q_stride_pad, q_block_pad, k_block_pad]

        cl.finish()
        kernels.enqueue("post_proc_mask", [t_merged_mask.shape[2], HQ, 1], [1, 1, 1], *params)
        ns = cl.finish()
        t_merged_mask_np = t_merged_mask.numpy()

        t_mask_np = t_mask.numpy()
        if MERGED_Q_NUM == 2:
            if q_block_valid != q_block_pad:
                assert np.all(t_mask_np[:,:,-(q_block_pad - q_block_valid):,:] == 1), f"new pad value must be one, {q=}, {k=}"
                org_pad = np.ones([1, HQ, q_block_pad + 1, k_block_pad], dtype=np.int8)
                org_pad[:,:,:-1,:] = org
                org = org_pad
            t_merged_mask_ref = org[:,:,0::2,:] | org[:,:,1::2,:]
        else:
            t_merged_mask_ref = org.copy()
        compare(t_merged_mask_ref, t_merged_mask_np)
        assert np.all(t_merged_mask_ref == t_merged_mask_np), f"merged mask not equal to ref, {q=} {k=}"

        for i, time_opt in enumerate(ns):
            print(f'(POSTPROC)TPUT_{i}: {time_opt*1e-3:,.0f} us')
    print(f'{Colors.GREEN}test_post_proc done.{Colors.END}')

def main():
    test_post_proc(BLOCK_SIZE = 128)
    test_post_proc(BLOCK_SIZE = 256)

if __name__ == "__main__":
    torch.manual_seed(3)
    torch.set_printoptions(linewidth=1024)
    
    cl.profiling(True)
    
    main()