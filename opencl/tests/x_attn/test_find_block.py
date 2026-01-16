import os

import torch
import numpy as np

import functools

from clops import cl
from clops import compare
from clops.utils import Colors

from references import get_partial_softmax_ref, find_blocks_ref

def div_up(a, b):
    return (a + b - 1) // b
def rnd_up(a, b):
    return (a + b - 1) // b * b

FIND_DEBUG_ACC = 1

SOFTMAX_TYPE = 'float' # 'half'
STRIDE = 16

BLOCK_SG_M = 64
BLOCK_SG_N = 32
SG_M = 4
SG_N = 8
BLOCK_WG_M = BLOCK_SG_M * SG_M
BLOCK_WG_N = BLOCK_SG_N * SG_N
KV_BLOCK_SIZE = 256

class xattn_find_block:
    def __init__(self, num_heads, num_kv_heads, head_size, xattn_block_size, is_causal):
        NUM_THREADS = 32 if xattn_block_size == 128 else 16
        
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.xattn_block_size = xattn_block_size
        self.is_causal = is_causal

        src = r'''#include "xattn_find_block.hpp"'''
        cwd = os.path.dirname(os.path.realpath(__file__))
        print(f"compiling {cwd} ...")

        jit_option = '-abortonspill -noschedule '
        self.kernels = cl.kernels(src, f'''-cmc -Qxcm_jit_option="{jit_option}"
                            -mCM_printregusage -mdump_asm -g2
                            -Qxcm_register_file_size=256 -I{cwd}
                            -DSTRIDE={STRIDE} -DHQ={int(num_heads)} -DHK={int(num_kv_heads)} -DHEAD_SIZE={int(head_size)} 
                            -DBLOCK_SIZE={int(xattn_block_size)} -DBLOCK_SHARE_MAX={BLOCK_WG_N} 
                            -DDEBUG_ACC={FIND_DEBUG_ACC} -DIS_CAUSAL={int(is_causal)} -DSOFTMAX_TYPE={SOFTMAX_TYPE}
                            -DNUM_THREADS={int(NUM_THREADS)}
                            ''')
        
    def __call__(self, t_kq_max_wg, t_kq_exp_partial_sum, q_len, q_stride, k_stride, xattn_thresh, n_repeats = 1):
        sum_per_n_token_in_block = self.xattn_block_size // STRIDE
        q_block = div_up(q_stride, sum_per_n_token_in_block)
        k_block = div_up(k_stride, sum_per_n_token_in_block)
        assert k_block >= q_block, "k block should be larger than q_block"

        batch, num_heads, q_stride_pad, k_block_pad = t_kq_exp_partial_sum.shape
        assert k_block_pad == (k_stride + BLOCK_WG_N - 1) // BLOCK_WG_N * BLOCK_WG_N // sum_per_n_token_in_block, "k_block padded to BLOCK_WG_N / xattn_block_size"
        assert q_stride_pad == (q_stride + BLOCK_WG_M - 1) // BLOCK_WG_M * BLOCK_WG_M, "q_stride_pad padded to BLOCK_WG_M / STRIDE"
        q_block_input = q_stride_pad // sum_per_n_token_in_block
        q_block_pad = div_up(q_len, self.xattn_block_size)
        
        t_kq_sum = cl.tensor(np.zeros([batch, num_heads, q_block_input, k_block_pad], dtype=np.float16))
        
        # mask shape: [rnd_up(q_stride, WG_M) / 8, rnd_up(k_stride, WG_N) / 8]
        t_mask = cl.tensor(np.ones([batch, self.num_heads, q_block_pad, k_block_pad], np.int8) * 100)
        params = [t_kq_max_wg, t_kq_exp_partial_sum, t_mask, q_len, q_stride, q_stride_pad, q_block_pad, k_block_pad, k_block-q_block, xattn_thresh]
        if FIND_DEBUG_ACC:
            params += [t_kq_sum]

        cl.finish()
        for i in range(n_repeats):
            self.kernels.enqueue("find_block", [q_block_pad, num_heads, batch], [1, 1, 1], *params)
        latency = cl.finish()
        return latency, t_mask, t_kq_sum

    @staticmethod
    @functools.cache
    def create_instance(num_heads, num_kv_heads, head_size, xattn_block_size, is_causal):
        return xattn_find_block(num_heads, num_kv_heads, head_size, xattn_block_size, is_causal)
 
    def cmp_mask(self, ref_mask_np, cur_mask_np, ref_sum, cur_exp_partial_sum_np, xattn_thresh):
        sum_per_token_in_block = self.xattn_block_size // STRIDE
        diff_idx = np.where(ref_mask_np != cur_mask_np)
        if diff_idx[0].shape[0]:
            # case 1: if 2+ values are very close, anyone selected is valid although its index is different
            diff_idx_t = torch.tensor(np.array(diff_idx)).transpose(0, 1)
            unique_rows, counts_rows = torch.unique(diff_idx_t[:,:-1], dim=0, return_counts=True)
            idxes = []
            repeated_rows = unique_rows[counts_rows > 1]
            for repeated_row in repeated_rows:
                vals = []
                full_pos = []
                for idx in range(diff_idx_t.shape[0]):
                    if torch.all(diff_idx_t[idx][:-1] == repeated_row):
                        pos = diff_idx_t[idx].tolist()
                        vals.append(ref_sum[*pos])
                        full_pos.append(pos)
                        idxes.append(idx)
                if torch.allclose(torch.tensor(vals, dtype=vals[0].dtype), vals[0], atol=0.01):
                    print(f'{Colors.YELLOW}similar float detected:{Colors.END} idx={full_pos}, vals={vals}')
                else:
                    print(f'{Colors.RED}not close float detected:{Colors.END} idx={full_pos}, vals={vals}')
                    raise Exception('failed')
            if len(idxes):
                indices_to_remove = torch.tensor(idxes)
                mask_one = np.ones_like(diff_idx[0], dtype=np.bool)
                mask_one[indices_to_remove] = False
                diff_idx = (diff_idx[0][mask_one], diff_idx[1][mask_one], diff_idx[2][mask_one], diff_idx[3][mask_one])

            ref_err_rows = torch.from_numpy(ref_sum.detach().numpy()[diff_idx[:3]])
            #cur_err_rows = torch.from_numpy(cur_sum_np[diff_idx[:3]])
            #cur_thresh = cur_err_rows.sum(dim=-1) * xattn_thresh
            cur_thresh = ref_thresh = ref_err_rows.sum(dim=-1) * xattn_thresh

            kq_exp_partial_sum_np = cur_exp_partial_sum_np
            cur_sorted_value = kq_exp_partial_sum_np[:, :, 1::sum_per_token_in_block, :].view(dtype=np.float16)
            cur_sorted_index = kq_exp_partial_sum_np[:, :, 3::sum_per_token_in_block, :].view(dtype=np.ushort)
            cur_accum_value  = kq_exp_partial_sum_np[:, :, 6::sum_per_token_in_block, :].view(dtype=np.float16)
            if not FIND_DEBUG_ACC:
                print(f'{Colors.RED}please set {Colors.BLUE}FIND_DEBUG_ACC=1{Colors.END} to get `cur_accum_value` for checking if there is an accuracy problem{Colors.END}')
                raise Exception('please set FIND_DEBUG_ACC=1')
            if SOFTMAX_TYPE == 'float':
                # results only uses 2 bytes
                cur_sorted_index = cur_sorted_index[...,:cur_sorted_index.shape[-1] // 2].copy()
                cur_accum_value = cur_accum_value[...,:cur_accum_value.shape[-1] // 2].copy()
            # first 2 elements are reserved for first and diag position
            cur_sorted_index[...,:2] = 65535
            # index(X) of different k_blocks 
            error_last_pos_np = np.array(diff_idx[3]).reshape([-1, 1])
            # the index(Y) of X in sorted list
            error_last_idx_np = np.where(cur_sorted_index[diff_idx[:3]] == error_last_pos_np)
            error_idx_np = (*diff_idx[:3], error_last_idx_np[-1])
            # lookup accum value using index Y in cumsum list
            error_accum = cur_accum_value[error_idx_np]
            # case 2: if accum is very close to thresh, it should be a minor float error
            if np.allclose(error_accum, cur_thresh, rtol=0.01, atol=0.01):
                print(f'{Colors.YELLOW}minor float error detected:{Colors.END} idx={diff_idx}, accum={error_accum}, thresh={cur_thresh}')
            else:
                print(f'{Colors.RED}mask is not same:{Colors.END} idx={diff_idx}, accum={error_accum}, thresh={cur_thresh}')
                raise Exception('mask is not same')


def test_find(num_heads, num_kv_heads, head_size, xattn_block_size, xattn_thresh, q_len, k_len, perf = False, causal = True):
    q_stride, k_stride = q_len // STRIDE, k_len // STRIDE

    qk = torch.randint(-2000, 3000, size=[1, num_heads, q_stride, k_stride], dtype=torch.int16).to(dtype=torch.float32)
    assert BLOCK_WG_N % xattn_block_size == 0, "BLOCK_WG_N should be multiple of xattn_block_size then there is no tails from xattn_block_size"
    assert BLOCK_WG_M % xattn_block_size == 0, "BLOCK_WG_M should be multiple of xattn_block_size then there is no tails from xattn_block_size"

    qk_max, kq_5d_max, qk_exp_partial_sum, qk_sum = get_partial_softmax_ref(qk, xattn_block_size, STRIDE, BLOCK_WG_N, BLOCK_WG_M, valid_q=q_stride)
    softmax_type = np.float16 if SOFTMAX_TYPE == 'half' else np.float32
    t_kq_max_wg = cl.tensor(kq_5d_max.detach().numpy().astype(softmax_type))
    t_kq_exp_partial_sum = cl.tensor(qk_exp_partial_sum.detach().numpy().astype(softmax_type))

    find_block_cm = xattn_find_block.create_instance(num_heads, num_kv_heads, head_size, xattn_block_size, causal)
    n_repeats = 100 if perf else 1
    ns, t_mask, t_kq_sum = find_block_cm(t_kq_max_wg, t_kq_exp_partial_sum, q_len, q_stride, k_stride, xattn_thresh, n_repeats)

    if not perf:
        q_block_pad = div_up(q_len, xattn_block_size)
        sum_per_n_token_in_block = xattn_block_size // STRIDE
        q_block = div_up(q_stride, sum_per_n_token_in_block)
        k_block = div_up(k_stride, sum_per_n_token_in_block)
        assert k_block >= q_block, "k block should be larger than q_block"
        mask_ref, sorted_value_ref, sorted_index_ref, cumulative_sum_without_self, required_sum = find_blocks_ref(qk_sum, xattn_thresh, q_block, k_block, causal=causal, current_index=k_block-q_block)

        if FIND_DEBUG_ACC == 1:
            compare(qk_sum.detach().numpy(), t_kq_sum.numpy())
            print(f'{Colors.GREEN}find:sum passed{Colors.END}')

        kq_exp_partial_sum_np = t_kq_exp_partial_sum.numpy()
        cur_sorted_value = kq_exp_partial_sum_np[:, :, 1::sum_per_n_token_in_block, :].view(dtype=np.float16)
        cur_sorted_index = kq_exp_partial_sum_np[:, :, 3::sum_per_n_token_in_block, :].view(dtype=np.ushort)
        cur_accum_value  = kq_exp_partial_sum_np[:, :, 6::sum_per_n_token_in_block, :].view(dtype=np.float16)
        sorted_value_ref = sorted_value_ref.detach().numpy()
        cur_sorted_value = cur_sorted_value[...,:q_block,:k_block]
        compare(sorted_value_ref, cur_sorted_value)
        print(f'{Colors.GREEN}find:sort_value passed{Colors.END}')
        sorted_index_ref = sorted_index_ref.to(torch.uint16).detach().numpy()

        # minor error may get different index
        if causal:
            # index of causal mask for line 0 is also 0, reference will only skip 1 point instead of 2 points(causal+leftmost)
            if q_block == k_block:
                sorted_index_ref[...,0, 2:] = sorted_index_ref[...,0, 1:-1]
            # index 0,1 are for diag+leftmost
            sorted_index_ref[...,:2] = 0
            cur_sorted_index[...,:2] = 0
        cur_sorted_index = cur_sorted_index[...,:q_block,:k_block]
        error_idx = np.where(sorted_index_ref != cur_sorted_index)
        if error_idx[0].shape[0]:
            # print(f'{error_idx=}\nidx_ref={sorte_index_ref[error_idx]}\nidx_cur={cur_sorted_index[error_idx]}')
            error_idx_ref = (error_idx[0], error_idx[1], error_idx[2], sorted_index_ref[error_idx])
            error_idx_cur = (error_idx[0], error_idx[1], error_idx[2], cur_sorted_index[error_idx])
            if not np.allclose(qk_sum[error_idx_ref], qk_sum[error_idx_cur], atol=0.01, rtol=0.01):
                pos = np.where(np.abs(qk_sum[error_idx_ref] - qk_sum[error_idx_cur]) > 0.01)
                print(f'ref={qk_sum[error_idx_ref][pos]}\ncur={qk_sum[error_idx_cur][pos]}')
                raise('error')

        #compare(sorte_index_ref, cur_sorted_index)
        print(f'{Colors.GREEN}find:sort_index passed{Colors.END}')

        if FIND_DEBUG_ACC == 1:
            cur_accum_value = cur_accum_value[...,:q_block,:k_block-2]
            cumulative_sum_without_self_np = cumulative_sum_without_self.detach().numpy()
            compare(cumulative_sum_without_self_np[...,:-2], cur_accum_value)
            print(f'{Colors.GREEN}find:accum passed{Colors.END}')

        cur_mask = t_mask.numpy()
        if q_block != q_block_pad:
            assert np.all(cur_mask[:,:,-(q_block_pad - q_block):,:] == 1), f"new pad value must be one"
        cur_mask = cur_mask[...,:q_block,:k_block]
        if not np.all(cur_mask == mask_ref.to(torch.int8).detach().numpy()):
            find_block_cm.cmp_mask(mask_ref.to(torch.int8).detach().numpy(), cur_mask, qk_sum, kq_exp_partial_sum_np, xattn_thresh)
        # minor error may get different mask index
        # mask_ref_np =  mask_ref.detach().numpy()
        # cur_mask = tMask.numpy()
        # error_idx = np.where(mask_ref_np != cur_mask)
        # if error_idx[0].shape[0]:
        #     if not np.allclose(A_sum[error_idx], tC_sum.numpy()[error_idx], atol=0.01, rtol=0.01):
        #         print(f'{error_idx=}\nidx_ref={mask_ref_np[error_idx]}\nidx_cur={cur_mask[error_idx]}')
        #         raise Exception(f'the diff value should be smaller than 0.01, ref={A_sum[error_idx]} cur={tC_sum.numpy()[error_idx]}')
        # print(f'{Colors.GREEN}find:mask passed{Colors.END}')
    else:
        for i, time_opt in enumerate(ns):
            print(f'(FIND)TPUT_{i}: {time_opt*1e-3:,.0f} us')

def test_func(xattn_block_size, xattn_thresh, HQ = 32, HK = 8, HEAD_SIZE = 128, is_causal = True):   
    dim = HEAD_SIZE
    sizes = [
        # normal case
        (512 * STRIDE, 512 * STRIDE, '8k'),
        (256 * STRIDE, 256 * STRIDE, 'normal+causal start == 0'),           # normal case:causal start == 0
        (4 * 1024, 128 * 1024, 'normal'),                                   # normal case:4k * 128k
        (128 * STRIDE, 256 * STRIDE, 'normal+smallest block'),              # smallest block just suitable for one workgroup
        # query tails: tails are less than STRIDE
        (128 * STRIDE + 7, 256 * STRIDE, 'M tail=7 of query'),
        (4 * 1024 + 5, 128 * 1024, 'M tail=5 of query'),
        # query tails: tails are multiple of STRIDE
        (STRIDE, 256 * STRIDE, 'M tail=16 of query'),                       # query has tail which is not enough for a workgroup
        ((128+1) * STRIDE, 256 * STRIDE, 'M tail of smallest query'),       # query has tail which is not full for its last block in M dimension
        (4 * 1024 + STRIDE*3, 128 * 1024, 'M tail of bigger query'),        # query has tail which is not full for its last block in M dimension
        (4 * 1024 + STRIDE*7+2, 128 * 1024, 'M tail=x*STRIDE+y of query'),  # query has tail which is not full for its last block in M dimension
        # key tails: tails are less than STRIDE
        (128 * STRIDE, 256 * STRIDE + 7, 'N tail=7 of key'),
        (4 * 1024 + 5, 128 * 1024 + 3, 'M tail=5&N tail=3 of key'),
        # key tails: tails are multiple of STRIDE
        (128 * STRIDE, STRIDE * (128 + 1), 'N tail=16 of key'),                # key has tail which is not enough for a workgroup
        (4 * 1024, 128*1024+STRIDE*2, 'N tail=32 of key'),                     # key has tail which is not enough for a workgroup
        (4 * 1024 + 3*STRIDE+5, 128 * 1024 + 7*STRIDE+ 3, 'M tail=3.5&N tail=7.3 of key'),
    ]
    for q_len, k_len, prompt in sizes:
        assert q_len // STRIDE >= 1, "there should be at least 1 row for gemm"
        print(f'{Colors.BLUE}test gemm("{prompt}") query: [{q_len}, {dim}*{STRIDE}] key:[{k_len}, {dim}*{STRIDE}] xattn_block_size:{xattn_block_size}, xattn_threshold:{xattn_thresh} ...{Colors.END}')

        if is_causal:
            q_start_strided = k_len // STRIDE - q_len // STRIDE
            assert q_start_strided >= 0, "length of key cache must be greater or equal than query"
        else:
            q_start_strided = 0
        test_find(HQ, HK, HEAD_SIZE, xattn_block_size, xattn_thresh, q_len, k_len, perf=False, causal=is_causal)

def test_perf(xattn_block_size, xattn_thresh, HQ = 32, HK = 8, HEAD_SIZE = 128, is_causal = True):
    # 106 T/s:
    # bsz = 1
    # q_head = 1
    # k_head = 1
    # q_len = 1024*16*2
    # k_len = 1024*128
    # dim = 128
    # xattn_block_size = 128
    # STRIDE = 16

    q_len = 1024*4*1  # if q_len=1024*4*2 ==> 95 T/s
    k_len = 1024*128

    assert (q_len // STRIDE) >= 128 and (k_len // STRIDE) >= 128, "minimum block size should be 128*128"
    test_find(HQ, HK, HEAD_SIZE, xattn_block_size, xattn_thresh, q_len, k_len, perf=True, causal=is_causal)

def main():
    for xattn_block_size in [128, 256]:
        for xattn_thresh in [0.9]:
            test_func(xattn_block_size, xattn_thresh)
            test_perf(xattn_block_size, xattn_thresh)

if __name__ == "__main__":
    torch.manual_seed(3)
    torch.set_printoptions(linewidth=1024)
    torch.set_printoptions(precision=2, sci_mode=False)
    np.set_printoptions(precision=2, suppress=True)

    cl.profiling(True)

    main()