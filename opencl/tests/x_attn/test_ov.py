#from xattn.src.utils import *
import torch
import math
import torch.nn.functional as F

import numpy as np

from clops import cl
import clops
import time
from clops import compare
import os
from clops.utils import Colors

from test_find_block import xattn_find_block
from test_gemm_qk import xattn_gemmQK

def div_up(a, b):
    return (a + b - 1) // b
def rnd_up(a, b):
    return (a + b - 1) // b * b

FIND_DEBUG_ACC = 1
STRIDE = 16

def test_ov():
    cl.profiling(True)
    torch.manual_seed(0)
    torch.set_printoptions(linewidth=1024)
    def get_tensor(name, dtype=np.float16):
        with open(name, 'rb') as f:
            data = f.read()
            np_data = np.frombuffer(data, dtype=dtype).copy()
            return torch.from_numpy(np_data)
    
    compressed_kvcache = False
    is_causal = True
    xattn_block_size, kv_block_size, trunk_sz = 256, 256, 4096
    xattn_thresh = 0.899902
    num_heads, num_kv_heads, head_size = 32, 8, 128
    base = '/home/ceciliapeng/try2/dump_debug_binary/'
    
    query = get_tensor(base + 'program1_network1_0_pagedattentionextension_PagedAttentionExtension_27964_src0__f16__612_4096_1_1__bfyx.bin').reshape([612, num_heads*head_size])
    key   = get_tensor(base + 'program1_network1_0_pagedattentionextension_PagedAttentionExtension_27964_updated_src_3__f16__3_8_256_128__bfyx.bin', np.int8 if compressed_kvcache else np.float16).reshape([-1, num_kv_heads, kv_block_size, head_size+4 if compressed_kvcache else head_size])

    q_len = query.shape[0]
    k_len = q_len  #??
    q_block_pad = (q_len + xattn_block_size - 1) // xattn_block_size
    mask  = get_tensor(base + 'program1_network1_0_pagedattentionextension_PagedAttentionExtension_27964_intermediates_4__boolean__1536_1_1_1__bfyx.bin', dtype=np.int8).reshape([num_heads, q_block_pad, -1])

    def check_sanity(mask):
        def is_binary_mask(mask):
            invalid = (mask != 0) & (mask != 1)
            has_invalid = invalid.any().item()
            if has_invalid:
                count = invalid.sum().item()
                bad_idx = invalid.nonzero(as_tuple=False)
                print(f'{Colors.RED}Coords with no bool:{Colors.END}', bad_idx.tolist())
            return not has_invalid

        def is_sanity(mask):
            # check sanity: Indices of (head, row) where the row is all False
            per_row_has_true = mask.any(dim=-1)
            bad_idx = (~per_row_has_true).nonzero(as_tuple=False)
            if bad_idx.numel() > 0:
                print(f'{Colors.RED}Rows with no True:{Colors.END}', bad_idx.tolist())
            
            # check if the first column of each row is True
            first_col_is_true = mask[:, :, 0].bool()
            bad_idx = (~first_col_is_true).nonzero(as_tuple=False)
            if bad_idx.numel() > 0:
                print(f'{Colors.RED}Rows with no True:{Colors.END}', bad_idx.tolist())

            return bad_idx.numel() == 0
        
        is_binary_mask(mask)
        is_sanity(mask)
        
        DUMP_MASK = True
        if DUMP_MASK:
            for i in range(num_heads):
                print(f'{i}:{mask[i,:,:]}')
    check_sanity(mask)
  
    valid_num_blks = key.shape[0] # genai usually generates one more blocks than required
    block_indices = get_tensor(base + 'program1_network1_0_pagedattentionextension_PagedAttentionExtension_27964_src7__i32__3_1_1_1__bfyx.bin', dtype=np.int32).reshape([valid_num_blks])
    block_indices_begins = get_tensor(base + 'program1_network1_0_pagedattentionextension_PagedAttentionExtension_27964_src8__i32__2_1_1_1__bfyx.bin', dtype=np.int32).reshape([2])

    # stage1 gemmQK
    M = q_len // STRIDE
    ## ?? which N is right?
    # N = valid_num_blks * key.shape[2] // STRIDE
    N = k_len // STRIDE
    K = head_size * STRIDE
    q_start_strided = k_len // STRIDE - q_len // STRIDE
    assert q_start_strided >= 0, "length of key cache must be greater or equal than query"

    t_query = cl.tensor(query.detach().numpy())
    t_key_cache = cl.tensor(key.detach().numpy())
    t_block_indices = cl.tensor(block_indices.to(torch.int32).detach().numpy())
    t_block_indices_begins = cl.tensor(block_indices_begins.to(torch.int32).detach().numpy())
    
    assert query.is_contiguous(), "query tensor must be contiguous... otherwise  here needs change query_stride correspondingly."
    query_stride = num_heads*head_size*STRIDE

    xattn_gemmQK_cm = xattn_gemmQK.create_instance(num_heads, num_kv_heads, head_size, xattn_block_size, is_causal, compressed_kvcache)
    ns, t_kq_max_wg, t_kq_exp_partial_sum = xattn_gemmQK_cm(t_key_cache, t_query, t_block_indices, t_block_indices_begins, q_start_strided, M, N, K, query_stride, 1)
    ut_kq_max_wg, ut_kq_exp_partial_sum =  torch.from_numpy(t_kq_max_wg.numpy()).clone(), torch.from_numpy(t_kq_exp_partial_sum.numpy()).clone()  # clone them to avoid changed by find_blocks kernel

    # stage2 find_block
    find_block_cm = xattn_find_block.create_instance(num_heads, num_kv_heads, head_size, xattn_block_size, is_causal)
    q_stride, k_stride = M, N
    ns, t_mask, t_kq_sum = find_block_cm(t_kq_max_wg, t_kq_exp_partial_sum, q_len, q_stride, k_stride, xattn_thresh, 1)
       
    sum_per_token_in_block = xattn_block_size // STRIDE
    q_block = div_up(q_stride, sum_per_token_in_block)

    t_mask_np_all = t_mask.numpy()
    t_mask_np = t_mask_np_all[:,:,:q_block,:]
    if q_block != q_block_pad:
        assert np.all(t_mask_np_all[:,:,-(q_block_pad - q_block):,:] == 1), f"new pad value must be one for ov"
    
    if not np.all(t_mask_np_all == mask.detach().numpy()):
        error_idx = np.where(t_mask_np_all != mask.detach().numpy())
        print(f'{Colors.RED} mask result of unit test is not same with ov{Colors.END}: diff idx={error_idx}')
        
        # Then further check internals
        xattn_internals = '/home/ceciliapeng/openvino/dump_debug_xattn2/'
        ov_kq_max_wg  = get_tensor(xattn_internals + 'xattn_internals_2__pagedattentionextension:PagedAttentionExtension_27964__f32__8192_1_1_1__bfyx.bin', dtype=np.float32).reshape(ut_kq_max_wg.shape)
        ov_kq_exp_partial_sum = get_tensor(xattn_internals + 'xattn_internals_3__pagedattentionextension:PagedAttentionExtension_27964__f32__131072_1_1_1__bfyx.bin', dtype=np.float32).reshape(ut_kq_exp_partial_sum.shape)
        compare(ut_kq_max_wg.detach().numpy()[..., :M], ov_kq_max_wg.detach().numpy()[..., :M])
        print(f'{Colors.GREEN}gemm:max_wg passed{Colors.END}')
        compare(ut_kq_exp_partial_sum.detach().numpy()[:,:,:M,:], ov_kq_exp_partial_sum.detach().numpy()[:,:,:M,:])
        print(f'{Colors.GREEN}gemm:exp_partial passed{Colors.END}')

        ov_mask  = get_tensor(xattn_internals + 'xattn_internals_4__pagedattentionextension:PagedAttentionExtension_27964__boolean__1536_1_1_1__bfyx.bin', dtype=np.int8).reshape([num_heads, q_block_pad, -1])
        check_sanity(ov_mask)

        if FIND_DEBUG_ACC == 1:
            ov_kq_sum = get_tensor(xattn_internals + 'xattn_internals_6__pagedattentionextension:PagedAttentionExtension_27964__f16__8192_1_1_1__bfyx.bin', dtype=np.float16).reshape(t_kq_sum.shape)
            compare(t_kq_sum.numpy(), ov_kq_sum.detach().numpy())
            print(f'{Colors.GREEN}find:sum passed{Colors.END}')
    else:
        print(f'{Colors.GREEN}test_ov result of unit test is same with ov.{Colors.END}')
        
    from test_xattn import xattn_estimate
    query_states = query.reshape([1, -1, num_heads, head_size]).permute(0, 2, 1, 3)
    key_states = key.permute(1, 0, 2, 3).reshape([1, num_kv_heads, -1, head_size]).repeat_interleave(num_heads // num_kv_heads, 1)[...,:k_len // STRIDE * STRIDE,:]
    query_states = query_states[:,:,:q_len // STRIDE * STRIDE, :]
    attn_sums, approx_simple_mask, reshaped_querry, reshaped_key = xattn_estimate(query_states=query_states,
                   key_states=key_states,
                   block_size=xattn_block_size,
                   stride=STRIDE,
                   norm=1,
                   threshold=xattn_thresh,
                   select_mode="inverse",
                   use_triton=False,
                   causal=is_causal,
                   chunk_size=trunk_sz)

    if FIND_DEBUG_ACC == 1:
        compare(attn_sums.detach().numpy(), t_kq_sum.numpy())
        print(f'{Colors.GREEN}find:sum passed{Colors.END}')
    
    approx_simple_mask = approx_simple_mask[...,:q_block,:]
    assert approx_simple_mask.shape[0] == 1, "approx_simple_mask.shape[0] is expected to be 1."
    check_sanity(approx_simple_mask[-1,...].to(torch.int8))
    approx_simple_mask_np = approx_simple_mask.to(torch.int8).detach().numpy()
    if not np.all(t_mask_np == approx_simple_mask_np):
        find_block_cm.cmp_mask(approx_simple_mask_np, t_mask_np, attn_sums, t_kq_exp_partial_sum.numpy(), xattn_thresh)
    print(f'{Colors.GREEN}test_ov done.{Colors.END}')

# def test_cuda_input():
#     # change global configs here
#     global FIND_DEBUG_ACC, num_kv_heads, THRESH
#     FIND_DEBUG_ACC = 1
#     THRESH = 0.6
#     num_kv_heads = 32

#     create_kernels(True)

#     def get_tensor(name, dtype=torch.float16):
#         data = np.load(name)
#         data = torch.from_numpy(data)
#         return data.to(dtype=dtype)
    
#     base = '/mnt/llm_irs/cuda_data/mnt/data_sda/ruonan/x-attention-fork/Qwen3-8B-demo-prompt-32k-16-0.6-dump/'
#     query = get_tensor(base + 'query_0.bin.npy')
#     key   = get_tensor(base + 'key_0.bin.npy')
#     mask  = get_tensor(base + 'Qwen3-8B-demo-prompt-32k-16-0.6-int4lowbit.npy', dtype=torch.int8)
#     q_len = query.shape[2]
#     M = q_len // STRIDE
#     Lk = key.shape[2]
#     N = Lk // STRIDE
#     K = head_size * STRIDE
#     q_start_strided = 0
#     q_stride_pad = rnd_up(M, BLOCK_WG_M)
#     Hq = 32
#     N_kq_groups = div_up(N, BLOCK_WG_N)
#     global_size = [N_kq_groups * (q_stride_pad // BLOCK_WG_M) * SG_N * WALK_HQ, SG_M, Hq // WALK_HQ]
#     query = query.permute(0, 2, 1, 3).reshape([1, -1, num_heads * head_size]).contiguous()
#     key = key.reshape([num_kv_heads, -1, KV_BLOCK_SIZE, head_size]).permute(1, 0, 2, 3).contiguous()
#     # start block 0
#     block_indices_begins = torch.zeros(1).to(torch.int32)
#     block_indices = torch.arange((Lk + KV_BLOCK_SIZE - 1) // KV_BLOCK_SIZE)
#     t_query = cl.tensor(query.detach().numpy())
#     t_key_cache = cl.tensor(key.detach().numpy())
#     t_block_indices = cl.tensor(block_indices.to(torch.int32).detach().numpy())
#     t_block_indices_begins = cl.tensor(block_indices_begins.to(torch.int32).detach().numpy())
#     softmax_type = np.float16 if SOFTMAX_TYPE == 'half' else np.float32
#     t_kq_max_wg = cl.tensor(np.zeros([1, Hq, N_kq_groups, q_stride_pad], softmax_type))
#     # [1, 32, 256, 64 * 16]
#     sum_per_token_in_block = xattn_block_size // STRIDE
#     k_block_in_group = BLOCK_WG_N // sum_per_token_in_block
#     k_block_pad = k_block_in_group * N_kq_groups
#     t_kq_exp_partial_sum = cl.tensor(np.zeros([1, Hq, q_stride_pad, k_block_pad], softmax_type))

#     kernels.enqueue(kernel_name, global_size, [SG_N, SG_M, 1], t_key_cache, t_query, t_block_indices, t_block_indices_begins, t_kq_max_wg, t_kq_exp_partial_sum, M, N, K, K * num_heads, 0, 0, q_start_strided)

#     q_stride = M
#     k_stride = N
#     q_block_input = q_stride_pad // sum_per_token_in_block
#     q_block_pad = div_up(q_len, xattn_block_size)
#     t_mask = cl.tensor(np.ones([1, num_heads, q_block_pad, k_block_pad], np.int8) * 100)
#     q_block = div_up(q_stride, sum_per_token_in_block)
#     k_block = div_up(k_stride, sum_per_token_in_block)
#     t_kq_sum = cl.tensor(np.zeros([1, num_heads, q_block_input, k_block_pad], dtype=np.float16))
#     params = [t_kq_max_wg, t_kq_exp_partial_sum, t_mask, q_len, q_stride, q_stride_pad, q_block_pad, k_block_pad, THRESH, k_block-q_block]
#     if FIND_DEBUG_ACC:
#         params += [t_kq_sum]
#     kernels.enqueue("find_block", [q_block_pad, num_heads, 1], [1, 1, 1], *params)
#     cl.finish()

#     from test_xattn import xattn_estimate
#     query_states = query.reshape([1, -1, num_heads, head_size]).permute(0, 2, 1, 3)
#     key_states = key.permute(1, 0, 2, 3).reshape([1, num_kv_heads, -1, head_size]).repeat_interleave(num_heads // num_kv_heads, 1)
#     query_states = query_states[:,:,:q_len // STRIDE * STRIDE, :]
#     attn_sums, approx_simple_mask, reshaped_querry, reshaped_key = xattn_estimate(query_states=query_states,
#                    key_states=key_states,
#                    block_size=xattn_block_size,
#                    stride=STRIDE,
#                    norm=1,
#                    threshold=THRESH,
#                    select_mode="inverse",
#                    use_triton=False,
#                    causal=IS_CAUSAL,
#                    chunk_size=query_states.shape[2])

#     if FIND_DEBUG_ACC == 1:
#         compare(attn_sums.detach().numpy(), t_kq_sum.numpy())
#         print(f'{Colors.GREEN}find:sum passed{Colors.END}')

#     t_mask_np_all = t_mask.numpy()
#     t_mask_np = t_mask_np_all[:,:,:q_block,:]
#     if q_block != q_block_pad:
#         assert np.all(t_mask_np_all[:,:,-(q_block_pad - q_block):,:] == 1), f"new pad value must be one for ov"

#     if not np.all(t_mask_np_all == mask[0].detach().numpy()):
#         error_idx = np.where(t_mask_np_all != mask[0].detach().numpy())
#         print(f'{Colors.RED}result of unit test is not same with ov{Colors.END}: diff idx={error_idx}')
#     cmp_mask(approx_simple_mask.to(torch.int8).detach().numpy(), t_mask_np, attn_sums, t_kq_exp_partial_sum.numpy())
#     print(f'{Colors.GREEN}test_ov done.{Colors.END}')

# def test_post_proc():
#     create_kernels()
#     sum_per_token_in_block = 8
#     qk = [
#         (4096, 4096),
#         (4096+1, 4096)
#     ]
#     for q, k in qk:
#         q_stride_pad = q // STRIDE
#         q_block_valid = q_stride_pad // sum_per_token_in_block
#         q_block_pad = div_up(q, xattn_block_size)
#         k_block_pad = div_up(k, xattn_block_size)
#         org = np.random.randint(low=0, high=2, size=[1, num_heads, q_block_pad, k_block_pad], dtype=np.int8)
#         if q_block_valid != q_block_pad:
#             org[:,:,-1,:] = 1

#         t_mask = cl.tensor(org)
#         t_merged_mask = cl.tensor(np.ones([1, num_heads, div_up(q_block_pad, MERGED_Q_NUM), k_block_pad], dtype=np.int8) * 100)
#         # block_mask, merged_block_mask, q_stride_pad, q_block_pad, k_block_pad
#         params = [t_mask, t_merged_mask, q_stride_pad, q_block_pad, k_block_pad]
#         kernels.enqueue("post_proc_mask", [t_merged_mask.shape[2], num_heads, 1], [1, 1, 1], *params)
#         cl.finish()
#         t_merged_mask_np = t_merged_mask.numpy()
#         t_mask_np = t_mask.numpy()
#         if q_block_valid != q_block_pad:
#             assert np.all(t_mask_np[:,:,-(q_block_pad - q_block_valid):,:] == 1), f"new pad value must be one, {q=}, {k=}"
#             org_pad = np.ones([1, num_heads, q_block_pad + 1, k_block_pad], dtype=np.int8)
#             org_pad[:,:,:-1,:] = org
#             org = org_pad
#         t_merged_mask_ref = org[:,:,0::2,:] & org[:,:,1::2,:]
#         assert np.all(t_merged_mask_ref == t_merged_mask_np), f"merged mask not equal to ref, {q=} {k=}"
#     print(f'{Colors.GREEN}test_post_proc done.{Colors.END}')

def main():
    test_ov()
    # test_post_proc()
    # test_cuda_input()

if __name__ == "__main__":
    torch.manual_seed(3)
    torch.set_printoptions(linewidth=1024)
    
    # requirements:
    # num_q_head == num_kv_head
    # chunk size alignment
    # causal_mask
    main()