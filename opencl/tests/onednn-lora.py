#!/usr/bin/python3
import numpy as np
import time, sys
import torch
# from torch import nn

from clops import cl
from clops.lora import blocking_1nd, LORA_1ST
from clops import compare

enable_debug = True
class onednnLoRA:
    def __init__(self, loraA, loraB, alpha, OC, IC, rank):
        self.w_dtype = cl.onednn_dtype.f16
        self.OC = OC
        self.IC = IC
        self.rank = rank
        self.M = -1
        self.K_group_size = -1

        self.loraA = loraA
        self.loraB = loraB
        self.alpha = alpha

    def update_batch(self, batch):
        if self.M != batch:
            empty_cl_tensor = cl.tensor()
            if enable_debug: print("======== create linear_A =========")
            self.linear_A = cl.onednn_linear(cl.onednn_dtype.f16, self.w_dtype, batch, self.IC, self.rank,   # M, K, N
                                        self.K_group_size, cl.onednn_matmul_type.with_bin_mul,
                                        cl.onednn_dtype.f16, 
                                        empty_cl_tensor, empty_cl_tensor, empty_cl_tensor, True)
            if enable_debug: print("======== create linear_B =========")
            self.linear_B = cl.onednn_linear(cl.onednn_dtype.f16, self.w_dtype, batch, self.rank, self.OC,
                                        self.K_group_size, cl.onednn_matmul_type.with_bin_add,
                                        cl.onednn_dtype.f16, 
                                        empty_cl_tensor, empty_cl_tensor, empty_cl_tensor, False)
            self.M = batch

    def forward(self, input, mainInput):
        M = input.shape[0]
        self.update_batch(M)
        temp_resA = cl.tensor(np.zeros([M, self.rank], dtype=np.float16))
        dst = cl.tensor(np.zeros([M, self.OC], dtype=np.float16))

        self.linear_A.forward(cl.tensor(input), temp_resA, cl.tensor(self.alpha), cl.tensor(self.loraA))
        self.linear_B.forward(temp_resA, dst, cl.tensor(mainInput), cl.tensor(self.loraB))
        return dst

def calc_ref(main_input, input, loraA, loraB, alpha):
    tloraA = loraA.transpose().copy()
    tloraB = loraB.transpose().copy()
    dst_ref = input @ tloraA
    dst_ref *= alpha
    dst_ref = dst_ref @ tloraB
    print(f'{dst_ref.shape=}, {tloraB.shape=}')
    dst_ref += main_input
    print("ref is calculated!")
    return dst_ref

def test_lora0():
    np.random.seed(0)
    OC, IC, rank = (512, 1536, 16)
    n_tokens = 8

    # alpha = np.ones([1, rank]).astype(np.float16)
    alpha = np.random.randint(-1,2,[1, rank]).astype(np.float16)
    loraA = np.random.randint(-1,2,[rank, IC]).astype(np.float16)
    loraB = np.random.randint(-1,2,[OC, rank]).astype(np.float16)

    lora = onednnLoRA(loraA, loraB, alpha, OC, IC, rank)

    for _ in range(3):
        # input = np.random.randint(-1,2,[n_tokens, IC]).astype(np.float16)
        input = torch.randn([n_tokens, IC], dtype=torch.float16).numpy()
        main_input = np.random.randint(-1,2,[n_tokens, OC]).astype(np.float16)

        dst_cur = lora.forward(input, main_input)
        dst_cur = dst_cur.numpy()
        print("cur is calculated!")
        
        dst_ref = calc_ref(main_input, input, loraA, loraB, alpha)
        compare(dst_ref, dst_cur)
        
test_lora0()



################################################################################################################
# cl.profiling(True)

# batch = 8
# input_state = 1024
# output_state_idx = 256//16
# rank = 16

# output_state = output_state_idx*16
# def oclLoRA(batch, rank, input_state, output_state, check_acc = False):
#     if check_acc:
#         REPEAT = 1
#     else:
#         REPEAT = 100        
        
#     A_regM, A_regN, A_sgM, A_sgN, B_regM, B_regN, B_sgM, B_sgN = blocking_1nd(batch, rank, input_state, output_state_idx*16)

#     Aoutput = np.zeros([batch, rank]).astype(np.float16)
    
#     stateA_list= [cl.tensor(stateA) for _ in range(REPEAT)]
#     alpha_list = [cl.tensor(alpha) for _ in range(REPEAT)]
#     stateB_list = [cl.tensor(stateB)for _ in range(REPEAT)]
#     loraInput_list = [cl.tensor(loraInput)for _ in range(REPEAT)]
#     mainInput_list = [cl.tensor(mainInput)for _ in range(REPEAT)]
#     # Must set the output to be zeros to avoid not all the data is updated in GEMMA. 
#     A_output_list = [cl.tensor(Aoutput)for _ in range(REPEAT)]
#     res_list = [cl.tensor([batch, output_state], np.dtype(np.float16))for _ in range(REPEAT)]

#     opt = LORA_1ST( batch, rank, input_state, output_state,  A_regM, A_regN, A_sgM, A_sgN, B_regM, B_regN, B_sgM, B_sgN, False)
    
#     opt(mainInput_list[0], loraInput_list[0], stateA_list[0], alpha_list[0], stateB_list[0], A_output_list[0], res_list[0])
#     cl.finish()
    
#     return res_list[0].numpy()

# def test_lora2():   
#     # generate inputs
#     vRANGE = 1
#     # np.random.seed(0)
#     stateA = np.random.randint(-vRANGE, vRANGE+1, [input_state, rank]).astype(np.float16)
#     alpha = np.random.rand(rank).astype(np.float16)
#     stateB = np.random.randint(-vRANGE, vRANGE+1, [rank, output_state]).astype(np.float16)
#     loraInput = np.random.randint(-vRANGE, vRANGE+1, [batch, input_state]).astype(np.float16)
#     mainInput = np.random.randint(-vRANGE, vRANGE+1, [batch, output_state]).astype(np.float16)
    
#     ref = lora(batch, rank, input_state, output_state, check_acc=True)
    
#     compare(ref, res_list[0].numpy())
#     print(f'BATCH:{batch} INPUT_STATE:{input_state}, RANK:{rank}, OUPUT_STATE:{output_state} ACC PASS!')