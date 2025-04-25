#!/usr/bin/python3
import numpy as np
import time, sys
import torch
# from torch import nn

from clops import cl
from clops import compare

enable_debug = True

def test_mm(M, K, N, K_group_size, w_dtype):
    np.random.seed(0)
    A = np.random.randint(-1,2,[M, K]).astype(np.float16)
    tA = cl.tensor(A)
    tC = cl.tensor(np.zeros([M, N], dtype=np.float16))
    tP1 = cl.tensor()

    if w_dtype == cl.onednn_dtype.f16:
        B = np.random.randint(-1,2,[K, N]).astype(np.float16)
        C = A @ B
        print("ref is calculated!")

        tB = cl.tensor(B.transpose().copy())
        linear = cl.onednn_linear(cl.onednn_dtype.f16, w_dtype, M, K, N, -1, cl.onednn_matmul_type.none,
                                  cl.onednn_dtype.f16, tB, cl.tensor(), cl.tensor(), False, True)

    linear.forward(tA, tC, tP1, cl.tensor())

    cl.finish()

    C1 = tC.numpy()

    if not np.allclose(C, C1):
        print(C)
        print(C1)
    else:
        print("================ PASSED ==================" , M, K, N, w_dtype)

# test_mm(M = 1, K = 768, N = 2048, K_group_size = 0, w_dtype = cl.onednn_dtype.f16)
# sys.exit()

class onednnLoRA:
    def __init__(self, loraA, loraB, alpha, OC, IC, rank, from_linear = True):
        self.w_dtype = cl.onednn_dtype.f16
        self.OC = OC
        self.IC = IC
        self.rank = rank
        self.M = -1
        self.K_group_size = -1

        self.loraA = loraA
        self.loraB = loraB
        self.alpha = alpha
        
        self.from_linear = from_linear

    def update_batch(self, n_tokens):
        if self.M != n_tokens:
            empty_cl_tensor = cl.tensor()
            if enable_debug: print("======== create linear_A =========")
            self.linear_A = cl.onednn_linear(cl.onednn_dtype.f16, self.w_dtype, n_tokens, self.IC, self.rank,   # M, K, N
                                        self.K_group_size, cl.onednn_matmul_type.with_bin_mul,
                                        cl.onednn_dtype.f16, 
                                        empty_cl_tensor, empty_cl_tensor, empty_cl_tensor, True, self.from_linear)
            if enable_debug: print("======== create linear_B =========")
            self.linear_B = cl.onednn_linear(cl.onednn_dtype.f16, self.w_dtype, n_tokens, self.rank, self.OC,
                                        self.K_group_size, cl.onednn_matmul_type.with_bin_add,
                                        cl.onednn_dtype.f16, 
                                        empty_cl_tensor, empty_cl_tensor, empty_cl_tensor, False, self.from_linear)
            self.M = n_tokens

    def forward(self, lora_input, main_input):
        M = lora_input.shape[0]
        self.update_batch(M)
        temp_resA = cl.tensor(np.zeros([M, self.rank], dtype=np.float16))
        dst = cl.tensor(np.zeros([M, self.OC], dtype=np.float16))

        self.linear_A.forward(cl.tensor(lora_input), temp_resA, cl.tensor(self.alpha), cl.tensor(self.loraA))
        self.linear_B.forward(temp_resA, dst, cl.tensor(main_input), cl.tensor(self.loraB))
        return dst

def calc_ref(main_input, lora_input, loraA, loraB, alpha, transpose_w = True):
    if transpose_w:
        loraA = loraA.transpose().copy()
        loraB = loraB.transpose().copy()
    print(f'{lora_input.shape=}, {loraA.shape=}')
    dst_ref = lora_input @ loraA
    dst_ref *= alpha
    print(f'{dst_ref.shape=}, {loraB.shape=}')
    dst_ref = dst_ref @ loraB
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

    lora = onednnLoRA(loraA, loraB, alpha, OC, IC, rank, True)

    for _ in range(3):
        # lora_input = np.random.randint(-1,2,[n_tokens, IC]).astype(np.float16)
        lora_input = torch.randn([n_tokens, IC], dtype=torch.float16).numpy()
        main_input = np.random.randint(-1,2,[n_tokens, OC]).astype(np.float16)

        dst_cur = lora.forward(lora_input, main_input)
        dst_cur = dst_cur.numpy()
        print("cur is calculated!")
        
        dst_ref = calc_ref(main_input, lora_input, loraA, loraB, alpha, True)
        compare(dst_ref, dst_cur)
        
# test_lora0()


def test_lora1():
    np.random.seed(0)
    OC, IC, rank = (512, 1536, 16)
    n_tokens = 8

    # generate inputs
    vRANGE = 1
    loraA = np.random.randint(-vRANGE, vRANGE+1, [IC, rank]).astype(np.float16)
    # loraA = np.ones([IC, rank]).astype(np.float16)
    alpha = np.random.rand(rank).astype(np.float16)
    loraB = np.random.randint(-vRANGE, vRANGE+1, [rank, OC]).astype(np.float16)
    # loraB = np.ones([rank, OC]).astype(np.float16)
    
    lora_input = np.random.randint(-vRANGE, vRANGE+1, [n_tokens, IC]).astype(np.float16)
    main_input = np.random.randint(-vRANGE, vRANGE+1, [n_tokens, OC]).astype(np.float16)
    lora_input = np.ones([n_tokens, IC]).astype(np.float16)

    lora = onednnLoRA(loraA, loraB, alpha, OC, IC, rank, False)

    for _ in range(1):
        dst_cur = lora.forward(lora_input, main_input)
        dst_cur = dst_cur.numpy()
        print("cur is calculated!")

        dst_ref = calc_ref(main_input, lora_input, loraA, loraB, alpha, False)
        compare(dst_ref, dst_cur)
        
# test_lora1()



################################################################################################################
from clops.lora import blocking_1nd, LORA_1ST

cl.profiling(True)

def oclLoRA(main_input, lora_input, loraA, loraB, alpha, n_tokens, rank, IC, OC, check_acc = True):
    if check_acc:
        REPEAT = 1
    else:
        REPEAT = 100        
        
    A_regM, A_regN, A_sgM, A_sgN, B_regM, B_regN, B_sgM, B_sgN = blocking_1nd(n_tokens, rank, IC, OC)

    Aoutput = np.zeros([n_tokens, rank]).astype(np.float16)
    
    stateA_list= [cl.tensor(loraA) for _ in range(REPEAT)]
    alpha_list = [cl.tensor(alpha) for _ in range(REPEAT)]
    stateB_list = [cl.tensor(loraB)for _ in range(REPEAT)]
    loraInput_list = [cl.tensor(lora_input)for _ in range(REPEAT)]
    mainInput_list = [cl.tensor(main_input)for _ in range(REPEAT)]
    # Must set the output to be zeros to avoid not all the data is updated in GEMMA. 
    A_output_list = [cl.tensor(Aoutput)for _ in range(REPEAT)]
    res_list = [cl.tensor([n_tokens, OC], np.dtype(np.float16))for _ in range(REPEAT)]

    opt = LORA_1ST( n_tokens, rank, IC, OC,  A_regM, A_regN, A_sgM, A_sgN, B_regM, B_regN, B_sgM, B_sgN, False)
    
    opt(mainInput_list[0], loraInput_list[0], stateA_list[0], alpha_list[0], stateB_list[0], A_output_list[0], res_list[0])
    duration = cl.finish()
    
    return res_list[0].numpy(), duration

def test_lora2(n_tokens, rank, IC, OC, check_acc = False):   
    # generate inputs
    vRANGE = 1
    # np.random.seed(0)
    loraA = np.random.randint(-vRANGE, vRANGE+1, [IC, rank]).astype(np.float16)
    alpha = np.random.rand(rank).astype(np.float16)
    loraB = np.random.randint(-vRANGE, vRANGE+1, [rank, OC]).astype(np.float16)
    
    lora = onednnLoRA(loraA, loraB, alpha, OC, IC, rank, False)

    num_iters = 1 if check_acc else 10
    for r in range(num_iters):
        # cl.finish()
        lora_input = np.random.randint(-vRANGE, vRANGE+1, [n_tokens, IC]).astype(np.float16)
        main_input = np.random.randint(-vRANGE, vRANGE+1, [n_tokens, OC]).astype(np.float16)

        ocl_res, durs = oclLoRA(main_input, lora_input, loraA, loraB, alpha, n_tokens, rank, IC, OC, check_acc=True)
        for ns in durs:
            print(f'ocl_lora is calculated!, {ns*1e-6:.3f} ms')

        # cl.finish()
        # t0 = time.time()
        dst_cur = lora.forward(lora_input, main_input)
        # latency_ns = cl.finish()
        dst_cur = dst_cur.numpy()
        # t1 = time.time()
        # for t in latency_ns:
        #     print(f"\t onednn_lora is calculated! {t*1e-3:.3f} us")
        # print(f" onednn_lora is calculated![{r}] :  {(t1 - t0)*1e3 : .3f} ms")

        if check_acc:
            dst_ref = calc_ref(main_input, lora_input, loraA, loraB, alpha, False)
            compare(dst_ref, dst_cur)

            compare(ocl_res, dst_cur)
            print(f'BATCH:{n_tokens} INPUT_STATE:{IC}, RANK:{rank}, OUPUT_STATE:{OC} ACC PASS!')

# test_lora2(8, 16, 1536, 512)
# test_lora2(8, 16, 7*16, 256, True)
test_lora2(3019, 64, 8960, 1536)