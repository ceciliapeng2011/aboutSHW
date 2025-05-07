#!/usr/bin/python3
import numpy as np
import time, sys
import torch
# from torch import nn

from clops import cl
from clops import compare

enable_debug = False

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

        t0 = time.time()
        self.linear_A.forward(cl.tensor(lora_input), temp_resA, cl.tensor(self.alpha), cl.tensor(self.loraA))
        self.linear_B.forward(temp_resA, dst, cl.tensor(main_input), cl.tensor(self.loraB))
        t1 = time.time()

        return dst, (t1 - t0)*1e3

    @staticmethod
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

        dst_cur, _ = lora.forward(lora_input, main_input)
        dst_cur = dst_cur.numpy()
        print("cur is calculated!")
        
        dst_ref = onednnLoRA.calc_ref(main_input, lora_input, loraA, loraB, alpha, True)
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
        dst_cur, _ = lora.forward(lora_input, main_input)
        dst_cur = dst_cur.numpy()
        print("cur is calculated!")

        dst_ref = onednnLoRA.calc_ref(main_input, lora_input, loraA, loraB, alpha, False)
        compare(dst_ref, dst_cur)
        
# test_lora1()



################################################################################################################
from clops.lora import blocking_1nd, LORA_1ST, blocking_2nd, LORA_2ND

cl.profiling(True)

def oclLoRA(main_input, lora_input, loraA, loraB, alpha, n_tokens, rank, IC, OC, check_acc = True):
    if check_acc:
        REPEAT = 1
    else:
        REPEAT = 100        
        
    Aoutput = np.zeros([n_tokens, rank]).astype(np.float16)
    
    stateA_list= [cl.tensor(loraA) for _ in range(REPEAT)]
    alpha_list = [cl.tensor(alpha) for _ in range(REPEAT)]
    stateB_list = [cl.tensor(loraB)for _ in range(REPEAT)]
    loraInput_list = [cl.tensor(lora_input)for _ in range(REPEAT)]
    mainInput_list = [cl.tensor(main_input)for _ in range(REPEAT)]
    # Must set the output to be zeros to avoid not all the data is updated in GEMMA. 
    A_output_list = [cl.tensor(Aoutput)for _ in range(REPEAT)]
    res_list = [cl.tensor([n_tokens, OC], np.dtype(np.float16))for _ in range(REPEAT)]
    
    if n_tokens == 1:
        gemma_sg_BK, gemma_sgK, gemmb_sgN = blocking_2nd(rank, IC, OC)
        opt = LORA_2ND(rank, IC, OC, gemma_sg_BK, gemma_sgK, gemmb_sgN,False)
    else:
        A_regM, A_regN, A_sgM, A_sgN, B_regM, B_regN, B_sgM, B_sgN = blocking_1nd(n_tokens, rank, IC, OC)
        opt = LORA_1ST( n_tokens, rank, IC, OC,  A_regM, A_regN, A_sgM, A_sgN, B_regM, B_regN, B_sgM, B_sgN, False)
    
    cl.finish()
    t0 = time.time()
    opt(mainInput_list[0], loraInput_list[0], stateA_list[0], alpha_list[0], stateB_list[0], A_output_list[0], res_list[0])
    duration = cl.finish()
    t1 = time.time()
    
    return res_list[0].numpy(), duration, (t1 - t0)*1e3

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

        ocl_res, durs, elapse = oclLoRA(main_input, lora_input, loraA, loraB, alpha, n_tokens, rank, IC, OC, check_acc=True)
        print(f" ocl_lora is calculated![{r}] :  {elapse: .3f} ms")
        for ns in durs:
            print(f'cl kernel durations, {ns*1e-6:.3f} ms')
       
        dst_cur, elapse = lora.forward(lora_input, main_input)
        dst_cur = dst_cur.numpy()
        print(f" onednn_lora is calculated![{r}] :  {elapse : .3f} ms")

        if check_acc:
            dst_ref = onednnLoRA.calc_ref(main_input, lora_input, loraA, loraB, alpha, False)
            compare(dst_ref, dst_cur)

            compare(ocl_res, dst_cur)
            print(f'BATCH:{n_tokens} INPUT_STATE:{IC}, RANK:{rank}, OUPUT_STATE:{OC} ACC PASS!')

# test_lora2(1, 64, 1536, 512)
# test_lora2(8, 16, 7*16, 256, True)
# test_lora2(3019, 64, 8960, 1536)

def test_qwen(n_tokens):
    # test perf based on qwen parameters
    # MLP up, gate
    test_lora2(n_tokens, 64, 1536, 8960)

    # MLP down
    test_lora2(n_tokens, 64, 8960, 1536)

    # Q, O
    test_lora2(n_tokens, 64, 1536, 1536)

    # KV
    test_lora2(n_tokens, 64, 256, 256)
# test_qwen(1)  # 2nd token
# test_qwen(3019)  # 1st token
# test_qwen(1024)  # 1st token

################################################################################################################
from test_qkv_lora_fusiong import qkv_blocking_1st, QKV_LORA_1ST, qkv_blocking_2nd, QKV_LORA_2ND

cl.profiling(True)

def oclLoRA_QKVFused(main_input, lora_input, loraA : list, loraB : list, alpha : list, batch, rank, input_state, kv_state):
    qkv_state = input_state + kv_state*2

    Aoutput = cl.tensor(np.zeros([batch, rank*3]).astype(np.float16))
    result = cl.tensor([batch, qkv_state], np.dtype(np.float16))
    
    if batch == 1:
        gemma_sg_BK, gemma_sgK, gemmb_sgN = qkv_blocking_2nd(rank, input_state, input_state+2*kv_state)
        opt = QKV_LORA_2ND(rank, input_state, kv_state, gemma_sg_BK, gemma_sgK, gemmb_sgN,False)
    else:
        A_regM, A_regN, A_sgM, A_sgN, B_regM, B_regN, B_sgM, B_sgN = qkv_blocking_1st(batch, rank, input_state, input_state)
        opt = QKV_LORA_1ST( batch, rank, input_state, kv_state,  A_regM, A_regN, A_sgM, A_sgN, B_regM, B_regN, B_sgM, B_sgN, False)

    loraA = [cl.tensor(t) for t in loraA]
    loraB = [cl.tensor(t) for t in loraB]
    alpha = [cl.tensor(t) for t in alpha]

    main_input = cl.tensor(main_input)
    lora_input = cl.tensor(lora_input)

    cl.finish()
    t0 = time.time()
    opt(main_input, lora_input,
        loraA[0], loraA[1], loraA[2], None, 
        alpha[0], alpha[1], alpha[2], None,
        loraB[0], loraB[1], loraB[2],
        Aoutput, result)
    duration = cl.finish()
    t1 = time.time()
    
    return result.numpy(), duration, (t1 - t0)*1e3

class onednnLoRAFusedQKV:
    def __init__(self, loraA : list, loraB : list, alpha : list, head_q, head_kv, rank, from_linear = False):
        self.head_q = head_q
        self.head_kv = head_kv
        self.rank = rank
        self.M = -1

        self.from_linear = from_linear
        
        self.w_dtype = cl.onednn_dtype.f16 if loraA[0].dtype == np.float16 else cl.onednn_dtype.f32
        self.act_dtype = self.w_dtype

        self.loraA = [cl.tensor(t) for t in loraA]
        self.loraB = [cl.tensor(t) for t in alpha]
        self.alpha = [cl.tensor(t) for t in loraB]

    def update_batch(self, n_tokens):
        if self.M != n_tokens:
            if enable_debug: print("======== create onednn_lorafused =========")
            self.lora_op = cl.onednn_lorafused(self.act_dtype, self.w_dtype, n_tokens, self.rank, self.head_q, self.head_kv)
            self.M = n_tokens

    def forward(self, lora_input, main_input):
        M = lora_input.shape[0]
        self.update_batch(M)
        dst = cl.tensor(np.zeros(main_input.shape, dtype=np.float16 if self.act_dtype == cl.onednn_dtype.f16 else np.float32))
        interm_A_output = cl.tensor(np.zeros([self.M, self.rank*3], dtype=np.float16 if self.act_dtype == cl.onednn_dtype.f16 else np.float32))

        main_input = cl.tensor(main_input)
        lora_input = cl.tensor(lora_input)

        t0 = time.time()
        self.lora_op.forward(main_input, lora_input, dst, interm_A_output, self.loraA, self.loraB, self.alpha)
        t1 = time.time()

        return dst, (t1 - t0)*1e3
    
    @staticmethod
    def calc_ref(main_input, lora_input, loraA : list, loraB : list, alpha : list, transpose_w = True):
        if transpose_w:
            loraA = [t.transpose().copy() for t in loraA]
            loraB = [t.transpose().copy() for t in loraB]
        # print(f'{loraA[0].shape=}, {loraB[0].shape=}')
        # concat A and alpha
        loraA = np.hstack(loraA)
        alpha = np.hstack(alpha)
        
        # print(f'{lora_input.shape=}, {loraA.shape=}')

        # gemm A
        intermediate = lora_input @ loraA
        intermediate *= alpha
        # print(f"{intermediate=}")
        
        # gemm B x3
        intermediate = np.hsplit(intermediate, 3)
        # print(f"{intermediate[0]=}, {intermediate[1]=}, {intermediate[2]=}")
        dst_ref = [intermediate[k] @ loraB[k] for k in range(3)]
        # print(f"{dst_ref[0]=}, {dst_ref[1]=}, {dst_ref[2]=}")
        dst_ref = np.hstack(dst_ref)
        dst_ref += main_input
        print("ref is calculated!")
        return dst_ref

def test_lora_fusedqkv(n_tokens, rank, head_q, head_kv, rt_dtype = np.float16, check_acc = False):   
    # generate inputs
    vRANGE = 1
    # np.random.seed(0)
    loraA = [np.random.randint(-vRANGE, vRANGE+1, [head_q, rank]).astype(rt_dtype),
             np.random.randint(-vRANGE, vRANGE+1, [head_q, rank]).astype(rt_dtype),
             np.random.randint(-vRANGE, vRANGE+1, [head_q, rank]).astype(rt_dtype)]
    # loraA = [np.ones([head_q, rank]).astype(rt_dtype), 2*np.ones([head_q, rank]).astype(rt_dtype), 3*np.ones([head_q, rank]).astype(rt_dtype)]
    alpha = [np.random.rand(rank).astype(rt_dtype),
             np.random.rand(rank).astype(rt_dtype),
             np.random.rand(rank).astype(rt_dtype)]
    # alpha = [np.ones([rank]).astype(rt_dtype), np.ones([rank]).astype(rt_dtype), np.ones([rank]).astype(rt_dtype)]
    loraB = [np.random.randint(-vRANGE, vRANGE+1, [rank, head_q]).astype(rt_dtype),
             np.random.randint(-vRANGE, vRANGE+1, [rank, head_kv]).astype(rt_dtype),
             np.random.randint(-vRANGE, vRANGE+1, [rank, head_kv]).astype(rt_dtype)]
    # loraB = [np.ones([rank, head_q]).astype(rt_dtype), np.ones([rank, head_kv]).astype(rt_dtype), np.ones([rank, head_kv]).astype(rt_dtype)]
    
    lora = onednnLoRAFusedQKV(loraA, loraB, alpha, head_q, head_kv, rank)

    num_iters = 1 if check_acc else 10
    for r in range(num_iters):
        # cl.finish()
        lora_input = np.random.randint(-vRANGE, vRANGE+1, [n_tokens, head_q]).astype(rt_dtype)
        # lora_input = np.ones([n_tokens, head_q]).astype(rt_dtype)
        main_input = np.random.randint(-vRANGE, vRANGE+1, [n_tokens, head_q+head_kv*2]).astype(rt_dtype)
      
        ocl_res, durs, elapse = oclLoRA_QKVFused(main_input, lora_input, loraA, loraB, alpha, n_tokens, rank, head_q, head_kv)
        print(f" oclLoRA_QKVFused is calculated![{r}] :  {elapse: .3f} ms")
        for ns in durs:
            print(f'cl kernel durations, {ns*1e-6:.3f} ms')

        dst_cur, elapse = lora.forward(lora_input, main_input)
        dst_cur = dst_cur.numpy()
        print(f" onednnLoRA_QKVFused is calculated![{r}] :  {elapse : .3f} ms")

        if check_acc:
            dst_ref = onednnLoRAFusedQKV.calc_ref(main_input.copy(), lora_input.copy(), loraA.copy(), loraB.copy(), alpha.copy(), False)
            compare(dst_ref, dst_cur)

            compare(ocl_res, dst_cur, atol=1.0, rtol=0.01)
            print(f'BATCH:{n_tokens} Q_STATE:{head_q}, RANK:{rank}, KV_STATE:{head_kv} ACC PASS!')

# test_lora_fusedqkv(1, 64, 1536, 256, check_acc=False)
# test_lora_fusedqkv(32, 64, 1536, 256, check_acc=False)
test_lora_fusedqkv(1024, 64, 1536, 256, check_acc=False)
# test_lora_fusedqkv(3192, 64, 1536, 256, check_acc=False)
# test_lora_fusedqkv(8*1024, 64, 1536, 256, check_acc=False)