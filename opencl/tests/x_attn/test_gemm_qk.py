import os

import torch
import numpy as np
import pytest

import functools

from clops import cl
from clops import compare
from clops.utils import Colors

from references import get_gemm_ref

def div_up(a, b):
    return (a + b - 1) // b
def rnd_up(a, b):
    return (a + b - 1) // b * b

def get_cm_grf_width():
    cm_kernels = cl.kernels(r'''
    extern "C" _GENX_MAIN_ void cm_get_grf_width(int * info [[type("svmptr_t")]]) {
        info[0] = CM_GRF_WIDTH;
    }''', f"-cmc")
    t_info = cl.tensor([2], np.dtype(np.int32))
    cm_kernels.enqueue("cm_get_grf_width", [1], [1], t_info)
    return t_info.numpy()[0]

CM_GRF_WIDTH = get_cm_grf_width()
print(f"{CM_GRF_WIDTH=}")
if CM_GRF_WIDTH == 256:
    xe_arch = 1
else:
    xe_arch = 2
print(f"xe_arch: {xe_arch}")

DUMP_ENQUEUE_ARGUMENTS = True
THRESH = 0.9 # useless in gemmQK actually
SOFTMAX_TYPE = 'float' # 'half'
STRIDE = 16
BLOCK_SG_M = 64
BLOCK_SG_N = 32
SUB_BLOCK_SIZE = 16

if xe_arch == 1:
    BLOCK_SG_M = 32
    BLOCK_SG_N = 16

SG_M = 4
SG_N = 8
BLOCK_WG_M = BLOCK_SG_M * SG_M
BLOCK_WG_N = BLOCK_SG_N * SG_N
KV_BLOCK_SIZE = 256
PA_WG_SIZE = 256 if CM_GRF_WIDTH == 512 else 128


def setup_module(module):
    torch.manual_seed(3)
    torch.set_printoptions(linewidth=1024)
    cl.profiling(True)

class xattn_gemmQK:
    def __init__(self, num_heads, num_kv_heads, head_size, xattn_block_size, is_causal, kvcache_compressed):
        BLOCK_WG_K = 64 if head_size % 64 == 0 else 32

        if isinstance(kvcache_compressed, bool):
            kv_cache_compression = 1 if kvcache_compressed else 0
        else:
            kv_cache_compression = int(kvcache_compressed)

        if kv_cache_compression == 1:
            self.HEAD_SIZE_KEY = head_size + 2 * 2
        elif kv_cache_compression == 2:
            self.HEAD_SIZE_KEY = head_size + head_size // 4
        else:
            self.HEAD_SIZE_KEY = head_size

        # loop order walks HQ first and the step is WALK_HQ, 1 means not walk HQ, 2 means walks 2 heads first. Valid value: 1, 2, 4...
        self.WALK_HQ = 2 if num_heads != num_kv_heads else 1
        
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.xattn_block_size = xattn_block_size
        self.is_causal = is_causal
        self.kv_cache_compression = kv_cache_compression

        INV_S = 1 / torch.sqrt(torch.tensor([head_size], dtype=torch.float32)) / STRIDE
        INV_S = int(INV_S.view(dtype=torch.uint32))

        src = r'''#include "xattn_gemm_qk.hpp"'''
        cwd = os.path.dirname(os.path.realpath(__file__))
        print(f"compiling {cwd} ...")

        jit_option = '-abortonspill -noschedule '
        self.kernels = cl.kernels(src, f'''-cmc -Qxcm_jit_option="{jit_option}"
                            -mCM_printregusage -mdump_asm -g2
                            -Qxcm_register_file_size=256 -I{cwd}
                            -DSTRIDE={STRIDE} -DHQ={num_heads} -DHK={num_kv_heads} -DHEAD_SIZE={head_size} -DSG_M={SG_M} -DSG_N={SG_N} -DBLOCK_SG_N={BLOCK_SG_N} -DBLOCK_SG_M={BLOCK_SG_M}
                            -DBLOCK_SIZE={int(xattn_block_size)} -DINV_S={INV_S} -DKV_BLOCK_SIZE={KV_BLOCK_SIZE} -DBLOCK_SHARE_MAX={BLOCK_WG_N} -DWALK_HQ={self.WALK_HQ}
                            -DIS_CAUSAL={int(is_causal)} -DKV_CACHE_COMPRESSION={self.kv_cache_compression} -DUSE_INT8={int(self.kv_cache_compression != 0)} -DHEAD_SIZE_KEY={self.HEAD_SIZE_KEY} -DSOFTMAX_TYPE={SOFTMAX_TYPE}
                            -DSUB_BLOCK_SIZE={SUB_BLOCK_SIZE}
                            -DBLOCK_WG_K={int(BLOCK_WG_K)}
                            ''')
        
    def __call__(self, t_key_cache, t_query, t_block_indices, t_block_indices_begins, q_start_strided, M, N, K, query_stride, n_repeats = 1):
        batch = 1
        q_stride_pad = rnd_up(M, BLOCK_WG_M)
        # [1, 32, 64, 256]
        softmax_type = np.float16 if SOFTMAX_TYPE == 'half' else np.float32
        N_kq_groups = div_up(N, BLOCK_WG_N)
        t_kq_max_wg = cl.tensor(np.ones([batch, self.num_heads, N_kq_groups, q_stride_pad], softmax_type))
        # [1, 32, 256, 64 * 16]
        sum_per_token_in_block = self.xattn_block_size // STRIDE
        k_block_in_group = BLOCK_WG_N // sum_per_token_in_block
        k_block_pad = k_block_in_group * N_kq_groups
        t_kq_exp_partial_sum = cl.tensor(np.ones([batch, self.num_heads, q_stride_pad, k_block_pad], softmax_type))

        # loop N first:[0, 1], loop M first:[0, 0]; block M first[slice_no, slice(>0)], block N first[slice_no, slice(<0)]
        #default linear
        slice_no = 0
        block_m = M // BLOCK_WG_M
        block_n = N // BLOCK_WG_N
        devinfo = cl.dev_info()
        eu_xecore = 8
        xecores = devinfo["CL_DEVICE_MAX_COMPUTE_UNITS"] // eu_xecore
        slice_no = 0
        slice = 0
        print(f'{xecores=} {block_m=} {block_n=} {slice=} {slice_no=}')

        GWS = [N_kq_groups * (q_stride_pad // BLOCK_WG_M) * SG_N * self.WALK_HQ, SG_M, self.num_heads // self.WALK_HQ]
        LWS = [SG_N, SG_M, 1]
        print(f"calling CM_gemm_qk {GWS=} {LWS=} ...")

        if DUMP_ENQUEUE_ARGUMENTS:
            LABEL_WIDTH = 32
            cltensors = [
                ("t_key_cache",            t_key_cache),
                ("t_query",                t_query),
                ("t_block_indices",        t_block_indices),
                ("t_block_indices_begins", t_block_indices_begins),
                ("t_kq_max_wg",            t_kq_max_wg),
                ("t_kq_exp_partial_sum",   t_kq_exp_partial_sum),
            ]
            lines = [(name, value.numel * value.dtype.itemsize) for name, value in cltensors]
            print("gemm_qk size of memories:")
            for name, value in lines:
                print(f"  {name:<{LABEL_WIDTH}} {value}")
            print("\gemm_qk scalers:")
            print(f"  M:{M:<10}  N:{N:<10}  "
                f"mask_H:{K:<10}  query_stride:{query_stride:<10}  "
                f"q_start_strided:{q_start_strided:<10} ")

        cl.finish()
        # gemm
        for i in range(n_repeats):
            self.kernels.enqueue('gemm_qk', GWS, LWS, t_key_cache, t_query, t_block_indices, t_block_indices_begins,
                                 t_kq_max_wg, t_kq_exp_partial_sum, M, N, K, query_stride, slice_no, slice, q_start_strided)
        latency = cl.finish()
        return latency, t_kq_max_wg, t_kq_exp_partial_sum
        

    @staticmethod
    @functools.cache
    def create_instance(num_heads, num_kv_heads, head_size, xattn_block_size, is_causal, kvcache_compressed):
        return xattn_gemmQK(num_heads, num_kv_heads, head_size, xattn_block_size, is_causal, kvcache_compressed)

    @staticmethod
    def _round_to_even(tensor: torch.Tensor) -> torch.Tensor:
        rounded = torch.floor(tensor + 0.5)
        adjustment = (rounded % 2 != 0) & (torch.abs(tensor - rounded) == 0.5000)
        adjustment = adjustment | (rounded > 255)
        result = rounded - adjustment.to(rounded.dtype)
        return torch.clamp(result, min=0, max=255)

    @staticmethod
    def _quant_per_channel(kv_cache_blocks: torch.Tensor, tail_sub_block: int, tail_token: int):
        blk_num, kv_heads, num_sub_blocks, sub_block_size, head_size = kv_cache_blocks.shape
        mask = torch.ones_like(kv_cache_blocks, dtype=torch.bool)
        if tail_token:
            mask[:, :, tail_sub_block:tail_sub_block + 1, tail_token:, :] = False
        kv_max = torch.where(mask, kv_cache_blocks, torch.tensor(float("-inf"), dtype=torch.float16)).amax(dim=3, keepdim=True)
        kv_min = torch.where(mask, kv_cache_blocks, torch.tensor(float("inf"), dtype=torch.float16)).amin(dim=3, keepdim=True)
        qrange = kv_max - kv_min

        kv_scale = 255.0 / qrange.to(dtype=torch.float)
        zero_mask = qrange == 0
        if zero_mask.any():
            kv_scale = torch.where(zero_mask, torch.ones_like(kv_scale), kv_scale)

        kv_scale_div = (1.0 / kv_scale).to(dtype=torch.float16)
        kv_zp = ((0.0 - kv_min) * kv_scale + 0.0).to(dtype=torch.float16)

        kv_u8 = xattn_gemmQK._round_to_even((kv_cache_blocks * kv_scale).to(dtype=torch.float16) + kv_zp).to(dtype=torch.uint8)
        kv_u8 = kv_u8.reshape(blk_num, kv_heads, -1)
        dq_scale = kv_scale_div.view(dtype=torch.uint8).reshape(blk_num, kv_heads, -1)
        kv_zp = kv_zp.view(dtype=torch.uint8).reshape(blk_num, kv_heads, -1)
        return kv_u8, dq_scale, kv_zp

    def quant_i8(self, k:torch.Tensor):
        HEAD_SIZE = self.head_size
        B, Hk, Lk, S = k.shape
        k_pad = torch.zeros([B, Hk, rnd_up(Lk, KV_BLOCK_SIZE), HEAD_SIZE], dtype=k.dtype)
        k_pad[:, :, :Lk,:] = k

        B, Hk, Lk, S = k_pad.shape
        k_i8 = torch.zeros([B, Hk, Lk, self.HEAD_SIZE_KEY], dtype=torch.uint8)
        k_i8_4d = k_i8.reshape([B, Hk, -1, KV_BLOCK_SIZE * self.HEAD_SIZE_KEY])
        if self.kv_cache_compression == 2:
            num_sub_blocks = KV_BLOCK_SIZE // SUB_BLOCK_SIZE
            num_blocks = Lk // KV_BLOCK_SIZE
            k_groups = k_pad.reshape(B * Hk * num_blocks, 1, num_sub_blocks, SUB_BLOCK_SIZE, HEAD_SIZE)
            kv_u8, dq_scale, kv_zp = self._quant_per_channel(k_groups, 0, 0)
            k_i8_4d[:, :, :, :KV_BLOCK_SIZE * HEAD_SIZE] = kv_u8.reshape(B, Hk, -1, KV_BLOCK_SIZE * HEAD_SIZE)
            scale_start = KV_BLOCK_SIZE * HEAD_SIZE
            scale_end = scale_start + num_sub_blocks * HEAD_SIZE * 2
            k_i8_4d[:, :, :, scale_start:scale_end] = dq_scale.reshape(B, Hk, -1, num_sub_blocks * HEAD_SIZE * 2)
            k_i8_4d[:, :, :, scale_end:] = kv_zp.reshape(B, Hk, -1, num_sub_blocks * HEAD_SIZE * 2)
            return k_i8, k_pad[:, :, :k.shape[2], :]
        if 0:
            max = torch.max(k_pad, dim=-1, keepdim=True)[0].to(torch.float32)
            min = torch.min(k_pad, dim=-1, keepdim=True)[0].to(torch.float32)
            diff_value = torch.masked_fill(max - min, max == min, 0.001)
            scale = 255/ diff_value
            zp = -min * scale
            quanted = k_pad * scale + zp
            quanted = quanted.clamp(0, 255)
            scale = 1.0 / scale
        else:
            scale = torch.randint(-1, 3, [B, Hk, Lk, 1], dtype=torch.float16)
            zp = torch.randint(0, 3, [B, Hk, Lk, 1], dtype=torch.float16)
            quanted = torch.randint(0, 3, [B, Hk, Lk, S], dtype=torch.float16)
            k_pad = (quanted - zp) * scale
            k = k_pad[:, :, :k.shape[2], :]
            k_pad[:,:,k.shape[2]:,:] = 0
        # weights
        k_i8_4d[:, :, :, :KV_BLOCK_SIZE * HEAD_SIZE] = torch.reshape(quanted, [B, Hk, Lk // KV_BLOCK_SIZE, -1])
        # scale
        k_i8_4d[:, :, :, KV_BLOCK_SIZE * HEAD_SIZE : (KV_BLOCK_SIZE * (HEAD_SIZE + 2))] = (scale).to(torch.float16).reshape([B, Hk, Lk // KV_BLOCK_SIZE, -1]).view(dtype=torch.uint8)
        # zp
        k_i8_4d[:, :, :, (KV_BLOCK_SIZE * (HEAD_SIZE + 2)) : ] = zp.to(torch.float16).reshape([B, Hk, Lk // KV_BLOCK_SIZE, -1]).view(dtype=torch.int8)

        return k_i8, k

# q: [B, Hq, L_q, S]
# k: [B, Hk, L_k, S]
def run_gemm(q:torch.Tensor, k:torch.Tensor, q_start_strided, xattn_block_size, num_heads, num_kv_heads, head_size, kvcache_compressed, causal=True, perf=True):
    B, Hq, Lq, S = q.shape
    _, Hk, Lk, _ = k.shape
    Lk = Lk // STRIDE * STRIDE
    M = Lq // STRIDE                  # will slient drop the tails which is less than `STRIDE`
    N = Lk // STRIDE
    K = STRIDE * S

    k = k[:,:,:Lk,:]                  # will slient drop the tails which is less than `STRIDE`

    # k -> key_cache blocked layout
    block_indices_begins = torch.zeros(B + 1).to(torch.int32)
    block_indices = torch.arange((Lk + KV_BLOCK_SIZE - 1) // KV_BLOCK_SIZE)
    # simulate random block allocation
    perm_idx = torch.randperm(block_indices.shape[0])
    inv_per_idx = torch.argsort(perm_idx)
    block_indices = block_indices[inv_per_idx]

    t_block_indices = cl.tensor(block_indices.to(torch.int32).detach().numpy())
    # t_past_lens = cl.tensor(past_lens.to(torch.int32).detach().numpy())                   # M_kq has already been calculated
    t_block_indices_begins = cl.tensor(block_indices_begins.to(torch.int32).detach().numpy())
    # t_subsequence_begins = cl.tensor(subsequence_begins.to(torch.int32).detach().numpy()) # N_kq has already been calculated
    
    xattn_cm = xattn_gemmQK.create_instance(num_heads, num_kv_heads, head_size, xattn_block_size, causal, kvcache_compressed)

    if xattn_cm.kv_cache_compression:
        k_pad, k = xattn_cm.quant_i8(k)
        key_cache = k_pad.reshape([B, Hk, -1, KV_BLOCK_SIZE, xattn_cm.HEAD_SIZE_KEY]).permute(0, 2, 1, 3, 4)
    else:
        k_pad = torch.zeros([B, Hk, (Lk + KV_BLOCK_SIZE - 1) // KV_BLOCK_SIZE * KV_BLOCK_SIZE, head_size], dtype=k.dtype)
        k_pad[:, :, :Lk,:] = k
        key_cache = k_pad.reshape([B, Hk, -1, KV_BLOCK_SIZE, head_size]).permute(0, 2, 1, 3, 4)
    key_cache = key_cache[:,perm_idx,:,:,:].contiguous()
    t_key_cache = cl.tensor(key_cache.detach().numpy())

    # [B, Hq, L, S] -> [B, L, Hq * S]
    q_3d = q.permute(0, 2, 1, 3).reshape([B, Lq, -1])
    # simulate layout of q,k,v is [B, L, Hq * S..Hkv * S]
    q_3d_with_padding = torch.zeros([B, Lq, Hq * S * 2], dtype=q_3d.dtype)
    q_3d_with_padding[:, :, : Hq * S] = q_3d
    t_query = cl.tensor(q_3d_with_padding.detach().numpy())
    query_stride = K * num_heads * 2

    n_repeats = 100 if perf else 1
    ns, t_kq_max_wg, t_kq_exp_partial_sum = xattn_cm(t_key_cache, t_query, t_block_indices, t_block_indices_begins, q_start_strided, M, N, K, query_stride, n_repeats)

    if not perf:
        # [1, 32, 256], [1, 32, 64, 256], [1, 32, 256, 64 * 16], A_sum:[1, 32, 32, 64 * 16]
        kq_max_ref, kq_5d_max_ret_ref, kq_exp_partial_sum_ret_ref, _ = get_gemm_ref(q, k, q_start_strided=q_start_strided, block_size = xattn_block_size,
                                                                                    S=STRIDE, threshold=THRESH, causal=causal, wg_k=BLOCK_WG_N, wg_q=BLOCK_WG_M)
        kq_5d_max_ret_ref_np = kq_5d_max_ret_ref.detach().numpy()[..., :M]
        t_kq_max_wg_np = t_kq_max_wg.numpy()[..., :M]
        compare(kq_5d_max_ret_ref_np, t_kq_max_wg_np)
        print(f'{Colors.GREEN}gemm:max_wg passed{Colors.END}')
        kq_exp_partial_sum_ret_ref_np = kq_exp_partial_sum_ret_ref.detach().numpy()[:,:,:M,:]
        t_kq_exp_partial_sum_np = t_kq_exp_partial_sum.numpy()[:,:,:M,:]
        compare(kq_exp_partial_sum_ret_ref_np, t_kq_exp_partial_sum_np)
        print(f'{Colors.GREEN}gemm:exp_partial passed{Colors.END}')
    else:
        flops = B * Hq * M * N * K * 2
        for i, time_opt in enumerate(ns):
            print(f'(GEMM)TPUT_{i}:{flops/time_opt:,.0f} GFLOPS, BW:{(M*K+K*N+M*N)*2/time_opt:,.0f} GB/s {time_opt*1e-3:,.0f} us')

    return t_kq_max_wg, t_kq_exp_partial_sum

def run_func(xattn_block_size, num_heads = 32, num_kv_heads = 8, head_size = 128, kvcache_compressed = True, is_causal = True):
    dim = head_size
    sizes = [
        # real cases
        (612, 612, '612'),
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
        print(f'{Colors.BLUE}test gemm("{prompt}") query: [{q_len}, {dim}*{STRIDE}] key:[{k_len}, {dim}*{STRIDE}] xattn_block_size:{xattn_block_size} ...{Colors.END}')

        q = torch.randint(-2, 4, size=[1, num_heads, q_len, dim], dtype=torch.int16).to(dtype=torch.float16)
        k = torch.randint(-2, 4, size=[1, num_kv_heads, k_len, dim], dtype=torch.int16).to(dtype=torch.float16)

        if is_causal:
            q_start_strided = k_len // STRIDE - q_len // STRIDE
            assert q_start_strided >= 0, "length of key cache must be greater or equal than query"
        else:
            q_start_strided = 0
        run_gemm(q, k, q_start_strided, xattn_block_size, num_heads, num_kv_heads, head_size, kvcache_compressed, is_causal, perf=False)

def run_perf(xattn_block_size, num_heads = 32, num_kv_heads = 8, head_size = 128, kvcache_compressed = True, is_causal = True):
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

    q = torch.randint(-2, 4, size=[1, num_heads, q_len, head_size], dtype=torch.int16).to(dtype=torch.float16)
    k = torch.randint(-2, 4, size=[1, num_kv_heads, k_len, head_size], dtype=torch.int16).to(dtype=torch.float16)
    q_start_strided=k_len // STRIDE - q_len // STRIDE

    run_gemm(q, k, q_start_strided, xattn_block_size, num_heads, num_kv_heads, head_size, kvcache_compressed, is_causal, perf=True)

# ============================================================================
# Multi-subsequence support
# ============================================================================

XATTN_META_STRIDE = 16

def build_xattn_subseq_meta(subsequences, num_heads, xattn_block_size, merged_q_num=2):
    """Build per-subsequence metadata buffer for multi-subseq kernel dispatch.

    Args:
        subsequences: list of (subseq_q_begin, q_len, kv_len, block_indices_begin) tuples
        num_heads: number of query heads
        xattn_block_size: xattention block size (128 or 256)
        merged_q_num: MERGED_Q_NUM (PA_WG_SIZE // xattn_block_size)
    Returns:
        (meta_np, total_wg_count, buf_sizes) where:
        - meta_np: int32 numpy array [num_subseqs, XATTN_META_STRIDE]
        - total_wg_count: total workgroups across all subsequences
        - buf_sizes: dict with total cumulative buffer sizes
    """
    softmax_type = np.float16 if SOFTMAX_TYPE == 'half' else np.float32
    sizeof_softmax = softmax_type().itemsize
    sum_per_token_in_block = xattn_block_size // STRIDE
    k_block_in_group = BLOCK_WG_N // sum_per_token_in_block

    meta = []
    offset_kq_max = 0
    offset_exp_sum = 0
    offset_mask = 0
    offset_mask_wg = 0
    wg_offset = 0

    for subseq_q_begin, q_len_s, kv_len_s, bi_begin in subsequences:
        M = q_len_s // STRIDE
        N = kv_len_s // STRIDE
        q_stride_pad = rnd_up(M, BLOCK_WG_M)
        N_kq_groups = div_up(N, BLOCK_WG_N)
        q_block_pad = div_up(q_len_s, xattn_block_size)
        k_block_pad = k_block_in_group * N_kq_groups
        k_block = div_up(kv_len_s, xattn_block_size)
        q_block = div_up(q_len_s, xattn_block_size)
        causal_start = k_block - q_block
        q_start_strided = (kv_len_s - q_len_s) // STRIDE
        wg_count_s = N_kq_groups * (q_stride_pad // BLOCK_WG_M)

        meta.append([
            subseq_q_begin,     # [0]  SUBSEQ_Q_BEGIN
            q_len_s,            # [1]  SUBSEQ_Q_LEN
            M,                  # [2]  M
            N,                  # [3]  N
            q_stride_pad,       # [4]  Q_STRIDE_PAD
            N_kq_groups,        # [5]  N_KQ_GROUPS
            q_block_pad,        # [6]  Q_BLOCK_PAD
            k_block_pad,        # [7]  K_BLOCK_PAD
            causal_start,       # [8]  CAUSAL_START
            q_start_strided,    # [9]  Q_START_STRIDED
            offset_kq_max,      # [10] BUF_OFF_KQ_MAX (byte offset to start of this subseq's region)
            offset_exp_sum,     # [11] BUF_OFF_EXP_SUM
            offset_mask,        # [12] BUF_OFF_MASK
            offset_mask_wg,     # [13] BUF_OFF_MASK_WG
            bi_begin,           # [14] BLOCK_IDX_BEGIN
            wg_offset,          # [15] WG_OFFSET
        ])

        # Accumulate total buffer sizes (all heads)
        offset_kq_max += N_kq_groups * q_stride_pad * sizeof_softmax * num_heads
        offset_exp_sum += q_stride_pad * k_block_pad * sizeof_softmax * num_heads
        offset_mask += q_block_pad * k_block_pad * num_heads
        offset_mask_wg += div_up(q_block_pad, merged_q_num) * k_block_pad * num_heads
        wg_offset += wg_count_s

    meta_np = np.array(meta, dtype=np.int32)
    return meta_np, wg_offset, {
        'kq_max': offset_kq_max,
        'exp_sum': offset_exp_sum,
        'mask': offset_mask,
        'mask_wg': offset_mask_wg,
    }


class xattn_gemmQK_multi:
    """Multi-subsequence variant of xattn_gemmQK. Compiles with -DMULTI_SUBSEQ=1."""
    def __init__(self, num_heads, num_kv_heads, head_size, xattn_block_size, is_causal, kvcache_compressed):
        BLOCK_WG_K = 64 if head_size % 64 == 0 else 32

        if isinstance(kvcache_compressed, bool):
            kv_cache_compression = 1 if kvcache_compressed else 0
        else:
            kv_cache_compression = int(kvcache_compressed)

        if kv_cache_compression == 1:
            self.HEAD_SIZE_KEY = head_size + 2 * 2
        elif kv_cache_compression == 2:
            self.HEAD_SIZE_KEY = head_size + head_size // 4
        else:
            self.HEAD_SIZE_KEY = head_size

        self.WALK_HQ = 2 if num_heads != num_kv_heads else 1

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.xattn_block_size = xattn_block_size
        self.is_causal = is_causal
        self.kv_cache_compression = kv_cache_compression

        INV_S = 1 / torch.sqrt(torch.tensor([head_size], dtype=torch.float32)) / STRIDE
        INV_S = int(INV_S.view(dtype=torch.uint32))

        src = r'''#include "xattn_gemm_qk.hpp"'''
        cwd = os.path.dirname(os.path.realpath(__file__))
        print(f"compiling multi-subseq {cwd} ...")

        jit_option = '-abortonspill -noschedule '
        self.kernels = cl.kernels(src, f'''-cmc -Qxcm_jit_option="{jit_option}"
                            -mCM_printregusage -mdump_asm -g2
                            -Qxcm_register_file_size=256 -I{cwd}
                            -DSTRIDE={STRIDE} -DHQ={num_heads} -DHK={num_kv_heads} -DHEAD_SIZE={head_size} -DSG_M={SG_M} -DSG_N={SG_N} -DBLOCK_SG_N={BLOCK_SG_N} -DBLOCK_SG_M={BLOCK_SG_M}
                            -DBLOCK_SIZE={int(xattn_block_size)} -DINV_S={INV_S} -DKV_BLOCK_SIZE={KV_BLOCK_SIZE} -DBLOCK_SHARE_MAX={BLOCK_WG_N} -DWALK_HQ={self.WALK_HQ}
                            -DIS_CAUSAL={int(is_causal)} -DKV_CACHE_COMPRESSION={self.kv_cache_compression} -DUSE_INT8={int(self.kv_cache_compression != 0)} -DHEAD_SIZE_KEY={self.HEAD_SIZE_KEY} -DSOFTMAX_TYPE={SOFTMAX_TYPE}
                            -DSUB_BLOCK_SIZE={SUB_BLOCK_SIZE}
                            -DBLOCK_WG_K={int(BLOCK_WG_K)}
                            -DMULTI_SUBSEQ=1
                            ''')

    def __call__(self, t_key_cache, t_query, t_block_indices, t_meta, total_wg_count, num_subseqs, K, query_stride, n_repeats=1):
        softmax_type = np.float16 if SOFTMAX_TYPE == 'half' else np.float32
        # Allocate output buffers sized for all subsequences combined
        # The metadata contains per-head byte offsets; total size = last offset + last subseq size
        # For simplicity, we pass pre-allocated buffers from the caller

        GWS = [total_wg_count * SG_N * self.WALK_HQ, SG_M, self.num_heads // self.WALK_HQ]
        LWS = [SG_N, SG_M, 1]
        print(f"calling multi-subseq CM_gemm_qk {GWS=} {LWS=} {num_subseqs=} ...")

        cl.finish()
        for i in range(n_repeats):
            self.kernels.enqueue('gemm_qk', GWS, LWS, t_key_cache, t_query, t_block_indices, t_meta,
                                 *t_key_cache._extra_bufs, K, query_stride, num_subseqs)
        latency = cl.finish()
        return latency

    @staticmethod
    @functools.cache
    def create_instance(num_heads, num_kv_heads, head_size, xattn_block_size, is_causal, kvcache_compressed):
        return xattn_gemmQK_multi(num_heads, num_kv_heads, head_size, xattn_block_size, is_causal, kvcache_compressed)


def test_multi_subseq(xattn_block_size, num_heads=32, num_kv_heads=8, head_size=128, kvcache_compressed=True, is_causal=True):
    """Test gemm_qk with multiple subsequences in a single kernel dispatch.

    Strategy: run multi-subseq kernel, then validate each subsequence's outputs
    independently against the single-subseq reference.
    """
    dim = head_size
    softmax_type = np.float16 if SOFTMAX_TYPE == 'half' else np.float32
    sizeof_softmax = np.dtype(softmax_type).itemsize
    sum_per_token_in_block = xattn_block_size // STRIDE
    k_block_in_group = BLOCK_WG_N // sum_per_token_in_block

    cases = [
        # (q_len_list, description)
        ([4096], "1 subseq baseline"),
        ([2048, 4096], "2 prefill subseqs"),
        ([1024, 2048, 4096], "3 prefill subseqs"),
        ([512 * STRIDE, 256 * STRIDE], "2 subseqs, equal block"),
    ]

    for q_lens, desc in cases:
        print(f'{Colors.BLUE}test multi-subseq gemm("{desc}") xattn_block_size:{xattn_block_size} ...{Colors.END}')

        num_subseqs = len(q_lens)
        # For simplicity, each subseq has kv_len == q_len (no past context, causal start from 0)
        kv_lens = q_lens[:]

        # Build per-subsequence data
        subseq_data = []  # (q, k, q_start_strided, subseq_q_begin)
        total_q_len = 0
        for i, (q_len, kv_len) in enumerate(zip(q_lens, kv_lens)):
            q = torch.randint(-2, 4, size=[1, num_heads, q_len, dim], dtype=torch.int16).to(dtype=torch.float16)
            k = torch.randint(-2, 4, size=[1, num_kv_heads, kv_len, dim], dtype=torch.int16).to(dtype=torch.float16)
            Lk = kv_len // STRIDE * STRIDE
            k = k[:, :, :Lk, :]
            if is_causal:
                q_start_strided = Lk // STRIDE - q_len // STRIDE
            else:
                q_start_strided = 0
            subseq_data.append((q, k, q_start_strided, total_q_len))
            total_q_len += q_len

        # Build key caches with random block allocation per subsequence
        # and concatenated query tensor
        all_key_cache_blocks = []
        all_block_indices = []
        block_indices_begins = [0]
        subsequences_for_meta = []

        xattn_cm_multi = xattn_gemmQK_multi.create_instance(num_heads, num_kv_heads, head_size, xattn_block_size, is_causal, kvcache_compressed)
        xattn_cm_single = xattn_gemmQK.create_instance(num_heads, num_kv_heads, head_size, xattn_block_size, is_causal, kvcache_compressed)
        HEAD_SIZE_KEY = xattn_cm_multi.HEAD_SIZE_KEY

        total_blocks_so_far = 0
        for i, (q, k, q_start_strided, subseq_q_begin) in enumerate(subseq_data):
            _, Hk, Lk, _ = k.shape
            num_blocks = (Lk + KV_BLOCK_SIZE - 1) // KV_BLOCK_SIZE

            if xattn_cm_single.kv_cache_compression:
                k_pad, k_ref = xattn_cm_single.quant_i8(k)
                subseq_data[i] = (q, k_ref, q_start_strided, subseq_q_begin)
                key_cache = k_pad.reshape([1, Hk, -1, KV_BLOCK_SIZE, HEAD_SIZE_KEY]).permute(0, 2, 1, 3, 4)
            else:
                k_pad = torch.zeros([1, Hk, num_blocks * KV_BLOCK_SIZE, head_size], dtype=k.dtype)
                k_pad[:, :, :Lk, :] = k
                key_cache = k_pad.reshape([1, Hk, num_blocks, KV_BLOCK_SIZE, head_size]).permute(0, 2, 1, 3, 4)

            perm_idx = torch.randperm(num_blocks)
            key_cache = key_cache[:, perm_idx, :, :, :].contiguous()
            inv_perm = torch.argsort(perm_idx)

            all_key_cache_blocks.append(key_cache)
            all_block_indices.append(inv_perm + total_blocks_so_far)
            bi_begin = block_indices_begins[-1]
            block_indices_begins.append(bi_begin + num_blocks)

            subsequences_for_meta.append((subseq_q_begin, q_lens[i], kv_lens[i], bi_begin))
            total_blocks_so_far += num_blocks

        all_key_cache = torch.cat(all_key_cache_blocks, dim=1).contiguous()
        t_key_cache = cl.tensor(all_key_cache.detach().numpy())

        # Concatenate block indices
        all_block_indices_cat = torch.cat(all_block_indices).to(torch.int32)
        t_block_indices = cl.tensor(all_block_indices_cat.detach().numpy())

        # Build concatenated query: [1, total_q_len, Hq * S * 2] (with padding for qkv layout)
        Hq = num_heads
        S = head_size
        q_3d_list = []
        for q, _, _, _ in subseq_data:
            q_3d = q.permute(0, 2, 1, 3).reshape([1, q.shape[2], -1])
            q_3d_list.append(q_3d)
        q_3d_cat = torch.cat(q_3d_list, dim=1)
        q_3d_with_padding = torch.zeros([1, total_q_len, Hq * S * 2], dtype=q_3d_cat.dtype)
        q_3d_with_padding[:, :, :Hq * S] = q_3d_cat
        t_query = cl.tensor(q_3d_with_padding.detach().numpy())
        query_stride = S * STRIDE * num_heads * 2
        K_dim = S * STRIDE

        # Build metadata
        merged_q_num = PA_WG_SIZE // xattn_block_size
        meta_np, total_wg_count, buf_sizes = build_xattn_subseq_meta(
            subsequences_for_meta, num_heads, xattn_block_size, merged_q_num)
        t_meta = cl.tensor(meta_np)

        # Allocate combined output buffers (buf_sizes already include all heads)
        total_kq_max_bytes = buf_sizes['kq_max']
        total_exp_sum_bytes = buf_sizes['exp_sum']
        t_kq_max_wg = cl.tensor(np.ones(total_kq_max_bytes // sizeof_softmax, softmax_type))
        t_kq_exp_partial_sum = cl.tensor(np.ones(total_exp_sum_bytes // sizeof_softmax, softmax_type))

        # Monkey-patch extra buffers onto t_key_cache for dispatch
        # (The multi-subseq __call__ needs kq_max_wg and kq_exp_partial_sum as extra args)
        # Actually, let's just call enqueue directly:
        xattn_cm_multi = xattn_gemmQK_multi.create_instance(num_heads, num_kv_heads, head_size, xattn_block_size, is_causal, kvcache_compressed)

        GWS = [total_wg_count * SG_N * xattn_cm_multi.WALK_HQ, SG_M, num_heads // xattn_cm_multi.WALK_HQ]
        LWS = [SG_N, SG_M, 1]
        print(f"calling multi-subseq CM_gemm_qk {GWS=} {LWS=} {num_subseqs=} ...")

        cl.finish()
        xattn_cm_multi.kernels.enqueue('gemm_qk', GWS, LWS,
                                        t_key_cache, t_query, t_block_indices, t_meta,
                                        t_kq_max_wg, t_kq_exp_partial_sum,
                                        K_dim, query_stride, num_subseqs)
        cl.finish()

        # Validate each subsequence independently against single-subseq reference
        for i, (q, k, q_start_strided, subseq_q_begin) in enumerate(subseq_data):
            q_len = q_lens[i]
            kv_len = kv_lens[i]
            Lk = kv_len // STRIDE * STRIDE
            M = q_len // STRIDE
            N = Lk // STRIDE
            q_stride_pad = rnd_up(M, BLOCK_WG_M)
            N_kq_groups = div_up(N, BLOCK_WG_N)
            k_block_pad = k_block_in_group * N_kq_groups

            # Reference
            kq_max_ref, kq_5d_max_ret_ref, kq_exp_partial_sum_ret_ref, _ = get_gemm_ref(
                q, k, q_start_strided=q_start_strided, block_size=xattn_block_size,
                S=STRIDE, threshold=THRESH, causal=is_causal, wg_k=BLOCK_WG_N, wg_q=BLOCK_WG_M)

            # Extract this subsequence's results from combined buffer
            meta_row = meta_np[i]
            off_kq_max = meta_row[10]  # BUF_OFF_KQ_MAX
            off_exp_sum = meta_row[11]  # BUF_OFF_EXP_SUM

            kq_max_wg_all = t_kq_max_wg.numpy()
            kq_exp_all = t_kq_exp_partial_sum.numpy()

            per_head_max_size = N_kq_groups * q_stride_pad
            per_head_exp_size = q_stride_pad * k_block_pad

            kq_max_subseq = np.zeros([1, num_heads, N_kq_groups, q_stride_pad], dtype=softmax_type)
            kq_exp_subseq = np.zeros([1, num_heads, q_stride_pad, k_block_pad], dtype=softmax_type)

            for h in range(num_heads):
                start_max = (off_kq_max // sizeof_softmax) + h * per_head_max_size
                kq_max_subseq[0, h] = kq_max_wg_all[start_max:start_max + per_head_max_size].reshape(N_kq_groups, q_stride_pad)

                start_exp = (off_exp_sum // sizeof_softmax) + h * per_head_exp_size
                kq_exp_subseq[0, h] = kq_exp_all[start_exp:start_exp + per_head_exp_size].reshape(q_stride_pad, k_block_pad)

            # Compare
            kq_5d_max_ret_ref_np = kq_5d_max_ret_ref.detach().numpy()[..., :M]
            compare(kq_5d_max_ret_ref_np, kq_max_subseq[..., :M])
            print(f'{Colors.GREEN}  subseq[{i}] gemm:max_wg passed{Colors.END}')

            kq_exp_partial_sum_ret_ref_np = kq_exp_partial_sum_ret_ref.detach().numpy()[:, :, :M, :]
            compare(kq_exp_partial_sum_ret_ref_np, kq_exp_subseq[:, :, :M, :])
            print(f'{Colors.GREEN}  subseq[{i}] gemm:exp_partial passed{Colors.END}')

        print(f'{Colors.GREEN}multi-subseq test "{desc}" passed{Colors.END}')


@pytest.mark.parametrize("xattn_block_size", [128, 256])
@pytest.mark.parametrize("head_size", [64, 128])
@pytest.mark.parametrize("kvcache_compressed", [0, 1, 2])
@pytest.mark.parametrize(
    "num_heads,num_kv_heads",
    [
        (1, 1),
        (2, 1),
        (4, 2),
        (32, 8),
        (16, 16),
        (28, 4),
    ],
)
def test_func_parametrized(xattn_block_size, head_size, kvcache_compressed, num_heads, num_kv_heads):
    run_func(
        xattn_block_size=xattn_block_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        kvcache_compressed=kvcache_compressed,
        is_causal=True,
    )

def main():
    for xattn_block_size in [128, 256]:
        run_func(xattn_block_size)

def main_multi():
    for xattn_block_size in [128, 256]:
        test_multi_subseq(xattn_block_size)

if __name__ == "__main__":
    torch.manual_seed(3)
    torch.set_printoptions(linewidth=1024)

    cl.profiling(True)

    import sys
    if '--multi-subseq' in sys.argv:
        main_multi()
    elif '--perf' in sys.argv:
        for xattn_block_size in [128, 256]:
            run_perf(xattn_block_size)
    else:
        main()

# Usage:
# - python -m pytest opencl/tests/x_attn/test_gemm_qk.py -s -vv
