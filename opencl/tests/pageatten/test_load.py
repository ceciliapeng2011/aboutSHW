import argparse
import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

torch.manual_seed(0)
torch.set_printoptions(linewidth=1024)

load_dict = {1: "cm_svm_block_read",
             2: "cm_ptr_load",
             3: "cm_load",
             4: "load_ref"}

parser = argparse.ArgumentParser('')
parser.add_argument('-i', "--impl", type=int, default=1)
parser.add_argument('-b', "--batch", type=int, default=1)
parser.add_argument('-nh', "--num-heads", type=int, default=32)
parser.add_argument('-nkvh', "--num-kv-heads", type=int, default=8)
parser.add_argument('-ql', "--q-len", type=int, default=1)
parser.add_argument('-kvl', "--kv-len", type=int, default=32769)
parser.add_argument('-hs', "--head-size", type=int, default=128)
parser.add_argument('-rkv', "--reset_kv_cache", type=int, default=1)
parser.add_argument('-v', "--verbose", type=int, default=-1)
parser.add_argument('-lm', "--load-mode", type=int, default=1)
args = parser.parse_args()
print(args)

enable_vprint = False

def vprint(*all_args):
    global enable_vprint
    if enable_vprint:
        print(*all_args)

batch = args.batch
q_len, q_step = args.q_len, 32
kv_len, kv_step = args.kv_len, 8
num_heads = args.num_heads
num_kv_heads = args.num_kv_heads
head_size = args.head_size
enable_gqa = num_heads > num_kv_heads

load_mode = args.load_mode
load_mode_string = load_dict.get(load_mode, None)
if load_mode_string == None:
    print("This load mode is not supproted.")
    exit(0)
print(f"Load mode is {load_mode}: {load_mode_string}")

# define KV_BLOCK_SIZE = 32,64,128,256
kv_block_size = 256

enable_kvcache_compression = 0
kv_cache_quantization_mode = os.environ.get("KV_CACHE_QUANT_MODE", "by_token")

def _validate_quant_mode(mode: str) -> str:
    mode = mode.strip().lower()
    if mode not in {"by_token", "by_channel"}:
        raise ValueError(f"Unsupported kv-cache quantization mode: {mode}")
    return mode

kv_cache_quantization_mode = _validate_quant_mode(kv_cache_quantization_mode)
kvcache_quantization_by_token = int(kv_cache_quantization_mode == "by_token")
print(f"{kv_cache_quantization_mode=}, {kvcache_quantization_by_token=}")

enable_clean_unused_kvcache = args.reset_kv_cache

def get_tensor(name, dtype=np.float16):
    with open(name, 'rb') as f:
        data = f.read()
        np_data = np.frombuffer(data, dtype=dtype).copy()
        return torch.from_numpy(np_data)

#xe_arch: 1: xe, 2: xe2
xe_arch=1
assert (xe_arch==1, "This test only for xe1.")

if xe_arch == 1:
    kv_step = 8
else:
    kv_step = 16

low = -127
high = 128
act_dtype = torch.float16
new_kv_len = (kv_len + kv_block_size - 1) // kv_block_size * kv_block_size
q = torch.randint(low, high, [batch, q_len, num_heads, head_size]).to(dtype=act_dtype)/high
k = torch.randint(low, high, [batch, new_kv_len, num_kv_heads, head_size]).to(dtype=act_dtype)/high
v = torch.randint(low, high, [batch, new_kv_len, num_kv_heads, head_size]).to(dtype=act_dtype)/high

run_real_data_test=False

# sdpa split size, must be multiple of kv_step and kv_len should be multiple of kv_partition_size
k_partition_block_num = kv_len//8192
if k_partition_block_num < 1:
    k_partition_block_num = 1
k_partition_block_num = 1  # test cm_sdpa_2nd
# k_partition_block_num = 0.5  # test cm_sdpa_2nd_half_block, now output is wrong for enable_kvcache_compression
kv_partition_size = int(kv_block_size * k_partition_block_num)

print("kv_step:", kv_step)
print("k_partition_block_num:", k_partition_block_num)
print("kv_partition_size:", kv_partition_size)

new_kv_len = (kv_len + kv_block_size - 1) // kv_block_size * kv_block_size
kv_partition_num = (new_kv_len + kv_partition_size - 1) // kv_partition_size
total_partition_num = kv_partition_num * num_heads

assert(kv_partition_size % kv_step == 0)

# support single sequence
seq_num = 1
total_blk_num = new_kv_len//kv_block_size

past_lens=torch.tensor([kv_len-1]).to(torch.int32)

# the first block of each sequence
block_indices_begins=torch.tensor([0, total_blk_num]).to(torch.int32)

# block physical indices from logic index 
block_indices =  torch.randperm(total_blk_num).to(torch.int32)
# print("block_indices:", block_indices)

subsequence_begins=torch.tensor([0, seq_num]).to(torch.int32)
gws_subseq_mapping=torch.tensor([0]).to(torch.int32)

# BLHS=>BHLS
k = k.transpose(1,2)

print("k:", k.shape, k.dtype)

def check_close(input, other, atol=1e-3, rtol=1e-3):
    print(f"[check_close] {input.shape}{input.dtype} vs {other.shape}{other.dtype}")
    #print("ref = ", input)
    #print("res = ", other)
    rtol_max = (((input - other).abs() - 1e-5)/other.abs())[other != 0].max()
    atol_max = (((input - other).abs()) - 1e-5*other.abs()).max()
    # print(f"[check_close] rtol_max: {rtol_max}")
    # print(f"[check_close] atol_max: {atol_max}")
    if not torch.allclose(input, other, atol=atol, rtol=rtol):
        close_check = torch.isclose(input, other, atol=atol, rtol=rtol)
        not_close_indices = torch.where(~close_check) # Invert the close check to find failures
        print(f"Not close indices: {not_close_indices}")
        print(f"    ref_tensor: {input[not_close_indices]}")
        print(f"    res_tensor: {other[not_close_indices]}")
        assert 0

# transpose back to orginal shape: [batch, q_len, num_heads, head_size] for padding
k = k.transpose(1,2)

if kv_len % kv_block_size != 0:
    # pad k,v to multiple of kv_block_size
    pad_len = ((kv_len + kv_block_size - 1)//kv_block_size)*kv_block_size - kv_len
    print(f"pad k,v from {kv_len} to {k.shape[1]}")
    new_kv_len = k.shape[1]
    total_blk_num = new_kv_len//kv_block_size
    assert(new_kv_len % kv_block_size == 0)

print("k.shape:", k.shape, k.dtype)
print("new_kv_len = ", new_kv_len)

# transpose shape: [batch, num_heads, q_len, head_size] for get_flash3
k = k.transpose(1,2)

print()
print("GPU cm kernels for testing loading logic:")

#====================================================================================================
# using the same parameter & inputs, develop cm kernels which produces the same output
# prototyping CM kernels
from clops import cl
import numpy as np
import time

# transpose back to orginal shape: [batch, q_len, num_heads, head_size]
k = k.transpose(1,2)
print("k:", k.shape, k.dtype)

if enable_clean_unused_kvcache:
    print("before blocking k:", k.shape, k.dtype)
    print("before blocking v:", v.shape, v.dtype)
    if kvcache_quantization_by_token:
        k.view(torch.uint16)[0,kv_len:,:,:] = 0xFE00
    else:
        k.view(torch.uint16)[0,kv_len:,:,:] = 0
    v.view(torch.uint16)[0,kv_len:,:,:] = 0xFE00

# change from [batch, kv_len, num_kv_heads, head_size] to [total_blk_num, num_kv_heads, kv_block_size, head_size]
k = k.reshape(total_blk_num, kv_block_size, num_kv_heads, head_size).transpose(1,2).contiguous()
print("after blocking k:", k.shape, k.dtype)

vprint("k[0,0,0,:] = ", k[0,0,0,:])
print()

blocked_k = k.clone()

for i in range(len(block_indices)):
    k[block_indices[i],:] =  blocked_k[i,:]

k = k.contiguous()

print(f"final k shape: {k.shape}")

def pyeval(src):
    result_src = ""
    for line in src.splitlines():
        if line.startswith("#pyeval"):
            new_line = eval(line[8:])
            result_src += new_line + "\n"
            # print(f"[pyeval] {new_line}")
        else:
            result_src += line + "\n"
    return result_src

scale_factor = 1.0/(head_size**0.5)

# each WG processes a partition
GWS = [seq_num, num_kv_heads, (new_kv_len + kv_partition_size - 1) // kv_partition_size]
WG_SIZE = 1
LWS = [1, 1, WG_SIZE]

print("GWS=", GWS)
print("LWS=", LWS)

#========================================================================
# Optimization Log
r'''
'''
#========================================================================

src1 = r'''
//# CM kernel for test loading, reference

#pyeval f"#define HEADS_NUM {num_heads}"
#pyeval f"#define KV_HEADS_NUM {num_kv_heads}"
#pyeval f"#define HEAD_SIZE {head_size}"
#pyeval f"#define Q_STEP {q_step}"
#pyeval f"#define KV_STEP {kv_step}"
#pyeval f"#define SCALE_FACTOR {scale_factor}"
#pyeval f"#define args_verbose {args.verbose}"
#pyeval f"#define WG_SIZE {WG_SIZE}"

#pyeval f"#define KV_BLOCK_SIZE {kv_block_size}"
#pyeval f"#define KV_PARTITION_SIZE {kv_partition_size}"

#pyeval f"#define CLEAN_UNUSED_KVCACHE {enable_clean_unused_kvcache}"

#pyeval f"#define KV_CACHE_COMPRESSION {enable_kvcache_compression}"
#pyeval f"#define KV_CACHE_COMPRESSION_BY_TOKEN {kvcache_quantization_by_token}"

// xe-1:8, xe-2:16
#pyeval f"#define XE_ARCH {xe_arch}"
#pyeval f"#define load_mode {load_mode}"

#if XE_ARCH==1
#define REG_N 8
#define USE_LSC_BLOCK_2D_DESC 0
#else
#define REG_N 16
#define USE_LSC_BLOCK_2D_DESC 1
#endif

#define SystolicDepth 8
#define RepeatCount 1
#define VNNI_WIDTH 2
#define REG_K (SystolicDepth * VNNI_WIDTH)
#define REG_M RepeatCount

#define PRINT_THR_ID 1000
#define PRINT_HEAD_ID 1000

#define LOAD_BY_SVM_BLOCK_READ 1
#define LOAD_BY_CM_PTR_LOAD 2
#define LOAD_BY_CM_READ 3
#define LOAD_BY_REF 4

#define KV_PARTITION_STEP_NUM  (KV_PARTITION_SIZE / KV_STEP)

// static_assert(Q_STEP == 16);
static_assert(KV_STEP == 8 || KV_STEP == 16);

template<typename T, int M, int N>
void show(matrix<T, M, N> mat) {
    for(int m = 0; m < M; m ++) {
        printf("\t[");
        for(int n = 0; n < N; n ++) {
            printf("%8.4f,", mat[m][n]);
        }
        printf("],\n");
    }
    printf("]\n");
}


template<typename T, int M, int N>
void show_u8(matrix<T, M, N> mat) {
    for(int m = 0; m < M; m ++) {
        printf("\t[");
        for(int n = 0; n < N; n ++) {
            printf("%4d", mat[m][n]);
        }
        printf("],\n");
    }
    printf("]\n");
}

template<typename T, int N>
void show(vector<T, N> vec) {
    printf("\t[");
    for(int n = 0; n < N; n ++) {
        printf("%8.4f,", vec[n]);
    }
    printf("]\n");
}

CM_INLINE uint64_t get_clock() {
    auto clk = cm_clock();
    return ((uint64_t)clk[1]) << 32 | clk[0];
}

template <typename T1, typename T2>
CM_INLINE void Transpose_16x16(matrix_ref<T1, 16, 16> in,
                               matrix_ref<T2, 16, 16> out) {
  matrix<T2, 16, 16> bBuf;
  bBuf.row(0) = in.template select<4, 1, 4, 4>(0, 0);   // 0,4,8,c
  bBuf.row(1) = in.template select<4, 1, 4, 4>(4, 0);   // 0,4,8,c
  bBuf.row(2) = in.template select<4, 1, 4, 4>(8, 0);   // 0,4,8,c
  bBuf.row(3) = in.template select<4, 1, 4, 4>(12, 0);  // 0,4,8,c
  bBuf.row(4) = in.template select<4, 1, 4, 4>(0, 1);   // 1,5,9,d
  bBuf.row(5) = in.template select<4, 1, 4, 4>(4, 1);   // 1,5,9,d
  bBuf.row(6) = in.template select<4, 1, 4, 4>(8, 1);   // 1,5,9,d
  bBuf.row(7) = in.template select<4, 1, 4, 4>(12, 1);  // 1,5,9,d
  bBuf.row(8) = in.template select<4, 1, 4, 4>(0, 2);   // 2,6,a,e
  bBuf.row(9) = in.template select<4, 1, 4, 4>(4, 2);   // 2,6,a,e
  bBuf.row(10) = in.template select<4, 1, 4, 4>(8, 2);  // 2,6,a,e
  bBuf.row(11) = in.template select<4, 1, 4, 4>(12, 2); // 2,6,a,e
  bBuf.row(12) = in.template select<4, 1, 4, 4>(0, 3);  // 3,7,b,f
  bBuf.row(13) = in.template select<4, 1, 4, 4>(4, 3);  // 3,7,b,f
  bBuf.row(14) = in.template select<4, 1, 4, 4>(8, 3);  // 3,7,b,f
  bBuf.row(15) = in.template select<4, 1, 4, 4>(12, 3); // 3,7,b,f

  out.row(0) = bBuf.template select<4, 1, 4, 4>(0, 0);   // 0
  out.row(1) = bBuf.template select<4, 1, 4, 4>(4, 0);   // 1
  out.row(2) = bBuf.template select<4, 1, 4, 4>(8, 0);   // 2
  out.row(3) = bBuf.template select<4, 1, 4, 4>(12, 0);  // 3
  out.row(4) = bBuf.template select<4, 1, 4, 4>(0, 1);   // 4
  out.row(5) = bBuf.template select<4, 1, 4, 4>(4, 1);   // 5
  out.row(6) = bBuf.template select<4, 1, 4, 4>(8, 1);   // 6
  out.row(7) = bBuf.template select<4, 1, 4, 4>(12, 1);  // 7
  out.row(8) = bBuf.template select<4, 1, 4, 4>(0, 2);   // 8
  out.row(9) = bBuf.template select<4, 1, 4, 4>(4, 2);   // 9
  out.row(10) = bBuf.template select<4, 1, 4, 4>(8, 2);  // a
  out.row(11) = bBuf.template select<4, 1, 4, 4>(12, 2); // b
  out.row(12) = bBuf.template select<4, 1, 4, 4>(0, 3);  // c
  out.row(13) = bBuf.template select<4, 1, 4, 4>(4, 3);  // d
  out.row(14) = bBuf.template select<4, 1, 4, 4>(8, 3);  // e
  out.row(15) = bBuf.template select<4, 1, 4, 4>(12, 3); // f
}

template <typename T1, typename T2>
CM_INLINE void Transpose_8x8(matrix_ref<T1, 8, 8> in, matrix_ref<T2, 8, 8> out) {
  matrix<T2, 8, 8> temp;
  temp.row(0) = in.template select<2, 1, 4, 2>(0, 0);
  temp.row(1) = in.template select<2, 1, 4, 2>(2, 0);
  temp.row(2) = in.template select<2, 1, 4, 2>(4, 0);
  temp.row(3) = in.template select<2, 1, 4, 2>(6, 0);
  temp.row(4) = in.template select<2, 1, 4, 2>(0, 1);
  temp.row(5) = in.template select<2, 1, 4, 2>(2, 1);
  temp.row(6) = in.template select<2, 1, 4, 2>(4, 1);
  temp.row(7) = in.template select<2, 1, 4, 2>(6, 1);

  out.row(0) = temp.template select<4, 1, 2, 4>(0, 0);
  out.row(2) = temp.template select<4, 1, 2, 4>(0, 1);
  out.row(4) = temp.template select<4, 1, 2, 4>(0, 2);
  out.row(6) = temp.template select<4, 1, 2, 4>(0, 3);
  out.row(1) = temp.template select<4, 1, 2, 4>(4, 0);
  out.row(3) = temp.template select<4, 1, 2, 4>(4, 1);
  out.row(5) = temp.template select<4, 1, 2, 4>(4, 2);
  out.row(7) = temp.template select<4, 1, 2, 4>(4, 3);
}

//prepack [K, N] to [K/2, N, 2] layout.
template <typename T1, typename T2, int K, int N>
inline void prepackAsVNNIWidth2(matrix_ref<T1, K, N> input, matrix_ref<T2, K/2, N*2> out) {
    #pragma unroll
    for (int r = 0; r < K/2; r++) {
        out.row(r).select<N, 2>(0) = input.row(r*2);
        out.row(r).select<N, 2>(1) = input.row(r*2+1);
    }
}

#if KV_CACHE_COMPRESSION
    // scale/zp is half-precision, so size = 2 * 2 = 4 bytes
    #define KV_SCALE_ZP_SIZE 4 // scale/zp bytes
    #define KV_ELEMENT_TYPE uint8_t
#else
    #define KV_SCALE_ZP_SIZE 0 // no scale/zp
    #define KV_ELEMENT_TYPE half
#endif

extern "C" _GENX_MAIN_ void cm_sdpa_2nd_loading(
    KV_ELEMENT_TYPE* key [[type("svmptr_t")]],
    int* past_lens [[type("svmptr_t")]],
    int* block_indices [[type("svmptr_t")]],
    int* block_indices_begins [[type("svmptr_t")]],
    int* subsequence_begins [[type("svmptr_t")]]
    ) {
    //# batch=1, seq_num=1 or >1
    //#   key [block_num, kv_head_num, block_size, head_size] + [block_num, kv_head_num, block_size, 4] (scale/zp)

    //# KV_PARTITION_SIZE should be multiple of kv_block_size(KV_BLOCK_SIZE)
    //# kv_len dimision will be split into multiple partitions, each WG process a partition
    //# total_partitions_num = kv_len // KV_PARTITION_SIZE
    //# GWS=[seq_num, num_kv_heads, total_partitions_num]
    //# LWS=[1, 1, 1]

    //# Each WG processes a partition, which is KV_PARTITION_SIZE long and multiple of KV_BLOCK_SIZE.
    //# KV_BLOCK_SIZE can be 32/64/128/256, etc.
    const auto seq_idx = cm_global_id(0);
    const auto kv_head_num_idx = cm_global_id(1);
    const auto head_num_idx = kv_head_num_idx * (HEADS_NUM/KV_HEADS_NUM);
    //# KV_PARTITION_SIZE --> EU thread
    const auto wg_thread_id = cm_global_id(2);
    const uint kv_partition_num = cm_group_count(2);
    const uint kv_partition_idx = cm_group_id(2);

    const uint kv_len = past_lens[seq_idx] + 1;
    // The code here requires KV_PARTITION_SIZE to be an integer multiple of KV_BLOCK_SIZE.
    const uint start_block_idx = block_indices_begins[seq_idx] + kv_partition_idx * (KV_PARTITION_SIZE / KV_BLOCK_SIZE);

    if(kv_partition_idx * KV_PARTITION_SIZE > kv_len) {
        // printf("WG exit: kv_partition_idx=%d, KV_PARTITION_SIZE=%d, kv_len=%d\n", kv_partition_idx, KV_PARTITION_SIZE, kv_len);
        return;
    }
    const uint total_blocks_num = (kv_len + KV_BLOCK_SIZE - 1) / KV_BLOCK_SIZE;
    constexpr uint kv_pitch = HEAD_SIZE * sizeof(KV_ELEMENT_TYPE);

    constexpr uint per_v_block_element_num = KV_BLOCK_SIZE * KV_HEADS_NUM * (HEAD_SIZE + KV_SCALE_ZP_SIZE); // 4 bytes: scale/zp
    #if KV_CACHE_COMPRESSION_BY_TOKEN
        constexpr uint per_k_block_element_num = KV_BLOCK_SIZE * KV_HEADS_NUM * (HEAD_SIZE + KV_SCALE_ZP_SIZE); // 4 bytes: scale/zp
    #else
        constexpr uint per_k_block_element_num = KV_HEADS_NUM * HEAD_SIZE * (KV_BLOCK_SIZE + KV_SCALE_ZP_SIZE); // 4 bytes: scale/zp
    #endif
    uint block_num = KV_PARTITION_SIZE / KV_BLOCK_SIZE;

    uint leftover_size = 0;
    if(kv_partition_idx == kv_partition_num - 1) {
        // last partition
        leftover_size = (kv_len - KV_PARTITION_SIZE * kv_partition_idx) % KV_PARTITION_SIZE;
    }
    if(block_num > total_blocks_num - start_block_idx) {
        block_num = total_blocks_num - start_block_idx;
    }

    // # Each SG can process multiple blocks
    #pragma unroll
    for(uint block_idx = 0, ki = 0; block_idx < block_num; block_idx++) {
        // split kv_partition into multi kv_block
        uint blk_indices = block_indices[start_block_idx + block_idx];
        uint k_base_offset = blk_indices * per_k_block_element_num + kv_head_num_idx * (per_k_block_element_num / KV_HEADS_NUM);
        uint k_scale_zp_offset = k_base_offset + KV_BLOCK_SIZE * HEAD_SIZE; // scale/zp offset

    #if USE_LSC_BLOCK_2D_DESC
        #if KV_CACHE_COMPRESSION
            // Transpose only support dword and qwork
            lsc::block_2d_desc<uint, 1, REG_N, REG_K/4> b2dK(reinterpret_cast<uint*>(key + k_base_offset),  KV_BLOCK_SIZE - 1, HEAD_SIZE*sizeof(uint8_t) - 1, kv_pitch - 1, 0, 0);
        #else
            lsc::block_2d_desc<uint, 1, REG_N, REG_K/2> b2dK(reinterpret_cast<uint*>(key + k_base_offset),  KV_BLOCK_SIZE - 1, HEAD_SIZE*sizeof(half) - 1, kv_pitch - 1, 0, 0);
        #endif
    #else
        uint kv_offset = k_base_offset;
        uint kv_stride = HEAD_SIZE;
        uint kv_x0 = 0, kv_y0 = 0;
        uint kv_x1 = HEAD_SIZE*sizeof(KV_ELEMENT_TYPE);
        uint kv_y1 = KV_BLOCK_SIZE;
    #endif

        uint kv_pos_end = KV_BLOCK_SIZE;
        if(block_idx == block_num - 1 && leftover_size > 0) {
            kv_pos_end = leftover_size % KV_BLOCK_SIZE;
            if(kv_pos_end == 0) kv_pos_end = KV_BLOCK_SIZE;
        }

        #if KV_CACHE_COMPRESSION
            #if KV_CACHE_COMPRESSION_BY_TOKEN
            // load scale/zp
            vector<half, KV_BLOCK_SIZE> scale_vec;
            vector<half, KV_BLOCK_SIZE> zp_vec;
            cm_svm_block_read(reinterpret_cast<svmptr_t>(key + k_scale_zp_offset), scale_vec);
            cm_svm_block_read(reinterpret_cast<svmptr_t>(key + k_scale_zp_offset + KV_BLOCK_SIZE * sizeof(half)), zp_vec);
            if(kv_pos_end < KV_BLOCK_SIZE) {
                // fill leftover with last valid scale/zp
                #pragma unroll
                for(int i = kv_pos_end; i < KV_BLOCK_SIZE; i++) {
                    scale_vec[i] = 0.0;
                    zp_vec[i] = 0.0;
                }
            }
            #else
            // load scale/zp
            vector<half, HEAD_SIZE> scale_vec;
            vector<half, HEAD_SIZE> zp_vec;
            cm_svm_block_read(reinterpret_cast<svmptr_t>(key + k_scale_zp_offset), scale_vec);
            cm_svm_block_read(reinterpret_cast<svmptr_t>(key + k_scale_zp_offset + HEAD_SIZE * sizeof(half)), zp_vec);
            #endif
        #endif

        vector<half, REG_N> sum = 0;

        for(int kv_pos = 0; kv_pos < kv_pos_end; kv_pos += KV_STEP, ki++) {

            #if KV_CACHE_COMPRESSION && KV_CACHE_COMPRESSION_BY_TOKEN
                vector<half, REG_N * 2> temp_scale, temp_zp;
                temp_scale.select<REG_N,2>(0) = scale_vec.select<REG_N,1>(kv_pos);
                temp_scale.select<REG_N,2>(1) = scale_vec.select<REG_N,1>(kv_pos);
                temp_zp.select<REG_N,2>(0) = zp_vec.select<REG_N,1>(kv_pos);
                temp_zp.select<REG_N,2>(1) = zp_vec.select<REG_N,1>(kv_pos);
            #endif

            #pragma unroll
            #if KV_CACHE_COMPRESSION
            for(int k = 0, ri = 0; k < HEAD_SIZE/4; k += REG_K/4, ri ++ ) {
            #else
            for(int k = 0, ri = 0; k < HEAD_SIZE/2; k += REG_K/2, ri ++ ) {
            #endif
                matrix<half, REG_K, REG_N> Kt = 0;
            #if KV_CACHE_COMPRESSION
                matrix<uint8_t, REG_K, REG_N> Kt_quant_temp, Kt_quant;
            #endif
            #if USE_LSC_BLOCK_2D_DESC
                //# Load Kt into register & pack as VNNI(as dpas-B tile)
                //# DWORD transposed load == (transposed + VNNI) load
                b2dK.set_block_x(k);
                #if KV_CACHE_COMPRESSION
                    cm_load<lsc::Transpose>(Kt_quant_temp.format<uint>(), b2dK.set_block_y(kv_pos));
                    auto quant_src = Kt_quant_temp.format<ushort, REG_K/2, REG_N>();
                    auto quant_dst = Kt_quant.format<ushort, REG_K/2, REG_N>();

                    #pragma unroll
                    for(int r = 0; r < REG_K / 2; r += 2) {
                        quant_dst.row(r  ) = quant_src.select<2,1,8,2>(r,0);
                        quant_dst.row(r+1) = quant_src.select<2,1,8,2>(r,1);
                    }
                #else
                    cm_load<lsc::Transpose>(Kt.format<uint>(), b2dK.set_block_y(kv_pos));
                #endif
            #else
                #if KV_CACHE_COMPRESSION
                    matrix<uint16_t, REG_N, REG_K/2> temp; // 8 x 8
                    uint cur_kv_offset = kv_offset + kv_pos * kv_stride + k * 4;
                    #pragma unroll
                    for(int kk = 0; kk < REG_N; kk++) {
                        cm_svm_block_read<uint16_t, REG_K/2>((svmptr_t)(key + cur_kv_offset + kk * kv_stride), temp[kk].format<uint16_t>());
                    }
                    #if XE_ARCH==1
                    // Transpose_8x8(temp, Kt_quant.format<uint16_t, REG_K/2, REG_N>());
                    Kt = temp.format<half, REG_K, REG_N>();
                    #else
                    Transpose_8x8(temp.select<8,1,8,1>(0,0), Kt_quant_temp.format<uint16_t, REG_K/2, REG_N/2>().select<8,1,8,1>(0,0));
                    Transpose_8x8(temp.select<8,1,8,1>(8,0), Kt_quant_temp.format<uint16_t, REG_K/2, REG_N/2>().select<8,1,8,1>(0,8));
                    #endif
                #else
                    matrix<uint, REG_N, REG_K/2> temp;
                    unsigned cur_kv_offset = kv_offset + kv_pos * kv_stride + k * 2;
                    half* key_base = key + cur_kv_offset;
                    #pragma unroll
                    for(int kk = 0; kk < REG_N; kk++) {
                        #if load_mode == LOAD_BY_SVM_BLOCK_READ
                        cm_svm_block_read<uint, REG_K/2>((svmptr_t)(key_base + kk * kv_stride), temp[kk].format<uint>());
                        #elif load_mode == LOAD_BY_CM_PTR_LOAD
                        temp[kk] = cm_ptr_load<uint, REG_K/2>((const unsigned int *const)key_base, (kk * kv_stride) * sizeof(half));
                        #elif load_mode == LOAD_BY_REF
                        vector<uint, REG_K/2> *v_ptr = (vector<uint, REG_K/2> *)(key_base + kk * kv_stride);
                        temp[kk] = *v_ptr;
                        #endif
                    }
                    #if XE_ARCH==1
                    Transpose_8x8(temp.select<8,1,8,1>(0,0), Kt.format<uint, REG_K/2, REG_N>().select<8,1,8,1>(0,0));
                    #else
                    Transpose_8x8(temp.select<8,1,8,1>(0,0), Kt.format<uint, REG_K/2, REG_N>().select<8,1,8,1>(0,0));
                    Transpose_8x8(temp.select<8,1,8,1>(8,0), Kt.format<uint, REG_K/2, REG_N>().select<8,1,8,1>(0,8));
                    #endif
                #endif
            #endif

            #if KV_CACHE_COMPRESSION
                #if KV_CACHE_COMPRESSION_BY_TOKEN
                #pragma unroll
                for(int r = 0; r < REG_K; r++) {
                    Kt[r] = Kt_quant[r] - temp_zp.format<half, 2, REG_N>()[r%2]; //vector - vector
                    Kt[r] = cm_mul<half>(Kt[r], temp_scale.format<half, 2, REG_N>()[r%2]);    // vector * vector
                }
                #else
                vector<half, REG_K> temp_scale, temp_zp;
                temp_scale.select<REG_K, 1>(0) = scale_vec.select<REG_K, 1>(k * 4);
                temp_zp.select<REG_K, 1>(0) = zp_vec.select<REG_K, 1>(k * 4);

                auto Kt_dequant_out = Kt.format<half, REG_K/2, 2*REG_N>();
                auto Kt_dequant_tmp = Kt_quant.format<uint8_t, REG_K/2, 2*REG_N>();
                #pragma unroll
                for(int r = 0; r < REG_K/2; r++) {
                    Kt_dequant_out[r].select<REG_N, 2>(0) = Kt_dequant_tmp[r].select<REG_N, 2>(0) - temp_zp[r*2];
                    Kt_dequant_out[r].select<REG_N, 2>(0) = cm_mul<half>(Kt_dequant_out[r].select<REG_N, 2>(0), temp_scale[r*2]);
                    Kt_dequant_out[r].select<REG_N, 2>(1) = Kt_dequant_tmp[r].select<REG_N, 2>(1) - temp_zp[r*2+1];
                    Kt_dequant_out[r].select<REG_N, 2>(1) = cm_mul<half>(Kt_dequant_out[r].select<REG_N, 2>(1), temp_scale[r*2+1]);
                }
                #endif
            #endif
            
            sum.select<REG_N, 1>(0) += Kt.row(0);
            sum.select<REG_N, 1>(0) += Kt.row(11);
            }
        }

        cm_svm_block_write<half, REG_N>((svmptr_t)(key + k_base_offset), sum.format<half>());
    }
}

'''

cl.profiling(True)

t_k = cl.tensor(k.contiguous().detach().numpy())
t_past_lens = cl.tensor(past_lens.detach().numpy())
t_block_indices = cl.tensor(block_indices.detach().numpy())
t_block_indices_begins = cl.tensor(block_indices_begins.detach().numpy())
t_subsequence_begins = cl.tensor(subsequence_begins.detach().numpy())

cwd = os.path.dirname(os.path.realpath(__file__))
print("compiling ...")
# cm_kernels = cl.kernels(pyeval(src1), f"-cmc -Qxcm_register_file_size=256 -mCM_printregusage -I{cwd}")
cm_kernels = cl.kernels(pyeval(src1), f"-cmc -Qxcm_register_file_size=256 -mCM_printregusage -Qxcm_jit_option='-abortonspill' -mdump_asm -g2 -I{cwd}")
print("first call ...")

if kv_partition_size * 2 == kv_block_size:
    print("Not support for now.")
else:
    cm_kernels.enqueue("cm_sdpa_2nd_loading", GWS, LWS, t_k,
                       t_past_lens, t_block_indices, t_block_indices_begins,
                       t_subsequence_begins)

loop_cnt = 100
all_layers = []
mem_size = 0
print(k.shape)
while len(all_layers) < loop_cnt and mem_size < 8e9:
    t_k = cl.tensor(k.detach().numpy())
    all_layers.append([
        t_k,
    ])
    mem_size += k.numel() * k.element_size()

for i in range(loop_cnt):
    j  = i % len(all_layers)
    cm_kernels.enqueue("cm_sdpa_2nd_loading", GWS, LWS,
                        all_layers[j][0],
                        t_past_lens, t_block_indices, t_block_indices_begins, t_subsequence_begins)

latency = cl.finish()
# only load k for loading speed test
if enable_kvcache_compression:
    kvcache_size = new_kv_len * num_kv_heads * head_size
else:
    kvcache_size = new_kv_len * num_kv_heads * head_size * 2

kvcache_loading_total_time=0
num_runs = 0
for ns in latency:
    kvcache_loading_total_time += ns
    num_runs += 1
    print(f"  {ns*1e-6:.3f} ms,  Bandwidth = {kvcache_size/(ns):.3f} GB/s, kvcache_size = {kvcache_size*1e-6:.1f} MB")

print()
print("kvcache_loading_total_time = ", kvcache_loading_total_time*1e-6, "ms")

print(f"num_runs = {num_runs}, avg kvcache loading = {kvcache_size*num_runs/kvcache_loading_total_time:.3f} GB/s")
print()

sys.exit(0)
