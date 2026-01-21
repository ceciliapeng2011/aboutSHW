import argparse
import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

torch.manual_seed(0)
torch.set_printoptions(linewidth=1024)

parser = argparse.ArgumentParser('')
parser.add_argument('-i', "--impl", type=int, default=1)
parser.add_argument('-b', "--batch", type=int, default=1)
parser.add_argument('-nh', "--num-heads", type=int, default=32)
parser.add_argument('-nkvh', "--num-kv-heads", type=int, default=8)
parser.add_argument('-ql', "--q-len", type=int, default=1)
parser.add_argument('-kvl', "--kv-len", type=int, default=32769)
parser.add_argument('-hs', "--head-size", type=int, default=128)
parser.add_argument('-rkv', "--reset_kv_cache", type=int, default=1)
parser.add_argument("--enable-kvcache-compression", type=int, default=0)
parser.add_argument(
    "--kv-cache-quant-mode",
    type=str,
    default=os.environ.get("KV_CACHE_QUANT_MODE", "by_token"),
)
parser.add_argument("--q-dist", type=str, default="random")
parser.add_argument('-v', "--verbose", type=int, default=-1)
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

# define KV_BLOCK_SIZE = 32,64,128,256
kv_block_size = 256

enable_kvcache_compression = args.enable_kvcache_compression
kv_cache_quantization_mode = args.kv_cache_quant_mode

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
xe_arch = 2

if xe_arch == 1:
    kv_step = 8
else:
    kv_step = 16

# dpas number for each split_len
# split_subblock_num = kv_partition_size // kv_step

# reduce step size
reduce_split_step = 8

low = -127
high = 128
act_dtype = torch.float16
new_kv_len = (kv_len + kv_block_size - 1) // kv_block_size * kv_block_size
if args.q_dist == "random":
    q = torch.randint(low, high, [batch, q_len, num_heads, head_size]).to(dtype=act_dtype) / high
elif args.q_dist == "ones":
    q = torch.ones([batch, q_len, num_heads, head_size], dtype=act_dtype)
else:
    raise ValueError(f"Unsupported q distribution: {args.q_dist}")
k = torch.randint(low, high, [batch, new_kv_len, num_kv_heads, head_size]).to(dtype=act_dtype)/high
v = torch.randint(low, high, [batch, new_kv_len, num_kv_heads, head_size]).to(dtype=act_dtype)/high

run_real_data_test=False

if run_real_data_test:
    q = get_tensor("./qwen3_q_f16_1_4096.bin").reshape([1,1,32,128])
    k = get_tensor("./qwen3_k_f16_129_8_16_128.bin").reshape([129,8,16,128]).transpose(1,2).reshape([1,129*16,8,128])[:,:128*16+1,:,:]
    v = get_tensor("./qwen3_v_f16_129_8_16_128.bin").reshape([129,8,16,128]).transpose(1,2).reshape([1,129*16,8,128])[:,:128*16+1,:,:]
    kv_len = 128*16+1
    # q = torch.randint(low, high, [1, 1, 32, 128]).to(dtype=act_dtype)/high
    # k = torch.randint(low, high, [1, kv_len, 8, 128]).to(dtype=act_dtype)/high
    # v = torch.randint(low, high, [1, kv_len, 8, 128]).to(dtype=act_dtype)/high
    print("q.shape = ", q.shape)
    print("k.shape = ", k.shape)
    print("v.shape = ", v.shape)

# sdpa split size, must be multiple of kv_step and kv_len should be multiple of kv_partition_size
k_partition_block_num = kv_len//8192
if k_partition_block_num < 1:
    k_partition_block_num = 1
k_partition_block_num = 1  # test cm_sdpa_2nd
kv_partition_size = int(kv_block_size * k_partition_block_num)

print("kv_step:", kv_step)
print("k_partition_block_num:", k_partition_block_num)
print("kv_partition_size:", kv_partition_size)

new_kv_len = (kv_len + kv_block_size - 1) // kv_block_size * kv_block_size
kv_partition_num = (new_kv_len + kv_partition_size - 1) // kv_partition_size
total_partition_num = kv_partition_num * num_heads
assert(kv_partition_size % kv_step == 0)

# random attnmask
attention_mask = torch.full([batch, 1, q_len, kv_len], torch.finfo(act_dtype).min).to(dtype=act_dtype)
#attention_mask[torch.rand(batch, 1, q_len, kv_len) > 0.5] = 0
attention_mask[...] = 0
att = torch.from_numpy(attention_mask.numpy())
#print("attention_mask:", att)

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

# BLHS=>BHLS
q = q.transpose(1,2)
k = k.transpose(1,2)
v = v.transpose(1,2)
print("q:", q.shape, q.dtype)
print("k:", k.shape, k.dtype)
print("v:", v.shape, v.dtype)
print("attention_mask:", attention_mask.shape, attention_mask.dtype)

def get_org(Q, K, V, attention_mask):
    B,H,L,S = Q.shape
    _,Hkv,_,_ = K.shape
    out = torch.zeros([B,H,L,S], dtype=Q.dtype)
    scale_factor = S**(-0.5)
    for b in range(B):
        for h in range(H):
            hkv = h // (H//Hkv)
            attn_score = Q[b, h, :, :].to(dtype=torch.float32) @ (K[b, hkv, :,:].transpose(0,1)).to(dtype=torch.float32)
            attn_score *= scale_factor
            attn_score += attention_mask[b,0,:,:]
            #print(attn_score.shape)
            attn_weights = F.softmax(attn_score, 1)
            out[b,h,:,:] = attn_weights @ V[b, hkv, :, :].to(dtype=torch.float32)
    return out

print("k = ", k.shape, k.dtype)
ref = F.scaled_dot_product_attention(q, k[:,:,:kv_len,:], v[:,:,:kv_len,:], attention_mask, dropout_p=0.0, enable_gqa = enable_gqa)
org = get_org(q, k[:,:,:kv_len,:], v[:,:,:kv_len,:], attention_mask)

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

check_close(ref, org, atol=1e-3, rtol=1e-2)

# [batch, seq-len, heads, size] BLHS
print("ref:", ref.shape, ref.dtype)

# blocking on kv-len dimension with online-softmax
# Softmax(Q@Kt)@V
def get_flash1(query, key, value, attention_mask):
    global enable_vprint
    B,H,q_len,hs = query.shape
    _,Hkv,kv_len,_ = key.shape
    out = torch.zeros([B,H,q_len,hs], dtype=value.dtype)
    scale_factor = hs**(-0.5)
    for b in range(B):
        for h in range(H):
            hkv = h // (H//Hkv)
            Q = query[b, h, :, :]
            K = key[b, hkv, :, :]
            V = value[b, hkv, :, :]
            mask = attention_mask[b,0,:,:]

            # loop one time
            # q_len == 1, q_step == 1
            for i in range(0, 1, 1):
                i1 = 1
                # online softmax states:
                #     per-row max-value     : [1, 1]
                #     per-row sum           : [1, 1]
                #     current accumulated V : [1, S]
                #cur_max = torch.full([i1-i, 1], torch.finfo(torch.float32).min, dtype=torch.float32)
                #cur_sum = torch.full([i1-i, 1], 0, dtype=torch.float32)
                #cur_O = torch.full([i1-i, hs], 0, dtype=torch.float32)

                rQ = Q[0, :].reshape(1,hs) # [1,128] sub Q block VNNI packed

                cur_O = torch.full([kv_len // kv_step, 1, hs], 0, dtype=torch.float32)
                max_comp_0 = torch.full([kv_len // kv_step], 1, dtype=torch.float32)

                for j in range(0, kv_len, kv_step):
                    j1 = min(j + kv_step, kv_len)

                    if (j == args.verbose): enable_vprint = True
                    # compute in local SRAM
                    # Step8: On chip, compute S(_𝑗)𝑖= Q𝑖K𝑇 𝑗∈ R𝐵𝑟 ×𝐵𝑐.
                    rKt = K[j:j1,:].transpose(0,1)  #[16, 128] -> [128, 16], suppose kv_step is 16
                    rS = (rQ @ rKt).to(dtype=torch.float32).reshape(1,kv_step)  # [1,16]
                    rMask = mask[i:i1, j:j1] # [1,16]

                    vprint("rK=", rKt.shape)
                    vprint("rQt=",rQ.shape)
                    vprint("rS=",rS.shape)
                    vprint("rMask=",rMask.shape)

                    rS *= scale_factor
                    rS += rMask
                    vprint("rS=",rS.shape)

                    rowmax = rS.max(1, keepdim=True).values # [1,1]
                    if j == 0:
                        cur_max = rowmax
                    else:
                        rowmax = torch.maximum(cur_max, rowmax)
                    vprint("rowmax=", rowmax.shape)

                    # compute in local SRAM
                    rS = torch.exp(rS - rowmax) # [1,16]
                    vprint("St(Pt)=", rS.shape)

                    rowsumP = rS.sum(1, keepdim=True) # [1,1]
                    vprint("rowsumP=", rowsumP.shape)

                    # corrected sum of previous block
                    if j > 0:
                        max_comp = torch.exp(cur_max - rowmax)
                        vprint("max_comp=", max_comp.shape)
                        max_comp_0[j//kv_step] = max_comp

                    if j == 0:
                        cur_sum = rowsumP
                    else:
                        cur_sum = cur_sum * max_comp + rowsumP

                    # softmax normalize is saved accoridng to flash-attn2 section 3.1.1
                    # We can instead maintain an “un-scaled” version of O(2) and keep around the statistics ℓ(2)
                    partial_attn_weight = rS.to(dtype=torch.float16) # [1,16]
                    
                    vprint("P=", partial_attn_weight.shape)

                    rV = V[j:j1, :]  # [16,128]
                    vprint("rV=",rV.shape)

                    # correct last Output to current statistics
                    cur_O[j//kv_step,:,:] = partial_attn_weight @ rV # [:,1,128]
                    vprint("cur_O2=", cur_O.shape)

                    cur_max = rowmax
                    if (j == args.verbose): assert 0

                cur_O_f32 = cur_O[0,:,:]
                for j in range(1, kv_len//kv_step):
                    cur_O_f32 = cur_O_f32 *  max_comp_0[j] + cur_O[j,:,:]
                vprint("cur_O_f32=", cur_O_f32.shape)
                vprint("cur_sum=", cur_sum.shape)
                cur_O_f16 = (cur_O_f32/cur_sum).to(torch.float16)

                if (i == args.verbose):
                    enable_vprint = True
                    print("cur_O_f16=", cur_O_f16.shape, cur_O_f16)
                    assert 0

                out[b, h, i:i1, :] = cur_O_f16
    return out

# Split KV online-softmax
def get_flash2(query, key, value, attention_mask, real_kv_len=0):
    global enable_vprint
    B,H,q_len,hs = query.shape
    _,Hkv,kv_len,_ = key.shape
    if real_kv_len == 0:
        real_kv_len = kv_len
    out = torch.zeros([B,H,q_len,hs], dtype=value.dtype)
    scale_factor = hs**(-0.5)
    for b in range(B):
        for h in range(H):
            hkv = h // (H//Hkv)
            Q = query[b, h, :, :]
            K = key[b, hkv, :, :]
            V = value[b, hkv, :, :]
            mask = attention_mask[b,0,:,:]

            # loop kv_split
            cur_O=torch.full([kv_len//kv_partition_size, 1, hs], 0, dtype=torch.float32)
            cur_O_f32=torch.full([1, hs], 0, dtype=torch.float32)
            cur_O_f16=torch.full([1, hs], 0, dtype=torch.float16)
            lse=torch.full([kv_len//kv_partition_size], 0, dtype=torch.float32)
            for i in range(0, kv_len, kv_partition_size):
                i1 = min(i + kv_partition_size, kv_len)
                # online softmax states:
                #     per-row max-value     : [1, 1]
                #     per-row sum           : [1, 1]
                #     current accumulated V : [1, S]
                #cur_max = torch.full([i1-i, 1], torch.finfo(torch.float32).min, dtype=torch.float32)
                #cur_sum = torch.full([i1-i, 1], 0, dtype=torch.float32)
                #cur_O = torch.full([i1-i, hs], 0, dtype=torch.float32)

                rQ = Q[0, :].reshape(1,hs) # [1,128] sub Q block VNNI packed
                vprint("rQ = ", rQ)
                vprint("mask = ", mask)

                cur_lse = 0.0
                cur_sum = 0.0
                for j in range(i, i1, kv_step):
                    j1 = min(j + kv_step, kv_len)
                    if j > kv_len:
                        break

                    if (j == args.verbose): enable_vprint = True
                    # compute in local SRAM
                    # Step8: On chip, compute S(_𝑗)𝑖= Q𝑖K𝑇 𝑗∈ R𝐵𝑟 ×𝐵𝑐.
                    rKt = K[j:j1,:].transpose(0,1) #[16,128]->[128,16]
                    rS = (rQ @ rKt).to(dtype=torch.float32).reshape(1,kv_step) #[1,16]
                    rMask = mask[0:1, j:j1] #[1,16]

                    vprint("rK=", rKt)
                    vprint("rQt=",rQ)
                    vprint("rS=",rS)
                    vprint("rMask=",rMask)

                    rS *= scale_factor
                    vprint("rS * scale_factor =",rS)
                    rS += rMask

                    cur_lse += torch.exp(rS).sum(1, keepdim=True).item() # [1,1]
                    vprint("rS=", rS)
                    vprint("exp(rS)=", torch.exp(rS))
                    vprint("cur_lse=", cur_lse)

                    rowmax = rS.max(1, keepdim=True).values # [1,1]
                    if j == 0:
                        cur_max = rowmax
                    else:
                        rowmax = torch.maximum(cur_max, rowmax)
                    vprint("rowmax=", rowmax.shape)

                    vprint("rS=", rS)
                    # compute in local SRAM
                    rS = torch.exp(rS - rowmax) # [1,16]
                    vprint("rowmax = ", rowmax)
                    vprint("St(Pt)=", rS)

                    rowsumP = rS.sum(1, keepdim=True) # [1,1]
                    vprint("rowsumP=", rowsumP)

                    # corrected sum of previous block
                    if j > 0:
                        max_comp = torch.exp(cur_max - rowmax)
                        vprint("max_comp=", max_comp.shape)

                    if j == 0:
                        cur_sum = rowsumP
                    else:
                        cur_sum = cur_sum * max_comp + rowsumP

                    # softmax normalize is saved accoridng to flash-attn2 section 3.1.1
                    # We can instead maintain an “un-scaled” version of O(2) and keep around the statistics ℓ(2)
                    partial_attn_weight = rS.to(dtype=torch.float16) # [1,16]
                    
                    vprint("P=", partial_attn_weight)

                    rV = V[j:j1, :] # [16,128]
                    vprint("rV=",rV)

                    # correct last Output to current statistics
                    if j== 0:
                        cur_O[i//kv_partition_size,:,:] = partial_attn_weight @ rV
                    else:
                        cur_O[i//kv_partition_size,:,:] = cur_O[i//kv_partition_size,:,:] * max_comp;
                        cur_O[i//kv_partition_size,:,:] += partial_attn_weight @ rV # [:,1,128]
                    vprint("j = ", j)
                    vprint("cur_O2=",  cur_O[i//kv_partition_size,:,:])

                    cur_max = rowmax
                    if (j == args.verbose): assert 0

                lse[i//kv_partition_size] = cur_lse
                if i > real_kv_len:
                    cur_O[i//kv_partition_size,:,:] = 0
                else:
                    cur_O[i//kv_partition_size,:,:] = cur_O[i//kv_partition_size,:,:] / cur_sum
                vprint("cur_sum=", cur_sum.shape)
                vprint("cur_O=",  cur_O[i//kv_partition_size,:,:])

            # reduce
            # for i in range(0, kv_len//kv_partition_size, 1):
            #     for j in range(0, hs, kv_step):
            #         stop = min(j + kv_step, hs)
            #         print("i=", i, ", j = ", j, ": cur_O[i,:,:]=", cur_O[i,0,j:stop])
            # vprint("lse=", lse)
            # print("lse=", lse.shape) # [4]
            sum_lse = lse.sum(0)
            # print("cur_O=", cur_O.shape) # 
            # print("cur_O_f32=", cur_O_f32.shape) #
            for i in range(0, kv_len//kv_partition_size, 1):
                if i * kv_partition_size > real_kv_len:
                    break
                cur_O_f32 += cur_O[i,:,:] * lse[i] / sum_lse
            cur_O_f16 = cur_O_f32.to(torch.float16)
            out[b, h, :, :] = cur_O_f16
            # print("cur_O_f16=", cur_O_f16.shape) #
            # print("out=", out.shape) #
    #print("out = ", out[0,0,0,:])
    #print("out = ", out)
    return out

# Split KV online-softmax
def get_flash3(query, key, value, attention_mask):
    global enable_vprint
    B,H,q_len,hs = query.shape
    _,Hkv,kv_len,_ = key.shape
    out = torch.zeros([B,H,q_len,hs], dtype=value.dtype)
    scale_factor = hs**(-0.5)
    for b in range(B):
        for h in range(H):
            hkv = h // (H//Hkv)
            Q = query[b, h, :, :]  # [q_len, head_size]
            K = key[b, hkv, :, :]  # [kv_len, head_size]
            V = value[b, hkv, :, :]  # [kv_len, head_size]
            mask = attention_mask[b,0,:,:]  # [q_len, kv_len]

            # loop kv_split
            cur_O=torch.full([kv_len//kv_partition_size, 1, hs], 0, dtype=torch.float32)  # [kv_len//kv_partition_size, 1, head_size]
            cur_O_f32=torch.full([1, hs], 0, dtype=torch.float32)  # [1, head_size]
            cur_O_f16=torch.full([1, hs], 0, dtype=torch.float16)  # [1, head_size]
            lse=torch.full([kv_len//kv_partition_size], 0, dtype=torch.float32)  # [kv_len//kv_partition_size]
            for i in range(0, kv_len, kv_partition_size):
                i1 = min(i + kv_partition_size, kv_len)
                # online softmax states:
                #     per-row max-value     : [1, 1]
                #     per-row sum           : [1, 1]
                #     current accumulated V : [1, S]
                #cur_max = torch.full([i1-i, 1], torch.finfo(torch.float32).min, dtype=torch.float32)
                #cur_sum = torch.full([i1-i, 1], 0, dtype=torch.float32)
                #cur_O = torch.full([i1-i, hs], 0, dtype=torch.float32)

                rQ = Q[0, :].reshape(1,hs) # [1,128] sub Q block VNNI packed
                if i==0:
                    vprint("rQ = ", rQ)
                vprint("mask = ", mask)

                cur_lse = 0.0
                cur_sum = 0.0
                for j in range(i, i1, kv_partition_size):
                    j1 = min(j + kv_partition_size, kv_len)

                    if (j == args.verbose): enable_vprint = True
                    # compute in local SRAM
                    # Step8: On chip, compute S(_𝑗)𝑖= Q𝑖K𝑇 𝑗∈ R𝐵𝑟 ×𝐵𝑐.
                    rKt = K[j:j1,:].transpose(0,1)  # [head_size, kv_partition_size]
                    rS = (rQ @ rKt).to(dtype=torch.float32).reshape(1,kv_partition_size)  #[1, kv_partition_size]
                    rMask = mask[0:1, j:j1]  #[1, kv_partition_size]

                    vprint("rK=", rKt)
                    vprint("rQt=",rQ)
                    vprint("rS=",rS)
                    vprint("rMask=",rMask)

                    rS *= scale_factor
                    vprint("rS * scale_factor =",rS)
                    rS += rMask

                    cur_lse += torch.exp(rS).sum(1, keepdim=True).item() # [1,1]
                    vprint("rS=", rS)
                    vprint("exp(rS)=", torch.exp(rS))
                    vprint("cur_lse=", cur_lse)

                    rowmax = rS.max(1, keepdim=True).values # [1,1]

                    # compute in local SRAM
                    rS = torch.exp(rS - rowmax) # [1,16]
                    vprint("rowmax = ", rowmax)
                    vprint("St(Pt)=", rS)

                    rowsumP = rS.sum(1, keepdim=True) # [1,1]
                    vprint("rowsumP=", rowsumP)

                    # corrected sum of previous block
                    cur_sum = rowsumP

                    # softmax normalize is saved accoridng to flash-attn2 section 3.1.1
                    # We can instead maintain an “un-scaled” version of O(2) and keep around the statistics ℓ(2)
                    partial_attn_weight = rS.to(dtype=torch.float16) # [1,16]

                    vprint("P=", partial_attn_weight)

                    rV = V[j:j1, :]  # [kv_partition_size, head_size]
                    vprint("rV=",rV)

                    # correct last Output to current statistics
                    cur_O[i//kv_partition_size,:,:] = partial_attn_weight @ rV
                    vprint("cur_O2=",  cur_O[i//kv_partition_size,:,:])

                lse[i//kv_partition_size] = cur_lse
                cur_O[i//kv_partition_size,:,:] = cur_O[i//kv_partition_size,:,:] / cur_sum
                vprint("cur_sum=", cur_sum.shape)
                vprint("Omat=",  cur_O[i//kv_partition_size,:,:])

            # reduce
            # for i in range(0, kv_len//kv_partition_size, 1):
            #     for j in range(0, hs, kv_step):
            #         stop = min(j + kv_step, hs)
            #         print("i=", i, ", j = ", j, ": cur_O[i,:,:]=", cur_O[i,0,j:stop])
            vprint("lse=", torch.log(lse))
            vprint("lse=", lse.shape) # [4]
            sum_lse = lse.sum(0)
            # print("cur_O=", cur_O.shape) # 
            # print("cur_O_f32=", cur_O_f32.shape) #
            vprint("lse = ", lse)
            vprint("sum_lse = ", sum_lse)
            for i in range(0, kv_len//kv_partition_size, 1):
                cur_O_f32 += cur_O[i,:,:] * lse[i] / sum_lse
            cur_O_f16 = cur_O_f32.to(torch.float16)
            out[b, h, :, :] = cur_O_f16
            # print("cur_O_f16=", cur_O_f16.shape) #
            # print("out=", out.shape) #
    #print("out = ", out[0,0,0,:])
    #print("out = ", out)
    return out

# transpose back to orginal shape: [batch, q_len, num_heads, head_size] for padding
q = q.transpose(1,2)
k = k.transpose(1,2)
v = v.transpose(1,2)

if kv_len % kv_block_size != 0:
    # pad k,v to multiple of kv_block_size
    pad_len = ((kv_len + kv_block_size - 1)//kv_block_size)*kv_block_size - kv_len
    # k = F.pad(k, (0,0,0,0,0,pad_len,0,0), "constant", 0)
    # v = F.pad(v, (0,0,0,0,0,pad_len,0,0), "constant", 0)
    attention_mask = F.pad(attention_mask, (0,pad_len,0,0), "constant", torch.finfo(act_dtype).min)
    print(f"pad k,v from {kv_len} to {k.shape[1]}")
    new_kv_len = k.shape[1]
    total_blk_num = new_kv_len//kv_block_size
    assert(new_kv_len % kv_block_size == 0)

print("k.shape:", k.shape, k.dtype)
print("v.shape:", v.shape, v.dtype)
print("new_kv_len = ", new_kv_len)

# transpose shape: [batch, num_heads, q_len, head_size] for get_flash3
q = q.transpose(1,2)
k = k.transpose(1,2)
v = v.transpose(1,2)

if args.impl == 0:
    f0 = get_flash3(q,k,v,attention_mask)
    check_close(org, f0, atol=1e-2, rtol=1e-3)
    print("=========== PASS ===========")
    sys.exit(0)

# org1 = get_flash1(q,k,v,attention_mask)
# check_close(ref, org1, atol=1e-3, rtol=1e-2)
# print("org of get_flash1 passed !")

# org2 = get_flash2(q,k,v,attention_mask, real_kv_len=kv_len)
# check_close(ref, org2, atol=1e-3, rtol=1e-2)
# print("org of get_flash2 passed !")

org = get_flash3(q,k,v,attention_mask)
check_close(ref, org, atol=1e-3, rtol=1e-2)
print("org of get_flash3 passed !")
# check_close(org, org2, atol=1e-3, rtol=1e-2)

print()
print("GPU cm kernels for flash attn2:")

#====================================================================================================
# using the same parameter & inputs, develop cm kernels which produces the same output
# prototyping CM kernels
from clops import cl
import numpy as np
import time

# transpose back to orginal shape: [batch, q_len, num_heads, head_size]
q = q.transpose(1,2)
k = k.transpose(1,2)
v = v.transpose(1,2)
print("q:", q.shape, q.dtype)
print("k:", k.shape, k.dtype)
print("v:", v.shape, v.dtype)
#print("attention_mask:", attention_mask.shape, attention_mask.dtype)

#print("original k:", k)

# if kv_len % kv_block_size != 0:
#     # pad k,v to multiple of kv_block_size
#     pad_len = ((kv_len + kv_block_size - 1)//kv_block_size)*kv_block_size - kv_len
#     k = F.pad(k, (0,0,0,0,0,pad_len,0,0), "constant", 0)
#     v = F.pad(v, (0,0,0,0,0,pad_len,0,0), "constant", 0)
#     attention_mask = F.pad(attention_mask, (0,pad_len,0,0), "constant", torch.finfo(act_dtype).min)
#     print(f"pad k,v from {kv_len} to {k.shape[1]}")
#     new_kv_len = k.shape[1]
#     total_blk_num = new_kv_len//kv_block_size
#     assert(new_kv_len % kv_block_size == 0)

def quan_per_token(kv):
    blk_num, kv_heads, blksz, *_ = kv.shape
    kv_max = kv.amax(dim=-1, keepdim = True)
    kv_min = kv.amin(dim=-1, keepdim = True)
    qrange = kv_max - kv_min

    INTMAX = 255.0
    INTMIN = 0.0
    INTRAGNE = INTMAX - INTMIN
    kv_scale = ((INTRAGNE)/qrange).to(dtype=torch.half)
    kv_zp = ((0.0-kv_min)*kv_scale+INTMIN).to(dtype=torch.half)

    kv_INT8 = torch.round((kv*kv_scale+kv_zp)).to(dtype=torch.uint8).reshape(blk_num,kv_heads,-1)
    # print("################################################################################")
    # print(f'KV\n:{kv.reshape(16, 32)}')
    # print(f'kv_INT8\n:{kv_INT8.reshape(16, 32)}')
    # print(f'kv_scale\n:{kv_scale.reshape( 16, 1)}')
    # print(f'kv_zp\n:{kv_zp.reshape( 16, 1)}')
    # print("################################################################################")

    # print("quant_scale =", (1.0/kv_scale).reshape(blk_num,kv_heads,-1))
    # print("quant_zp    =", kv_zp.reshape(blk_num,kv_heads,-1))

    dq_scale = (1.0/kv_scale).view(dtype=torch.uint8).reshape(blk_num,kv_heads,-1)
    kv_zp = kv_zp.view(dtype=torch.uint8).reshape(blk_num,kv_heads,-1)
    # print("dq_scale: ", dq_scale)
    # print("kz_zp: ", kv_zp)
    return torch.concat((kv_INT8, dq_scale, kv_zp), dim=-1)

def dequant_per_token(kv, head_size, blk_size):
    blk_num, kv_head_num, _ = kv.shape
    kv_u8 = kv[:,:,:head_size * blk_size].to(dtype=torch.float16).reshape(blk_num, kv_head_num, blk_size, head_size)
    kv_scale = kv[:,:,head_size * blk_size: (head_size * blk_size + blk_size * 2)].view(dtype=torch.float16).reshape(blk_num, kv_head_num, blk_size, 1)
    kv_zp = kv[:,:, (head_size * blk_size + blk_size * 2):(head_size * blk_size + blk_size * 4)].view(dtype=torch.float16).reshape(blk_num, kv_head_num, blk_size, 1)

    # print("dequant_kv_u8 = ", kv_u8)
    # print("dequant_kv_scale = ", kv_scale.reshape(blk_num, kv_head_num, blk_size))
    # print("dequant_kv_zp    = ", kv_zp.reshape(blk_num, kv_head_num, blk_size))

    kv_dequant = torch.empty([blk_num, kv_head_num, blk_size, head_size], dtype=torch.float16)

    for m in range(blk_num):
        for n in range(kv_head_num):
            for i in range(blk_size):
                kv_dequant[m,n,i,:] = (kv_u8[m,n,i,:].to(dtype=torch.float16) - kv_zp[m,n,i,0].to(dtype=torch.float16)) * kv_scale[m,n,i,0].to(dtype=torch.float16)

    return kv_dequant

def quan_per_channel(kv):
    blk_num, kv_heads, blksz, head_size = kv.shape
    kv_max = kv.amax(dim=2, keepdim=True)
    kv_min = kv.amin(dim=2, keepdim=True)
    qrange = kv_max - kv_min

    INTMAX = 255.0
    INTMIN = 0.0
    INTRANGE = INTMAX - INTMIN

    # need to consider qrange equals to zero
    kv_scale = torch.zeros(blk_num, kv_heads, 1, head_size, dtype=torch.float16)
    kv_scale[qrange!=0] = (INTRANGE / qrange[qrange!=0]).to(dtype=torch.float16)
    kv_zp = ((0.0 - kv_min) * kv_scale + INTMIN).to(dtype=torch.float16)
    kv_INT8 = torch.round(kv * kv_scale + kv_zp).clamp(INTMIN, INTMAX).to(dtype=torch.uint8).reshape(blk_num,kv_heads,-1)
    dq_scale = torch.zeros(blk_num, kv_heads, 1, head_size, dtype=torch.float16)
    dq_scale[kv_scale!=0] = (1.0 / kv_scale[kv_scale!=0]).to(dtype=torch.float16)
    dq_scale = dq_scale.view(dtype=torch.uint8).reshape(blk_num,kv_heads,-1)
    kv_zp = kv_zp.view(dtype=torch.uint8).reshape(blk_num,kv_heads,-1)

    return torch.concat((kv_INT8, dq_scale, kv_zp), dim=-1)

def dequant_per_channel(kv, head_size, blk_size):
    blk_num, kv_head_num, _ = kv.shape
    kv_u8 = kv[:,:,:head_size * blk_size].to(dtype=torch.float16).reshape(blk_num, kv_head_num, blk_size, head_size)
    kv_scale = kv[:,:,head_size * blk_size: (head_size * blk_size + head_size * 2)].view(dtype=torch.float16).reshape(blk_num, kv_head_num, 1, head_size)
    kv_zp = kv[:,:, (head_size * blk_size + head_size * 2):(head_size * blk_size + head_size * 4)].view(dtype=torch.float16).reshape(blk_num, kv_head_num, 1, head_size)

    # print("dequant_kv_u8 = ", kv_u8)
    # print("dequant_kv_scale = ", kv_scale.reshape(blk_num, kv_head_num, blk_size))
    # print("dequant_kv_zp    = ", kv_zp.reshape(blk_num, kv_head_num, blk_size))

    kv_dequant = torch.empty([blk_num, kv_head_num, blk_size, head_size], dtype=torch.float16)

    for m in range(blk_num):
        for n in range(kv_head_num):
            for i in range(head_size):
                kv_dequant[m,n,:,i] = (kv_u8[m,n,:,i].to(dtype=torch.float16) - kv_zp[m,n,0,i].to(dtype=torch.float16)) * kv_scale[m,n,0,i].to(dtype=torch.float16)

    return kv_dequant

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
v = v.reshape(total_blk_num, kv_block_size, num_kv_heads, head_size).transpose(1,2).contiguous()
print("after blocking k:", k.shape, k.dtype)
print("after blocking v:", v.shape, v.dtype)

vprint("k[0,0,0,:] = ", k[0,0,0,:])
vprint("v[0,0,0,:] = ", v[0,0,0,:])
print()

if enable_kvcache_compression:
    # print("quant = ", k.reshape(total_blk_num, num_kv_heads, kv_block_size, head_size))
    k_origin = k.clone()
    print("k = ", k.shape)
    if kvcache_quantization_by_token:
        k = quan_per_token(k)
    else:
        k = quan_per_channel(k)
    v = quan_per_token(v)
    print(f"quant k shape: {k.shape}, dtype={k.dtype}")
    print(f"quant v shape: {v.shape}, dtype={v.dtype}")

    enable_dequant_check = 1
    if enable_dequant_check:
        if kvcache_quantization_by_token:
            k_dequan = dequant_per_token(k, head_size, kv_block_size)
        else:
            k_dequan = dequant_per_channel(k, head_size, kv_block_size)
        v_dequan = dequant_per_token(v, head_size, kv_block_size)
        # print("de-quant = ", k_dequan.reshape(total_blk_num, num_kv_heads, kv_block_size, head_size))
        # print("diff = ", (k_dequan - k_origin).abs())
        # print("k_dequan = ", k_dequan.shape)

        q_input = q.transpose(1,2).contiguous()
        k_input = k_dequan.transpose(1,2).reshape(batch, new_kv_len, num_kv_heads, head_size).transpose(1,2).contiguous()
        v_input = v_dequan.transpose(1,2).reshape(batch, new_kv_len, num_kv_heads, head_size).transpose(1,2).contiguous()

        # print("q = ", q_input.shape)
        # print("k_input = ", k_input.shape)
        org = get_org(q_input, k_input[:, :, :kv_len, :], v_input[:, :, :kv_len, :], attention_mask[:, :, :, :kv_len])
        check_close(ref, org, atol=1e-3, rtol=1e-2)


# print("quanted k[0,0,16*32:16*32+2*16]:", k[0,0,16*32:16*32+2*16])
# scale_slice=k[0,0,16*32:16*32+2*16]
# print("scale_slice:", scale_slice, scale_slice.dtype)
# scale_temp=scale_slice[0:2].view(torch.half)
# zp_slice=k[0,0,16*34:16*34+2*16]
# zp_temp=zp_slice[0:2].view(torch.half)[0]

# print("quanted scale = ",scale_temp)
# print("quanted zp    = ",zp_temp)
# print()

# k_test = k[0,0,:34]
# k_test = (k_test.to(dtype=torch.half) - zp_temp).to(dtype=torch.half) * scale_temp.to(dtype=torch.half)
# print("de-quant k_test:", k_test)
# print()

# print("reshape k:", k.shape)
# print("k[8,1,:,:]=", k[8,1,:,:])

blocked_k = k.clone()
blocked_v = v.clone()

for i in range(len(block_indices)):
    k[block_indices[i],:] =  blocked_k[i,:]
    v[block_indices[i],:] =  blocked_v[i,:]

k = k.contiguous()
v = v.contiguous()

print(f"final k shape: {k.shape}")
print(f"final v shape: {v.shape}")

#print("block_indices:", block_indices)
#print("blocked k:", k)

'''
ugemm_qk: [q_step, head_size] x [head_size, kv_step]
ugemm_kq: [kv_step, head_size] x [head_size, q_step]
ugemm_pv: [q_step, kv_step] x [kv_step, head_size]
'''

scale_factor = 1.0/(head_size**0.5)

# each WG processes a partition, and a chunk of q_head
MaxRepeatCount=8
q_heads_per_kv_head = num_heads // num_kv_heads
q_head_chunks_per_kv_head = (q_heads_per_kv_head + (MaxRepeatCount - 1)) // MaxRepeatCount
q_head_chunk_size = num_heads // (num_kv_heads * q_head_chunks_per_kv_head)
print(f"{num_heads=}, {num_kv_heads=}, {q_head_chunk_size=}, {q_head_chunks_per_kv_head=}")
GWS=[seq_num, num_kv_heads * q_head_chunks_per_kv_head, (new_kv_len + kv_partition_size - 1) // kv_partition_size]
WG_SIZE = 1;#max(kv_len // kv_partition_size//2, 1)
LWS=[1, 1, WG_SIZE]
print("GWS=", GWS)
print("LWS=", LWS)

#========================================================================
# Optimization Log
r'''
increase WG_SIZE to 16 (-70ms)
use GRF to store temp Output(rO) instead of SLM, requires `-Qxcm_register_file_size=256` (-40ms)
avoid type-promotion:   St = cm_mul<float>(St, scale_factor);    =>   St = cm_mul<float>(St, (float)scale_factor);   (-10ms) 
change dtype of attention_mask from float to half (-14ms)
avoid indirect register access: unroll for loop which access matrix rows using loop-index
use GRF to store temp Input rQ instead of SLM, this allows more Work-Groups to be packed into same Xe-core!!!
'''
#========================================================================

def create_kernels():
    # kernel
    src = r'''#include "pa_single_token.cm"'''
    cwd = os.path.dirname(os.path.realpath(__file__))
    print(f"compiling {cwd} ...")

    # jit_option = '-abortonspill -noschedule '
    jit_option = ''
    kernels = cl.kernels(src, f'''-cmc -Qxcm_jit_option="{jit_option}"
                        -mCM_printregusage -mdump_asm -g2
                        -Qxcm_register_file_size=256 -I{cwd}
                        -DHEADS_NUM={num_heads} -DKV_HEADS_NUM={num_kv_heads} -DHEAD_SIZE={head_size}
                        -DQ_STEP={q_step} -DKV_STEP={kv_step}
                        -DWG_SIZE={WG_SIZE} -DKV_BLOCK_SIZE={kv_block_size}
                        -DKV_PARTITION_SIZE={kv_partition_size} -DREDUCE_SPLIT_SIZE={reduce_split_step}
                        -DCLEAN_UNUSED_KVCACHE={enable_clean_unused_kvcache}
                        -DKV_CACHE_COMPRESSION={enable_kvcache_compression}
                        -DKV_CACHE_COMPRESSION_BY_TOKEN={kvcache_quantization_by_token}
                        -DXE_ARCH={xe_arch}
                        -DQ_head_chunks_per_kv_head={int(q_head_chunks_per_kv_head)}
                        -DQ_head_chunk_size={int(q_head_chunk_size)}
                        -DSCALE_FACTOR={scale_factor}''')
    return kernels

cl.profiling(True)

t_q = cl.tensor(q.detach().numpy())
t_k = cl.tensor(k.contiguous().detach().numpy())
t_v = cl.tensor(v.contiguous().detach().numpy())
t_past_lens = cl.tensor(past_lens.detach().numpy())
t_block_indices = cl.tensor(block_indices.detach().numpy())
t_block_indices_begins = cl.tensor(block_indices_begins.detach().numpy())
t_subsequence_begins = cl.tensor(subsequence_begins.detach().numpy())
t_out = cl.tensor([batch, num_heads, kv_partition_num, head_size], np.dtype(np.float32))
t_out_final = cl.tensor([batch, 1, num_heads, head_size], np.dtype(np.float16))
t_lse = cl.tensor([batch, num_heads, kv_partition_num], np.dtype(np.float32))


intermedia_mem_size=batch*num_heads*kv_partition_num*(head_size+1)*4
print("intermediate memory size = ", intermedia_mem_size/1024/1024, "MB")

if enable_kvcache_compression and False:
    print("k = ", k.shape, k.dtype)
    key_u8_data=k.reshape(new_kv_len//kv_block_size, num_kv_heads, kv_block_size * (head_size + 4))[:,:,:kv_block_size*head_size]
    print(key_u8_data.reshape(new_kv_len//kv_block_size, num_kv_heads, kv_block_size, head_size))

    print("v = ", v.shape, v.dtype)
    value_u8_data=v.reshape(new_kv_len//kv_block_size, num_kv_heads, kv_block_size * (head_size + 4))[:,:,:kv_block_size*head_size]
    print(value_u8_data.reshape(new_kv_len//kv_block_size, num_kv_heads, kv_block_size, head_size))


# f"-cmc -mdump_asm -g2 "
cwd = os.path.dirname(os.path.realpath(__file__))
print("compiling ...")
cm_kernels = create_kernels()

print("first call ...")
cm_kernels.enqueue("cm_sdpa_2nd", GWS, LWS, t_q, t_k, t_v,
                       t_past_lens, t_block_indices, t_block_indices_begins,
                       t_subsequence_begins,t_out, t_lse, q_len)

# np.save("bmg_out", t_out.numpy())

# f0 = torch.from_numpy(t_out.numpy())
# print("f0 = ", f0.shape, f0.dtype)
# for i in range(num_heads):
#     for j in range(kv_partition_num):
#         for k in range(0, head_size, kv_step):
#             stop = min(k + kv_step, head_size)
#             print(f"f0[{i}, {j}, {k}:{stop}] = ", f0[0, i, j, k:stop])

# lse0 = torch.from_numpy(t_lse.numpy())
# print("lse0 = ", lse0.shape, lse0.dtype, lse0)
# print("lse_sum = ", lse0.sum(2))

GWS_2 = [batch, num_heads, head_size//reduce_split_step]
LWS_2 = [1, 1, 1]
print("GWS_2=", GWS_2)
print("LWS_2=", LWS_2)
cm_kernels.enqueue("cm_sdpa_2nd_reduce", GWS_2, LWS_2, t_out, t_out_final, t_lse, kv_partition_num)
f1 = torch.from_numpy(t_out_final.numpy())

# lse_final = torch.from_numpy(t_lse.numpy())
# print("lse_final = ", lse_final.shape, lse_final.dtype)
# print("lse_final sum = ", lse_final)


print("ref = ", org.transpose(1,2)[0,0,-1,:])
print("res = ", f1[0,0,-1,:])
check_close(org.transpose(1,2), f1, atol=1e-2, rtol=1e-3)
#sys.exit(0)

loop_cnt = 100
all_layers = []
mem_size = 0
while len(all_layers) < loop_cnt and mem_size < 8e9:
    all_layers.append([
        cl.tensor(q.detach().numpy()),
        cl.tensor(k.detach().numpy()),
        cl.tensor(v.detach().numpy()),
        cl.tensor([batch, num_heads, kv_partition_num, head_size], np.dtype(np.float32)),
        cl.tensor([batch, 1, num_heads, head_size], np.dtype(np.float16)),
    ])
    mem_size += q.numel() * q.element_size()
    mem_size += k.numel() * k.element_size()
    mem_size += v.numel() * v.element_size()
    # print(f"nlayers={len(all_layers)} mem_size={mem_size*1e-9:.3f} GB")

for i in range(loop_cnt):
    j  = i % len(all_layers)
    cm_kernels.enqueue("cm_sdpa_2nd", GWS, LWS,
                        all_layers[j][0],
                        all_layers[j][1],
                        all_layers[j][2],
                        t_past_lens, t_block_indices, t_block_indices_begins,t_subsequence_begins,
                        all_layers[j][3],
                        t_lse, q_len)
    cm_kernels.enqueue("cm_sdpa_2nd_reduce", GWS_2, LWS_2, all_layers[j][3],all_layers[j][4], t_lse, kv_partition_num)

latency = cl.finish()
first_kernel = 1
if enable_kvcache_compression:
    kvcache_size = new_kv_len * num_kv_heads * head_size * 1 * 2
else:
    kvcache_size = new_kv_len * num_kv_heads * head_size * 2 * 2
intermedia_size = batch * num_heads * kv_partition_num * (head_size + 1) * 4

kvcache_total_time=0
intermedia_total_time=0
num_runs = 0
num_warmup = 0
for iter, ns in enumerate(latency):
    if first_kernel:
        if num_warmup <= 4:
            num_warmup += 1
        else:
            kvcache_total_time += ns
            num_runs += 1
        print(f" {iter} {ns*1e-6:.3f} ms,  Bandwidth = {kvcache_size/(ns):.3f} GB/s, kvcache_size = {kvcache_size*1e-6:.1f} MB")
        #print(f"  {ns*1e-6:.3f} ms,  Bandwidth = {kvcache_size/(ns):.3f} GB/s")
    else:
        if num_warmup <= 4:
            num_warmup += 1
        else:
            intermedia_total_time += ns
        #print(f"  {ns*1e-6:.3f} ms,  Bandwidth = {intermedia_size/(ns):.3f} GB/s, intermedia = {intermedia_size*1e-6:.1f} MB")
    first_kernel =  1 - first_kernel


print()
print("intermedia_total_time = ", intermedia_total_time*1e-6, "ms")
print("kvcache_total_time = ", kvcache_total_time*1e-6, "ms")

print(f"num_runs = {num_runs}, avg kvcache = {kvcache_size*num_runs/kvcache_total_time:.3f} GB/s, avg intermedia = {intermedia_size*num_runs/(intermedia_total_time):.3f} GB/s")
print(f"num_runs = {num_runs}, avg pa kernel time for Qwen3-8B = {(kvcache_total_time + intermedia_total_time) * 1e-6 / num_runs * 36:.3f} ms")
print()

f1 = torch.from_numpy(all_layers[0][4].numpy())

check_close(org.transpose(1,2), f1, atol=1e-2, rtol=1e-3)
print(f"=========== cm_sdpa_2nd PASS GWS={GWS} LWS={LWS} GWS_2={GWS_2} LWS_2={LWS_2} ===========")

sys.exit(0)
