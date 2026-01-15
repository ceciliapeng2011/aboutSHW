from clops.utils import *

import torch
import torch.nn as nn
import torch.nn.functional as F

VERBOSE = -1
enable_vprint = False
def vprint(*all_args):
    global enable_vprint
    if enable_vprint:
        print(*all_args)
        
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

                    if (j == VERBOSE): enable_vprint = True
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
                    if (j == VERBOSE): assert 0

                cur_O_f32 = cur_O[0,:,:]
                for j in range(1, kv_len//kv_step):
                    cur_O_f32 = cur_O_f32 *  max_comp_0[j] + cur_O[j,:,:]
                vprint("cur_O_f32=", cur_O_f32.shape)
                vprint("cur_sum=", cur_sum.shape)
                cur_O_f16 = (cur_O_f32/cur_sum).to(torch.float16)

                if (i == VERBOSE):
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

                    if (j == VERBOSE): enable_vprint = True
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
                    if (j == VERBOSE): assert 0

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

                    if (j == VERBOSE): enable_vprint = True
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

if __name__ == "__main__":
    import argparse
    import os

    torch.manual_seed(0)
    torch.set_printoptions(linewidth=1024)
   
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

    enable_kvcache_compression = 1
    kv_cache_quantization_mode = os.environ.get("KV_CACHE_QUANT_MODE", "by_token")
    kv_cache_quantization_mode = "by_channel"

    def _validate_quant_mode(mode: str) -> str:
        mode = mode.strip().lower()
        if mode not in {"by_token", "by_channel"}:
            raise ValueError(f"Unsupported kv-cache quantization mode: {mode}")
        return mode

    kv_cache_quantization_mode = _validate_quant_mode(kv_cache_quantization_mode)
    kvcache_quantization_by_token = int(kv_cache_quantization_mode == "by_token")
    print(f"{kv_cache_quantization_mode=}, {kvcache_quantization_by_token=}")

    enable_clean_unused_kvcache = args.reset_kv_cache
    
    low = -127
    high = 128
    act_dtype = torch.float16
    new_kv_len = (kv_len + kv_block_size - 1) // kv_block_size * kv_block_size
    q = torch.randint(low, high, [batch, q_len, num_heads, head_size]).to(dtype=act_dtype)/high
    k = torch.randint(low, high, [batch, new_kv_len, num_kv_heads, head_size]).to(dtype=act_dtype)/high
    v = torch.randint(low, high, [batch, new_kv_len, num_kv_heads, head_size]).to(dtype=act_dtype)/high
    attention_mask = torch.full([batch, 1, q_len, kv_len], torch.finfo(act_dtype).min).to(dtype=act_dtype)
    
    ref = F.scaled_dot_product_attention(q, k[:,:,:kv_len,:], v[:,:,:kv_len,:], attention_mask, dropout_p=0.0, enable_gqa = enable_gqa)
    org = get_org(q, k[:,:,:kv_len,:], v[:,:,:kv_len,:], attention_mask)
    check_close(ref, org, atol=1e-3, rtol=1e-2)
    
    org1 = get_flash1(q,k,v,attention_mask)
    check_close(ref, org1, atol=1e-3, rtol=1e-2)
    print("org of get_flash1 passed !")

    org2 = get_flash2(q,k,v,attention_mask, real_kv_len=kv_len)
    check_close(ref, org2, atol=1e-3, rtol=1e-2)
    print("org of get_flash2 passed !")

    org = get_flash3(q,k,v,attention_mask)
    check_close(ref, org, atol=1e-3, rtol=1e-2)
    print("org of get_flash3 passed !")
    check_close(org, org2, atol=1e-3, rtol=1e-2)