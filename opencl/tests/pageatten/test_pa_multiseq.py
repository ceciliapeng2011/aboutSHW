import torch
import torch.nn as nn
import torch.nn.functional as F

import functools

from clops import cl
from clops.utils import Colors
import os

import numpy as np
import sys

from flashattn import get_flash0

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
    #round half to even
    kv_INT8 = torch.round((kv*kv_scale+kv_zp)).to(dtype=torch.uint8).reshape(blk_num,kv_heads,-1)
    # print("################################################################################")
    # print(f'KV\n:{kv.reshape(64, 16)}')
    # print(f'kv_INT8\n:{kv_INT8.reshape(64, 16)}')
    # print(f'kv_scale\n:{kv_scale.reshape( 1, 64)}')
    # print(f'kv_zp\n:{kv_zp.reshape( 1, 64)}')
    # print("################################################################################")

    dq_scale = (1.0/kv_scale).view(dtype=torch.uint8).reshape(blk_num,kv_heads,-1)
    kv_zp = kv_zp.view(dtype=torch.uint8).reshape(blk_num,kv_heads,-1)
    return torch.concat((kv_INT8, dq_scale, kv_zp), dim=-1)

def dequant_per_token(kv, head_size, blk_size):
    blk_num, kv_head_num, _ = kv.shape
    kv_u8 = kv[:,:,:head_size * blk_size].to(dtype=torch.float16).reshape(blk_num, kv_head_num, blk_size, head_size)
    kv_scale = kv[:,:,head_size * blk_size:head_size * blk_size + blk_size * 2].view(dtype=torch.float16).reshape(blk_num, kv_head_num, blk_size, 1)
    kv_zp = kv[:,:,head_size * blk_size + blk_size * 2:head_size * blk_size + blk_size * 4].view(dtype=torch.float16).reshape(blk_num, kv_head_num, blk_size, 1)

    # print("dequant_kv_u8 = ", kv_u8)
    # print("dequant_kv_scale = ", kv_scale.reshape(blk_num, kv_head_num, blk_size))
    # print("dequant_kv_zp    = ", kv_zp.reshape(blk_num, kv_head_num, blk_size))

    kv_dequant = torch.empty([blk_num, kv_head_num, blk_size, head_size], dtype=torch.float16)

    for m in range(blk_num):
        for n in range(kv_head_num):
            for i in range(blk_size):
                kv_dequant[m,n,i,:] = (kv_u8[m,n,i,:].to(dtype=torch.float16) - kv_zp[m,n,i,0].to(dtype=torch.float16)) * kv_scale[m,n,i,0].to(dtype=torch.float16)

    return kv_dequant

def ALIGN_UP(x, y):
    return (x + y -1) // y * y

def DIV_UP(x, y):
    return (x + y -1) // y


DUMP_ENQUEUE_ARGUMENTS = True
USE_RANDOM_MASK_BY_FORCE = True
class page_atten_cm:
    def __init__(self, num_heads, num_kv_heads, head_size, block_sz, trunk_sz, compressed_kvcache, is_causal = True):
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.is_causal = is_causal
        assert trunk_sz % block_sz == 0, f'Error: trunk_sz must be multiple of block_sz'
        self.block_sz = block_sz
        self.trunk_sz = trunk_sz
        self.compressed_kvcache = compressed_kvcache

        wg_size = 16
        q_step = CM_GRF_WIDTH // 32
        self.wg_seq_len = wg_size * q_step

        src1 = r'''#include "pa_multi_token.cm"'''
        cwd = os.path.dirname(os.path.realpath(__file__))
        print(f"compiling {cwd} {num_heads=} {head_size=}...")

        scale_factor = 1.0/(head_size**0.5)
        self.kernels = cl.kernels(src1,
                     (f'-cmc -Qxcm_jit_option="-abortonspill" -Qxcm_register_file_size=256  -mCM_printregusage -I{cwd}'
                      f' -DKERNEL_NAME=cm_page_attention'
                      f" -DCMFLA_NUM_HEADS={num_heads}"
                      f" -DCMFLA_NUM_KV_HEADS={num_kv_heads}"
                      f" -DCMFLA_HEAD_SIZE={head_size}"
                      f" -DCMFLA_SCALE_FACTOR={scale_factor}"
                      f" -DCMFLA_IS_CAUSAL={int(is_causal)}"
                      f" -DCMPA_BLOCK_SZ={self.block_sz}"
                      f" -DSPARSE_BLOCK_SIZE=1"
                      f" -DCMPA_WG_SEQ_LEN={int(self.wg_seq_len)}"
                      f" -DCMPA_KVCACHE_U8={int(compressed_kvcache)}"
                      f" -mdump_asm -g2")
                     )

    def __call__(self, q, k, v, n_repeats = 1):
        seq_len, _, head_size = q.shape
        padded_k = k
        padded_v = v
        old_dtype = q.dtype
        total_heads = (self.num_heads + self.num_kv_heads * 2)

        assert head_size == self.head_size
        #align seqlen with block_sz.  block is the PA K/V cache minimum unit.
        aligned_seqlen = seq_len
        #pad the K, V to align with block_sz
        if seq_len % self.block_sz != 0:
            padding_tokens = self.block_sz -  seq_len % self.block_sz
            assert len(k.shape) == 3
            kv_padding_dims = (0,0,0,0,0,padding_tokens)
            aligned_seqlen = seq_len + padding_tokens
            padded_k = torch.nn.functional.pad(k,kv_padding_dims, "constant", 1)
            padded_v = torch.nn.functional.pad(v,kv_padding_dims, "constant", 1)
            #padding all to NAN to simulate the NAN case  when fp16
            if self.compressed_kvcache == False:
                padded_k.view(torch.uint16)[seq_len:aligned_seqlen] = 0xfe00
                padded_v.view(torch.uint16)[seq_len:aligned_seqlen] = 0xfe00

        # print(f'k.shape:{k.shape}, padded_k.shape:{padded_k.shape}')
        # reorder K,V from [L, H, S] to [block_num, H, block_size, S]
        k_cache = padded_k.reshape(aligned_seqlen//self.block_sz, self.block_sz, self.num_kv_heads, self.head_size).transpose(1,2).contiguous()
        v_cache = padded_v.reshape(aligned_seqlen//self.block_sz, self.block_sz, self.num_kv_heads, self.head_size).transpose(1,2).contiguous()
        if self.compressed_kvcache:
            k_cache = quan_per_token(k_cache)
            v_cache = quan_per_token(v_cache)
        else:
            k_cache = k_cache.reshape(aligned_seqlen//self.block_sz, self.num_kv_heads, -1)
            v_cache = v_cache.reshape(aligned_seqlen//self.block_sz, self.num_kv_heads, -1)
        #output memory for the whole SDPA
        output = torch.zeros(seq_len, self.num_heads, self.head_size).to(torch.float16)
        blks_per_trunk = self.trunk_sz // self.block_sz
        assert aligned_seqlen % self.block_sz==0, f'Error: aligned_seqlen must be multiple of block_sz'
        # Q[L, H, S]
        # K/V: [blk_num, H, blk_sz, S]
        trunk_num = (aligned_seqlen+ self.trunk_sz - 1) // self.trunk_sz
        max_blks = aligned_seqlen // self.block_sz

        kv_dtype = torch.uint8 if self.compressed_kvcache else torch.half
        #extra half zp and half scale per token. totally 4 bytes.
        token_sz = (head_size+4) if self.compressed_kvcache else (head_size)

        cl.finish()

        for _ in range(n_repeats):
            for trunk_idx in range(trunk_num):
                # print(f'{Colors.GREEN}=============== {trunk_idx}/{trunk_num}  ==============={Colors.END}')
                blk_num = max_blks if blks_per_trunk*(trunk_idx + 1) > max_blks else blks_per_trunk*(trunk_idx + 1)
                block_indices =  torch.randperm(blk_num)
                # block_indices =  torch.arange(blk_num)
                # print(f'==============={block_indices=}')
                sub_k = torch.zeros(blk_num, self.num_kv_heads, self.block_sz*token_sz).to(kv_dtype)
                sub_v = torch.zeros(blk_num, self.num_kv_heads, self.block_sz*token_sz).to(kv_dtype)
                for i in  range(len(block_indices)):
                    sub_k[block_indices[i],:] = k_cache[i,:]
                    sub_v[block_indices[i],:] = v_cache[i,:]
                q_start = trunk_idx*self.trunk_sz
                q_end =  min(q_start + self.trunk_sz, seq_len)
                q_len = q_end - q_start
                sub_q = q[q_start:q_end, :]

                wg_size = 16
                wg_seq_len = self.wg_seq_len
                wg_count = (q_len + wg_seq_len - 1) // wg_seq_len

                t_q = cl.tensor(sub_q.to(torch.float16).detach().numpy())
                t_k= cl.tensor(sub_k.to(kv_dtype).detach().numpy())
                t_v = cl.tensor(sub_v.to(kv_dtype).detach().numpy())
                t_out = cl.tensor([q_len, self.num_heads, self.head_size], np.dtype(np.float16))

                GWS = [1, self.num_heads, int(wg_count * wg_size)]
                LWS = [1, 1, wg_size]
                # block_indices = int[blk_num], past_lens = 0, block_indices_begins = 0,
                past_lens=torch.tensor([trunk_idx*self.trunk_sz]).to(torch.int32)
                block_indices_begins=torch.tensor([0, blk_num]).to(torch.int32)
                subsequence_begins=torch.tensor([0,q_len]).to(torch.int32)

                t_block_indices=cl.tensor(block_indices.to(torch.int32).detach().numpy())
                t_past_lens=cl.tensor(past_lens.to(torch.int32).detach().numpy())
                t_block_indices_begins=cl.tensor(block_indices_begins.to(torch.int32).detach().numpy())
                t_subsequence_begins=cl.tensor(subsequence_begins.to(torch.int32).detach().numpy())

                self.kernels.enqueue(
                    "cm_page_attention",
                    GWS,
                    LWS,
                    t_q,
                    t_k,
                    t_v,
                    t_past_lens,
                    t_block_indices,
                    t_block_indices_begins,
                    t_subsequence_begins,
                    t_out,
                    q_len,
                )
                output[q_start:q_end] = torch.from_numpy(t_out.numpy())

        return output

    @staticmethod
    @functools.cache
    def create_instance(num_heads, num_kv_heads, head_size,block_sz, trunk_sz, compressed_kvcache, is_causal):
        return page_atten_cm(num_heads, num_kv_heads, head_size, block_sz, trunk_sz, compressed_kvcache, is_causal)

def flash_attn_vlen_ref(q, k, v, cu_seqlens, is_causal = True, attention_mask = None):
    seq_length, num_heads, head_size = q.shape
    kv_seq_length, num_kv_heads, head_size = k.shape
    old_dtype = q.dtype
    q = q.transpose(0, 1)
    k = k.transpose(0, 1)
    v = v.transpose(0, 1)

    # print(f"============2 {cu_seqlens=} {seq_length=} {num_heads=}")
    # print(f"============2 {q.shape=} {q.is_contiguous()=} {k.shape=} {k.is_contiguous()=} {v.shape=} {v.is_contiguous()=}")
    if attention_mask is not None:
        attn_output = F.scaled_dot_product_attention(
            q.unsqueeze(0).to(torch.float16), k.unsqueeze(0).to(torch.float16), v.unsqueeze(0).to(torch.float16),
            attn_mask = attention_mask,
            dropout_p=0.0,
            enable_gqa = (num_kv_heads != num_heads)
        )
        attn_output = attn_output.squeeze(0).transpose(0, 1)
        # print(f"============2 {attn_output.shape=} ")
        print(".")
        return attn_output.to(old_dtype)

    if is_causal:
        attn_output = F.scaled_dot_product_attention(
            q.unsqueeze(0).to(torch.float16), k.unsqueeze(0).to(torch.float16), v.unsqueeze(0).to(torch.float16),
            is_causal = True,
            dropout_p=0.0,
            enable_gqa = (num_kv_heads != num_heads)
        )
    else:
        attention_mask = torch.zeros([1, seq_length, seq_length], device=q.device, dtype=torch.bool)
        if len(cu_seqlens):
            for i in range(1, len(cu_seqlens)):
                attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = True
        else:
            attention_mask[...] = True

        attn_output = F.scaled_dot_product_attention(
            q.unsqueeze(0).to(torch.float16), k.unsqueeze(0).to(torch.float16), v.unsqueeze(0).to(torch.float16),
            attn_mask = attention_mask,
            dropout_p=0.0,
            enable_gqa = (num_kv_heads != num_heads)
        )
    attn_output = attn_output.squeeze(0).transpose(0, 1)
    # print(f"============2 {attn_output.shape=} ")
    print(".")
    return attn_output.to(old_dtype)

def get_attention_mask(q, k, v, is_causal = True):
    q_len, num_heads, head_size = q.shape
    kv_len, num_kv_heads, head_size = k.shape

    attention_mask = torch.full([num_heads, q_len, kv_len], True).to(dtype=torch.bool)
    if is_causal:
        causal_pattern = torch.tril(torch.ones(q_len, kv_len, dtype=torch.bool, device=q.device))
        causal_pattern = causal_pattern.unsqueeze(0).repeat_interleave(num_heads, dim=0)
        attention_mask = attention_mask & causal_pattern
    attention_mask = torch.where(attention_mask, 0, torch.finfo(q.dtype).min)
    attention_mask = attention_mask.unsqueeze(0)
    # print(f"============flash0 {attention_mask.shape=} {attention_mask.is_contiguous()=}")
    # print(f"============flash0 {attention_mask=}")
    return attention_mask

def check_close(input, other, atol=1e-2, rtol=1e-2):
    print(f"[check_close] {input.shape}{input.dtype} vs {other.shape}{other.dtype}")
    rtol_max = (((input - other).abs() - 1e-5)/other.abs())[other != 0].max()
    atol_max = (((input - other).abs()) - 1e-5*other.abs()).max()
    print(f"[check_close] rtol_max: {rtol_max}")
    print(f"[check_close] atol_max: {atol_max}")
    if not torch.allclose(input, other, atol=atol, rtol=rtol, equal_nan=True):
        close_check = torch.isclose(input, other, atol=atol, rtol=rtol)
        not_close_indices = torch.where(~close_check) # Invert the close check to find failures
        print(f"Not close indices: {not_close_indices}")
        print(f"    input_tensor: {input[not_close_indices]}")
        print(f"    other_tensor: {other[not_close_indices]}")
        assert 0

def test_page_attn_causal_batch1(seq_len, num_heads = 16, num_kv_heads = 16, head_size = 80, block_sz=128, trunk_sz=512, compressed_kvcache=False, check_acc = True):
    cl.profiling(True)
    torch.manual_seed(0)
    torch.set_printoptions(linewidth=1024)

    low = -1
    high = 2
    act_dtype = torch.float16
    q = torch.randint(low, high, [seq_len, num_heads, head_size]).to(dtype=act_dtype)

    if compressed_kvcache:
        k = torch.randint(low, high, [seq_len, num_kv_heads, head_size]).to(dtype=act_dtype) / 4.0
        k[0:seq_len:3, :, :] = (k[0:seq_len:3, :, :] + 0.25)/ 2.0
        v = torch.randint(low, high, [seq_len, num_kv_heads, head_size]).to(dtype=act_dtype)/high
        v[0:seq_len:3, :, :] = (v[0:seq_len:3, :, :] + 0.25)/ 2.0
    else:
        k = torch.rand(seq_len, num_kv_heads, head_size).to(dtype=act_dtype)
        v = torch.rand(seq_len, num_kv_heads, head_size).to(dtype=act_dtype)/high

    is_causal = True  # PageAttention implictly means causal_mask

    # // warmup
    pa_cm = page_atten_cm.create_instance(num_heads, num_kv_heads, head_size, block_sz, trunk_sz, compressed_kvcache, is_causal)
    out = pa_cm(q, k, v)
    latency = cl.finish()

    if check_acc:
        attention_mask = get_attention_mask(q, k, v, is_causal)
        ref = flash_attn_vlen_ref(q, k, v, [], is_causal, attention_mask)
        if torch.isinf(ref).any() or torch.isnan(ref).any():
            raise AssertionError("reference has inf or nan values")
        check_close(ref, out)

if __name__ == "__main__":

    test_page_attn_causal_batch1(256, num_heads = 2, num_kv_heads = 1, head_size = 32, block_sz=256, trunk_sz=256, compressed_kvcache=False, check_acc=True)
