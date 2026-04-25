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
from check_density import paired_adjacent_row_diff_pct, load_ov_model_block_mask, verify_block_mask_integrity
from generate_block_mask import generate_block_mask_with_ratio, count_false_percentage
from kv_cache_quant_utils import (
    DEFAULT_SUB_BLOCK_SIZE,
    dequant_per_channel,
    dequant_per_token,
    quant_per_channel as quan_per_channel,
    quant_per_token as quan_per_token,
)

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

KV_CACHE_COMPRESSION_NONE = 0
KV_CACHE_COMPRESSION_BY_TOKEN = 1
KV_CACHE_COMPRESSION_BY_CHANNEL = 2
def normalize_kv_cache_compression(mode):
    if isinstance(mode, bool):
        return KV_CACHE_COMPRESSION_BY_TOKEN if mode else KV_CACHE_COMPRESSION_NONE

    mode = int(mode)
    if mode not in {
        KV_CACHE_COMPRESSION_NONE,
        KV_CACHE_COMPRESSION_BY_TOKEN,
        KV_CACHE_COMPRESSION_BY_CHANNEL,
    }:
        raise ValueError(f"unsupported kv-cache compression mode: {mode}")
    return mode


def get_k_cache_layout(head_size, block_sz, compression_mode, sub_block_sz):
    if compression_mode == KV_CACHE_COMPRESSION_BY_TOKEN:
        return block_sz, head_size + 4
    if compression_mode == KV_CACHE_COMPRESSION_BY_CHANNEL:
        return block_sz + block_sz // sub_block_sz * 4, head_size
    return block_sz, head_size


def get_v_cache_layout(head_size, block_sz, compression_mode):
    if compression_mode != KV_CACHE_COMPRESSION_NONE:
        return block_sz, head_size + 4
    return block_sz, head_size


def ALIGN_UP(x, y):
    return (x + y -1) // y * y

def DIV_UP(x, y):
    return (x + y -1) // y


DUMP_ENQUEUE_ARGUMENTS = True
USE_RANDOM_MASK_BY_FORCE = True
class page_atten_cm:
    def __init__(self, num_heads, num_kv_heads, head_size, block_sz, trunk_sz, compressed_kvcache, sub_block_sz=DEFAULT_SUB_BLOCK_SIZE, is_causal = True, sparse_block_sz = 128):
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.is_causal = is_causal
        assert trunk_sz % block_sz == 0, f'Error: trunk_sz must be multiple of block_sz'
        self.block_sz = block_sz
        self.trunk_sz = trunk_sz
        self.sparse_block_sz = sparse_block_sz
        self.compressed_kvcache = normalize_kv_cache_compression(compressed_kvcache)
        self.sub_block_sz = sub_block_sz
        # assert sparse_block_sz == 1 or sparse_block_sz == 128 or sparse_block_sz == 256, f"unsupported sparse_block_sz:{sparse_block_sz}"

        wg_size = 16
        q_step = CM_GRF_WIDTH // 32
        self.wg_seq_len = wg_size * q_step

        src1 = r'''#include "pa_multi_token.cm"'''
        cwd = os.path.dirname(os.path.realpath(__file__))
        print(f"compiling {cwd} {num_heads=} {head_size=} {sparse_block_sz=}...")

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
                      f" -DSPARSE_BLOCK_SIZE={int(sparse_block_sz)}"
                      f" -DCMPA_WG_SEQ_LEN={int(self.wg_seq_len)}"
                      f" -DKV_CACHE_COMPRESSION={self.compressed_kvcache}"
                      f" -DSUB_BLOCK_SIZE={self.sub_block_sz}"
                      f" -mdump_asm -g2")
                     )

    def __call__(self, q, k, v, block_mask, n_repeats = 1):
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
            if self.compressed_kvcache == KV_CACHE_COMPRESSION_NONE:
                padded_k.view(torch.uint16)[seq_len:aligned_seqlen] = 0xfe00
                padded_v.view(torch.uint16)[seq_len:aligned_seqlen] = 0xfe00

        # print(f'k.shape:{k.shape}, padded_k.shape:{padded_k.shape}')
        # reorder K,V from [L, H, S] to [block_num, H, block_size, S]
        k_cache = padded_k.reshape(aligned_seqlen//self.block_sz, self.block_sz, self.num_kv_heads, self.head_size).transpose(1,2).contiguous()
        v_cache = padded_v.reshape(aligned_seqlen//self.block_sz, self.block_sz, self.num_kv_heads, self.head_size).transpose(1,2).contiguous()
        if self.compressed_kvcache == KV_CACHE_COMPRESSION_BY_TOKEN:
            k_cache = quan_per_token(k_cache)
            v_cache = quan_per_token(v_cache)
        elif self.compressed_kvcache == KV_CACHE_COMPRESSION_BY_CHANNEL:
            k_cache = quan_per_channel(k_cache, self.sub_block_sz)
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

        kv_dtype = torch.uint8 if self.compressed_kvcache != KV_CACHE_COMPRESSION_NONE else torch.half
        k_block_sz, k_token_sz = get_k_cache_layout(head_size, self.block_sz, self.compressed_kvcache, self.sub_block_sz)
        v_block_sz, v_token_sz = get_v_cache_layout(head_size, self.block_sz, self.compressed_kvcache)

        if self.sparse_block_sz > 1:
            block_mask_list = []
            block_mask_in_wg_list = []
            for trunk_idx in range(trunk_num):
                q_start = trunk_idx*self.trunk_sz
                q_end =  min(q_start + self.trunk_sz, seq_len)
                q_len = q_end - q_start

                wg_size = 16
                wg_seq_len = self.wg_seq_len
                wg_count = (q_len + wg_seq_len - 1) // wg_seq_len

                kv_len = trunk_idx*self.trunk_sz + q_len
                q_start_block = 0
                q_block_num = (q_len + self.sparse_block_sz -1) // self.sparse_block_sz
                kv_block_num = (kv_len + self.sparse_block_sz -1) // self.sparse_block_sz
                sub_block_mask = torch.zeros(self.num_heads, q_block_num, kv_block_num).to(torch.bool)
                sub_block_mask[...] = block_mask[trunk_idx, :, q_start_block : q_start_block + q_block_num, : kv_block_num]

                sub_block_mask_in_wg = torch.zeros(self.num_heads, wg_count, kv_block_num).to(torch.bool)

                for head_idx in range(0, self.num_heads):
                    for wg_id in range(0, wg_count):
                        # default skip all the blocks.
                        sub_block_mask_in_wg[head_idx, wg_id, :] = False
                        qblk_start_wg = wg_id*wg_seq_len // self.sparse_block_sz
                        qblk_end_wg = DIV_UP(min(wg_id*wg_seq_len + wg_seq_len, q_len), self.sparse_block_sz)
                        # print(f'############qblk_start_wg = {qblk_start_wg}, qblk_end_wg = {qblk_end_wg}, submask shape = {sub_block_mask.shape}')
                        for kv_blk_idx in range(0, kv_block_num):
                            for qblk_idx in range(qblk_start_wg, qblk_end_wg):
                                if sub_block_mask[head_idx, qblk_idx, kv_blk_idx] == True:
                                    sub_block_mask_in_wg[head_idx, wg_id, kv_blk_idx] = True
                # print(f'{sub_block_mask=}')
                # print(f'{sub_block_mask_in_wg=}')
                block_mask_list.append(sub_block_mask)
                block_mask_in_wg_list.append(sub_block_mask_in_wg)

        cl.finish()

        for _ in range(n_repeats):
            for trunk_idx in range(trunk_num):
                # print(f'{Colors.GREEN}=============== {trunk_idx}/{trunk_num}  ==============={Colors.END}')
                blk_num = max_blks if blks_per_trunk*(trunk_idx + 1) > max_blks else blks_per_trunk*(trunk_idx + 1)
                block_indices =  torch.randperm(blk_num)
                # block_indices =  torch.arange(blk_num)
                # print(f'==============={block_indices=}')
                sub_k = torch.zeros(blk_num, self.num_kv_heads, k_block_sz * k_token_sz).to(kv_dtype)
                sub_v = torch.zeros(blk_num, self.num_kv_heads, v_block_sz * v_token_sz).to(kv_dtype)
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

                # if self.sparse_block_sz > 1:
                #     kv_len = trunk_idx*self.trunk_sz + q_len
                #     q_start_block = 0
                #     q_block_num = (q_len + self.sparse_block_sz -1) // self.sparse_block_sz
                #     kv_block_num = (kv_len + self.sparse_block_sz -1) // self.sparse_block_sz
                #     sub_block_mask = torch.zeros(self.num_heads, q_block_num, kv_block_num).to(torch.bool)
                #     sub_block_mask[...] = block_mask[trunk_idx, :, q_start_block : q_start_block + q_block_num, : kv_block_num]

                #     sub_block_mask_in_wg = torch.zeros(self.num_heads, wg_count, kv_block_num).to(torch.bool)
                #     for head_idx in range(0, self.num_heads):
                #         for wg_id in range(0, wg_count):
                #             # default skip all the blocks.
                #             sub_block_mask_in_wg[head_idx, wg_id, :] = False
                #             qblk_start_wg = DIV_UP(wg_id*wg_seq_len, self.sparse_block_sz)
                #             qblk_end_wg = DIV_UP(min(wg_id*wg_seq_len + wg_seq_len, q_len), self.sparse_block_sz)
                #             # print(f'############qblk_start_wg = {qblk_start_wg}, qblk_end_wg = {qblk_end_wg}, submask shape = {sub_block_mask.shape}')
                #             for kv_blk_idx in range(0, kv_block_num):
                #                 for qblk_idx in range(qblk_start_wg, qblk_end_wg):
                #                     if sub_block_mask[head_idx, qblk_idx, kv_blk_idx] == True:
                #                         sub_block_mask_in_wg[head_idx, wg_id, kv_blk_idx] = True

                    # print(f"============ {sub_block_mask.shape=} {sub_block_mask.is_contiguous()=}")
                    # print(f"============ {sub_block_mask=}")

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

                # print(f"calling cm_page_attention {GWS=} {LWS=} x {n_repeats} times, q:[{q_start}, {q_end}], past_lens:{int(past_lens)}, kv_blk_num:{blk_num}, sparse_block_sz:{self.sparse_block_sz} kv_cache:{"U8" if self.compressed_kvcache else "F16"}")
                if self.sparse_block_sz > 1:
                    t_block_mask = cl.tensor(block_mask_list[trunk_idx].to(torch.bool).detach().numpy())
                    num_q_blocks = t_block_mask.shape[1]
                    num_k_blocks = t_block_mask.shape[2]
                else:
                    t_block_mask = None
                    num_q_blocks = None
                    num_k_blocks = None

                validate = True
                if self.sparse_block_sz > 1:
                    t_block_mask_in_wg  = cl.tensor(block_mask_in_wg_list[trunk_idx].to(torch.bool).detach().numpy())
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
                        t_block_mask,
                        t_block_mask_in_wg,
                        num_q_blocks,
                        num_k_blocks,
                    )
                else:
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


    def run_perf(self, q, k, v, block_mask, n_warmup: int = 20, n_iters: int = 200, deterministic_block_indices: bool = True):
        """Perf runner that avoids per-iter host<->device copies.

        Returns: list of per-enqueue durations (ns) from cl.finish().
        """
        seq_len, _, head_size = q.shape
        assert head_size == self.head_size

        padded_k = k
        padded_v = v

        aligned_seqlen = seq_len
        if seq_len % self.block_sz != 0:
            padding_tokens = self.block_sz - seq_len % self.block_sz
            kv_padding_dims = (0, 0, 0, 0, 0, padding_tokens)
            aligned_seqlen = seq_len + padding_tokens
            padded_k = torch.nn.functional.pad(k, kv_padding_dims, "constant", 1)
            padded_v = torch.nn.functional.pad(v, kv_padding_dims, "constant", 1)
            if self.compressed_kvcache == KV_CACHE_COMPRESSION_NONE:
                padded_k.view(torch.uint16)[seq_len:aligned_seqlen] = 0xfe00
                padded_v.view(torch.uint16)[seq_len:aligned_seqlen] = 0xfe00

        k_cache = padded_k.reshape(aligned_seqlen // self.block_sz, self.block_sz, self.num_kv_heads, self.head_size).transpose(1, 2).contiguous()
        v_cache = padded_v.reshape(aligned_seqlen // self.block_sz, self.block_sz, self.num_kv_heads, self.head_size).transpose(1, 2).contiguous()
        if self.compressed_kvcache == KV_CACHE_COMPRESSION_BY_TOKEN:
            k_cache = quan_per_token(k_cache)
            v_cache = quan_per_token(v_cache)
        elif self.compressed_kvcache == KV_CACHE_COMPRESSION_BY_CHANNEL:
            k_cache = quan_per_channel(k_cache, self.sub_block_sz)
            v_cache = quan_per_token(v_cache)
        else:
            k_cache = k_cache.reshape(aligned_seqlen // self.block_sz, self.num_kv_heads, -1)
            v_cache = v_cache.reshape(aligned_seqlen // self.block_sz, self.num_kv_heads, -1)

        blks_per_trunk = self.trunk_sz // self.block_sz
        trunk_num = (aligned_seqlen + self.trunk_sz - 1) // self.trunk_sz
        max_blks = aligned_seqlen // self.block_sz

        kv_dtype = torch.uint8 if self.compressed_kvcache != KV_CACHE_COMPRESSION_NONE else torch.half
        k_block_sz, k_token_sz = get_k_cache_layout(head_size, self.block_sz, self.compressed_kvcache, self.sub_block_sz)
        v_block_sz, v_token_sz = get_v_cache_layout(head_size, self.block_sz, self.compressed_kvcache)

        # Precompute per-trunk masks once (CPU)
        block_mask_list = None
        block_mask_in_wg_list = None
        if self.sparse_block_sz > 1:
            block_mask_list = []
            block_mask_in_wg_list = []
            for trunk_idx in range(trunk_num):
                q_start = trunk_idx * self.trunk_sz
                q_end = min(q_start + self.trunk_sz, seq_len)
                q_len = q_end - q_start

                wg_size = 16
                wg_seq_len = self.wg_seq_len
                wg_count = (q_len + wg_seq_len - 1) // wg_seq_len

                kv_len = trunk_idx * self.trunk_sz + q_len
                q_block_num = (q_len + self.sparse_block_sz - 1) // self.sparse_block_sz
                kv_block_num = (kv_len + self.sparse_block_sz - 1) // self.sparse_block_sz

                sub_block_mask = torch.zeros(self.num_heads, q_block_num, kv_block_num, dtype=torch.bool)
                sub_block_mask[...] = block_mask[trunk_idx, :, 0:0 + q_block_num, :kv_block_num]

                sub_block_mask_in_wg = torch.zeros(self.num_heads, wg_count, kv_block_num, dtype=torch.bool)

                for head_idx in range(0, self.num_heads):
                    for wg_id in range(0, wg_count):
                        sub_block_mask_in_wg[head_idx, wg_id, :] = False
                        qblk_start_wg = wg_id * wg_seq_len // self.sparse_block_sz
                        qblk_end_wg = DIV_UP(min(wg_id * wg_seq_len + wg_seq_len, q_len), self.sparse_block_sz)
                        for kv_blk_idx in range(0, kv_block_num):
                            for qblk_idx in range(qblk_start_wg, qblk_end_wg):
                                if bool(sub_block_mask[head_idx, qblk_idx, kv_blk_idx].item()):
                                    sub_block_mask_in_wg[head_idx, wg_id, kv_blk_idx] = True
                block_mask_list.append(sub_block_mask)
                block_mask_in_wg_list.append(sub_block_mask_in_wg)

        # Pre-create per-trunk device buffers and scalars
        per_trunk_args = []
        for trunk_idx in range(trunk_num):
            blk_num = max_blks if blks_per_trunk * (trunk_idx + 1) > max_blks else blks_per_trunk * (trunk_idx + 1)
            if deterministic_block_indices:
                block_indices = torch.arange(blk_num, dtype=torch.int64)
            else:
                block_indices = torch.randperm(blk_num)

            sub_k = torch.zeros(blk_num, self.num_kv_heads, k_block_sz * k_token_sz, dtype=kv_dtype)
            sub_v = torch.zeros(blk_num, self.num_kv_heads, v_block_sz * v_token_sz, dtype=kv_dtype)
            for i in range(len(block_indices)):
                sub_k[block_indices[i], :] = k_cache[i, :]
                sub_v[block_indices[i], :] = v_cache[i, :]

            q_start = trunk_idx * self.trunk_sz
            q_end = min(q_start + self.trunk_sz, seq_len)
            q_len = q_end - q_start
            sub_q = q[q_start:q_end, :]

            wg_size = 16
            wg_seq_len = self.wg_seq_len
            wg_count = (q_len + wg_seq_len - 1) // wg_seq_len

            GWS = [1, self.num_heads, int(wg_count * wg_size)]
            LWS = [1, 1, wg_size]

            past_lens = torch.tensor([trunk_idx * self.trunk_sz], dtype=torch.int32)
            block_indices_begins = torch.tensor([0, blk_num], dtype=torch.int32)
            subsequence_begins = torch.tensor([0, q_len], dtype=torch.int32)

            t_q = cl.tensor(sub_q.to(torch.float16).detach().numpy())
            t_k = cl.tensor(sub_k.detach().numpy())
            t_v = cl.tensor(sub_v.detach().numpy())
            t_out = cl.tensor([q_len, self.num_heads, self.head_size], np.dtype(np.float16))

            t_block_indices = cl.tensor(block_indices.to(torch.int32).detach().numpy())
            t_past_lens = cl.tensor(past_lens.detach().numpy())
            t_block_indices_begins = cl.tensor(block_indices_begins.detach().numpy())
            t_subsequence_begins = cl.tensor(subsequence_begins.detach().numpy())

            if self.sparse_block_sz > 1:
                t_block_mask = cl.tensor(block_mask_list[trunk_idx].detach().numpy())
                num_q_blocks = t_block_mask.shape[1]
                num_k_blocks = t_block_mask.shape[2]
            else:
                t_block_mask = None
                num_q_blocks = None
                num_k_blocks = None

            if self.sparse_block_sz > 1:
                t_block_mask_in_wg = cl.tensor(block_mask_in_wg_list[trunk_idx].detach().numpy())
                per_trunk_args.append(
                    (GWS, LWS, t_q, t_k, t_v, t_past_lens, t_block_indices, t_block_indices_begins, t_subsequence_begins, t_out,
                     q_len, t_block_mask, t_block_mask_in_wg, num_q_blocks, num_k_blocks)
                )
            else:
                per_trunk_args.append(
                    (GWS, LWS, t_q, t_k, t_v, t_past_lens, t_block_indices, t_block_indices_begins, t_subsequence_begins, t_out,
                     q_len)
                )

        # Flush any setup copies so timing only reflects kernels
        cl.finish()

        # Warmup
        for _ in range(n_warmup):
            for args in per_trunk_args:
                if self.sparse_block_sz > 1:
                    (GWS, LWS, t_q, t_k, t_v, t_past_lens, t_block_indices, t_block_indices_begins, t_subsequence_begins, t_out,
                     q_len, t_block_mask, t_block_mask_in_wg, nq, nk) = args
                    self.kernels.enqueue("cm_page_attention", GWS, LWS, t_q, t_k, t_v, t_past_lens, t_block_indices, t_block_indices_begins, t_subsequence_begins, t_out, q_len, t_block_mask, t_block_mask_in_wg, nq, nk)
                else:
                    (GWS, LWS, t_q, t_k, t_v, t_past_lens, t_block_indices, t_block_indices_begins, t_subsequence_begins, t_out,
                     q_len) = args
                    self.kernels.enqueue("cm_page_attention", GWS, LWS, t_q, t_k, t_v, t_past_lens, t_block_indices, t_block_indices_begins, t_subsequence_begins, t_out, q_len)
        cl.finish()

        # Timed
        for _ in range(n_iters):
            for args in per_trunk_args:
                if self.sparse_block_sz > 1:
                    (GWS, LWS, t_q, t_k, t_v, t_past_lens, t_block_indices, t_block_indices_begins, t_subsequence_begins, t_out,
                     q_len, t_block_mask, t_block_mask_in_wg, nq, nk) = args
                    self.kernels.enqueue("cm_page_attention", GWS, LWS, t_q, t_k, t_v, t_past_lens, t_block_indices, t_block_indices_begins, t_subsequence_begins, t_out, q_len, t_block_mask, t_block_mask_in_wg, nq, nk)
                else:
                    (GWS, LWS, t_q, t_k, t_v, t_past_lens, t_block_indices, t_block_indices_begins, t_subsequence_begins, t_out,
                     q_len) = args
                    self.kernels.enqueue("cm_page_attention", GWS, LWS, t_q, t_k, t_v, t_past_lens, t_block_indices, t_block_indices_begins, t_subsequence_begins, t_out, q_len)

        return cl.finish()

    @staticmethod
    @functools.cache
    def create_instance(num_heads, num_kv_heads, head_size,block_sz, trunk_sz, compressed_kvcache, sub_block_sz=DEFAULT_SUB_BLOCK_SIZE, is_causal=True, sparse_block_sz=128):
        return page_atten_cm(num_heads, num_kv_heads, head_size, block_sz, trunk_sz, compressed_kvcache, sub_block_sz, is_causal, sparse_block_sz)

# sparse to dense mask
def block_mask_to_attention_mask(block_mask: torch.Tensor, q_len: int, kv_len: int, sparse_block_size: int, trunk_sz: int) -> torch.Tensor:
    # block_mask shape [num_trunks, num_head, q_block_num, k_block_num] dtype bool ->
    # attention_mask shape [num_head, q_len, kv_len] dtype bool
    trunk_num, num_head, q_block_num, kv_block_num = block_mask.shape
    assert((q_len + trunk_sz -1) // trunk_sz == trunk_num)
    assert((trunk_sz + sparse_block_size -1) // sparse_block_size == q_block_num)
    assert((kv_len + sparse_block_size -1) // sparse_block_size == kv_block_num)

    attention_mask = torch.zeros([num_head, q_len, kv_len], device=block_mask.device, dtype=torch.bool)
    for trunk_idx in range(trunk_num):
        expanded = block_mask[trunk_idx].repeat_interleave(sparse_block_size, dim=1)
        expanded = expanded.repeat_interleave(sparse_block_size, dim=2)
        q_start = trunk_idx * trunk_sz
        q_end = min(q_start + trunk_sz, q_len)
        remaining = q_end - q_start
        attention_mask[ : , q_start : q_end, : ] = expanded[:, :remaining, :kv_len]

    return attention_mask


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

def get_attention_mask(q, k, v, approx_simple_mask, sparse_block_size, trunk_sz, is_causal = True):
    q_len, num_heads, head_size = q.shape
    kv_len, num_kv_heads, head_size = k.shape

    # [num_head, q_len, kv_len] dtype bool ->
    # [1, num_head, q_len, kv_len] dtype float16
    if sparse_block_size > 1:
        attention_mask = block_mask_to_attention_mask(approx_simple_mask, q_len, kv_len, sparse_block_size, trunk_sz)
    else:
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

def flash0_ref(q, k, v, attention_mask):
    attn_output = get_flash0(
            q.unsqueeze(0).to(torch.float16), k.unsqueeze(0).to(torch.float16), v.unsqueeze(0).to(torch.float16),
            attention_mask = attention_mask)
    attn_output = attn_output.squeeze(0)
    # print(f"============2 {attn_output.shape=} ")
    print(".")
    return attn_output.to(q.dtype)


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

def test_page_attn_causal_batch1(seq_len, num_heads = 16, num_kv_heads = 16, head_size = 80, block_sz=128, trunk_sz=512, compressed_kvcache=KV_CACHE_COMPRESSION_NONE, sub_block_sz=DEFAULT_SUB_BLOCK_SIZE, sparse_block_sz=128, density=0.5, check_acc = True, return_output = False):
    cl.profiling(True)
    torch.manual_seed(0)
    torch.set_printoptions(linewidth=1024)

    compressed_kvcache = normalize_kv_cache_compression(compressed_kvcache)

    low = -1
    high = 2
    act_dtype = torch.float16
    q = torch.randint(low, high, [seq_len, num_heads, head_size]).to(dtype=act_dtype)

    if compressed_kvcache != KV_CACHE_COMPRESSION_NONE:
        k = torch.randint(low, high, [seq_len, num_kv_heads, head_size]).to(dtype=act_dtype) / 4.0
        k[0:seq_len:3, :, :] = (k[0:seq_len:3, :, :] + 0.25)/ 2.0
        v = torch.randint(low, high, [seq_len, num_kv_heads, head_size]).to(dtype=act_dtype)/high
        v[0:seq_len:3, :, :] = (v[0:seq_len:3, :, :] + 0.25)/ 2.0
    else:
        k = torch.rand(seq_len, num_kv_heads, head_size).to(dtype=act_dtype)
        v = torch.rand(seq_len, num_kv_heads, head_size).to(dtype=act_dtype)/high

    # Generate approximate sparse block mask
    is_causal = True  # PageAttention implictly means causal_mask
    requested_density = density
    if sparse_block_sz > 1:
        if seq_len != 32768 or num_heads != 32 or sparse_block_sz != 256 or USE_RANDOM_MASK_BY_FORCE == True:
            approx_simple_mask, effective_density = generate_block_mask_with_ratio(num_heads, seq_len, trunk_sz, sparse_block_sz, requested_density, is_causal)
        else:
            assert seq_len == 32768 and num_heads == 32 and sparse_block_sz == 256, "only support 32k prompt!"
            block_mask_tmp = load_ov_model_block_mask('/home/ceciliapeng/toolbox/linux/xattn_thresh0.99/dump_xattn_mask_bs256', sparse_block_sz, seq_len, 4096, 36, num_heads)
            # L03 H04 | density=98.36%  'xattn_thresh0.99/dump_xattn_mask_bs256'
            # L02 H15 | density=66.35%  'xattn_thresh0.99/dump_xattn_mask_bs256'
            # L02 H11 | density=33.12%  'xattn_thresh0.99/dump_xattn_mask_bs256'
            # L14 H01 | density=11.45%  'xattn_thresh0.99/dump_xattn_mask_bs256'
            L, H, Q, K = block_mask_tmp.shape
            trunk_num = (seq_len + trunk_sz - 1) // trunk_sz
            assert Q % trunk_num == 0, "Q should be divisible by trunk_num for correct reshaping"
            if requested_density >= 1.0:
                approx_simple_mask = torch.full([1, 1, Q, K], True).to(dtype=torch.bool)
            elif requested_density > 0.9:
                approx_simple_mask = block_mask_tmp[3, 4, ...]
            elif requested_density > 0.6:
                approx_simple_mask = block_mask_tmp[2, 15, ...]
            elif requested_density > 0.3:
                approx_simple_mask = block_mask_tmp[2, 11, ...]
            else:
                approx_simple_mask = block_mask_tmp[14, 1, ...]
            approx_simple_mask = approx_simple_mask.reshape(trunk_num, 1, Q//trunk_num, K).repeat(1, num_heads, 1, 1)
            for h in range(num_heads):
                verify_block_mask_integrity(approx_simple_mask[:, h:h+1, :, :])
            percentage = count_false_percentage(approx_simple_mask, is_stacked_by_trunk=True)
            # print(f"Percentage of False elements: {percentage:.2f}%")
            # print(f"============ block_mask.shape={approx_simple_mask.shape}")
            # print(f"============ block_mask={approx_simple_mask}")
            effective_density = 1.0 - percentage / 100.0
    else:
        approx_simple_mask, effective_density = None, 1.0

    if sparse_block_sz == 128 and False:
        pct_per_pair, mean_pct, max_pct = paired_adjacent_row_diff_pct(approx_simple_mask)
        L, H, Q, K = approx_simple_mask.shape
        print(f"block_mask shape={approx_simple_mask.shape}, num_pairs={Q//2} (dropping last row if Q is odd)")
        for l in range(L):
            for h in range(H):
                print(f"L{l:02d} H{h:02d} | mean={mean_pct[l,h].item():.2f}% max={max_pct[l,h].item():.2f}%")
                # If you want per-pair detail:
                # for p in range(pct_per_pair.shape[2]):
                #     q0, q1 = 2*p, 2*p + 1
                #     print(f"  pair rows ({q0},{q1}) : {pct_per_pair[l,h,p].item():.2f}%")

    # // warmup
    pa_cm = page_atten_cm.create_instance(num_heads, num_kv_heads, head_size, block_sz, trunk_sz, compressed_kvcache, sub_block_sz, is_causal, sparse_block_sz)
    out = pa_cm(q, k, v, approx_simple_mask)
    latency = cl.finish()

    if check_acc:
        attention_mask = get_attention_mask(q, k, v, approx_simple_mask, sparse_block_sz, trunk_sz)
        ref = flash_attn_vlen_ref(q, k, v, [], is_causal, attention_mask)
        if torch.isinf(ref).any() or torch.isnan(ref).any():
            raise AssertionError("reference has inf or nan values")
        # ref0 = flash0_ref(q, k, v, attention_mask)
        # check_close(ref, ref0, atol=1e-2, rtol=1e-3)
        # print(ref)
        # print(out)
        if compressed_kvcache == KV_CACHE_COMPRESSION_NONE and sparse_block_sz == 128:
            check_close(ref, out, atol=5e-2, rtol=2e-1)
        else:
            check_close(ref, out)
    else:
        roofline = 293.27 if compressed_kvcache != KV_CACHE_COMPRESSION_NONE else 293.20
        warmup = 5
        rep = 15
        latency = pa_cm.run_perf(q, k, v, approx_simple_mask, n_warmup=warmup, n_iters=rep, deterministic_block_indices=True)
        # pa_cm(q, k, v, approx_simple_mask, n_repeats = rep)
        # latency = cl.finish()
        num_trunks = len(latency) // rep if rep > 0 else 0
        trunk_lat = []
        # for i,ns in enumerate(latency): print(f"[{i}]  {ns*1e-6:.3f} ms")
        if rep >= 1:
            # print(f'====================================================================================')
            total_flops = 1 * num_heads * seq_len * seq_len * head_size * 2 * effective_density

            if num_trunks > 0:
                total_lat_ms = [
                    sum(latency[i * num_trunks + t] for t in range(num_trunks)) * 1e-6
                    for i in range(len(latency) // num_trunks)
                ]
            else:
                total_lat_ms = []

            if len(total_lat_ms):
                total_min = min(total_lat_ms)
                total_max = max(total_lat_ms)
                total_avg = sum(total_lat_ms) / rep
            else:
                total_min = total_max = total_avg = 0.0
            mfu = total_flops / (total_avg * 1e6) if total_avg > 0 else 0.0
            meet = (roofline * effective_density / total_avg) if total_avg > 0 else 0.0
            density_note = (
                f"density req/eff {requested_density:.2f}/{effective_density:.2f}"
            )
            print(
                f"[total]: PA_causal {sparse_block_sz=} {seq_len=} , {num_trunks=}, {density_note}, compressKVCache {compressed_kvcache}, "
                f"MFU {mfu:,.0f} GFLOPS, latency(ms): min={total_min:.3f} avg={total_avg:.3f} max={total_max:.3f}, "
                f"meet: {meet:.2f}"
            )
            # print(f'====================================================================================')

    if return_output:
        return out

def test_ov():
    cl.profiling(True)
    torch.manual_seed(0)
    torch.set_printoptions(linewidth=1024)
    def get_tensor(name, dtype=np.float16):
        with open(name, 'rb') as f:
            data = f.read()
            np_data = np.frombuffer(data, dtype=dtype).copy()
            return torch.from_numpy(np_data)

    def check_tril(block_mask: torch.Tensor):
        """
        Checks if each [H, W] slice in a [B, H, W] boolean tensor is lower triangular.
        Returns a list of coordinates where the mask is True above the diagonal.
        """
        B, H, W = block_mask.shape
        assert H == W, "Only square matrices can be checked for tril structure."

        # Create a mask for upper triangular part (excluding diagonal)
        upper_tri_mask = torch.triu(torch.ones(H, W, dtype=torch.bool), diagonal=1)

        violations = []
        for b in range(B):
            # Get the upper triangle violations for this batch
            violation_coords = torch.nonzero(block_mask[b] & upper_tri_mask, as_tuple=False)
            for coord in violation_coords:
                h, w = coord.tolist()
                violations.append((b, h, w))

        if violations:
            print("Not a tril. Violations found at coordinates:")
            for v in violations:
                print(f"Batch {v[0]}, Row {v[1]}, Col {v[2]}")
        else:
            print("All slices are lower triangular.")

        return violations

    def check_tril_all(tensor_dir, file_suffix, block_mask_shape):
        files_checked = 0
        is_tril = True
        for filename in os.listdir(tensor_dir):
            if filename.endswith(file_suffix):
                file_path = os.path.join(tensor_dir, filename)
                block_mask  = get_tensor(file_path, dtype=np.int8).reshape(block_mask_shape)
                print(f'{block_mask.shape=}')
                B, H, W = block_mask.shape
                violations = check_tril(block_mask[:,:,:H])
                is_tril &= (len(violations) == 0)
                files_checked += 1
        print(f'checked {files_checked} files')
        return is_tril & files_checked > 0

    compressed_kvcache = KV_CACHE_COMPRESSION_NONE
    sub_block_sz = DEFAULT_SUB_BLOCK_SIZE
    xattn_thresh = 0.9
    sparse_block_sz, kv_block_size, trunk_sz = 256, 256, 4096 # trunk_sz no use
    num_heads, num_kv_heads, head_size = 32, 8, 128
    base = '/home/ceciliapeng/dump_debug_binary/'

    key_block_sz, key_token_sz = get_k_cache_layout(head_size, kv_block_size, compressed_kvcache, sub_block_sz)
    value_block_sz, value_token_sz = get_v_cache_layout(head_size, kv_block_size, compressed_kvcache)

    query = get_tensor(base + 'program1_network1_0_pagedattentionextension_PagedAttentionExtension_28408_src0__f16__612_4096_1_1__bfyx.bin').reshape([612, num_heads*head_size])
    key_cache = get_tensor(
        base + 'program1_network1_0_pagedattentionextension_PagedAttentionExtension_28408_updated_src_3__f16__3_8_256_128__bfyx.bin',
        np.int8 if compressed_kvcache != KV_CACHE_COMPRESSION_NONE else np.float16,
    ).reshape([-1, num_kv_heads, key_block_sz, key_token_sz])
    value_cache = get_tensor(
        base + 'program1_network1_0_pagedattentionextension_PagedAttentionExtension_28408_updated_src_4__f16__3_8_256_128__bfyx.bin',
        np.int8 if compressed_kvcache != KV_CACHE_COMPRESSION_NONE else np.float16,
    ).reshape([-1, num_kv_heads, value_block_sz, value_token_sz])
    # low = -1
    # high = 2
    # act_dtype = torch.float16
    # query = torch.ones(*query.shape).to(dtype=act_dtype)
    # key_cache = torch.ones(*key_cache.shape).to(dtype=act_dtype)
    # value_cache = torch.ones(*value_cache.shape).to(dtype=act_dtype)/high

    q_len = query.shape[0]
    valid_num_blks = key_cache.shape[0] - 1 # genai usually generates one more blocks than required
    valid_num_blks = key_cache.shape[0]
    q_block_pad = (q_len + sparse_block_sz - 1) // sparse_block_sz

    # check scale and zp of last head
    def show_scales_zp():
        blk_num, kv_heads, *_ = key_cache.shape
        key_cache_zps = key_cache.to(dtype=torch.uint8).reshape(blk_num,kv_heads,-1)[:,:,kv_block_size*head_size:].view(dtype=torch.half).reshape(blk_num,kv_heads,-1)
        print(f'{key_cache_zps.shape=}')   # [blk_num, num_kv_heads, kv_block_size*2]
        # key_cache_zps = torch.from_numpy(key_cache[0, -1,:-1, -4:].numpy().view(np.float16))
        # value_cache_zps = torch.from_numpy(value_cache[0, -1, :-1, -4:].numpy().view(np.float16))
        print(f"{key_cache_zps=}")

        value_cache_zps = value_cache.to(dtype=torch.uint8).reshape(blk_num,kv_heads,-1)[:,:,kv_block_size*head_size:].view(dtype=torch.half).reshape(blk_num,kv_heads,-1)
        print(f'{value_cache_zps.shape=}') 
        print(f"{value_cache_zps=}")

        def check_sanity(kv_cache_zps):
            nan_mask = torch.isnan(kv_cache_zps)
            inf_mask = torch.isinf(kv_cache_zps)

            # Indices where NaN or Inf occur (as [blk, head, pos])
            nan_indices = torch.nonzero(nan_mask, as_tuple=False)
            inf_indices = torch.nonzero(inf_mask, as_tuple=False)

            print("NaN indices (up to first 20):", nan_indices[:20])
            print("Inf indices (up to first 20):", inf_indices[:20])

            if inf_indices.numel():
                print(f'{kv_cache_zps[:,-1,:]=}')
        check_sanity(key_cache_zps)
        check_sanity(value_cache_zps)

    if compressed_kvcache != KV_CACHE_COMPRESSION_NONE:
        show_scales_zp()

    block_mask  = get_tensor(base + 'program1_network1_0_pagedattentionextension_PagedAttentionExtension_28408_intermediates_4__boolean__1536_1_1_1__bfyx.bin', dtype=np.int8).reshape([num_heads, q_block_pad, -1])
    block_mask_in_wg  = get_tensor(base + 'program1_network1_0_pagedattentionextension_PagedAttentionExtension_28408_intermediates_5__boolean__1536_1_1_1__bfyx.bin', dtype=np.int8).reshape([num_heads, -1, block_mask.shape[-1]])
    for i in range(num_heads):
        print(f'{i}:{block_mask[i,:,:]}')
    # block_mask = torch.ones(block_mask.shape, dtype=torch.bool)
    # block_mask_in_wg = torch.ones(block_mask_in_wg.shape, dtype=torch.bool)

    past_lens = get_tensor(base + 'program1_network1_0_pagedattentionextension_PagedAttentionExtension_28408_src5__i32__1_1_1_1__bfyx.bin', dtype=np.int32).reshape([1])
    subsequence_begins = get_tensor(base + 'program1_network1_0_pagedattentionextension_PagedAttentionExtension_28408_src6__i32__2_1_1_1__bfyx.bin', dtype=np.int32).reshape([2])
    block_indices = get_tensor(base + 'program1_network1_0_pagedattentionextension_PagedAttentionExtension_28408_src7__i32__3_1_1_1__bfyx.bin', dtype=np.int32).reshape([valid_num_blks])
    block_indices_begins = get_tensor(base + 'program1_network1_0_pagedattentionextension_PagedAttentionExtension_28408_src8__i32__2_1_1_1__bfyx.bin', dtype=np.int32).reshape([2])

    print(f'{past_lens=}, {subsequence_begins=}, {block_indices=}, {block_indices_begins=}')

    full_dense = False
    if xattn_thresh >= 1:
        full_dense = check_tril_all(base, "_intermediates_4__boolean__4096_1_1_1__bfyx.bin", block_mask.shape)
        assert full_dense, "SHOULD be full dense if XAttn thresh larger than 1.0"

    ov_out = get_tensor(base + 'program1_network1_0_pagedattentionextension_PagedAttentionExtension_28408_dst0__f16__612_4096_1_1__bfyx.bin').reshape([q_len, num_heads*head_size])

    # q [q_len, num_heads*head_size], k/v cache [num_blks, num_kv_heads, kv_block_size, head_size]
    print(f'{query.shape = }, {key_cache.shape = }, {value_cache.shape = },  {ov_out.shape = }')
    # print(f'{block_indices=}')

    is_causal = True
    pa_cm = page_atten_cm.create_instance(num_heads, num_kv_heads, head_size, kv_block_size, trunk_sz, compressed_kvcache, sub_block_sz, is_causal, sparse_block_sz)

    t_query = cl.tensor(query.detach().numpy())
    t_key_cache = cl.tensor(key_cache.detach().numpy())
    t_value_cache = cl.tensor(value_cache.detach().numpy())
    t_block_indices = cl.tensor(block_indices.to(torch.int32).detach().numpy())
    t_block_indices_begins = cl.tensor(block_indices_begins.to(torch.int32).detach().numpy())
    t_past_lens = cl.tensor(past_lens.to(torch.int32).detach().numpy())
    t_subsequence_begins = cl.tensor(subsequence_begins.to(torch.int32).detach().numpy())

    output = torch.zeros(q_len, num_heads*head_size).to(torch.float16)
    output = ov_out.clone()
    # has_inf_or_nan = (torch.isinf(output) | torch.isnan(output)).any()
    # print("output: Contains inf or nan:", has_inf_or_nan)
    t_out = cl.tensor(output.detach().numpy())

    wg_size = 16
    q_step = CM_GRF_WIDTH // 32 # or 8 on Xe1
    wg_seq_len = wg_size * q_step
    wg_count = (q_len + wg_seq_len - 1) // wg_seq_len
    GWS = [1, pa_cm.num_heads, int(wg_count * wg_size)]
    LWS = [1, 1, wg_size]

    # print(f"calling cm_page_attention {GWS=} {LWS=} sparse_block_sz:{pa_cm.sparse_block_sz} kv_cache:{"U8" if pa_cm.compressed_kvcache else "F16"}")
    if pa_cm.sparse_block_sz > 1:
        t_block_mask = cl.tensor(block_mask.to(torch.bool).detach().numpy())
        num_q_blocks = t_block_mask.shape[1]
        num_k_blocks = t_block_mask.shape[2]
    else:
        t_block_mask = None
        num_q_blocks = None
        num_k_blocks = None

    if pa_cm.sparse_block_sz > 1 and wg_seq_len != pa_cm.sparse_block_sz:
        t_block_mask_in_wg  = cl.tensor(block_mask_in_wg.to(torch.bool).detach().numpy())

    if DUMP_ENQUEUE_ARGUMENTS:
        LABEL_WIDTH = 32
        cltensors = [
            ("t_query",                t_query),
            ("t_key_cache",            t_key_cache),
            ("t_value_cache",          t_value_cache),
            ("t_past_lens",            t_past_lens),
            ("t_block_indices",        t_block_indices),
            ("t_block_indices_begins", t_block_indices_begins),
            ("t_subsequence_begins",   t_subsequence_begins),
            ("t_out",                  t_out),
            ("t_block_mask",           t_block_mask),
            ("t_block_mask_in_wg",     t_block_mask_in_wg) if pa_cm.sparse_block_sz > 1 else None,
        ]
        lines = [(name, value.numel * value.dtype.itemsize) for name, value in cltensors if value is not None]
        print("cm_page_attention size of memories:")
        for name, value in lines:
            print(f"  {name:<{LABEL_WIDTH}} {value}")
        print("\ncm_page_attention scalers:")
        print(f"  q_len:{q_len:<10}  mask_W:{num_q_blocks:<10}  "
            f"mask_H:{num_k_blocks:<10}  block_sz:{pa_cm.sparse_block_sz}")

    if pa_cm.sparse_block_sz > 1:
        if wg_seq_len != pa_cm.sparse_block_sz:
            t_block_mask_in_wg = cl.tensor(block_mask_in_wg.to(torch.bool).detach().numpy())
        else:
            t_block_mask_in_wg = cl.tensor(np.ones((1, 1, 1), dtype=np.bool_))
        pa_cm.kernels.enqueue(
            "cm_page_attention",
            GWS,
            LWS,
            t_query,
            t_key_cache,
            t_value_cache,
            t_past_lens,
            t_block_indices,
            t_block_indices_begins,
            t_subsequence_begins,
            t_out,
            q_len,
            t_block_mask,
            t_block_mask_in_wg,
            num_q_blocks,
            num_k_blocks,
        )
    else:
        pa_cm.kernels.enqueue(
            "cm_page_attention",
            GWS,
            LWS,
            t_query,
            t_key_cache,
            t_value_cache,
            t_past_lens,
            t_block_indices,
            t_block_indices_begins,
            t_subsequence_begins,
            t_out,
            q_len,
        )
    latency = cl.finish()

    ut_out = torch.from_numpy(t_out.numpy().reshape(-1, num_heads, head_size))
    ov_out = ov_out.reshape(-1, num_heads, head_size)

    # print(f"{ov_out[0, -1, :] = }")
    # print(f"{ut_out[0, -1, :] = }")

    has_inf_or_nan = (torch.isinf(ut_out) | torch.isnan(ut_out)).any()
    print("ut_out: Contains inf or nan:", has_inf_or_nan)
    has_inf_or_nan = (torch.isinf(ov_out) | torch.isnan(ov_out)).any()
    print("ov_out: Contains inf or nan:", has_inf_or_nan)
    check_close(ut_out, ov_out)
    sys.exit(0)

    enable_dequant_check = compressed_kvcache != KV_CACHE_COMPRESSION_NONE
    if enable_dequant_check: # TODO: there is bug in this check?
        if compressed_kvcache == KV_CACHE_COMPRESSION_BY_CHANNEL:
            k_dequan = dequant_per_channel(key_cache.reshape(-1, num_kv_heads, key_block_sz * key_token_sz), head_size, kv_block_size, sub_block_sz)
        else:
            k_dequan = dequant_per_token(key_cache.reshape(-1, num_kv_heads, key_block_sz * key_token_sz), head_size, kv_block_size)
        v_dequan = dequant_per_token(value_cache.reshape(-1, num_kv_heads, value_block_sz * value_token_sz), head_size, kv_block_size)
        # print(f'{k_dequan.shape = }, {v_dequan.shape = }')

        # => q [q_len, num_heads, head_size], k/v [kv_len, num_kv_heads, head_size]
        q_3d = query.reshape(q_len, num_heads, head_size).contiguous()
        k_3d = k_dequan[:valid_num_blks, :].transpose(1,2).reshape(-1, num_kv_heads, head_size).contiguous()
        v_3d = v_dequan[:valid_num_blks, :].transpose(1,2).reshape(-1, num_kv_heads, head_size).contiguous()
    else:
        # => q [q_len, num_heads, head_size], k/v [kv_len, num_kv_heads, head_size]
        q_3d = query.reshape(q_len, num_heads, head_size).contiguous()
        k_3d = key_cache[:valid_num_blks, :].transpose(1,2).reshape(-1, num_kv_heads, head_size).contiguous()
        v_3d = value_cache[:valid_num_blks, :].transpose(1,2).reshape(-1, num_kv_heads, head_size).contiguous()

    print(f'{q_3d.shape = }, {k_3d.shape = }, {v_3d.shape = }')

    attention_mask = None if full_dense else get_attention_mask(q_3d, k_3d, v_3d, block_mask.unsqueeze(0), sparse_block_sz, trunk_sz)
    ref = flash_attn_vlen_ref(q_3d, k_3d, v_3d, [], is_causal, attention_mask)
    # print(f'{ref.shape=}')

    # check_close(ref, ov_out)
    check_close(ref, ut_out)

    print(f'{Colors.GREEN}test_ov done.{Colors.END}')

if __name__ == "__main__":

    # test_page_attn_causal_batch1(seq_len, num_heads = 1, num_kv_heads = 1, head_size = 32, block_sz=block_sz, trunk_sz=blocks_per_trunk*block_sz, compressed_kvcache=KV_CACHE_COMPRESSION_BY_TOKEN, sparse_block_sz = sparse_block_sz, density=density, check_acc=True)
    #ACC test PA base
    if 0:
        for block_sz in range(32, 144, 16):
            for blocks_per_trunk in range(1, 30, 6):
                for seq_len in range(8192, 8248, 3):
                    for compressed_kv in [KV_CACHE_COMPRESSION_NONE, KV_CACHE_COMPRESSION_BY_TOKEN]:
                        print("----------------------------------------------------------------------------------------------------------------------------------------------------------------------")
                        print(f'[PA_BASE_ACC_TETS]: seq_len={seq_len} block_sz={block_sz} blocks_per_trunk={blocks_per_trunk} kv_cache=={"U8" if compressed_kv else "F16"} sparse_block_sz=1')
                        print("----------------------------------------------------------------------------------------------------------------------------------------------------------------------")
                        test_page_attn_causal_batch1(seq_len, num_heads = 1, num_kv_heads = 1, head_size = 32, block_sz=block_sz, trunk_sz=blocks_per_trunk*block_sz, compressed_kvcache=compressed_kv, sub_block_sz=block_sz, sparse_block_sz = 1, check_acc=True)
                        test_page_attn_causal_batch1(seq_len, num_heads = 1, num_kv_heads = 1, head_size = 32, block_sz=block_sz, trunk_sz=blocks_per_trunk*block_sz, compressed_kvcache=compressed_kv, sub_block_sz=block_sz, sparse_block_sz = 1, check_acc=False)

        for block_sz in range(128, 257, 32):
                for seq_len in range(32768, 32810):
                    for trunk_num in range(1, 21):
                        for compressed_kvcache in [KV_CACHE_COMPRESSION_BY_TOKEN, KV_CACHE_COMPRESSION_NONE]:
                            seq_in_blks = (seq_len + block_sz -1 ) // block_sz
                            blocks_per_trunk = seq_in_blks // trunk_num if seq_in_blks % trunk_num == 0 else seq_in_blks // (trunk_num - 1)
                            print("----------------------------------------------------------------------------------------------------------------------------------------------------------------------")
                            print(f'[PA_BASE_ACC_TETS]:seq_len={seq_len} block_sz={block_sz} blocks_per_trunk={blocks_per_trunk} kv_cache={"U8" if compressed_kvcache else "F16"}')
                            print("----------------------------------------------------------------------------------------------------------------------------------------------------------------------")
                            test_page_attn_causal_batch1(seq_len, num_heads = 1, num_kv_heads = 1, head_size = 128, block_sz=block_sz, trunk_sz=blocks_per_trunk*block_sz, compressed_kvcache=compressed_kvcache, sub_block_sz=block_sz, sparse_block_sz=1, check_acc=True)
                            test_page_attn_causal_batch1(seq_len, num_heads = 1, num_kv_heads = 1, head_size = 128, block_sz=block_sz, trunk_sz=blocks_per_trunk*block_sz, compressed_kvcache=compressed_kvcache, sub_block_sz=block_sz, sparse_block_sz=1, check_acc=True)

        if 1:
            seq_len = 32 * 1024
            block_sz = 256
            trunk_sz = seq_len
            compressed_kv = KV_CACHE_COMPRESSION_BY_CHANNEL
            for sub_block_sz in [16, 32, 64, 128]:
                print("----------------------------------------------------------------------------------------------------------------------------------------------------------------------")
                print(f'[PA_BY_CHANNEL_ACC_TESTS]: seq_len={seq_len} block_sz={block_sz} trunk_sz={trunk_sz} kv_cache={compressed_kv} sub_block_sz={sub_block_sz}')
                print("----------------------------------------------------------------------------------------------------------------------------------------------------------------------")
                test_page_attn_causal_batch1(seq_len, num_heads = 1, num_kv_heads = 1, head_size = 32, block_sz=block_sz, trunk_sz=trunk_sz, compressed_kvcache=compressed_kv, sub_block_sz=sub_block_sz, sparse_block_sz = 1, check_acc=True)
                test_page_attn_causal_batch1(seq_len, num_heads = 1, num_kv_heads = 1, head_size = 32, block_sz=block_sz, trunk_sz=trunk_sz, compressed_kvcache=compressed_kv, sub_block_sz=sub_block_sz, sparse_block_sz = 1, check_acc=True)
    #ACC test sparse X Attention:
    if 0:
        for sparse_block_sz in [128, 256, 1]:
            for block_sz in [256]:
                for density in [1.0, 0.5, 0.75]:
                    for blocks_per_trunk in [1, 15, 16, 17, 32, 300]:
                        for seq_len in [16*15, 16*16, 16*16+1, 1024, 1024+1, 8*1024, 8*1024+3, 16*1024]:
                            for head_size in [32, 96, 128]:
                                for compressed_kvcache in [KV_CACHE_COMPRESSION_BY_TOKEN, KV_CACHE_COMPRESSION_NONE]:
                                    print("----------------------------------------------------------------------------------------------------------------------------------------------------------------------")
                                    print(f'[XATTENION_ACC_TETS]:seq_len={seq_len} block_sz={block_sz} blocks_per_trunk={blocks_per_trunk} kv_cache={"U8" if compressed_kvcache else "F16"} {sparse_block_sz=} {density=}')
                                    print("----------------------------------------------------------------------------------------------------------------------------------------------------------------------")
                                    test_page_attn_causal_batch1(seq_len, num_heads = 4, num_kv_heads = 2, head_size = head_size, block_sz=block_sz, trunk_sz=blocks_per_trunk*block_sz, compressed_kvcache=compressed_kvcache, sub_block_sz=block_sz, sparse_block_sz = sparse_block_sz, density=density, check_acc=True)

    def smoke_accuracy_test(blocks_per_trunk = 128, compressed_kvcache = KV_CACHE_COMPRESSION_BY_TOKEN, sub_block_sz=DEFAULT_SUB_BLOCK_SIZE):
        seq_len, block_sz = 32*1024, 256
        trunk_sz = blocks_per_trunk*block_sz

        test_page_attn_causal_batch1(seq_len, num_heads = 2, num_kv_heads = 1, head_size = 128, block_sz=block_sz, trunk_sz=trunk_sz,  compressed_kvcache=compressed_kvcache, sub_block_sz=sub_block_sz, sparse_block_sz = 1, density=1.0, check_acc=True)
        test_page_attn_causal_batch1(seq_len, num_heads = 2, num_kv_heads = 1, head_size = 128, block_sz=block_sz, trunk_sz=trunk_sz,  compressed_kvcache=compressed_kvcache, sub_block_sz=sub_block_sz, sparse_block_sz = 256, density=0.33, check_acc=True)
        test_page_attn_causal_batch1(seq_len, num_heads = 2, num_kv_heads = 1, head_size = 128, block_sz=block_sz, trunk_sz=trunk_sz,  compressed_kvcache=compressed_kvcache, sub_block_sz=sub_block_sz, sparse_block_sz = 128, density=0.33, check_acc=True)

    # perf for sparse X attention, with QWen3 8K case
    def smoke_perf_test(blocks_per_trunk = 128, compressed_kvcache = KV_CACHE_COMPRESSION_BY_TOKEN, sub_block_sz=DEFAULT_SUB_BLOCK_SIZE):
        seq_len, block_sz = 32*1024, 256
        trunk_sz = blocks_per_trunk*block_sz

        test_page_attn_causal_batch1(seq_len, num_heads = 32, num_kv_heads = 8, head_size = 128, block_sz=block_sz, trunk_sz=trunk_sz,  compressed_kvcache=compressed_kvcache, sub_block_sz=sub_block_sz, sparse_block_sz = 1, density=1.0, check_acc=False)

        for sparse_block_sz in [256, 128]:
            for density in [1.0, 0.99, 0.66, 0.33, 0.11]:
            # for density in [1.0]:
                # print("-----------------------------------------------------------------------------------------------------------------------------------------")
                # print(f'seq_len={seq_len} block_sz={block_sz} blocks_per_trunk={blocks_per_trunk} sparse_block_sz={sparse_block_sz}')
                # print("-----------------------------------------------------------------------------------------------------------------------------------------")
                test_page_attn_causal_batch1(seq_len, num_heads = 32, num_kv_heads = 8, head_size = 128, block_sz=block_sz, trunk_sz=trunk_sz,  compressed_kvcache=compressed_kvcache, sub_block_sz=sub_block_sz, sparse_block_sz = sparse_block_sz, density=density, check_acc=False)

    smoke_accuracy_test()
    smoke_accuracy_test(16)
    smoke_accuracy_test(compressed_kvcache=KV_CACHE_COMPRESSION_NONE)
    smoke_accuracy_test(compressed_kvcache=KV_CACHE_COMPRESSION_BY_CHANNEL, sub_block_sz=DEFAULT_SUB_BLOCK_SIZE)
    smoke_accuracy_test(compressed_kvcache=KV_CACHE_COMPRESSION_BY_CHANNEL, sub_block_sz=32)

    smoke_perf_test()
    smoke_perf_test(16)
    smoke_perf_test(compressed_kvcache=KV_CACHE_COMPRESSION_NONE)
    smoke_perf_test(compressed_kvcache=KV_CACHE_COMPRESSION_BY_CHANNEL, sub_block_sz=DEFAULT_SUB_BLOCK_SIZE)
    smoke_perf_test(compressed_kvcache=KV_CACHE_COMPRESSION_BY_CHANNEL, sub_block_sz=32)

    # test_ov()
    
