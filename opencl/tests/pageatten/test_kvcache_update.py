import os
import time
import math

import torch
import functools
import numpy as np

from clops import cl
from clops import compare
from clops.utils import Colors

# "CM_FE_DIR": "c:\\ceciliapeng\\ComputeSDK_Windows_internal_2025_WW41\\compiler\bin"
# os.environ["CM_FE_DIR"] = "c:\\ceciliapeng\\ComputeSDK_Windows_internal_2025_WW41\\compiler\bin"

class pa_kvcache_update_cm:
    def __init__(self, num_kv_heads, k_head_size, v_head_size, block_size, enable_kvcache_compress):
        self.num_kv_heads = num_kv_heads
        self.k_head_size = k_head_size
        self.v_head_size = v_head_size
        self.block_size = block_size
        self.wg_size = 16

        self.enable_kvcache_compress = enable_kvcache_compress

        src = r'''#include "pa_kv_cache_update_ref.hpp"'''
        cwd = os.path.dirname(os.path.realpath(__file__))
        print(f"compiling {cwd} {num_kv_heads=} {k_head_size=} {v_head_size=} {enable_kvcache_compress=}...")

        if enable_kvcache_compress:
            adjusted_k_head_size = k_head_size + 4
            adjusted_v_head_size = v_head_size + 4
        else:
            adjusted_k_head_size = k_head_size
            adjusted_v_head_size = v_head_size

        jit_option = '-abortonspill -noschedule '
        self.kernels = cl.kernels(src,
                      (f' -cmc -Qxcm_jit_option="{jit_option}" -Qxcm_register_file_size=256 -mCM_printregusage -I{cwd}'
                      f" -DKV_HEADS_NUM={num_kv_heads}"
                      f" -DK_HEAD_SIZE={k_head_size}"
                      f" -DV_HEAD_SIZE={v_head_size}"
                      f" -DADJUSTED_K_HEAD_SIZE={adjusted_k_head_size}"
                      f" -DADJUSTED_V_HEAD_SIZE={adjusted_v_head_size}"
                      f" -DPAGED_ATTENTION_BLOCK_SIZE={self.block_size}"
                      f" -DWG_SIZE={self.wg_size}"
                      f" -DKV_CACHE_COMPRESSION_PER_TOKEN={int(enable_kvcache_compress)}"
                      f" -mdump_asm -g2")
                    )

    def __call__(self, key:torch.Tensor,
                 value:torch.Tensor,
                key_cache:torch.Tensor,
                value_cache:torch.Tensor,
                past_lens:list,
                subsequence_begins:list,
                block_indices:list,
                block_indices_begins:list,
                n_repeats = 1):
        batch_size_in_tokens, _ = key.shape
        batch_size_in_sequences = len(past_lens)
        key_pitch = key.stride()[0]
        val_pitch = value.stride()[0]
        key_offset = 0
        value_offset = 0 
        # print(f"============ {key.shape=} {key.is_contiguous()=}")
        # print(f"============ {value.shape=} {value.is_contiguous()=}")
        # print(f"============ {batch_size_in_tokens=} {batch_size_in_sequences=}")
        # print(f"============ {key.stride()=} {value.stride()=}")

        if self.enable_kvcache_compress:
            kv_cache_type = torch.uint8
        else:
            kv_cache_type = torch.float16

        wg_count = (batch_size_in_tokens + self.wg_size - 1) // self.wg_size
        GWS = [1, self.num_kv_heads, int(wg_count * self.wg_size)]
        LWS = [1, 1, self.wg_size]

        for i in range(0, n_repeats):
            print(f'{Colors.GREEN}calling pa_kv_cache_update with enable_kvcache_compress {self.enable_kvcache_compress} {GWS=} {LWS=} {key_pitch=} {val_pitch=} {batch_size_in_sequences=} at {i}/{n_repeats} times {Colors.END}')

            t_key = cl.tensor(key.to(torch.float16).detach().numpy())
            t_value = cl.tensor(value.to(torch.float16).detach().numpy())

            t_key_cache = cl.tensor(key_cache.to(kv_cache_type).detach().numpy())
            t_value_cache = cl.tensor(value_cache.to(kv_cache_type).detach().numpy())

            t_block_indices=cl.tensor(torch.tensor(block_indices).to(torch.int32).detach().numpy())
            t_past_lens=cl.tensor(torch.tensor(past_lens).to(torch.int32).detach().numpy())
            t_block_indices_begins=cl.tensor(torch.tensor(block_indices_begins).to(torch.int32).detach().numpy())
            t_subsequence_begins=cl.tensor(torch.tensor(subsequence_begins).to(torch.int32).detach().numpy())

            self.kernels.enqueue("pa_kv_cache_update", GWS, LWS, t_key, t_value,
                            t_past_lens, t_block_indices, t_block_indices_begins, t_subsequence_begins, 
                            t_key_cache, t_value_cache,
                            key_pitch, key_offset,
                            val_pitch, value_offset,
                            batch_size_in_sequences)
            ns = cl.finish()
            for i, time_opt in enumerate(ns):
                print(f'(pa_kv_cache_update)TPUT_{i}:[{key.numel()=}]+[{value.numel()=}] {time_opt*1e-3:,.0f} us')
                if self.enable_kvcache_compress:
                    total_bytes = batch_size_in_tokens * self.num_kv_heads * (3 * self.k_head_size + 3 * self.v_head_size + 8)
                else:
                    total_bytes = batch_size_in_tokens * self.num_kv_heads * (4 * self.k_head_size + 4 * self.v_head_size)
                tput = total_bytes / time_opt
                print(f'(pa_kv_cache_update)TPUT_{i}:[{total_bytes*1e-6:,} MB] {tput:,.2f} GB/s')

        return t_key_cache.numpy(), t_value_cache.numpy()
                    
    @staticmethod
    @functools.cache
    def create_instance(num_kv_heads, k_head_size, v_head_size, block_size, enable_kvcache_compress):
        return pa_kvcache_update_cm(num_kv_heads, k_head_size, v_head_size, block_size, enable_kvcache_compress)
    
class ContinuousBatchKVCacheGenerator:
    def __init__(self, num_tokens:list, past_lens:list, num_kv_heads, k_head_size, v_head_size, block_size, enable_kvcache_compress):
        self.batch_size_in_sequences = len(num_tokens)
        assert(self.batch_size_in_sequences == len(past_lens))

        self.enable_kvcache_compress = enable_kvcache_compress
        self.num_tokens = num_tokens
        self.past_lens = past_lens
        self.num_kv_heads = num_kv_heads
        self.k_head_size = k_head_size
        self.v_head_size = v_head_size
        self.block_size = block_size
    
        # prepare page attention inputs
        # key_data/value_data are lists of torch.Tensor with shape [subsequence_length, num_kv_heads, kv_head_size] for each sequence
        self.key_data = []
        self.value_data = []
        self.subsequence_begins = []
        self.block_indices_begins = []
    
        self.subsequence_begins.append(0)
        self.block_indices_begins.append(0)
        for i in range(self.batch_size_in_sequences):
            subsequence_length = num_tokens[i] + past_lens[i]

            k = torch.rand(subsequence_length, num_kv_heads*k_head_size).to(dtype=torch.float16)
            v = torch.rand(subsequence_length, num_kv_heads*v_head_size).to(dtype=torch.float16)
            # k[:,0] = 0
            # v[:,0] = 0
            self.key_data.append(k)
            self.value_data.append(v)
            
            subsequence_start_pos = self.subsequence_begins[i]
            subsequence_end_pos = subsequence_start_pos + num_tokens[i]
            self.subsequence_begins.append(subsequence_end_pos)

            required_blocks = (subsequence_length + block_size - 1) // block_size

            block_indices_start_pos = self.block_indices_begins[i]
            block_indices_end_pos = block_indices_start_pos + required_blocks
            self.block_indices_begins.append(block_indices_end_pos)

        # simulate random block allocation
        self.num_blocks = self.block_indices_begins[-1]
        block_indices = torch.arange(self.num_blocks)
        perm_idx = torch.randperm(block_indices.shape[0])
        inv_per_idx = torch.argsort(perm_idx)
        self.block_indices = block_indices[inv_per_idx]
        # print(f'{Colors.BLUE} ============ {self.subsequence_begins=} {Colors.END}')
        # print(f'{Colors.BLUE} ============ {self.block_indices_begins=} {Colors.END}')
        # print(f'{Colors.BLUE} ============ {block_indices=} {Colors.END}')

        # print(f'{Colors.BLUE} {self.key_data=} {Colors.END}')
        # print(f'{Colors.BLUE} {self.value_data=} {Colors.END}')

    def get_block_table(self):
        return self.subsequence_begins, self.block_indices, self.block_indices_begins
    
    # generate key / value inputs
    def get_current_kv(self):
        batch_size_in_tokens = self.subsequence_begins[-1]
        key_mem = torch.zeros(batch_size_in_tokens, self.num_kv_heads*self.k_head_size).to(torch.float16)
        value_mem = torch.zeros(batch_size_in_tokens, self.num_kv_heads*self.v_head_size).to(torch.float16)
        for i in range(self.batch_size_in_sequences):
            key_mem[self.subsequence_begins[i] : self.subsequence_begins[i+1], :] = self.key_data[i][self.past_lens[i]:, :]
            value_mem[self.subsequence_begins[i] : self.subsequence_begins[i+1], :] = self.value_data[i][self.past_lens[i]:, :]

        return key_mem, value_mem
    
    # generate key_cache / value_cache
    def get_kv_cache(self, skip_input = True):
        if self.enable_kvcache_compress is False:
            key_cache = self.__get_kv_cache_half(self.k_head_size, self.key_data, skip_input)
            value_cache = self.__get_kv_cache_half(self.v_head_size, self.value_data, skip_input)
        else:
            key_cache = self.__get_kv_cache_u8(self.k_head_size, self.key_data, skip_input)
            value_cache = self.__get_kv_cache_u8(self.v_head_size, self.value_data, skip_input)
        return key_cache, value_cache
            
    # private methods
    def __get_kv_cache_half(self, _head_size, input_data, skip_input):
        cache_data = torch.zeros(self.num_blocks, self.block_size, self.num_kv_heads*_head_size).to(torch.float16)
        for i in range(self.batch_size_in_sequences):
            process_len = self.past_lens[i] if skip_input else self.past_lens[i] + self.num_tokens[i]
            if process_len > 0:
                blocks_num = (process_len + self.block_size - 1) // self.block_size
                for block_idx in range(blocks_num):
                    last_token_idx = process_len % self.block_size if block_idx == blocks_num -1 else self.block_size
                    if last_token_idx == 0: last_token_idx = self.block_size
                    # print(f'{Colors.RED} {block_idx=} {blocks_num=} {process_len=} {last_token_idx=} {Colors.END}')
                    for token_idx in range(last_token_idx):
                        input_token_offset = block_idx * self.block_size + token_idx
                        block_pos = self.block_indices[self.block_indices_begins[i] + block_idx]
                        cache_data[block_pos, token_idx, :] = input_data[i][input_token_offset, :]
        return cache_data.reshape(self.num_blocks, self.block_size, self.num_kv_heads, _head_size).transpose(1, 2).contiguous()
    
    def __get_kv_cache_u8(self, _head_size, input_data, skip_input):
        cache_data = torch.zeros(self.num_blocks, self.num_kv_heads, self.block_size * (_head_size + 4)).to(torch.uint8)
        for i in range(self.batch_size_in_sequences):
            process_len = self.past_lens[i] if skip_input else self.past_lens[i] + self.num_tokens[i]
            if process_len > 0:
                blocks_num = (process_len + self.block_size - 1) // self.block_size
                for block_idx in range(blocks_num):
                    last_token_idx = process_len % self.block_size if block_idx == blocks_num -1 else self.block_size
                    if last_token_idx == 0: last_token_idx = self.block_size
                    # print(f'{Colors.RED} {block_idx=} {blocks_num=} {process_len=} {last_token_idx=} {Colors.END}')
                    for h in range(self.num_kv_heads):
                        # input_data[seq_num][token_num, head_size * kv_head_num]
                        token_start_idx = block_idx * self.block_size
                        token_end_idx = token_start_idx + last_token_idx
                        input_block_per_head = input_data[i][token_start_idx:token_end_idx, h*_head_size:(h+1)*_head_size].reshape(1, 1, -1, _head_size)
                        input_block_per_head_q = self.__quant_block_per_token(input_block_per_head, _head_size, self.block_size).reshape(-1)

                        # print()
                        # print(f'head_idx = {h} token_start_idx = {token_start_idx} token_end_idx = {token_end_idx} last_token_idx = {last_token_idx}')
                        # print('input_block_per_head.shape = {input_block_per_head.shape}')
                        # print('input_block_per_head = ',input_block_per_head)
                        # print('input_block_per_head_q.shape = ',input_block_per_head_q.reshape(1,1,-1,head_size).shape)
                        # print('input_block_per_head_q = ',input_block_per_head_q.reshape(1,1,-1,head_size))

                        block_pos = self.block_indices[self.block_indices_begins[i] + block_idx]
                        cache_data[block_pos, h, :] = input_block_per_head_q
        # if skip_input == False:
        #     print("cache_data =", cache_data)
        return cache_data
    
    # quantize a kv block in fashion of per_token
    # input_block_per_head [1, 1, block_size, head_size]
    def __quant_block_per_token(self, input_block_per_head, _head_size, block_size):
        blk_num, kv_heads, blksz, *_ = input_block_per_head.shape
        kv_u8, dq_scale, kv_zp = self.quant_per_token(input_block_per_head)
        if blksz < block_size:
            kv_pad = torch.zeros(blk_num, kv_heads, (block_size - blksz)*_head_size).to(dtype=torch.uint8)
            kv_u8 = torch.cat((kv_u8, kv_pad), dim=-1)
            scale_zp_pad = torch.zeros(blk_num, kv_heads, (block_size - blksz)*2).to(dtype=torch.uint8)
            dq_scale = torch.cat((dq_scale, scale_zp_pad), dim=-1)
            kv_zp = torch.cat((kv_zp, scale_zp_pad), dim=-1)

        # print("dq_scale: ", dq_scale)
        # print("kz_zp: ", kv_zp)
        return torch.concat((kv_u8, dq_scale, kv_zp), dim=-1)
    
    # quantize in fashion of per_token
    # kv_cache_blocks [num_blocks, num_kv_heads, block_size, head_size]
    @staticmethod
    def quant_per_token(kv_cache_blocks):
        blk_num, kv_heads, *_ = kv_cache_blocks.shape
        kv_max = kv_cache_blocks.amax(dim=-1, keepdim = True).to(dtype=torch.float)
        kv_min = kv_cache_blocks.amin(dim=-1, keepdim = True).to(dtype=torch.float)
        qrange = (kv_max - kv_min).to(dtype=torch.float)

        U8_MAX = torch.tensor(255.0, dtype=torch.float)
        U8_MIN = torch.tensor(0.0, dtype=torch.float)
        U8_RANGE = (U8_MAX - U8_MIN).to(dtype=torch.float)
        kv_scale = ((U8_RANGE)/qrange).to(dtype=torch.float)
        kv_scale_div = (1.0/kv_scale).to(dtype=torch.float)
        kv_zp = ((0.0-kv_min)*kv_scale+U8_MIN).to(dtype=torch.float)

        def round_to_even(tensor):
            rounded = torch.floor(tensor + 0.5)
            adjustment = (rounded % 2 != 0) & (torch.abs(tensor - rounded) == 0.5000)
            adjustment = adjustment | (rounded > 255)  # also handle overflow
            return rounded - adjustment.to(rounded.dtype)

        # kv_u8 = torch.round((kv_cache_blocks*kv_scale+kv_zp)).to(dtype=torch.uint8).reshape(blk_num,kv_heads,-1)
        kv_u8 = round_to_even((kv_cache_blocks*kv_scale+kv_zp)).to(dtype=torch.uint8).reshape(blk_num,kv_heads,-1)

        # torch.set_printoptions(precision=6)
        # print("kv_fp16:\n", kv_cache_blocks.reshape(blk_num, kv_heads, blksz, k_head_size))
        # print("kv_max:\n", kv_max.reshape(blk_num, kv_heads, -1))
        # print("kv_min:\n", kv_min.reshape(blk_num, kv_heads, -1))
        # print("kv_scale:\n", kv_scale.reshape(blk_num, kv_heads, -1))
        # print("kv_scale_div:\n", kv_scale_div.reshape(blk_num, kv_heads, -1))
        # print("kv_zp:\n", kv_zp.reshape(blk_num, kv_heads, -1))
        # print("kv_quant_fp16:\n", (kv_cache_blocks*kv_scale+kv_zp).reshape(blk_num, kv_heads, blksz, k_head_size))

        # print("quant_scale =", (1.0/kv_scale).reshape(blk_num,kv_heads,-1))
        # print("quant_zp    =", kv_zp.reshape(blk_num,kv_heads,-1))

        dq_scale = kv_scale_div.to(dtype=torch.half).view(dtype=torch.uint8).reshape(blk_num,kv_heads,-1)
        # kv_scale = ((U8_RANGE)/qrange).to(dtype=torch.half)
        # dq_scale = (kv_scale).view(dtype=torch.uint8).reshape(blk_num,kv_heads,-1)
        kv_zp = kv_zp.to(dtype=torch.half).view(dtype=torch.uint8).reshape(blk_num,kv_heads,-1)

        # print("dq_scale: ", dq_scale)
        # print("kz_zp: ", kv_zp)
        return kv_u8, dq_scale, kv_zp
    
    @staticmethod
    def print_kv_cache_u8(kv_cache_u8, blk_size, kv_head_size, name="key_cache"):
        blk_num, kv_heads, kv_block_bytes_per_head = kv_cache_u8.shape
        print("name =", name, ",blk_num =", blk_num, ",kv_heads =", kv_heads, ",blk_size =", blk_size, ",kv_head_size =", kv_head_size)
        for b in range(blk_num):
            for h in range(kv_heads):
                print(f'blk={b} head={h}')
                block_head_data = kv_cache_u8[b,h,:blk_size * kv_head_size].reshape(blk_size, kv_head_size)
                block_head_scale = kv_cache_u8[b,h,blk_size * kv_head_size : blk_size * kv_head_size + blk_size * 2].reshape(blk_size, 2).view(dtype=torch.float16)
                block_head_zp = kv_cache_u8[b,h,blk_size * kv_head_size + blk_size * 2 : ].reshape(blk_size, 2).view(dtype=torch.float16)
                print('data: shape = ', block_head_data.shape, "\n", block_head_data)
                print('scale: shape = ', block_head_scale.shape, '\n', block_head_scale.reshape(1,blk_size))
                print('zp: shape = ', block_head_zp.shape, '\n', block_head_zp.reshape(1,blk_size))

                # block_head_data_f16 = block_head_data.to(dtype=torch.float16)
                # for i in range(blk_size):
                #     block_head_data_f16[i,:] = (block_head_data_f16[i,:] - block_head_zp[i,0]) * block_head_scale[i,0]
                # print('dequant_data_f16: shape = ', block_head_data_f16.shape, '\n', block_head_data_f16)

def test_pa_kv_cache_update(num_tokens:list, past_lens:list, num_kv_heads=1, k_head_size=64, v_head_size=64, block_size=16, enable_kvcache_compress=True, check_perf=False):   
    cb_kvcache_gnr = ContinuousBatchKVCacheGenerator(num_tokens, past_lens, num_kv_heads, k_head_size, v_head_size, block_size, enable_kvcache_compress)
    subsequence_begins, block_indices, block_indices_begins = cb_kvcache_gnr.get_block_table()

    key, value = cb_kvcache_gnr.get_current_kv()
    # print(f'{Colors.BLUE} ============ {key.shape=} {key.is_contiguous()=}" {Colors.END}')
    # print(f'{Colors.BLUE} ============ {value.shape=} {value.is_contiguous()=}" {Colors.END}')
    # print(f'{Colors.BLUE} {key=} {Colors.END}')
    # print(f'{Colors.BLUE} {value=} {Colors.END}')

    key_cache, value_cache = cb_kvcache_gnr.get_kv_cache()
    key_cache_ref, value_cache_ref = cb_kvcache_gnr.get_kv_cache(False)

    # print(f'{Colors.BLUE} ============ {key_cache_ref.shape=} {key_cache_ref.is_contiguous()=} {Colors.END}')
    # print(f'{Colors.BLUE} ============ {value_cache_ref.shape=} {value_cache_ref.is_contiguous()=} {Colors.END}')
    # if enable_kvcache_compress:
    #     print_kv_cache_u8(key_cache_ref, block_size, k_head_size, "key_cache_ref")
    #     print_kv_cache_u8(value_cache_ref, block_size, v_head_size, "value_cache_ref")
    # else:
    #     print(f'{Colors.BLUE} {key_cache_ref=} {Colors.END}')
    #     print(f'{Colors.BLUE} {value_cache_ref=} {Colors.END}')
   
    # opt
    pa_cm = pa_kvcache_update_cm.create_instance(num_kv_heads, k_head_size, v_head_size, block_size, enable_kvcache_compress)
    n_repeats = 20 if check_perf else 1
    out_key_cache, out_value_cache = pa_cm(key, value, key_cache, value_cache, past_lens, subsequence_begins, block_indices, block_indices_begins, n_repeats)

    if enable_kvcache_compress:
        out_key_cache=torch.tensor(out_key_cache).to(dtype=torch.uint8).reshape(-1, num_kv_heads, block_size * (k_head_size + 4))
        out_value_cache=torch.tensor(out_value_cache).to(dtype=torch.uint8).reshape(-1, num_kv_heads, block_size * (v_head_size + 4))
        key_cache_ref = key_cache_ref.reshape(-1, num_kv_heads, block_size * (k_head_size + 4))
        value_cache_ref = value_cache_ref.reshape(-1, num_kv_heads, block_size * (v_head_size + 4))
        compare(key_cache_ref[:,:,:block_size * k_head_size].to(dtype=torch.int).detach().numpy(), out_key_cache[:,:,:block_size * k_head_size].to(dtype=torch.int).detach().numpy(),1)
        compare(key_cache_ref[:,:,block_size * k_head_size :].view(dtype=torch.half).detach().numpy(), out_key_cache[:,:,block_size * k_head_size : ].view(dtype=torch.half).detach().numpy(),1e-3)

        compare(value_cache_ref[:,:,:block_size * v_head_size].to(dtype=torch.int).detach().numpy(), out_value_cache[:,:,:block_size * v_head_size].to(dtype=torch.int).detach().numpy(),1)
        compare(value_cache_ref[:,:,block_size * v_head_size :].view(dtype=torch.half).detach().numpy(), out_value_cache[:,:,block_size * v_head_size : ].view(dtype=torch.half).detach().numpy(),1e-3)
    else:
        compare(key_cache_ref.detach().numpy(), out_key_cache)
        compare(value_cache_ref.detach().numpy(), out_value_cache)
    print(f'{Colors.GREEN}kv_cache_update passed{Colors.END}')

# reference impl
# // # cur_kv_data:     [batch_size_in_tokens, num_kv_heads * head_size]
# // # kv_cache_data:   [num_blocks, num_heads, block_size, (head_size + 4)]
def reference_kv_cache_update(kv_cache_data, cur_kv_data, past_lens, subsequence_begins, block_indices, block_indices_begins):
    batch_size_in_tokens, _ = cur_kv_data.shape
    num_blocks, num_kv_heads, block_size, adjusted_head_size = kv_cache_data.shape
    head_size = adjusted_head_size - 4
    batch_size_in_sequences = subsequence_begins.shape[0] - 1

    kv_cache_data = kv_cache_data.reshape(num_blocks, num_kv_heads, -1)

    for i in range(batch_size_in_sequences):
        tokens_num = subsequence_begins[i + 1] - subsequence_begins[i]
        for token_idx in range(tokens_num):
            cur_block_idx = (past_lens[i] + token_idx) // block_size
            block_pos = block_indices[block_indices_begins[i] + cur_block_idx]

            token_start_pos = (past_lens[i] + token_idx) % block_size
            scale_start_pos = block_size*head_size + token_start_pos*2
            zp_start_pos = block_size*head_size + block_size*2 + token_start_pos*2
            for h in range(num_kv_heads):
                cur_kv_row = cur_kv_data[subsequence_begins[i] + token_idx, h*head_size:(h+1)*head_size].reshape(1, 1, 1, head_size)
                kv_u8, dq_scale, dq_zp = ContinuousBatchKVCacheGenerator.quant_per_token(cur_kv_row)
                kv_cache_data[block_pos, h, token_start_pos*head_size : (token_start_pos+1)*head_size] = kv_u8.reshape(-1)
                # print(f"{scale_start_pos=}, {dq_scale.reshape(-1)=}, {kv_cache_data[block_pos, h, scale_start_pos:scale_start_pos+2]}, {token_start_pos=}")
                kv_cache_data[block_pos, h, scale_start_pos:scale_start_pos+2] = dq_scale.reshape(-1)
                kv_cache_data[block_pos, h, zp_start_pos:zp_start_pos+2] = dq_zp.reshape(-1)
    return kv_cache_data.reshape(num_blocks, num_kv_heads, block_size, adjusted_head_size)

# def test_ov(dump_dir, pa_node_name):
#     def get_tensor(name, dtype=np.float16):
#         with open(name, 'rb') as f:
#             data = f.read()
#             np_data = np.frombuffer(data, dtype=dtype).copy()
#             return torch.from_numpy(np_data)

#     compressed_kvcache = True
#     kv_block_size = 256
#     num_kv_heads, k_head_size, v_head_size = 8, 128, 128
#     base = f"c:\\ceciliapeng\\{dump_dir}_{pa_node_name}\\"

#     key = get_tensor(base + f'program1_network1_0_pagedattentionextension_{pa_node_name}_src1__f16__255_1024_1_1__bfyx.bin').reshape([-1, num_kv_heads*k_head_size])
#     value = get_tensor(base + f'program1_network1_0_pagedattentionextension_{pa_node_name}_src2__f16__255_1024_1_1__bfyx.bin').reshape([-1, num_kv_heads*v_head_size])

#     key_cache = get_tensor(base + f'program1_network1_0_pagedattentionextension_{pa_node_name}_updated_src_3__i8__1_8_256_132__bfyx.bin', np.int8 if compressed_kvcache else np.float16).reshape([-1, num_kv_heads, kv_block_size, v_head_size+4])
#     value_cache = get_tensor(base + f'program1_network1_0_pagedattentionextension_{pa_node_name}_updated_src_4__i8__1_8_256_132__bfyx.bin', np.int8 if compressed_kvcache else np.float16).reshape([-1, num_kv_heads, kv_block_size, v_head_size+4])
    
#     # o [q_len, num_heads*head_size], k/v cache [num_blks, num_kv_heads, kv_block_size, head_size]
#     print(f'{key_cache.shape = }, {value_cache.shape = }')

#     valid_num_blks = key_cache.shape[0] - 1 # genai usually generates one more blocks than required
#     valid_num_blks = 1

#     past_lens = get_tensor(base + f'program1_network1_0_pagedattentionextension_{pa_node_name}_src5__i32__1_1_1_1__bfyx.bin', dtype=np.int32).reshape([1])
#     subsequence_begins = get_tensor(base + f'program1_network1_0_pagedattentionextension_{pa_node_name}_src6__i32__2_1_1_1__bfyx.bin', dtype=np.int32).reshape([2])
#     block_indices = get_tensor(base + f'program1_network1_0_pagedattentionextension_{pa_node_name}_src7__i32__1_1_1_1__bfyx.bin', dtype=np.int32).reshape([valid_num_blks])
#     block_indices_begins = get_tensor(base + f'program1_network1_0_pagedattentionextension_{pa_node_name}_src8__i32__2_1_1_1__bfyx.bin', dtype=np.int32).reshape([2])
#     print(f'{past_lens=}, {subsequence_begins=}, {block_indices=}, {block_indices_begins=}')        

#     print(f'{Colors.BLUE} ============ {key_cache.shape=} {key_cache.is_contiguous()=} {Colors.END}')
#     print(f'{Colors.BLUE} ============ {value_cache.shape=} {value_cache.is_contiguous()=} {Colors.END}')
#     print(f'{Colors.BLUE} ============ {key.shape=} {key.is_contiguous()=} {Colors.END}')
#     print(f'{Colors.BLUE} ============ {value.shape=} {value.is_contiguous()=} {Colors.END}')
#     print(f'{Colors.BLUE} ============ {block_indices.shape=} {block_indices.is_contiguous()=} {Colors.END}')
    
#     # opt
#     pa_cm = pa_kvcache_update_cm.create_instance(num_kv_heads, k_head_size, v_head_size, kv_block_size, compressed_kvcache)
#     out_key_cache, out_value_cache = pa_cm(key, value, key_cache, value_cache, past_lens, subsequence_begins, block_indices, block_indices_begins, 1)

#     key_cache_ref = reference_kv_cache_update(key_cache.clone(), key.clone(), past_lens, subsequence_begins, block_indices, block_indices_begins)
#     value_cache_ref = reference_kv_cache_update(value_cache.clone(), value.clone(), past_lens, subsequence_begins, block_indices, block_indices_begins)

#     if compressed_kvcache:
#         out_key_cache=torch.tensor(out_key_cache).reshape(-1, num_kv_heads, kv_block_size * (k_head_size + 4)).to(dtype=torch.uint8)
#         out_value_cache=torch.tensor(out_value_cache).reshape(-1, num_kv_heads, kv_block_size * (v_head_size + 4)).to(dtype=torch.uint8)

#         key_cache_ref = key_cache_ref.reshape(-1, num_kv_heads, kv_block_size * (k_head_size + 4)).to(dtype=torch.uint8)
#         value_cache_ref = value_cache_ref.reshape(-1, num_kv_heads, kv_block_size * (v_head_size + 4)).to(dtype=torch.uint8)

#         compare(key_cache_ref[:,:,kv_block_size * k_head_size :].view(dtype=torch.half).detach().numpy(), out_key_cache[:,:,kv_block_size * k_head_size : ].view(dtype=torch.half).detach().numpy(),1e-3, 0.01, True)
#         compare(key_cache_ref[:,:,:kv_block_size * k_head_size].to(dtype=torch.uint8).detach().numpy(), out_key_cache[:,:,:kv_block_size * k_head_size].to(dtype=torch.uint8).detach().numpy(),1)

#         ref = value_cache_ref[:,:,kv_block_size * k_head_size :].view(dtype=torch.half).detach().numpy()
#         opt = out_value_cache[:,:,kv_block_size * v_head_size : ].view(dtype=torch.half).detach().numpy()
#         print(f"DEBUG  ref_zp={ref[0, 7, 265]}, ref_scale={ref[0, 7, 9]}")
#         print(f"DEBUG  opt_zp={opt[0, 7, 265]}, ref_scale={opt[0, 7, 9]}")
#         compare(value_cache_ref[:,:,kv_block_size * k_head_size :].view(dtype=torch.half).detach().numpy(), out_value_cache[:,:,kv_block_size * v_head_size : ].view(dtype=torch.half).detach().numpy(),1e-3, 0.01, True)
#         compare(value_cache_ref[:,:,:kv_block_size * k_head_size].to(dtype=torch.uint8).detach().numpy(), out_value_cache[:,:,:kv_block_size * v_head_size].to(dtype=torch.uint8).detach().numpy(),1)
#     else:
#         compare(key_cache_ref.detach().numpy(), out_key_cache)
#         compare(value_cache_ref.detach().numpy(), out_value_cache)
#     print(f'{Colors.GREEN}kv_cache_update passed{Colors.END}')

if __name__ == "__main__":
    torch.manual_seed(3)
    torch.set_printoptions(linewidth=1024)
    
    cl.profiling(True)

    # if 0:
    #     test_ov("dump_debug_bin_int4", "PagedAttentionExtension_40206")
    #     test_ov("dump_debug_bin_int8", "PagedAttentionExtension_38414")
    #     test_ov("dump_debug_bin_int8", "PagedAttentionExtension_38747")
    #     import sys
    #     sys.exit(0)
    
    for compress_kvcache in [False, True]:
        # test_pa_kv_cache_update([1024, 16, 17], [16, 0, 1])
        test_pa_kv_cache_update([32*1024], [0], num_kv_heads=8, k_head_size=128, v_head_size=128, block_size=256, enable_kvcache_compress=compress_kvcache, check_perf=True)
        test_pa_kv_cache_update([32*1024], [0], num_kv_heads=8, k_head_size=96, v_head_size=96, block_size=256, enable_kvcache_compress=compress_kvcache, check_perf=True)
        test_pa_kv_cache_update([32*1024], [0], num_kv_heads=8, k_head_size=48, v_head_size=48, block_size=256, enable_kvcache_compress=compress_kvcache, check_perf=True)
        test_pa_kv_cache_update([32*1024], [0], num_kv_heads=8, k_head_size=48, v_head_size=96, block_size=256, enable_kvcache_compress=compress_kvcache, check_perf=True)
        # test_pa_kv_cache_update([64*1024], [0], num_kv_heads=8, k_head_size=128, v_head_size=128, block_size=256, check_perf=True)
        # test_pa_kv_cache_update([128*1024], [0], num_kv_heads=8, k_head_size=128, v_head_size=128, block_size=256, check_perf=True)
        test_pa_kv_cache_update([32*1024], [4*1024], num_kv_heads=8, k_head_size=128, v_head_size=128, block_size=256, check_perf=True)
        # test_pa_kv_cache_update([128*1024], [1*1024], num_kv_heads=8, k_head_size=128, v_head_size=128, block_size=256, check_perf=True)

        test_pa_kv_cache_update([16], [0], num_kv_heads=1, k_head_size=16, v_head_size=16, block_size=16, enable_kvcache_compress=compress_kvcache, check_perf=False)
        
        # test_pa_kv_cache_update([1024], [0], num_kv_heads=8, k_head_size=128, v_head_size=128, block_size=256, check_perf=True)
        # test_pa_kv_cache_update([1], [0], num_kv_heads=8, k_head_size=128, v_head_size=128, block_size=256, enable_kvcache_compress=compress_kvcache, check_perf=False)
        # test_pa_kv_cache_update([1024], [0], num_kv_heads=2, k_head_size=16, v_head_size=16, block_size=32, check_perf=False)
        # test_pa_kv_cache_update([129], [0], num_kv_heads=2, k_head_size=64, v_head_size=64, block_size=16, check_perf=True)