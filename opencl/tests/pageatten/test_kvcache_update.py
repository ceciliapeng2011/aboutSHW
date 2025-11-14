import os
import time
import math

from numpy import diff
import numpy as np
import torch
import functools

from clops import cl
from clops import compare
from clops.utils import Colors

kv_cache_compression_enabled = 1
kv_cache_compression_by_channel = 1  # 0 = by-token, 1 = by-channel
TEST_GROUP_NUM = 16

# Host feature toggles for new incremental decode & stats instrumentation
ENABLE_INCREMENTAL_DECODE = True     # passes -DENABLE_KV_CACHE_INCREMENTAL_DECODE=1
ENABLE_INCREMENTAL_STATS  = False     # passes -DENABLE_KV_CACHE_STATS=1 and allocates stats buffers

# Debug instrumentation: enable kernel counters for by-channel path
ENABLE_KEY_BY_CHANNEL_DEBUG = False  # False to disable compiling debug counters (only meaningful for by-channel path)
class pa_kvcache_update_cm:
    def __init__(self, num_kv_heads, k_head_size, v_head_size, block_size):
        self.num_kv_heads = num_kv_heads
        self.k_head_size = k_head_size
        self.v_head_size = v_head_size
        self.block_size = block_size
        self.wg_size = 16

        src = r'''#include "pa_kv_cache_update_ref.hpp"'''
        cwd = os.path.dirname(os.path.realpath(__file__))
        print(f"compiling {cwd} {num_kv_heads=} {k_head_size=} {v_head_size=} ...")

        if kv_cache_compression_enabled:
            adjusted_k_head_size = k_head_size + 4
            adjusted_v_head_size = v_head_size + 4
        else:
            adjusted_k_head_size = k_head_size
            adjusted_v_head_size = v_head_size

        jit_option = '-abortonspill -noschedule '
        if kv_cache_compression_enabled and kv_cache_compression_by_channel:
            compression_flag = " -DKV_CACHE_COMPRESSION_PER_CHANNEL=1"
        elif kv_cache_compression_enabled:
            compression_flag = " -DKV_CACHE_COMPRESSION_PER_TOKEN=1"
        else:
            compression_flag = ""

        # Only compile debug counters for by-channel compression; by-token path omits macro to avoid extra kernel parameter.
        debug_flag = " -DENABLE_KEY_BY_CHANNEL_DEBUG" if (ENABLE_KEY_BY_CHANNEL_DEBUG and kv_cache_compression_by_channel) else ""
        incr_flag  = " -DENABLE_KV_CACHE_INCREMENTAL_DECODE=1" if (kv_cache_compression_by_channel and ENABLE_INCREMENTAL_DECODE) else ""
        stats_flag = " -DENABLE_KV_CACHE_STATS=1" if (kv_cache_compression_by_channel and ENABLE_INCREMENTAL_STATS) else ""

        self.kernels = cl.kernels(
            src,
            (f' -cmc -Qxcm_jit_option="{jit_option}" -Qxcm_register_file_size=256 -mCM_printregusage -I{cwd}'
             f" -DKV_HEADS_NUM={num_kv_heads}"
             f" -DK_HEAD_SIZE={k_head_size}"
             f" -DV_HEAD_SIZE={v_head_size}"
             f" -DADJUSTED_K_HEAD_SIZE={adjusted_k_head_size}"
             f" -DADJUSTED_V_HEAD_SIZE={adjusted_v_head_size}"
             f" -DPAGED_ATTENTION_BLOCK_SIZE={self.block_size}"
             f" -DWG_SIZE={self.wg_size}"
             f" -DGROUP_NUM={TEST_GROUP_NUM}"
             f"{compression_flag}"
             f"{incr_flag}{stats_flag}"
             #  f"{debug_flag}"
             f" -mdump_asm -g2")
        )

        if ENABLE_KEY_BY_CHANNEL_DEBUG and kv_cache_compression_enabled and kv_cache_compression_by_channel:
            # 12 uint32 slots per head (see kernel):
            # [0]=update_calls, [1]=prefill_windows,
            # [2]=vmin_blk0_grp0 (half bits), [3]=vmax_blk0_grp0,
            # [4]=vmin_blk0_grp1, [5]=vmax_blk0_grp1,
            # [6]=scale_inv_blk0_grp0, [7]=zp_blk0_grp0,
            # [8]=scale_inv_blk0_grp1, [9]=zp_blk0_grp1,
            # [10]=vmin_blk1_grp0, [11]=vmax_blk1_grp0
            self.debug_counters = cl.tensor(np.zeros(num_kv_heads * 12, dtype=np.uint32))
        else:
            self.debug_counters = None

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

        t_key = cl.tensor(key.to(torch.float16).detach().numpy())
        t_value = cl.tensor(value.to(torch.float16).detach().numpy())

        if kv_cache_compression_enabled:
            kv_cache_type = torch.uint8
        else:
            kv_cache_type = torch.float16
        t_key_cache = cl.tensor(key_cache.to(kv_cache_type).detach().numpy())
        t_value_cache = cl.tensor(value_cache.to(kv_cache_type).detach().numpy())

        t_block_indices=cl.tensor(torch.tensor(block_indices).to(torch.int32).detach().numpy())
        t_past_lens=cl.tensor(torch.tensor(past_lens).to(torch.int32).detach().numpy())
        t_block_indices_begins=cl.tensor(torch.tensor(block_indices_begins).to(torch.int32).detach().numpy())
        t_subsequence_begins=cl.tensor(torch.tensor(subsequence_begins).to(torch.int32).detach().numpy())

        is_prefill_stage = 1 if all(p == 0 for p in past_lens) else 0
        print(f'{Colors.GREEN} pa_kv_cache_update: {is_prefill_stage=}, {past_lens=} {subsequence_begins=} {block_indices_begins=} {block_indices=} {Colors.END}')

        if is_prefill_stage:
            prefill_block_records = []  # list of (seq_idx, phys_block_id, global_start, global_end)
            for seq_idx in range(batch_size_in_sequences):
                new_tokens = subsequence_begins[seq_idx+1] - subsequence_begins[seq_idx]
                if new_tokens == 0:
                    continue
                seq_begin = subsequence_begins[seq_idx]
                required_blocks = (new_tokens + self.block_size - 1) // self.block_size
                seq_block_indices_begin = block_indices_begins[seq_idx]
                for lb in range(required_blocks):
                    global_start = seq_begin + lb * self.block_size
                    global_end   = min(seq_begin + new_tokens, global_start + self.block_size)
                    phys_block_id = block_indices[seq_block_indices_begin + lb]
                    prefill_block_records.append((seq_idx, int(phys_block_id), int(global_start), int(global_end)))

            if len(prefill_block_records) == 0:
                is_prefill_stage = 0
                prefill_blocks = 0
                t_blocked_indexes_start          = cl.tensor(torch.zeros(1, dtype=torch.int32).numpy())
                t_blocked_indexes_end            = cl.tensor(torch.zeros(1, dtype=torch.int32).numpy())
                t_gws_seq_indexes_correspondence = cl.tensor(torch.zeros(1, dtype=torch.int32).numpy())
            else:
                prefill_blocks = len(prefill_block_records)
                arr_start = torch.tensor([rec[2] for rec in prefill_block_records], dtype=torch.int32)
                arr_end   = torch.tensor([rec[3] for rec in prefill_block_records], dtype=torch.int32)
                arr_seq   = torch.tensor([rec[0] for rec in prefill_block_records], dtype=torch.int32)
                t_blocked_indexes_start          = cl.tensor(arr_start.numpy())
                t_blocked_indexes_end            = cl.tensor(arr_end.numpy())
                t_gws_seq_indexes_correspondence = cl.tensor(arr_seq.numpy())

            GWS = [max(prefill_blocks, 1), self.num_kv_heads, 1]
            LWS = [1, 1, 1]
        else:
            wg_count = (batch_size_in_tokens + self.wg_size - 1) // self.wg_size
            GWS = [1, self.num_kv_heads, int(wg_count * self.wg_size)]
            LWS = [1, 1, self.wg_size]
            print("GWS:", GWS, "LWS:", LWS)
            zero_i32 = torch.zeros(1, dtype=torch.int32)
            t_blocked_indexes_start          = cl.tensor(zero_i32.numpy())
            t_blocked_indexes_end            = cl.tensor(zero_i32.numpy())
            t_gws_seq_indexes_correspondence = cl.tensor(zero_i32.numpy())

        # ================= Optional stats buffers (incremental decode) =================
        # Only allocate for by-channel & stats enabled
        stat_buffers = None
        if (kv_cache_compression_enabled and kv_cache_compression_by_channel and ENABLE_INCREMENTAL_STATS):
            num_blocks = len(block_indices)
            stat_len = num_blocks * self.num_kv_heads  # one slot per (phys_block, head)
            self.stat_inc_blocks      = cl.tensor(np.zeros(stat_len, dtype=np.uint32))
            self.stat_full_blocks     = cl.tensor(np.zeros(stat_len, dtype=np.uint32))
            self.stat_expanded_groups = cl.tensor(np.zeros(stat_len, dtype=np.uint32))
            self.stat_inc_max_abs_err = cl.tensor(np.zeros(stat_len, dtype=np.float32))
            self.stat_inc_mse_acc     = cl.tensor(np.zeros(stat_len, dtype=np.float32))
            stat_buffers = [self.stat_inc_blocks, self.stat_full_blocks, self.stat_expanded_groups,
                            self.stat_inc_max_abs_err, self.stat_inc_mse_acc]

        for i in range(0, n_repeats):
            print(f'{Colors.GREEN}calling pa_kv_cache_update {GWS=} {LWS=} {key_pitch=} {val_pitch=} {batch_size_in_sequences=} prefill={is_prefill_stage} at {i}/{n_repeats}{Colors.END}')
            enqueue_args = [
                "pa_kv_cache_update", GWS, LWS,
                t_key, t_value,
                t_past_lens, t_block_indices, t_block_indices_begins, t_subsequence_begins,
                t_key_cache, t_value_cache,
                key_pitch, val_pitch, batch_size_in_sequences,
                t_blocked_indexes_start,
                t_blocked_indexes_end,
                t_gws_seq_indexes_correspondence,
                int(is_prefill_stage)
            ]
            if self.debug_counters is not None:
                enqueue_args.append(self.debug_counters)
            if stat_buffers is not None:
                enqueue_args.extend(stat_buffers)
            self.kernels.enqueue(*enqueue_args)

            ns = cl.finish()
            for i_time, time_opt in enumerate(ns):
                print(f'(pa_kv_cache_update)TPUT_{i_time}:[{key.numel()=}]+[{value.numel()=}] {time_opt*1e-3:,.0f} us')
                if kv_cache_compression_enabled:
                    total_bytes = batch_size_in_tokens * self.num_kv_heads * (3 * self.k_head_size + 3 * self.v_head_size + 8)
                else:
                    total_bytes = batch_size_in_tokens * self.num_kv_heads * (4 * self.k_head_size + 4 * self.v_head_size)
                tput = total_bytes / time_opt
                print(f'(pa_kv_cache_update)TPUT_{i_time}:[{total_bytes*1e-6:,} MB] {tput/1e3:,.2f} GB/s')

        # ================== Stats readback & summary ==================
        if stat_buffers is not None:
            inc  = self.stat_inc_blocks.numpy().astype(np.uint32)
            full = self.stat_full_blocks.numpy().astype(np.uint32)
            expg = self.stat_expanded_groups.numpy().astype(np.uint32)
            maxe = self.stat_inc_max_abs_err.numpy().astype(np.float32)
            mse  = self.stat_inc_mse_acc.numpy().astype(np.float32)
            inc_sum  = int(inc.sum()); full_sum = int(full.sum()); exp_sum = int(expg.sum())
            maxe_sum = float(maxe.sum()); mse_sum = float(mse.sum())
            total_attempts = inc_sum + full_sum
            inc_ratio = (inc_sum / total_attempts) if total_attempts else 0.0
            avg_max_abs_err = (maxe_sum / inc_sum) if inc_sum else 0.0
            avg_mse = (mse_sum / inc_sum) if inc_sum else 0.0
            print(f"[INC-STATS] attempts={total_attempts} inc={inc_sum} full={full_sum} expanded_groups={exp_sum} inc_ratio={inc_ratio:.4f}")
            print(f"[INC-STATS] avg_max_abs_err={avg_max_abs_err:.6f} avg_mse={avg_mse:.6f}")

        if self.debug_counters is not None:
            dbg = self.debug_counters.numpy().astype(np.uint32)
            print('[DEBUG] per-head counters & quant stats:')
            import struct
            def unpack_half(u32):
                return struct.unpack('e', struct.pack('H', int(u32 & 0xFFFF)))[0]
            for h in range(self.num_kv_heads):
                base = h*12
                upd = int(dbg[base+0]); pre = int(dbg[base+1])
                vmin_g0 = unpack_half(dbg[base+2]); vmax_g0 = unpack_half(dbg[base+3])
                vmin_g1 = unpack_half(dbg[base+4]); vmax_g1 = unpack_half(dbg[base+5])
                scale_inv_g0 = unpack_half(dbg[base+6]); zp_g0 = unpack_half(dbg[base+7])
                scale_inv_g1 = unpack_half(dbg[base+8]); zp_g1 = unpack_half(dbg[base+9])
                vmin_blk1_g0 = unpack_half(dbg[base+10]); vmax_blk1_g0 = unpack_half(dbg[base+11])
                print(f'  head {h}: update_calls={upd}, prefill_windows={pre}')
                print(f'    blk0 grp0: vmin={vmin_g0:.6f} vmax={vmax_g0:.6f} scale_inv={scale_inv_g0:.6f} zp={zp_g0:.6f}')
                if (vmin_g1 != 0.0 or vmax_g1 != 0.0 or scale_inv_g1 != 0.0 or zp_g1 != 0.0):
                    print(f'    blk0 grp1: vmin={vmin_g1:.6f} vmax={vmax_g1:.6f} scale_inv={scale_inv_g1:.6f} zp={zp_g1:.6f}')
                if (vmin_blk1_g0 != 0.0 or vmax_blk1_g0 != 0.0):
                    print(f'    blk1 grp0: vmin={vmin_blk1_g0:.6f} vmax={vmax_blk1_g0:.6f}')

        return t_key_cache.numpy(), t_value_cache.numpy()
                    
    @staticmethod
    @functools.cache
    def create_instance(num_kv_heads, k_head_size, v_head_size, block_size):
        return pa_kvcache_update_cm(num_kv_heads, k_head_size, v_head_size, block_size)


def test_pa_kv_cache_update(num_tokens:list, past_lens:list, num_kv_heads=1, k_head_size=64, v_head_size=64, block_size=16, check_perf=False):   
    batch_size_in_sequences = len(num_tokens)
    assert(batch_size_in_sequences == len(past_lens))

    # prepare page attention inputs
    key_data = []
    value_data = []
    subsequence_begins = []
    block_indices_begins = []    
    subsequence_begins.append(0)
    block_indices_begins.append(0)
    full_lengths = []
    for i in range(batch_size_in_sequences):
        subsequence_length = num_tokens[i] + past_lens[i]
        full_lengths.append(subsequence_length)

        k = torch.rand(subsequence_length, num_kv_heads*k_head_size).to(dtype=torch.float16)
        v = torch.rand(subsequence_length, num_kv_heads*v_head_size).to(dtype=torch.float16)
        print(k.shape)
        print(v.shape)
        key_data.append(k)
        value_data.append(v)

        subsequence_start_pos = subsequence_begins[i]
        # Use full subsequence length (past + new) so kernel can load past tokens directly
        subsequence_end_pos = subsequence_start_pos + subsequence_length
        subsequence_begins.append(subsequence_end_pos)

        required_blocks = (subsequence_length + block_size - 1) // block_size

        block_indices_start_pos = block_indices_begins[i];
        block_indices_end_pos = block_indices_start_pos + required_blocks;
        block_indices_begins.append(block_indices_end_pos);

    # simulate random block allocation
    num_blocks = block_indices_begins[-1]
    print(num_blocks)
    block_indices = torch.arange(num_blocks)
    perm_idx = torch.randperm(block_indices.shape[0])
    inv_per_idx = torch.argsort(perm_idx)
    block_indices = block_indices[inv_per_idx]
    # print(f'{Colors.BLUE} ============ {subsequence_begins=} {Colors.END}')
    # print(f'{Colors.BLUE} ============ {block_indices_begins=} {Colors.END}')
    # print(f'{Colors.BLUE} ============ {block_indices=} {Colors.END}')

    # generate key / value inputs
    def get_kv_input(num_kv_heads, head_size, input_data):
        batch_size_in_tokens = subsequence_begins[-1]
        mem = torch.zeros(batch_size_in_tokens, num_kv_heads*head_size).to(torch.float16)
        for i in range(batch_size_in_sequences):
            # Provide full (past + new) tokens to kernel; past_lens portion will be skipped by early return logic
            mem[subsequence_begins[i] : subsequence_begins[i+1], :] = input_data[i][: full_lengths[i], :]
        return mem
    key = get_kv_input(num_kv_heads, k_head_size, key_data)
    value = get_kv_input(num_kv_heads, v_head_size, value_data)
    # print(key)
    # print(f'{Colors.BLUE} ============ {key.shape=} {key.is_contiguous()=}" {Colors.END}')
    # print(f'{Colors.BLUE} ============ {value.shape=} {value.is_contiguous()=}" {Colors.END}')
    # print(f'{Colors.BLUE} {key_data=} {Colors.END}')
    # print(f'{Colors.BLUE} {key=} {Colors.END}')
    # print(f'{Colors.BLUE} {value_data=} {Colors.END}')
    # print(f'{Colors.BLUE} {value=} {Colors.END}')

    def round_to_nearest_even(x):
        floor_x = torch.floor(x)
        frac = x - floor_x
        is_half = frac == 0.5
        return torch.where(
            is_half,
            floor_x + ((floor_x % 2) == 1).to(x.dtype),
            torch.round(x)
        )

    def quant_per_channel_cm_layout_exact(kv: torch.Tensor, group_num: int = 1):
        assert kv.dtype == torch.float16
        B,H,T,C = kv.shape
        assert C % group_num == 0
        G   = group_num
        GSZ = C // G

        kv_f32 = kv.to(torch.float32)
        q_all = torch.empty((B,H,T,C), dtype=torch.int8, device=kv.device)
        comp_bytes = torch.empty((B,H, 4*G), dtype=torch.uint8, device=kv.device)

        for g in range(G):
            c0, c1 = g*GSZ, (g+1)*GSZ
            blk = kv_f32[..., c0:c1]                           # [B,H,T,GSZ]

            vmin = blk.amin(dim=(2,3), keepdim=True)           # [B,H,1,1]
            vmax = blk.amax(dim=(2,3), keepdim=True)           # [B,H,1,1]
            rng  = vmax - vmin
            min_rng = vmax.abs() * 0.1
            rng  = torch.where(rng <= min_rng,
                            rng + torch.maximum(min_rng, torch.tensor(1.0, device=kv.device)),
                            rng)
            scale = 255.0 / rng                                # [B,H,1,1]
            zp    = -vmin * scale                              # [B,H,1,1]
            # q = torch.floor(blk * scale + zp + 0.5).clamp(0,255).to(torch.uint8)  # [B,H,T,GSZ]
            q = round_to_nearest_even(blk * scale + zp).clamp(0, 255).to(torch.uint8)
            q_all[..., c0:c1] = q.to(torch.int8)
            scale_inv = (1.0/scale).to(torch.float16).view(B,H)   # [B,H]
            zp_f16    = zp.to(torch.float16).view(B,H)            # [B,H]
            comp_pair = torch.stack([scale_inv, zp_f16], dim=-1)  # [B,H,2]
            comp_u8   = comp_pair.view(torch.uint8)               # [B,H,4]
            comp_bytes[..., 4*g:4*(g+1)] = comp_u8

        data_bytes = q_all.view(torch.uint8).reshape(B,H, T*C)     # [B,H, T*C]
        out = torch.cat([data_bytes, comp_bytes], dim=2)           # [B,H, T*C + 4*G]
        return out

    def get_kv_cache_ref_by_channel_cm_layout_exact(
        num_blocks, block_size, num_kv_heads, head_size, input_data,
        past_lens, block_indices, block_indices_begins, num_tokens,
        group_num=1, skip_input=True
    ):
        """
        output:
            [num_blocks, num_kv_heads, block_size*head_size + 4*group_num]  (uint8)
        """
        per_head_bytes = block_size * head_size + 4 * group_num
        out = torch.zeros(num_blocks, num_kv_heads, per_head_bytes, dtype=torch.uint8)

        for seq_idx in range(len(input_data)):
            process_len = past_lens[seq_idx] if skip_input else past_lens[seq_idx] + num_tokens[seq_idx]
            if process_len == 0:
                continue

            blocks_num = (process_len + block_size - 1) // block_size
            for b in range(blocks_num):
                blk_len = process_len % block_size if b == blocks_num - 1 else block_size
                if blk_len == 0: blk_len = block_size
                blk_start, blk_end = b*block_size, b*block_size + blk_len

                for h in range(num_kv_heads):
                    kv_block = input_data[seq_idx][blk_start:blk_end, h*head_size:(h+1)*head_size].to(torch.float16)
                    if blk_len < block_size:
                        pad = torch.zeros(block_size - blk_len, head_size, dtype=torch.float16)
                        kv_block = torch.cat([kv_block, pad], dim=0)

                    kv_bhtc = kv_block.unsqueeze(0).unsqueeze(0)  # [1,1,T,C]
                    packed  = quant_per_channel_cm_layout_exact(kv_bhtc, group_num=group_num)[0,0]  # [T*C + 4G]

                    blk_pos = int(block_indices[int(block_indices_begins[seq_idx]) + b])
                    out[blk_pos, h, :] = packed

        return out




    def round_to_even(tensor):
        rounded = torch.floor(tensor + 0.5)
        adjustment = (rounded % 2 != 0) & (torch.abs(tensor - rounded) == 0.5000)
        adjustment = adjustment | (rounded > 255)  # also handle overflow
        return rounded - adjustment.to(rounded.dtype)

    def quant_per_token(kv):
        blk_num, kv_heads, blksz, *_ = kv.shape
        kv_max = kv.amax(dim=-1, keepdim = True).to(dtype=torch.float)
        kv_min = kv.amin(dim=-1, keepdim = True).to(dtype=torch.float)
        qrange = (kv_max - kv_min).to(dtype=torch.float)

        U8_MAX = torch.tensor(255.0, dtype=torch.float)
        U8_MIN = torch.tensor(0.0, dtype=torch.float)
        U8_RANGE = (U8_MAX - U8_MIN).to(dtype=torch.float)
        kv_scale = ((U8_RANGE)/qrange).to(dtype=torch.float)
        kv_scale_div = (1.0/kv_scale).to(dtype=torch.float)
        kv_zp = ((0.0-kv_min)*kv_scale+U8_MIN).to(dtype=torch.float)

        # kv_u8 = torch.round((kv*kv_scale+kv_zp)).to(dtype=torch.uint8).reshape(blk_num,kv_heads,-1)
        kv_u8 = round_to_even((kv*kv_scale+kv_zp)).to(dtype=torch.uint8).reshape(blk_num,kv_heads,-1)

        dq_scale = kv_scale_div.to(dtype=torch.half).view(dtype=torch.uint8).reshape(blk_num,kv_heads,-1)
        kv_zp = kv_zp.to(dtype=torch.half).view(dtype=torch.uint8).reshape(blk_num,kv_heads,-1)
        if blksz < block_size:
            kv_pad = torch.zeros(blk_num, kv_heads, (block_size - blksz)*k_head_size).to(dtype=torch.uint8)
            kv_u8 = torch.cat((kv_u8, kv_pad), dim=-1)
            scale_zp_pad = torch.zeros(blk_num, kv_heads, (block_size - blksz)*2).to(dtype=torch.uint8)
            dq_scale = torch.cat((dq_scale, scale_zp_pad), dim=-1)
            kv_zp = torch.cat((kv_zp, scale_zp_pad), dim=-1)

        return torch.concat((kv_u8, dq_scale, kv_zp), dim=-1)

    # generate key_cache / value_cache
    # input_data list of torch.Tensor with shape [subsequence_length, num_kv_heads, kv_head_size] for each sequence
    def get_kv_cache(num_blocks, block_size, num_kv_heads, head_size, input_data, skip_input = True):
        cache_data = torch.zeros(num_blocks, block_size, num_kv_heads*head_size).to(torch.float16)
        for i in range(batch_size_in_sequences):
            process_len = past_lens[i] if skip_input else past_lens[i] + num_tokens[i]
            if process_len > 0:
                blocks_num = (process_len + block_size - 1) // block_size
                for block_idx in range(blocks_num):
                    last_token_idx = process_len % block_size if block_idx == blocks_num -1 else block_size
                    if last_token_idx == 0: last_token_idx = block_size
                    # print(f'{Colors.RED} {block_idx=} {blocks_num=} {process_len=} {last_token_idx=} {Colors.END}')
                    for token_idx in range(last_token_idx):
                        input_token_offset = block_idx * block_size + token_idx
                        block_pos = block_indices[block_indices_begins[i] + block_idx]
                        cache_data[block_pos, token_idx, :] = input_data[i][input_token_offset, :]
        return cache_data.reshape(num_blocks, block_size, num_kv_heads, head_size).transpose(1, 2).contiguous()
    
    def get_kv_cache_u8(num_blocks, block_size, num_kv_heads, head_size, input_data, skip_input = True):
        cache_data = torch.zeros(num_blocks, num_kv_heads, block_size * (head_size + 4)).to(torch.uint8)
        for i in range(batch_size_in_sequences):
            process_len = past_lens[i] if skip_input else past_lens[i] + num_tokens[i]
            if process_len > 0:
                blocks_num = (process_len + block_size - 1) // block_size
                for block_idx in range(blocks_num):
                    last_token_idx = process_len % block_size if block_idx == blocks_num -1 else block_size
                    if last_token_idx == 0: last_token_idx = block_size
                    # print(f'{Colors.RED} {block_idx=} {blocks_num=} {process_len=} {last_token_idx=} {Colors.END}')
                    for h in range(num_kv_heads):
                        # input_data[seq_num][token_num, head_size * kv_head_num]
                        token_start_idx = block_idx * block_size
                        token_end_idx = token_start_idx + last_token_idx
                        input_block_per_head = input_data[i][token_start_idx:token_end_idx, h*head_size:(h+1)*head_size].reshape(1, 1, -1, head_size)
                        input_block_per_head_q = quant_per_token(input_block_per_head).reshape(-1)

                        # print()
                        # print(f'head_idx = {h} token_start_idx = {token_start_idx} token_end_idx = {token_end_idx} last_token_idx = {last_token_idx}')
                        # print('input_block_per_head.shape = {input_block_per_head.shape}')
                        # print('input_block_per_head = ',input_block_per_head)
                        # print('input_block_per_head_q.shape = ',input_block_per_head_q.reshape(1,1,-1,head_size).shape)
                        # print('input_block_per_head_q = ',input_block_per_head_q.reshape(1,1,-1,head_size))

                        block_pos = block_indices[block_indices_begins[i] + block_idx]
                        cache_data[block_pos, h, :] = input_block_per_head_q
        # if skip_input == False:
        #     print("cache_data =", cache_data)
        return cache_data
    
    
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

    def compare_kcache(ref_cache, opt_cache,
                    block_size, head_size,
                    num_blocks, num_kv_heads,
                    GROUP_NUM,
                    by_channel,
                    past_lens=None,
                    num_tokens=None):
        """
        ref_cache, opt_cache: [B, H, bytes]  (uint8)
        """
        if not kv_cache_compression_enabled:
            compare(ref_cache.detach().numpy(), opt_cache.detach().numpy())
            return

        if by_channel:
            data_bytes = block_size * head_size
            full_ref = ref_cache[:, :, :data_bytes].to(torch.int32)
            full_opt = opt_cache[:, :, :data_bytes].to(torch.int32)

            if past_lens is not None and num_tokens is not None:
                new_rows_ref = []
                new_rows_opt = []
                for seq_idx in range(len(past_lens)):
                    past_len_seq = int(past_lens[seq_idx])
                    new_tok_seq  = int(num_tokens[seq_idx])
                    if new_tok_seq == 0:
                        continue
                    total_tokens_seq = past_len_seq + new_tok_seq
                    first_new = past_len_seq
                    last_new_excl = total_tokens_seq
                    new_start_blk = first_new // block_size
                    new_end_blk   = (last_new_excl - 1) // block_size
                    for lb in range(new_start_blk, new_end_blk + 1):
                        phys_block = int(block_indices[int(block_indices_begins[seq_idx]) + lb])
                        blk_token_beg = lb * block_size
                        blk_token_end_excl = blk_token_beg + block_size
                        local_beg = max(first_new, blk_token_beg) - blk_token_beg
                        local_end = min(last_new_excl, blk_token_end_excl) - blk_token_beg
                        for h in range(num_kv_heads):
                            beg_bytes = local_beg * head_size
                            end_bytes = local_end * head_size
                            slice_ref = full_ref[phys_block, h, beg_bytes:end_bytes]
                            slice_opt = full_opt[phys_block, h, beg_bytes:end_bytes]
                            new_rows_ref.append(slice_ref)
                            new_rows_opt.append(slice_opt)
                if len(new_rows_ref) == 0:
                    print('[WARN] no new rows extracted; nothing to compare.')
                    return
                cat_ref = torch.cat(new_rows_ref).to(torch.int32)
                cat_opt = torch.cat(new_rows_opt).to(torch.int32)
                compare(cat_ref.detach().numpy(), cat_opt.detach().numpy(), atol=1)
            else:
                compare(full_ref.detach().numpy(), full_opt.detach().numpy(), atol=1)
            return
        else:
            def decode(cache):
                data = cache[:, :, : block_size * head_size].to(torch.uint8)
                return data.to(torch.int32)
            ref = decode(ref_cache)
            opt = decode(opt_cache)
            compare(ref.detach().numpy(), opt.detach().numpy(), atol=1)

            ref_comp = ref_cache[:,:, block_size*head_size :].view(torch.float16)
            opt_comp = opt_cache[:,:, block_size*head_size :].view(torch.float16)
            compare(ref_comp.detach().numpy(), opt_comp.detach().numpy(), atol=1e-3)
            return




    if kv_cache_compression_enabled:
        if kv_cache_compression_by_channel:
            GROUP_NUM = TEST_GROUP_NUM
            print("num_blocks =", num_blocks, ", block_size =", block_size, ", num_kv_heads =", num_kv_heads, ", k_head_size =", k_head_size, ", v_head_size =", v_head_size, ", GROUP_NUM =", GROUP_NUM)
            key_cache = get_kv_cache_ref_by_channel_cm_layout_exact(
                num_blocks, block_size, num_kv_heads, k_head_size, key_data,
                past_lens, block_indices, block_indices_begins, num_tokens,
                group_num=GROUP_NUM, skip_input=True)
            
            print(key_cache.shape)

            value_cache = get_kv_cache_u8(num_blocks, block_size, num_kv_heads, v_head_size, value_data)

            print(value_cache.shape)

            key_cache_ref = get_kv_cache_ref_by_channel_cm_layout_exact(
                num_blocks, block_size, num_kv_heads, k_head_size, key_data,
                past_lens, block_indices, block_indices_begins, num_tokens,
                group_num=GROUP_NUM, skip_input=False)

            value_cache_ref = get_kv_cache_u8(num_blocks, block_size, num_kv_heads, v_head_size, value_data, False)

            # ---- Reference debug: per-group vmin/vmax for first block & head 0 ----
            try:
                if num_kv_heads > 0 and num_blocks > 0:
                    ref_block0_head0 = []  # list of token rows (full block_size or filled length)
                    subseq_len_total = num_tokens[0] + past_lens[0]
                    blk0_len = block_size if subseq_len_total >= block_size else subseq_len_total
                    raw_tokens = key_data[0][:blk0_len, :k_head_size]  # [blk0_len, k_head_size]
                    import torch as _torch
                    raw_tokens_f32 = _torch.tensor(raw_tokens, dtype=_torch.float16).to(_torch.float32)
                    GSZ = k_head_size // GROUP_NUM
                    for g in range(GROUP_NUM):
                        c0, c1 = g*GSZ, (g+1)*GSZ
                        grp_vals = raw_tokens_f32[:, c0:c1]
                        vmin = grp_vals.min().item()
                        vmax = grp_vals.max().item()
                        rng = vmax - vmin
                        min_rng = abs(vmax) * 0.1
                        rng_inflated = rng + (min_rng if min_rng > 1.0 else 1.0) if rng <= min_rng else rng
                        scale_inv = rng_inflated / 255.0
                        zp = -vmin * (255.0 / rng_inflated)
                        print(f"[REF-DBG] blk0 head0 grp{g}: vmin={vmin:.6f} vmax={vmax:.6f} diff={rng:.6f} min_rng={min_rng:.6f} scale_inv={scale_inv:.6f} zp={zp:.6f}")
            except Exception as e:
                print(f"[REF-DBG] exception gathering ref stats: {e}")
        else:
            key_cache = get_kv_cache_u8(num_blocks, block_size, num_kv_heads, k_head_size, key_data)
            value_cache = get_kv_cache_u8(num_blocks, block_size, num_kv_heads, v_head_size, value_data)

            # generate reference key/value cache
            key_cache_ref = get_kv_cache_u8(num_blocks, block_size, num_kv_heads, k_head_size, key_data, False)
            value_cache_ref = get_kv_cache_u8(num_blocks, block_size, num_kv_heads, v_head_size, value_data, False)
    else:
        key_cache = get_kv_cache(num_blocks, block_size, num_kv_heads, k_head_size, key_data)
        value_cache = get_kv_cache(num_blocks, block_size, num_kv_heads, v_head_size, value_data)
        # generate reference key/value cache
        key_cache_ref = get_kv_cache(num_blocks, block_size, num_kv_heads, k_head_size, key_data, False)
        value_cache_ref = get_kv_cache(num_blocks, block_size, num_kv_heads, v_head_size, value_data, False)
    
    # opt
    pa_cm = pa_kvcache_update_cm.create_instance(num_kv_heads, k_head_size, v_head_size, block_size)
    n_repeats = 20 if check_perf else 1
    out_key_cache, out_value_cache = pa_cm(key, value, key_cache, value_cache, past_lens, subsequence_begins, block_indices, block_indices_begins, n_repeats)

    if kv_cache_compression_enabled:
        out_key_cache=torch.tensor(out_key_cache).to(dtype=torch.uint8)
        out_value_cache=torch.tensor(out_value_cache).to(dtype=torch.uint8)

        data_bytes = block_size * k_head_size
        ref_data_region = key_cache_ref[:,:,:data_bytes]
        opt_data_region = out_key_cache[:,:,:data_bytes]
        masked_ref_preview = ref_data_region.clone()
        masked_opt_preview = opt_data_region.clone()
        if kv_cache_compression_by_channel and past_lens is not None and num_tokens is not None:
            for seq_idx in range(len(past_lens)):
                past_len_seq = int(past_lens[seq_idx]); new_tok_seq = int(num_tokens[seq_idx])
                total_tokens_seq = past_len_seq + new_tok_seq
                blocks_num_seq = (total_tokens_seq + block_size - 1) // block_size
                for lb in range(blocks_num_seq):
                    phys_block = int(block_indices[int(block_indices_begins[seq_idx]) + lb])
                    blk_token_beg = lb * block_size
                    blk_token_end_excl = min((lb + 1) * block_size, total_tokens_seq)
                    hist_beg = blk_token_beg
                    hist_end = min(blk_token_end_excl, past_len_seq)
                    if hist_end > hist_beg:
                        for h in range(num_kv_heads):
                            beg_bytes = (hist_beg - blk_token_beg) * k_head_size
                            end_bytes = (hist_end - blk_token_beg) * k_head_size
                            masked_ref_preview[phys_block, h, beg_bytes:end_bytes] = 0
                            masked_opt_preview[phys_block, h, beg_bytes:end_bytes] = 0
                    if blk_token_end_excl < (lb + 1) * block_size:
                        pad_beg_tok = blk_token_end_excl - blk_token_beg
                        for h in range(num_kv_heads):
                            beg_bytes = pad_beg_tok * k_head_size
                            end_bytes = block_size * k_head_size
                            masked_ref_preview[phys_block, h, beg_bytes:end_bytes] = 0
                            masked_opt_preview[phys_block, h, beg_bytes:end_bytes] = 0
        diff = (masked_opt_preview.to(torch.int32) - masked_ref_preview.to(torch.int32))
        idx = torch.nonzero(diff, as_tuple=False)
        if idx.numel() > 0:
            b,h,o = idx[0].tolist()
            tok = o // k_head_size
            ch  = o %  k_head_size
            print(f'[DEBUG] (masked) first mismatch: blk={b}, head={h}, tok={tok}, ch={ch}, ref={int(masked_ref_preview[b,h,o])}, opt={int(masked_opt_preview[b,h,o])}')
            lo = max(ch-8, 0); hi = min(ch+8, k_head_size-1)
            print('[DEBUG] (masked) ref token window:', masked_ref_preview[b,h,tok*k_head_size + lo: tok*k_head_size + hi + 1])
            print('[DEBUG] (masked) opt token window:', masked_opt_preview[b,h,tok*k_head_size + lo: tok*k_head_size + hi + 1])
            # Dump comp params (scale_inv, zp) per group for the mismatching block/head
            data_bytes_dbg = block_size * k_head_size
            if kv_cache_compression_by_channel:
                print('[DEBUG] comp params (scale_inv, zp) per group for ref vs opt:')
                for g in range(TEST_GROUP_NUM):
                    ref_comp_bytes = key_cache_ref[b, h, data_bytes_dbg + g*4 : data_bytes_dbg + (g+1)*4]
                    opt_comp_bytes = out_key_cache[b, h, data_bytes_dbg + g*4 : data_bytes_dbg + (g+1)*4]
                    ref_scale_inv, ref_zp = ref_comp_bytes.view(dtype=torch.float16).tolist()
                    opt_scale_inv, opt_zp = opt_comp_bytes.view(dtype=torch.float16).tolist()
                    print(f'  group {g}: ref(scale_inv={ref_scale_inv:.6f}, zp={ref_zp:.6f}) opt(scale_inv={opt_scale_inv:.6f}, zp={opt_zp:.6f})')
        else:
            print('[DEBUG] no mismatch in masked new-token rows before compare_kcache')

        compare_kcache(
            key_cache_ref, out_key_cache,
            block_size, k_head_size,
            num_blocks, num_kv_heads,
            1,
            by_channel = kv_cache_compression_by_channel,
            past_lens = past_lens,
            num_tokens = num_tokens
        )

        compare_kcache(
            value_cache_ref, out_value_cache,
            block_size, v_head_size,
            num_blocks, num_kv_heads,
            1,
            by_channel = kv_cache_compression_by_channel
        )
    else:
        compare(key_cache_ref.detach().numpy(), out_key_cache)
        compare(value_cache_ref.detach().numpy(), out_value_cache)
    print(f'{Colors.GREEN}kv_cache_update passed{Colors.END}')

if __name__ == "__main__":
    torch.manual_seed(3)
    torch.set_printoptions(linewidth=1024)
    
    cl.profiling(True)

    test_pa_kv_cache_update([32*1024], [0], num_kv_heads=8, k_head_size=128, v_head_size=128, block_size=256, check_perf=False)
    test_pa_kv_cache_update([64*1024], [0], num_kv_heads=8, k_head_size=128, v_head_size=128, block_size=256, check_perf=False)
    test_pa_kv_cache_update([128*1024], [0], num_kv_heads=8, k_head_size=128, v_head_size=128, block_size=256, check_perf=False)
    test_pa_kv_cache_update([32*1024], [4*1024], num_kv_heads=8, k_head_size=128, v_head_size=128, block_size=256, check_perf=False)
    test_pa_kv_cache_update([128*1024], [1*1024], num_kv_heads=8, k_head_size=128, v_head_size=128, block_size=256, check_perf=False)
    
    test_pa_kv_cache_update([1024*128], [1024], num_kv_heads=8, k_head_size=128, v_head_size=128, block_size=256, check_perf=True)
    test_pa_kv_cache_update([1023], [0], num_kv_heads=8, k_head_size=128, v_head_size=128, block_size=256, check_perf=False)
    test_pa_kv_cache_update([256], [0], num_kv_heads=8, k_head_size=128, v_head_size=128, block_size=256, check_perf=False)

    # Added targeted mixed decode scenario: past_lens != 0 and new tokens not aligned to block boundary
    # Expect debug counters (update_calls may still be 0 with current kernel logic) for inspection.
