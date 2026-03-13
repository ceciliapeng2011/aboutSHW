import functools
import os
from dataclasses import dataclass

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from clops import cl

cl.profiling(True)
torch.manual_seed(0)
torch.set_printoptions(linewidth=1024)

def get_cm_grf_width():
    cm_kernels = cl.kernels(r'''
    extern "C" _GENX_MAIN_ void cm_get_grf_width(int * info [[type("svmptr_t")]]) {
        info[0] = CM_GRF_WIDTH;
    }''', "-cmc")
    t_info = cl.tensor([2], np.dtype(np.int32))
    cm_kernels.enqueue("cm_get_grf_width", [1], [1], t_info)
    return t_info.numpy()[0]


CM_GRF_WIDTH = get_cm_grf_width()
print(f"{CM_GRF_WIDTH=}")


def quan_per_token(kv):
    blk_num, kv_heads, blksz, *_ = kv.shape
    kv_max = kv.amax(dim=-1, keepdim=True)
    kv_min = kv.amin(dim=-1, keepdim=True)
    qrange = kv_max - kv_min

    int_max = 255.0
    int_min = 0.0
    int_range = int_max - int_min
    kv_scale = (int_range / qrange).to(dtype=torch.half)
    kv_zp = ((0.0 - kv_min) * kv_scale + int_min).to(dtype=torch.half)
    kv_int8 = torch.round(kv * kv_scale + kv_zp).to(dtype=torch.uint8).reshape(blk_num, kv_heads, -1)

    dq_scale = (1.0 / kv_scale).view(dtype=torch.uint8).reshape(blk_num, kv_heads, -1)
    kv_zp = kv_zp.view(dtype=torch.uint8).reshape(blk_num, kv_heads, -1)
    return torch.concat((kv_int8, dq_scale, kv_zp), dim=-1)


def DIV_UP(x, y):
    return (x + y - 1) // y


@dataclass(frozen=True)
class SubsequenceDescriptor:
    num_tokens: int
    past_len: int = 0

    def __post_init__(self):
        if self.num_tokens <= 0:
            raise ValueError(f"num_tokens must be positive, got {self.num_tokens}")
        if self.past_len < 0:
            raise ValueError(f"past_len must be non-negative, got {self.past_len}")

    @property
    def total_seq_len(self) -> int:
        return self.past_len + self.num_tokens


@dataclass(frozen=True)
class SessionDescriptor:
    num_input_tokens: int
    num_output_tokens: int

    def __post_init__(self):
        if self.num_input_tokens < 0:
            raise ValueError(f"num_input_tokens must be non-negative, got {self.num_input_tokens}")
        if self.num_output_tokens < 0:
            raise ValueError(f"num_output_tokens must be non-negative, got {self.num_output_tokens}")
        if self.num_output_tokens > 1:
            # NOTE: Temporary kernel limitation in this harness:
            # prefill-stage path only, allowing at most one decode token per session.
            # TODO(enable_mixed_prefill_decode): remove this guard once kernel supports
            # true mixed prefill/decode iteration in a single scheduler flow.
            raise ValueError(
                "num_output_tokens must be <= 1 for this prefill-stage kernel path (kernel currently supports only one decode token)"
            )
        if self.num_input_tokens == 0 and self.num_output_tokens == 0:
            raise ValueError("SessionDescriptor cannot have both num_input_tokens and num_output_tokens as 0")

    @property
    def total_tokens(self) -> int:
        return self.num_input_tokens + self.num_output_tokens


@dataclass(frozen=True)
class RoundPlan:
    sequence_ids: tuple[int, ...]
    subsequences: tuple[SubsequenceDescriptor, ...]

    @property
    def batch_size_in_sequences(self) -> int:
        return len(self.subsequences)

    @property
    def batch_size_in_tokens(self) -> int:
        return sum(subsequence.num_tokens for subsequence in self.subsequences)

    @property
    def max_context_len(self) -> int:
        return max(subsequence.total_seq_len for subsequence in self.subsequences)


@dataclass(frozen=True)
class PagedAttentionInputs:
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    key_cache: torch.Tensor
    value_cache: torch.Tensor
    past_lens: torch.Tensor
    subsequence_begins: torch.Tensor
    block_indices: torch.Tensor
    block_indices_begins: torch.Tensor
    max_context_len: int
    sequence_ids: tuple[int, ...]
    subsequences: tuple[SubsequenceDescriptor, ...]


@dataclass(frozen=True)
class SequenceTensors:
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor


class ContinuousBatchingScheduler:
    def __init__(self, max_num_batched_tokens: int, max_subsequence_tokens: int | None = None, dynamic_split_fuse: bool = True):
        if max_num_batched_tokens <= 0:
            raise ValueError("max_num_batched_tokens must be positive")
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_subsequence_tokens = max_subsequence_tokens or max_num_batched_tokens
        self.dynamic_split_fuse = dynamic_split_fuse

    def schedule(self, sessions: list[SessionDescriptor]) -> list[RoundPlan]:
        remaining_input_tokens = [session.num_input_tokens for session in sessions]
        remaining_output_tokens = [session.num_output_tokens for session in sessions]
        emitted_tokens = [0 for _ in sessions]
        round_plans: list[RoundPlan] = []

        while any((remaining_input_tokens[i] > 0 or remaining_output_tokens[i] > 0) for i in range(len(sessions))):
            capacity = self.max_num_batched_tokens
            round_sequence_ids: list[int] = []
            round_subsequences: list[SubsequenceDescriptor] = []

            if self.dynamic_split_fuse:
                # Generation first: one token per session when input is exhausted.
                for sequence_id, _session in enumerate(sessions):
                    if capacity == 0:
                        break
                    if remaining_input_tokens[sequence_id] == 0 and remaining_output_tokens[sequence_id] > 0:
                        scheduled_tokens = 1
                        round_sequence_ids.append(sequence_id)
                        round_subsequences.append(
                            SubsequenceDescriptor(num_tokens=scheduled_tokens, past_len=emitted_tokens[sequence_id])
                        )
                        emitted_tokens[sequence_id] += scheduled_tokens
                        remaining_output_tokens[sequence_id] -= scheduled_tokens
                        capacity -= scheduled_tokens

                # Greedy prompt chunk split by request order.
                for sequence_id, _session in enumerate(sessions):
                    if capacity == 0:
                        break
                    if remaining_input_tokens[sequence_id] == 0:
                        continue
                    per_sequence_cap = min(self.max_subsequence_tokens, capacity)
                    scheduled_tokens = min(remaining_input_tokens[sequence_id], per_sequence_cap)
                    if scheduled_tokens <= 0:
                        continue
                    round_sequence_ids.append(sequence_id)
                    round_subsequences.append(
                        SubsequenceDescriptor(num_tokens=scheduled_tokens, past_len=emitted_tokens[sequence_id])
                    )
                    emitted_tokens[sequence_id] += scheduled_tokens
                    remaining_input_tokens[sequence_id] -= scheduled_tokens
                    capacity -= scheduled_tokens
            else:
                # vLLM-like: prompt phase (whole prompt, no split) then generation phase (1 token/session/iter).
                has_prompt_left = any(remaining > 0 for remaining in remaining_input_tokens)
                if has_prompt_left:
                    for sequence_id, _session in enumerate(sessions):
                        if capacity == 0:
                            break
                        remaining = remaining_input_tokens[sequence_id]
                        if remaining == 0:
                            continue
                        if remaining > self.max_subsequence_tokens or remaining > self.max_num_batched_tokens:
                            raise ValueError(
                                f"Session {sequence_id} prompt length {remaining} exceeds prompt scheduling limit"
                            )
                        if remaining > capacity:
                            continue

                        round_sequence_ids.append(sequence_id)
                        round_subsequences.append(
                            SubsequenceDescriptor(num_tokens=remaining, past_len=emitted_tokens[sequence_id])
                        )
                        emitted_tokens[sequence_id] += remaining
                        remaining_input_tokens[sequence_id] = 0
                        capacity -= remaining
                else:
                    for sequence_id, _session in enumerate(sessions):
                        if capacity == 0:
                            break
                        if remaining_output_tokens[sequence_id] <= 0:
                            continue
                        round_sequence_ids.append(sequence_id)
                        round_subsequences.append(
                            SubsequenceDescriptor(num_tokens=1, past_len=emitted_tokens[sequence_id])
                        )
                        emitted_tokens[sequence_id] += 1
                        remaining_output_tokens[sequence_id] -= 1
                        capacity -= 1

            if not round_subsequences:
                raise RuntimeError("Failed to build a non-empty continuous batching round")

            round_plans.append(RoundPlan(tuple(round_sequence_ids), tuple(round_subsequences)))

        return round_plans


def serialize_round_plans_for_single_subsequence(round_plans: list[RoundPlan]) -> list[RoundPlan]:
    # NOTE: Compatibility adapter for current kernel behavior.
    # Kernel dispatch currently assumes one logical subsequence per enqueue.
    # TODO(enable_real_multi_subsequence): remove this serialization and pass
    # scheduler-produced multi-subsequence round plans directly.
    serialized: list[RoundPlan] = []
    for round_plan in round_plans:
        if len(round_plan.subsequences) <= 1:
            serialized.append(round_plan)
            continue
        for sequence_id, subsequence in zip(round_plan.sequence_ids, round_plan.subsequences):
            serialized.append(RoundPlan(sequence_ids=(sequence_id,), subsequences=(subsequence,)))
    return serialized


class PagedAttentionRuntime:
    def __init__(self, num_heads, num_kv_heads, head_size, block_sz, compressed_kvcache, is_causal=True):
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.block_sz = block_sz
        self.compressed_kvcache = compressed_kvcache
        self.is_causal = is_causal

        wg_size = 16
        q_step = CM_GRF_WIDTH // 32
        self.wg_seq_len = wg_size * q_step

        src1 = r'''#include "pa_multi_token.cm"'''
        cwd = os.path.dirname(os.path.realpath(__file__))
        print(f"compiling {cwd} {num_heads=} {head_size=}...")

        scale_factor = 1.0 / (head_size ** 0.5)
        self.kernels = cl.kernels(
            src1,
            (
                f'-cmc -Qxcm_jit_option="-abortonspill" -Qxcm_register_file_size=256  -mCM_printregusage -I{cwd}'
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
                f" -mdump_asm -g2"
            ),
        )

    @staticmethod
    @functools.cache
    def create_instance(num_heads, num_kv_heads, head_size, block_sz, compressed_kvcache, is_causal):
        return PagedAttentionRuntime(num_heads, num_kv_heads, head_size, block_sz, compressed_kvcache, is_causal)

    def _reshape_cache_for_kernel(self, cache: torch.Tensor) -> torch.Tensor:
        if self.compressed_kvcache:
            return quan_per_token(cache)
        return cache.reshape(cache.shape[0], self.num_kv_heads, -1).contiguous()

    def _build_sequence_cache_blocks(self, key: torch.Tensor, value: torch.Tensor, context_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        padded_context_len = DIV_UP(context_len, self.block_sz) * self.block_sz
        if padded_context_len != context_len:
            padding_tokens = padded_context_len - context_len
            pad_dims = (0, 0, 0, 0, 0, padding_tokens)
            padded_key = F.pad(key[:context_len], pad_dims, "constant", 1)
            padded_value = F.pad(value[:context_len], pad_dims, "constant", 1)
            if not self.compressed_kvcache:
                padded_key.view(torch.uint16)[context_len:padded_context_len] = 0xfe00
                padded_value.view(torch.uint16)[context_len:padded_context_len] = 0xfe00
        else:
            padded_key = key[:context_len]
            padded_value = value[:context_len]

        key_blocks = padded_key.reshape(-1, self.block_sz, self.num_kv_heads, self.head_size).transpose(1, 2).contiguous()
        value_blocks = padded_value.reshape(-1, self.block_sz, self.num_kv_heads, self.head_size).transpose(1, 2).contiguous()
        return key_blocks, value_blocks

    def build_round_inputs(self, sequence_tensors: list[SequenceTensors], round_plan: RoundPlan) -> PagedAttentionInputs:
        flat_query: list[torch.Tensor] = []
        flat_key: list[torch.Tensor] = []
        flat_value: list[torch.Tensor] = []
        key_cache_blocks: list[torch.Tensor] = []
        value_cache_blocks: list[torch.Tensor] = []
        past_lens: list[int] = []
        subsequence_begins = [0]
        block_indices: list[int] = []
        block_indices_begins = [0]
        physical_block_offset = 0

        for sequence_id, subsequence in zip(round_plan.sequence_ids, round_plan.subsequences):
            tensors = sequence_tensors[sequence_id]
            context_len = subsequence.total_seq_len

            q_slice = tensors.query[subsequence.past_len:context_len].contiguous()
            k_slice = tensors.key[subsequence.past_len:context_len].contiguous()
            v_slice = tensors.value[subsequence.past_len:context_len].contiguous()

            flat_query.append(q_slice.reshape(subsequence.num_tokens, -1))
            flat_key.append(k_slice.reshape(subsequence.num_tokens, -1))
            flat_value.append(v_slice.reshape(subsequence.num_tokens, -1))

            sequence_key_cache, sequence_value_cache = self._build_sequence_cache_blocks(tensors.key, tensors.value, context_len)
            num_sequence_blocks = sequence_key_cache.shape[0]

            key_cache_blocks.append(sequence_key_cache)
            value_cache_blocks.append(sequence_value_cache)
            block_indices.extend(range(physical_block_offset, physical_block_offset + num_sequence_blocks))
            physical_block_offset += num_sequence_blocks

            past_lens.append(subsequence.past_len)
            subsequence_begins.append(subsequence_begins[-1] + subsequence.num_tokens)
            block_indices_begins.append(len(block_indices))

        return PagedAttentionInputs(
            query=torch.cat(flat_query, dim=0).contiguous(),
            key=torch.cat(flat_key, dim=0).contiguous(),
            value=torch.cat(flat_value, dim=0).contiguous(),
            key_cache=torch.cat(key_cache_blocks, dim=0).contiguous(),
            value_cache=torch.cat(value_cache_blocks, dim=0).contiguous(),
            past_lens=torch.tensor(past_lens, dtype=torch.int32),
            subsequence_begins=torch.tensor(subsequence_begins, dtype=torch.int32),
            block_indices=torch.tensor(block_indices, dtype=torch.int32),
            block_indices_begins=torch.tensor(block_indices_begins, dtype=torch.int32),
            max_context_len=round_plan.max_context_len,
            sequence_ids=round_plan.sequence_ids,
            subsequences=round_plan.subsequences,
        )

    def submit_round(self, round_inputs: PagedAttentionInputs, n_repeats: int = 1) -> torch.Tensor:
        batch_size_in_tokens = round_inputs.query.shape[0]
        output = torch.zeros(batch_size_in_tokens, self.num_heads * self.head_size, dtype=torch.float16)
        kv_dtype = torch.uint8 if self.compressed_kvcache else torch.float16

        cl.finish()

        for _ in range(n_repeats):
            q_tensor = round_inputs.query.reshape(batch_size_in_tokens, self.num_heads, self.head_size).contiguous()
            kernel_key_cache = self._reshape_cache_for_kernel(round_inputs.key_cache.contiguous())
            kernel_value_cache = self._reshape_cache_for_kernel(round_inputs.value_cache.contiguous())
            out_shape = (batch_size_in_tokens, self.num_heads, self.head_size)

            t_q = cl.tensor(q_tensor.to(torch.float16).detach().numpy())
            t_out = cl.tensor(list(out_shape), np.dtype(np.float16))
            t_key_cache = cl.tensor(kernel_key_cache.to(kv_dtype).detach().numpy())
            t_value_cache = cl.tensor(kernel_value_cache.to(kv_dtype).detach().numpy())
            t_past_lens = cl.tensor(round_inputs.past_lens.detach().numpy())
            t_block_indices = cl.tensor(round_inputs.block_indices.detach().numpy())
            t_block_indices_begins = cl.tensor(round_inputs.block_indices_begins.detach().numpy())
            t_subsequence_begins = cl.tensor(round_inputs.subsequence_begins.detach().numpy())

            wg_size = 16
            wg_count = DIV_UP(batch_size_in_tokens, self.wg_seq_len)
            gws = [1, self.num_heads, int(wg_count * wg_size)]
            lws = [1, 1, wg_size]

            print("[enqueue] gws=", gws, "lws=", lws)
            print("[enqueue] q.shape=", tuple(q_tensor.shape), "q.dtype=", q_tensor.dtype)
            print("[enqueue] key_cache.shape=", tuple(kernel_key_cache.shape), "key_cache.dtype=", kernel_key_cache.dtype)
            print("[enqueue] value_cache.shape=", tuple(kernel_value_cache.shape), "value_cache.dtype=", kernel_value_cache.dtype)
            print("[enqueue] past_lens=", round_inputs.past_lens.tolist())
            print("[enqueue] block_indices=", round_inputs.block_indices.tolist())
            print("[enqueue] block_indices_begins=", round_inputs.block_indices_begins.tolist())
            print("[enqueue] subsequence_begins=", round_inputs.subsequence_begins.tolist())
            print("[enqueue] q_len=", batch_size_in_tokens, "out.shape=", out_shape)

            self.kernels.enqueue(
                "cm_page_attention",
                gws,
                lws,
                t_q,
                t_key_cache,
                t_value_cache,
                t_past_lens,
                t_block_indices,
                t_block_indices_begins,
                t_subsequence_begins,
                t_out,
                batch_size_in_tokens,
            )

            output = torch.from_numpy(t_out.numpy().reshape(batch_size_in_tokens, -1))

        return output


def create_sequence_tensors(subsequence: SubsequenceDescriptor, num_heads, num_kv_heads, head_size, compressed_kvcache, act_dtype=torch.float16) -> SequenceTensors:
    total_seq_len = subsequence.total_tokens
    low = -1
    high = 2
    query = torch.randint(low, high, [total_seq_len, num_heads, head_size]).to(dtype=act_dtype)

    if compressed_kvcache:
        key = torch.randint(low, high, [total_seq_len, num_kv_heads, head_size]).to(dtype=act_dtype) / 4.0
        key[0:total_seq_len:3, :, :] = (key[0:total_seq_len:3, :, :] + 0.25) / 2.0
        value = torch.randint(low, high, [total_seq_len, num_kv_heads, head_size]).to(dtype=act_dtype) / high
        value[0:total_seq_len:3, :, :] = (value[0:total_seq_len:3, :, :] + 0.25) / 2.0
    else:
        key = torch.rand(total_seq_len, num_kv_heads, head_size).to(dtype=act_dtype)
        value = torch.rand(total_seq_len, num_kv_heads, head_size).to(dtype=act_dtype) / high

    return SequenceTensors(query=query, key=key, value=value)


def normalize_session_descriptors(sessions=None, num_output_tokens=0):
    if sessions is None:
        sessions = (128, 257, 511)

    normalized = []
    for entry in sessions:
        if isinstance(entry, SessionDescriptor):
            normalized.append(entry)
        elif isinstance(entry, int):
            normalized.append(SessionDescriptor(num_input_tokens=entry, num_output_tokens=num_output_tokens))
        elif isinstance(entry, (tuple, list)) and len(entry) == 2:
            normalized.append(SessionDescriptor(num_input_tokens=int(entry[0]), num_output_tokens=int(entry[1])))
        else:
            raise TypeError("Each session must be an int, a (num_input_tokens, num_output_tokens) pair, or a SessionDescriptor")
    return normalized


def get_attention_mask(q_len: int, kv_len: int, num_heads: int, q_dtype: torch.dtype, q_device: torch.device, past_len: int, is_causal=True):
    attention_mask = torch.full([num_heads, q_len, kv_len], True, dtype=torch.bool, device=q_device)
    if is_causal:
        query_positions = past_len + torch.arange(q_len, device=q_device).unsqueeze(1)
        key_positions = torch.arange(kv_len, device=q_device).unsqueeze(0)
        causal_pattern = key_positions <= query_positions
        causal_pattern = causal_pattern.unsqueeze(0).repeat_interleave(num_heads, dim=0)
        attention_mask = attention_mask & causal_pattern
    attention_mask = torch.where(attention_mask, 0, torch.finfo(q_dtype).min)
    return attention_mask.unsqueeze(0)


def flash_attn_vlen_ref(q, k, v, cu_seqlens, is_causal=True, attention_mask=None):
    seq_length, num_heads, head_size = q.shape
    kv_seq_length, num_kv_heads, head_size = k.shape
    old_dtype = q.dtype
    q = q.transpose(0, 1)
    k = k.transpose(0, 1)
    v = v.transpose(0, 1)

    if attention_mask is not None:
        attn_output = F.scaled_dot_product_attention(
            q.unsqueeze(0).to(torch.float16),
            k.unsqueeze(0).to(torch.float16),
            v.unsqueeze(0).to(torch.float16),
            attn_mask=attention_mask,
            dropout_p=0.0,
            enable_gqa=(num_kv_heads != num_heads),
        )
        print(".")
        return attn_output.squeeze(0).transpose(0, 1).to(old_dtype)

    if is_causal:
        attn_output = F.scaled_dot_product_attention(
            q.unsqueeze(0).to(torch.float16),
            k.unsqueeze(0).to(torch.float16),
            v.unsqueeze(0).to(torch.float16),
            is_causal=True,
            dropout_p=0.0,
            enable_gqa=(num_kv_heads != num_heads),
        )
    else:
        attention_mask = torch.zeros([1, seq_length, seq_length], device=q.device, dtype=torch.bool)
        if len(cu_seqlens):
            for i in range(1, len(cu_seqlens)):
                attention_mask[..., cu_seqlens[i - 1]:cu_seqlens[i], cu_seqlens[i - 1]:cu_seqlens[i]] = True
        else:
            attention_mask[...] = True

        attn_output = F.scaled_dot_product_attention(
            q.unsqueeze(0).to(torch.float16),
            k.unsqueeze(0).to(torch.float16),
            v.unsqueeze(0).to(torch.float16),
            attn_mask=attention_mask,
            dropout_p=0.0,
            enable_gqa=(num_kv_heads != num_heads),
        )

    print(".")
    return attn_output.squeeze(0).transpose(0, 1).to(old_dtype)


def check_close(input_tensor, other_tensor, atol=1e-2, rtol=1e-2):
    print(f"[check_close] {input_tensor.shape}{input_tensor.dtype} vs {other_tensor.shape}{other_tensor.dtype}")
    rtol_max = (((input_tensor - other_tensor).abs() - 1e-5) / other_tensor.abs())[other_tensor != 0].max()
    atol_max = (((input_tensor - other_tensor).abs()) - 1e-5 * other_tensor.abs()).max()
    print(f"[check_close] rtol_max: {rtol_max}")
    print(f"[check_close] atol_max: {atol_max}")
    if not torch.allclose(input_tensor, other_tensor, atol=atol, rtol=rtol, equal_nan=True):
        close_check = torch.isclose(input_tensor, other_tensor, atol=atol, rtol=rtol)
        not_close_indices = torch.where(~close_check)
        print(f"Not close indices: {not_close_indices}")
        print(f"    input_tensor: {input_tensor[not_close_indices]}")
        print(f"    other_tensor: {other_tensor[not_close_indices]}")
        raise AssertionError("Output mismatch")


def reference_round(runtime: PagedAttentionRuntime, round_inputs: PagedAttentionInputs) -> torch.Tensor:
    refs = []
    for sequence_index, subsequence in enumerate(round_inputs.subsequences):
        q_start = int(round_inputs.subsequence_begins[sequence_index].item())
        q_end = int(round_inputs.subsequence_begins[sequence_index + 1].item())
        blk_start = int(round_inputs.block_indices_begins[sequence_index].item())
        blk_end = int(round_inputs.block_indices_begins[sequence_index + 1].item())
        q_len = q_end - q_start
        context_len = subsequence.total_seq_len

        q = round_inputs.query[q_start:q_end].reshape(q_len, runtime.num_heads, runtime.head_size).contiguous()
        physical_blocks = round_inputs.block_indices[blk_start:blk_end]
        key_cache = round_inputs.key_cache[physical_blocks].transpose(1, 2).reshape(-1, runtime.num_kv_heads, runtime.head_size)[:context_len].contiguous()
        value_cache = round_inputs.value_cache[physical_blocks].transpose(1, 2).reshape(-1, runtime.num_kv_heads, runtime.head_size)[:context_len].contiguous()
        attention_mask = get_attention_mask(q_len, context_len, runtime.num_heads, q.dtype, q.device, past_len=subsequence.past_len, is_causal=runtime.is_causal)
        ref = flash_attn_vlen_ref(q, key_cache, value_cache, [], runtime.is_causal, attention_mask)
        refs.append(ref.reshape(q_len, -1))

    return torch.cat(refs, dim=0)


def run_page_attn_causal_batch1(seq_len, num_heads=2, num_kv_heads=1, head_size=32, block_sz=32, compressed_kvcache=False, max_num_batched_tokens=256, dynamic_split_fuse=True, check_acc=True):
    return run_page_attn_causal_multiseq(
        sessions=[SessionDescriptor(num_input_tokens=seq_len, num_output_tokens=0)],
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        block_sz=block_sz,
        compressed_kvcache=compressed_kvcache,
        enable_prefix_caching=False,
        max_num_batched_tokens=max_num_batched_tokens,
        dynamic_split_fuse=dynamic_split_fuse,
        check_acc=check_acc,
    )


def run_page_attn_causal_multiseq(sessions=None, num_heads=2, num_kv_heads=1, head_size=32, block_sz=32, compressed_kvcache=False, enable_prefix_caching=False, max_num_batched_tokens=256, dynamic_split_fuse=True, check_acc=True, return_rounds=False):
    if enable_prefix_caching:
        # NOTE: Prefix caching is not modeled in this standalone PA harness yet.
        # TODO(enable_prefix_caching): emulate hash-based shared KV block reuse.
        raise ValueError("enable_prefix_caching=True is not supported in this test harness")

    descriptors = normalize_session_descriptors(sessions=sessions)
    sequence_tensors = [
        create_sequence_tensors(descriptor, num_heads, num_kv_heads, head_size, compressed_kvcache)
        for descriptor in descriptors
    ]

    scheduler = ContinuousBatchingScheduler(
        max_num_batched_tokens=max_num_batched_tokens,
        max_subsequence_tokens=max_num_batched_tokens,
        dynamic_split_fuse=dynamic_split_fuse,
    )
    round_plans = scheduler.schedule(descriptors)
    runtime = PagedAttentionRuntime.create_instance(num_heads, num_kv_heads, head_size, block_sz, compressed_kvcache, True)

    round_outputs = []
    round_refs = []
    round_inputs_list = []

    for round_plan in round_plans:
        round_inputs = runtime.build_round_inputs(sequence_tensors, round_plan)
        round_inputs_list.append(round_inputs)
        round_outputs.append(runtime.submit_round(round_inputs))
        if check_acc:
            round_refs.append(reference_round(runtime, round_inputs))

    flat_output = torch.cat(round_outputs, dim=0) if round_outputs else torch.empty(0, num_heads * head_size, dtype=torch.float16)

    if check_acc:
        flat_ref = torch.cat(round_refs, dim=0) if round_refs else torch.empty_like(flat_output)
        check_close(flat_ref, flat_output)

    if return_rounds:
        return round_inputs_list, round_outputs
    return flat_output


@pytest.mark.parametrize(
    "num_heads,num_kv_heads,head_size,block_sz,compressed_kvcache,seq_len,trunk_size",
    [
        (2, 1, 32, 32, False, 128, 128),
        (4, 2, 64, 32, False, 128, 128),
        (4, 1, 64, 64, True, 128, 128),
        (2, 1, 32, 32, False, 96, 128),
    ],
)
def test_pa_batch1_pytest_configs(num_heads, num_kv_heads, head_size, block_sz, compressed_kvcache, seq_len, trunk_size):
    run_page_attn_causal_batch1(
        seq_len=seq_len,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        block_sz=block_sz,
        compressed_kvcache=compressed_kvcache,
        max_num_batched_tokens=trunk_size,
        dynamic_split_fuse=True,
        check_acc=True,
    )


@pytest.mark.parametrize(
    "num_heads,num_kv_heads,head_size,block_sz,compressed_kvcache,trunk_size",
    [
        (2, 1, 32, 32, False, 128),
        (4, 2, 64, 32, False, 128),
        (4, 1, 64, 64, True, 128),
    ],
)
def test_pa_multiseq_pytest_prefill_batch_size_in_sequences_1(
    num_heads,
    num_kv_heads,
    head_size,
    block_sz,
    compressed_kvcache,
    trunk_size,
):
    run_page_attn_causal_multiseq(
        sessions=[
            SessionDescriptor(num_input_tokens=trunk_size, num_output_tokens=0),
            SessionDescriptor(num_input_tokens=trunk_size + 64, num_output_tokens=0),
        ],
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        block_sz=block_sz,
        compressed_kvcache=compressed_kvcache,
        max_num_batched_tokens=trunk_size,
        check_acc=True,
    )


@pytest.mark.parametrize(
    "num_heads,num_kv_heads,head_size,block_sz,compressed_kvcache,trunk_size",
    [
        (2, 1, 32, 32, False, 64),
        (4, 2, 64, 32, False, 128),
        (4, 1, 64, 64, True, 128),
    ],
)
def test_pa_multiseq_pytest_prefill_batch_size_in_sequences_2(
    num_heads,
    num_kv_heads,
    head_size,
    block_sz,
    compressed_kvcache,
    trunk_size,
):
    run_page_attn_causal_multiseq(
        sessions=[
            SessionDescriptor(num_input_tokens=trunk_size // 2, num_output_tokens=0),
            SessionDescriptor(num_input_tokens=trunk_size // 2, num_output_tokens=0),
        ],
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        block_sz=block_sz,
        compressed_kvcache=compressed_kvcache,
        max_num_batched_tokens=trunk_size,
        check_acc=True,
    )

# '''
# Usage:

# # `py_compile` to verify syntax correctness of the test file before running pytest.
# python -m py_compile test_pa_multiseq.py

# # `-q` for quieter pytest output, and `-k` to select the specific test to run.
# python -m pytest -q test_pa_multiseq.py -k test_pa_multiseq_pytest_prefill_batch_size_in_sequences_1

# # `-s` to show captured print statement for debugging purposes.
# python -m pytest -s test_pa_multiseq.py -k test_pa_multiseq_pytest_prefill_batch_size_in_sequences_1

# '''

