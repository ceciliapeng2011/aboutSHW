import functools
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from clops import cl
from kv_cache_quant_utils import (
    DEFAULT_SUB_BLOCK_SIZE,
    dequant_per_channel as dequant_per_channel_ref,
    dequant_per_token as dequant_per_token_ref,
    quant_per_channel as quant_per_channel_ref,
    quant_per_token as quant_per_token_ref,
)


KernelInputs = dict[str, object]


KV_CACHE_COMPRESSION_NONE = 0
KV_CACHE_COMPRESSION_BY_TOKEN = 1
KV_CACHE_COMPRESSION_BY_CHANNEL = 2


def normalize_kv_cache_compression(mode: int) -> int:
    if isinstance(mode, bool):
        raise TypeError("kv_cache_compression must be int in {0, 1, 2}, got bool")
    mode = int(mode)
    if mode not in {
        KV_CACHE_COMPRESSION_NONE,
        KV_CACHE_COMPRESSION_BY_TOKEN,
        KV_CACHE_COMPRESSION_BY_CHANNEL,
    }:
        raise ValueError(f"unsupported kv_cache_compression mode: {mode}")
    return mode


@functools.cache
def get_cm_grf_width() -> int:
    cm_kernels = cl.kernels(r'''
    extern "C" _GENX_MAIN_ void cm_get_grf_width(int * info [[type("svmptr_t")]]) {
        info[0] = CM_GRF_WIDTH;
    }''', "-cmc")
    t_info = cl.tensor([2], np.dtype(np.int32))
    cm_kernels.enqueue("cm_get_grf_width", [1], [1], t_info)
    return int(t_info.numpy()[0])


def check_close(input_tensor: torch.Tensor, other_tensor: torch.Tensor, atol=1e-2, rtol=1e-2):
    if not torch.allclose(input_tensor, other_tensor, atol=atol, rtol=rtol, equal_nan=True):
        close_check = torch.isclose(input_tensor, other_tensor, atol=atol, rtol=rtol)
        not_close_indices = torch.where(~close_check)
        raise AssertionError(f"Output mismatch at {not_close_indices}")


def DIV_UP(x: int, y: int) -> int:
    return (x + y - 1) // y


def ss(num_tokens: int, past_len: int = 0) -> "SubsequenceDescriptor":
    return SubsequenceDescriptor(num_tokens=num_tokens, past_len=past_len)


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
class PagedAttentionTestCase:
    subsequences: tuple[SubsequenceDescriptor, ...]
    num_heads: int = 2
    num_kv_heads: int = 2
    k_head_size: int = 64
    v_head_size: int = 64
    block_size: int = 16
    sub_block_size: int = DEFAULT_SUB_BLOCK_SIZE
    kv_cache_compression: int = KV_CACHE_COMPRESSION_NONE


@dataclass(frozen=True)
class SubsequenceTensors:
    q_cur: torch.Tensor
    k_cur: torch.Tensor
    v_cur: torch.Tensor
    k_past: torch.Tensor
    v_past: torch.Tensor


@dataclass(frozen=True)
class KVCacheTable:
    block_indices: torch.Tensor
    block_indices_begins: torch.Tensor
    key_cache: torch.Tensor
    value_cache: torch.Tensor


class KVCacheQuantizer:
    @staticmethod
    def quant_per_token(kv: torch.Tensor) -> torch.Tensor:
        return quant_per_token_ref(kv)

    @staticmethod
    def quant_per_channel(kv: torch.Tensor, sub_block_size: int) -> torch.Tensor:
        return quant_per_channel_ref(kv, sub_block_size)

    @staticmethod
    def dequant_per_token(cache: torch.Tensor, block_size: int, head_size: int) -> torch.Tensor:
        if cache.dtype != torch.uint8:
            raise ValueError("dequant_cache_per_token expects uint8 cache")
        return dequant_per_token_ref(cache, head_size, block_size)

    @staticmethod
    def dequant_per_channel(cache: torch.Tensor, block_size: int, head_size: int, sub_block_size: int) -> torch.Tensor:
        if cache.dtype != torch.uint8:
            raise ValueError("dequant_cache_per_channel expects uint8 cache")
        return dequant_per_channel_ref(cache, head_size, block_size, sub_block_size)


def quan_per_token(kv: torch.Tensor) -> torch.Tensor:
    return KVCacheQuantizer.quant_per_token(kv)


def dequant_cache_per_token(cache: torch.Tensor, block_size: int, head_size: int) -> torch.Tensor:
    return KVCacheQuantizer.dequant_per_token(cache, block_size, head_size)


def dequant_cache_per_channel(cache: torch.Tensor, block_size: int, head_size: int, sub_block_size: int) -> torch.Tensor:
    return KVCacheQuantizer.dequant_per_channel(cache, block_size, head_size, sub_block_size)


class KVCacheUpdater:
    def __init__(
        self,
        num_heads: int,
        num_kv_heads: int,
        head_size: int,
        block_sz: int,
        compressed_kvcache: int,
        sub_block_size: int = DEFAULT_SUB_BLOCK_SIZE,
        is_causal: bool = True,
    ):
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.block_sz = block_sz
        self.compressed_kvcache = normalize_kv_cache_compression(compressed_kvcache)
        self.sub_block_size = sub_block_size
        self.is_causal = is_causal
        if self.compressed_kvcache == KV_CACHE_COMPRESSION_BY_CHANNEL and self.block_sz % self.sub_block_size != 0:
            raise ValueError(
                f"block_sz ({self.block_sz}) must be divisible by sub_block_size ({self.sub_block_size}) for KV_CACHE_COMPRESSION_BY_CHANNEL"
            )

    def _build_sequence_cache_blocks(self, key: torch.Tensor, value: torch.Tensor, context_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        padded_context_len = DIV_UP(context_len, self.block_sz) * self.block_sz
        if padded_context_len != context_len:
            padding_tokens = padded_context_len - context_len
            pad_dims = (0, 0, 0, 0, 0, padding_tokens)
            padded_key = F.pad(key[:context_len], pad_dims, "constant", 0)
            padded_value = F.pad(value[:context_len], pad_dims, "constant", 0)
        else:
            padded_key = key[:context_len]
            padded_value = value[:context_len]

        key_blocks = padded_key.reshape(-1, self.block_sz, self.num_kv_heads, self.head_size).transpose(1, 2).contiguous()
        value_blocks = padded_value.reshape(-1, self.block_sz, self.num_kv_heads, self.head_size).transpose(1, 2).contiguous()
        return key_blocks, value_blocks

    def update_cache_for_subsequence(
        self,
        current_key: torch.Tensor,
        current_value: torch.Tensor,
        past_len: int,
        context_len: int,
        past_key_cache_blocks: torch.Tensor,
        past_value_cache_blocks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        local_num_blocks = int(past_key_cache_blocks.shape[0])
        local_block_indices = torch.arange(local_num_blocks, dtype=torch.long)

        if self.compressed_kvcache == KV_CACHE_COMPRESSION_BY_TOKEN:
            past_key_cache_blocks = quan_per_token(past_key_cache_blocks)
            past_value_cache_blocks = quan_per_token(past_value_cache_blocks)
        elif self.compressed_kvcache == KV_CACHE_COMPRESSION_BY_CHANNEL:
            past_key_cache_blocks = KVCacheQuantizer.quant_per_channel(past_key_cache_blocks, self.sub_block_size)
            # V stays by-token in mode-2 kernels.
            past_value_cache_blocks = quan_per_token(past_value_cache_blocks)

        k_past = self.recover_context_from_cache(
            past_key_cache_blocks[local_block_indices].contiguous(),
            past_len,
            self.num_kv_heads,
            self.head_size,
            self.block_sz,
            self.compressed_kvcache,
            cache_kind="key",
            sub_block_size=self.sub_block_size,
        )
        v_past = self.recover_context_from_cache(
            past_value_cache_blocks[local_block_indices].contiguous(),
            past_len,
            self.num_kv_heads,
            self.head_size,
            self.block_sz,
            self.compressed_kvcache,
            cache_kind="value",
            sub_block_size=self.sub_block_size,
        )

        k_cache_source = torch.cat((k_past, current_key), dim=0).contiguous()
        v_cache_source = torch.cat((v_past, current_value), dim=0).contiguous()
        sequence_key_cache, sequence_value_cache = self._build_sequence_cache_blocks(k_cache_source, v_cache_source, context_len)
        if self.compressed_kvcache == KV_CACHE_COMPRESSION_BY_TOKEN:
            sequence_key_cache = quan_per_token(sequence_key_cache)
            sequence_value_cache = quan_per_token(sequence_value_cache)
        elif self.compressed_kvcache == KV_CACHE_COMPRESSION_BY_CHANNEL:
            sequence_key_cache = KVCacheQuantizer.quant_per_channel(sequence_key_cache, self.sub_block_size)
            # V stays by-token in mode-2 kernels.
            sequence_value_cache = quan_per_token(sequence_value_cache)
        return sequence_key_cache, sequence_value_cache

    @staticmethod
    def recover_context_from_cache(
        cache_blocks: torch.Tensor,
        context_len: int,
        num_kv_heads: int,
        head_size: int,
        block_size: int,
        compressed_kvcache: int,
        cache_kind: str,
        sub_block_size: int = DEFAULT_SUB_BLOCK_SIZE,
    ) -> torch.Tensor:
        mode = normalize_kv_cache_compression(compressed_kvcache)
        if mode == KV_CACHE_COMPRESSION_BY_TOKEN:
            cache_blocks = dequant_cache_per_token(cache_blocks, block_size, head_size)
        elif mode == KV_CACHE_COMPRESSION_BY_CHANNEL:
            if cache_kind == "key":
                cache_blocks = dequant_cache_per_channel(cache_blocks, block_size, head_size, sub_block_size)
            elif cache_kind == "value":
                cache_blocks = dequant_cache_per_token(cache_blocks, block_size, head_size)
            else:
                raise ValueError(f"unsupported cache_kind: {cache_kind}")
        return cache_blocks.transpose(1, 2).reshape(-1, num_kv_heads, head_size)[:context_len].contiguous()

    def __call__(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        past_lens: list[int],
        subsequence_begins: list[int],
        block_indices: torch.Tensor,
        block_indices_begins: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
    ) -> KernelInputs:
        num_sequences = len(past_lens)
        if len(subsequence_begins) != num_sequences + 1:
            raise ValueError("subsequence_begins must have len(past_lens) + 1 entries")

        total_tokens = int(subsequence_begins[-1])
        if int(key.shape[0]) != total_tokens or int(value.shape[0]) != total_tokens:
            raise ValueError("packed key/value token dimensions must match subsequence_begins")
        if int(key.shape[1]) != self.num_kv_heads * self.head_size:
            raise ValueError("packed key feature dimension must be num_kv_heads * head_size")
        if int(value.shape[1]) != self.num_kv_heads * self.head_size:
            raise ValueError("packed value feature dimension must be num_kv_heads * head_size")

        key_cache_blocks: list[torch.Tensor] = []
        value_cache_blocks: list[torch.Tensor] = []
        out_past_lens: list[int] = []
        num_total_blocks = int(block_indices.numel())
        max_context_len = 0

        if int(subsequence_begins[0]) != 0:
            raise ValueError("subsequence_begins must start from 0")

        for sequence_id in range(num_sequences):
            past_len = int(past_lens[sequence_id])
            begin = int(subsequence_begins[sequence_id])
            end = int(subsequence_begins[sequence_id + 1])
            if end <= begin:
                raise ValueError("subsequence_begins must be strictly increasing")
            num_tokens = end - begin
            if num_tokens <= 0:
                raise ValueError("num_tokens must be positive")

            k_slice = key[begin:end].reshape(num_tokens, self.num_kv_heads, self.head_size).contiguous()
            v_slice = value[begin:end].reshape(num_tokens, self.num_kv_heads, self.head_size).contiguous()
            context_len = past_len + num_tokens

            blk_start = int(block_indices_begins[sequence_id].item())
            blk_end = int(block_indices_begins[sequence_id + 1].item())
            sequence_physical_blocks = block_indices[blk_start:blk_end].to(dtype=torch.long)
            past_key_cache_blocks = key_cache[sequence_physical_blocks].contiguous()
            past_value_cache_blocks = value_cache[sequence_physical_blocks].contiguous()
            num_sequence_blocks = int(sequence_physical_blocks.numel())

            sequence_key_cache, sequence_value_cache = self.update_cache_for_subsequence(
                k_slice,
                v_slice,
                past_len,
                context_len,
                past_key_cache_blocks,
                past_value_cache_blocks,
            )
            if int(sequence_key_cache.shape[0]) != num_sequence_blocks:
                raise ValueError("Per-subsequence KV cache table size does not match cache block count")

            key_cache_blocks.append(sequence_key_cache)
            value_cache_blocks.append(sequence_value_cache)

            out_past_lens.append(past_len)
            if context_len > max_context_len:
                max_context_len = context_len

        key_cache = torch.cat(key_cache_blocks, dim=0).contiguous()
        value_cache = torch.cat(value_cache_blocks, dim=0).contiguous()
        if int(key_cache.shape[0]) != num_total_blocks:
            raise ValueError("Global KV cache table size does not match total cache blocks")

        return {
            "query": torch.empty((total_tokens, self.num_heads * self.head_size), dtype=key.dtype),
            "key_cache": key_cache,
            "value_cache": value_cache,
            "past_lens": torch.tensor(out_past_lens, dtype=torch.int32),
            "subsequence_begins": torch.tensor(subsequence_begins, dtype=torch.int32),
            "block_indices": block_indices.to(dtype=torch.int32),
            "block_indices_begins": block_indices_begins.to(dtype=torch.int32),
            "max_context_len": max_context_len,
        }


def create_subsequence_tensors(
    subsequence: SubsequenceDescriptor,
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    act_dtype=torch.float16,
) -> SubsequenceTensors:
    low = -1
    high = 2
    q_cur = torch.randint(low, high, [subsequence.num_tokens, num_heads, head_size]).to(dtype=act_dtype)

    k_cur = torch.rand(subsequence.num_tokens, num_kv_heads, head_size).to(dtype=act_dtype)
    v_cur = torch.rand(subsequence.num_tokens, num_kv_heads, head_size).to(dtype=act_dtype) / high

    k_past = torch.rand(subsequence.past_len, num_kv_heads, head_size).to(dtype=act_dtype)
    v_past = torch.rand(subsequence.past_len, num_kv_heads, head_size).to(dtype=act_dtype) / high

    return SubsequenceTensors(
        q_cur=q_cur,
        k_cur=k_cur,
        v_cur=v_cur,
        k_past=k_past,
        v_past=v_past,
    )


def create_kvcache_table(
    subsequences: tuple[SubsequenceDescriptor, ...],
    subsequence_tensors: list[SubsequenceTensors],
    num_kv_heads: int,
    head_size: int,
    block_size: int,
    keep_order: bool = False,
) -> KVCacheTable:
    block_indices_begins = [0]
    num_blocks_total = 0
    sequence_key_cache_blocks: list[torch.Tensor] = []
    sequence_value_cache_blocks: list[torch.Tensor] = []

    for subsequence, tensors in zip(subsequences, subsequence_tensors):
        num_seq_blocks = DIV_UP(subsequence.total_seq_len, block_size)
        padded_len = num_seq_blocks * block_size

        k_past_padded = torch.zeros(padded_len, num_kv_heads, head_size, dtype=tensors.k_past.dtype)
        v_past_padded = torch.zeros(padded_len, num_kv_heads, head_size, dtype=tensors.v_past.dtype)
        if subsequence.past_len > 0:
            k_past_padded[:subsequence.past_len] = tensors.k_past
            v_past_padded[:subsequence.past_len] = tensors.v_past

        sequence_key_cache_blocks.append(
            k_past_padded.reshape(num_seq_blocks, block_size, num_kv_heads, head_size).transpose(1, 2).contiguous()
        )
        sequence_value_cache_blocks.append(
            v_past_padded.reshape(num_seq_blocks, block_size, num_kv_heads, head_size).transpose(1, 2).contiguous()
        )

        num_blocks_total += num_seq_blocks
        block_indices_begins.append(num_blocks_total)

    if keep_order:
        block_indices = torch.arange(num_blocks_total, dtype=torch.int32)
    else:
        block_indices = torch.randperm(num_blocks_total, dtype=torch.int32)

    key_cache = torch.zeros(
        (num_blocks_total, num_kv_heads, block_size, head_size),
        dtype=sequence_key_cache_blocks[0].dtype,
    )
    value_cache = torch.zeros(
        (num_blocks_total, num_kv_heads, block_size, head_size),
        dtype=sequence_value_cache_blocks[0].dtype,
    )

    for sequence_index, (sequence_key_blocks, sequence_value_blocks) in enumerate(zip(sequence_key_cache_blocks, sequence_value_cache_blocks)):
        blk_start = block_indices_begins[sequence_index]
        blk_end = block_indices_begins[sequence_index + 1]
        physical_blocks = block_indices[blk_start:blk_end].to(dtype=torch.long)
        key_cache[physical_blocks] = sequence_key_blocks
        value_cache[physical_blocks] = sequence_value_blocks

    return KVCacheTable(
        block_indices=block_indices,
        block_indices_begins=torch.tensor(block_indices_begins, dtype=torch.int32),
        key_cache=key_cache.contiguous(),
        value_cache=value_cache.contiguous(),
    )


def create_paged_attention_inputs(
    case: PagedAttentionTestCase,
    subsequence_tensors: list[SubsequenceTensors],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int], list[int], KVCacheTable]:
    query = torch.cat([tensors.q_cur.reshape(tensors.q_cur.shape[0], -1) for tensors in subsequence_tensors], dim=0).contiguous()
    key = torch.cat([tensors.k_cur.reshape(tensors.k_cur.shape[0], -1) for tensors in subsequence_tensors], dim=0).contiguous()
    value = torch.cat([tensors.v_cur.reshape(tensors.v_cur.shape[0], -1) for tensors in subsequence_tensors], dim=0).contiguous()

    kvcache_table = create_kvcache_table(
        case.subsequences,
        subsequence_tensors,
        case.num_kv_heads,
        case.k_head_size,
        case.block_size,
    )
    past_lens = [descriptor.past_len for descriptor in case.subsequences]
    subsequence_begins = [0]
    for descriptor in case.subsequences:
        subsequence_begins.append(subsequence_begins[-1] + descriptor.num_tokens)

    return query, key, value, past_lens, subsequence_begins, kvcache_table


def get_attention_mask(
    q_len: int,
    kv_len: int,
    num_heads: int,
    q_dtype: torch.dtype,
    q_device: torch.device,
    past_len: int,
    is_causal: bool = True,
) -> torch.Tensor:
    attention_mask = torch.full([num_heads, q_len, kv_len], True, dtype=torch.bool, device=q_device)
    if is_causal:
        query_positions = past_len + torch.arange(q_len, device=q_device).unsqueeze(1)
        key_positions = torch.arange(kv_len, device=q_device).unsqueeze(0)
        causal_pattern = key_positions <= query_positions
        causal_pattern = causal_pattern.unsqueeze(0).repeat_interleave(num_heads, dim=0)
        attention_mask = attention_mask & causal_pattern
    attention_mask = torch.where(attention_mask, 0, torch.finfo(q_dtype).min)
    return attention_mask.unsqueeze(0)


def flash_attn_vlen_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens,
    is_causal: bool = True,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    seq_length, num_heads, _ = q.shape
    _, num_kv_heads, _ = k.shape
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


def get_sequence_ranges(kern_attn_inputs: KernelInputs, sequence_index: int) -> tuple[int, int, int, int]:
    subsequence_begins = kern_attn_inputs["subsequence_begins"]
    block_indices_begins = kern_attn_inputs["block_indices_begins"]
    assert isinstance(subsequence_begins, torch.Tensor)
    assert isinstance(block_indices_begins, torch.Tensor)
    q_start = int(subsequence_begins[sequence_index].item())
    q_end = int(subsequence_begins[sequence_index + 1].item())
    blk_start = int(block_indices_begins[sequence_index].item())
    blk_end = int(block_indices_begins[sequence_index + 1].item())
    return q_start, q_end, blk_start, blk_end
