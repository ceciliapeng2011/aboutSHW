import functools
import os
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from clops import cl
from clops.utils import Colors

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


def quan_per_token(kv: torch.Tensor) -> torch.Tensor:
    """Per-token affine quantization for KV cache blocks.

    Input layout: [num_blocks, num_kv_heads, block_size, head_size] (fp16).
    Output layout: [num_blocks, num_kv_heads, block_size * (head_size + 4)] (u8),
    where each token stores:
        - quantized data: head_size bytes
        - dequant scale: 2 bytes (fp16)
        - zero-point:    2 bytes (fp16)
    """
    blk_num, kv_heads, blksz, *_ = kv.shape
    kv_max = kv.amax(dim=-1, keepdim=True)
    kv_min = kv.amin(dim=-1, keepdim=True)
    qrange = kv_max - kv_min
    zero_range_mask = qrange == 0

    int_max = 255.0
    int_min = 0.0
    int_range = int_max - int_min
    safe_qrange = torch.where(zero_range_mask, torch.ones_like(qrange), qrange)
    kv_scale = (int_range / safe_qrange).to(dtype=torch.half)
    kv_zp = ((0.0 - kv_min) * kv_scale + int_min).to(dtype=torch.half)

    # For constant-value tokens (including zero-padding), force a finite affine map
    # to avoid inf/nan in quant/dequant metadata.
    kv_scale = torch.where(zero_range_mask, torch.ones_like(kv_scale), kv_scale)
    kv_zp = torch.where(zero_range_mask, (-kv_min).to(dtype=torch.half), kv_zp)

    # Quantized payload is uint8; clamp guards against small numerical overshoot.
    kv_int8 = torch.round(kv * kv_scale + kv_zp).clamp(int_min, int_max).to(dtype=torch.uint8).reshape(blk_num, kv_heads, -1)

    # Store dequant parameters as raw fp16 bytes in uint8 tensor tail.
    dq_scale = (1.0 / kv_scale).view(dtype=torch.uint8).reshape(blk_num, kv_heads, -1)
    kv_zp = kv_zp.view(dtype=torch.uint8).reshape(blk_num, kv_heads, -1)
    return torch.concat((kv_int8, dq_scale, kv_zp), dim=-1)


def DIV_UP(x, y):
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


class CacheQuantMode(str, Enum):
    BY_TOKEN = "BY_TOKEN"


@dataclass(frozen=True)
class PagedAttentionTestCase:
    subsequences: tuple[SubsequenceDescriptor, ...]
    num_heads: int = 2
    num_kv_heads: int = 2
    k_head_size: int = 64
    v_head_size: int = 64
    block_size: int = 16
    kv_cache_compression: bool = False
    key_cache_quant_mode: CacheQuantMode = CacheQuantMode.BY_TOKEN

# Helper dataclasses for test input generation,
# simulating the tensor shapes and data that would be generated
# in a real multi-sequence attention scenario with KV cache updates.
# All are in uncompressed fp16 format, as the KV cache updater will
# handle quantization if enabled.
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


KernelInputs = dict[str, object]


class KVCacheUpdater:
    def __init__(
        self,
        num_heads: int,
        num_kv_heads: int,
        head_size: int,
        block_sz: int,
        compressed_kvcache: bool,
        is_causal: bool = True,
    ):
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.block_sz = block_sz
        self.compressed_kvcache = compressed_kvcache
        self.is_causal = is_causal

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
        # Local KV cache table slice for this subsequence.
        local_num_blocks = int(past_key_cache_blocks.shape[0])
        local_block_indices = torch.arange(local_num_blocks, dtype=torch.long)

        if self.compressed_kvcache:
            past_key_cache_blocks = quan_per_token(past_key_cache_blocks)
            past_value_cache_blocks = quan_per_token(past_value_cache_blocks)

        # Simulate kv_cache_update input stage: dequantize/recover past tokens from cache.
        k_past = self.recover_context_from_cache(
            past_key_cache_blocks[local_block_indices].contiguous(),
            past_len,
            self.num_kv_heads,
            self.head_size,
            self.block_sz,
            self.compressed_kvcache,
        )
        v_past = self.recover_context_from_cache(
            past_value_cache_blocks[local_block_indices].contiguous(),
            past_len,
            self.num_kv_heads,
            self.head_size,
            self.block_sz,
            self.compressed_kvcache,
        )

        # Simulate kv_cache_update behavior: decode/dequantize past cache, append current KV,
        # then write back to cache layout (and quantize if compression is enabled).
        k_cache_source = torch.cat((k_past, current_key), dim=0).contiguous()
        v_cache_source = torch.cat((v_past, current_value), dim=0).contiguous()
        sequence_key_cache, sequence_value_cache = self._build_sequence_cache_blocks(k_cache_source, v_cache_source, context_len)
        if self.compressed_kvcache:
            sequence_key_cache = quan_per_token(sequence_key_cache)
            sequence_value_cache = quan_per_token(sequence_value_cache)
        return sequence_key_cache, sequence_value_cache

    @staticmethod
    def recover_context_from_cache(
        cache_blocks: torch.Tensor,
        context_len: int,
        num_kv_heads: int,
        head_size: int,
        block_size: int,
        compressed_kvcache: bool,
    ) -> torch.Tensor:
        if compressed_kvcache:
            cache_blocks = dequant_cache_per_token(cache_blocks, block_size, head_size)
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


class AttentionExecutor:
    def __init__(
        self,
        num_heads,
        num_kv_heads,
        head_size,
        block_sz,
        compressed_kvcache,
        is_causal=True,
        sparse_block_size=1,
        enable_hybrid_dispatch=True,
    ):
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.block_sz = block_sz
        self.compressed_kvcache = compressed_kvcache
        self.is_causal = is_causal
        self.sparse_block_size = sparse_block_size
        self.enable_hybrid_dispatch = enable_hybrid_dispatch

        self.q_step = CM_GRF_WIDTH // 32
        self.max_wg_size = 16       # Most optimal thread numbers per workgroup
        self.optimized_wg_seq_len = self.max_wg_size * self.q_step

        src1 = r'''#include "pa_multi_token.cm"'''
        cwd = os.path.dirname(os.path.realpath(__file__))
        print(
            f"{Colors.GREEN} [compile] "
            f"{num_heads=} {num_kv_heads=} {head_size=} {block_sz=} "
            f"{compressed_kvcache=} {is_causal=} {sparse_block_size=} {enable_hybrid_dispatch=} "
            f"{Colors.END}"
        )

        scale_factor = 1.0 / (head_size ** 0.5)
        base_options = (
            f' -cmc -Qxcm_register_file_size=256'
            f' -I{cwd}'
            f' -Qxcm_jit_option="-abortonspill" -mCM_printregusage'
            f" -mdump_asm -g2"
            f' -DKERNEL_NAME=cm_page_attention'
            f" -DCMFLA_NUM_HEADS={int(num_heads)}"
            f" -DCMFLA_NUM_KV_HEADS={int(num_kv_heads)}"
            f" -DCMFLA_HEAD_SIZE={int(head_size)}"
            f" -DCMFLA_SCALE_FACTOR={scale_factor}"
            f" -DCMFLA_IS_CAUSAL={int(is_causal)}"
            f" -DCMPA_BLOCK_SZ={int(self.block_sz)}"
            f" -DSPARSE_BLOCK_SIZE={int(self.sparse_block_size)}"
            f" -DCMPA_KVCACHE_U8={int(self.compressed_kvcache)}"
        )

        # Generic dynamic path (uses runtime local size to derive wg_seq_len in kernel).
        dynamic_options = base_options
        self.kernels_dynamic = cl.kernels(src1, dynamic_options)

        # Optional optimized path (compile-time fixed CMPA_WG_SEQ_LEN), only valid for sparse 128/256.
        self.kernels_optimized = None
        if self.sparse_block_size in (128, 256):
            optimized_options = base_options + f" -DCMPA_WG_SEQ_LEN={int(self.optimized_wg_seq_len)}"
            self.kernels_optimized = cl.kernels(src1, optimized_options)

    @staticmethod
    @functools.cache
    def create_instance(
        num_heads,
        num_kv_heads,
        head_size,
        block_sz,
        compressed_kvcache,
        is_causal,
        sparse_block_size=1,
        enable_hybrid_dispatch=True,
    ):
        return AttentionExecutor(
            num_heads,
            num_kv_heads,
            head_size,
            block_sz,
            compressed_kvcache,
            is_causal,
            sparse_block_size=sparse_block_size,
            enable_hybrid_dispatch=enable_hybrid_dispatch,
        )

    def _format_cache_for_kernel(self, cache: torch.Tensor) -> torch.Tensor:
        # Never fallback to fp16 kernel path when compression is enabled.
        if self.compressed_kvcache:
            return cache.contiguous()
        return cache.reshape(cache.shape[0], self.num_kv_heads, -1).contiguous()

    def __call__(self, kern_attn_inputs: KernelInputs, n_repeats: int = 1) -> torch.Tensor:
        query = kern_attn_inputs["query"]
        key_cache = kern_attn_inputs["key_cache"]
        value_cache = kern_attn_inputs["value_cache"]
        past_lens = kern_attn_inputs["past_lens"]
        block_indices = kern_attn_inputs["block_indices"]
        block_indices_begins = kern_attn_inputs["block_indices_begins"]
        subsequence_begins = kern_attn_inputs["subsequence_begins"]

        assert isinstance(query, torch.Tensor)
        assert isinstance(key_cache, torch.Tensor)
        assert isinstance(value_cache, torch.Tensor)
        assert isinstance(past_lens, torch.Tensor)
        assert isinstance(block_indices, torch.Tensor)
        assert isinstance(block_indices_begins, torch.Tensor)
        assert isinstance(subsequence_begins, torch.Tensor)

        batch_size_in_tokens = query.shape[0]
        output = torch.zeros(batch_size_in_tokens, self.num_heads * self.head_size, dtype=torch.float16)
        kv_dtype = torch.uint8 if self.compressed_kvcache else torch.float16

        cl.finish()

        for _ in range(n_repeats):
            q_tensor = query.reshape(batch_size_in_tokens, self.num_heads, self.head_size).contiguous()
            kernel_key_cache = self._format_cache_for_kernel(key_cache.contiguous())
            kernel_value_cache = self._format_cache_for_kernel(value_cache.contiguous())
            out_shape = (batch_size_in_tokens, self.num_heads, self.head_size)

            t_q = cl.tensor(q_tensor.to(torch.float16).detach().numpy())
            t_out = cl.tensor(list(out_shape), np.dtype(np.float16))
            t_key_cache = cl.tensor(kernel_key_cache.to(kv_dtype).detach().numpy())
            t_value_cache = cl.tensor(kernel_value_cache.to(kv_dtype).detach().numpy())
            t_past_lens = cl.tensor(past_lens.detach().numpy())
            t_block_indices = cl.tensor(block_indices.detach().numpy())
            t_block_indices_begins = cl.tensor(block_indices_begins.detach().numpy())
            t_subsequence_begins = cl.tensor(subsequence_begins.detach().numpy())

            max_subsequence_q_len = 0
            for sequence_index in range(int(past_lens.numel())):
                q_start, q_end, _, _ = get_sequence_ranges(kern_attn_inputs, sequence_index)
                subsequence_q_len = q_end - q_start
                if subsequence_q_len > max_subsequence_q_len:
                    max_subsequence_q_len = subsequence_q_len

            q_step = self.q_step
            wg_size = self.max_wg_size
            wg_seq_len = wg_size * q_step

            wg_count = 0
            for sequence_index in range(int(past_lens.numel())):
                q_start, q_end, _, _ = get_sequence_ranges(kern_attn_inputs, sequence_index)
                subsequence_q_len = q_end - q_start
                wg_count += DIV_UP(subsequence_q_len, wg_seq_len)
            gws = [1, self.num_heads, int(wg_count * wg_size)]
            lws = [1, 1, wg_size]

            use_optimized_dispatch = (
                self.enable_hybrid_dispatch
                and self.kernels_optimized is not None
                and self.sparse_block_size in (128, 256)
                and wg_size == self.max_wg_size
                and self.optimized_wg_seq_len == self.sparse_block_size
            )
            selected_kernels = self.kernels_optimized if use_optimized_dispatch else self.kernels_dynamic
            dispatch_mode = "optimized" if use_optimized_dispatch else "dynamic"

            print(
                f"{Colors.GREEN}[enqueue] gws={gws} lws={lws} dispatch_mode={dispatch_mode} "
                f"{Colors.END}"
            )
            print("[enqueue] q.shape=", tuple(q_tensor.shape), "q.dtype=", q_tensor.dtype)
            print("[enqueue] key_cache.shape=", tuple(kernel_key_cache.shape), "key_cache.dtype=", kernel_key_cache.dtype)
            print("[enqueue] value_cache.shape=", tuple(kernel_value_cache.shape), "value_cache.dtype=", kernel_value_cache.dtype)
            print("[enqueue] past_lens=", past_lens.tolist())
            print("[enqueue] block_indices=", block_indices.tolist())
            print("[enqueue] block_indices_begins=", block_indices_begins.tolist())
            print("[enqueue] subsequence_begins=", subsequence_begins.tolist())
            print("[enqueue] q_len=", batch_size_in_tokens, "out.shape=", out_shape)

            selected_kernels.enqueue(
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


class PagedAttentionRuntime:
    def __init__(
        self,
        num_heads: int,
        num_kv_heads: int,
        head_size: int,
        block_size: int,
        kv_cache_compression: bool,
        is_causal: bool = True,
    ):
        self.kvcache_updater = KVCacheUpdater(
            num_heads,
            num_kv_heads,
            head_size,
            block_size,
            kv_cache_compression,
            is_causal,
        )
        self.attn_executor = AttentionExecutor.create_instance(
            num_heads,
            num_kv_heads,
            head_size,
            block_size,
            kv_cache_compression,
            is_causal,
        )

    def stage_kvcache_update(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        past_lens: list[int],
        subsequence_begins: list[int],
        kvcache_table: KVCacheTable,
    ) -> KernelInputs:
        return self.kvcache_updater(
            key,
            value,
            past_lens,
            subsequence_begins,
            kvcache_table.block_indices,
            kvcache_table.block_indices_begins,
            kvcache_table.key_cache,
            kvcache_table.value_cache,
        )

    def stage_attention(
        self,
        kern_attn_inputs: KernelInputs,
    ) -> torch.Tensor:
        attn_outputs = self.attn_executor(kern_attn_inputs)
        return attn_outputs


def create_subsequence_tensors(
    subsequence: SubsequenceDescriptor,
    num_heads,
    num_kv_heads,
    head_size,
    act_dtype=torch.float16,
) -> SubsequenceTensors:
    low = -1
    high = 2
    q_cur = torch.randint(low, high, [subsequence.num_tokens, num_heads, head_size]).to(dtype=act_dtype)

    # Current KV are always uncompressed fp16 tensors.
    k_cur = torch.rand(subsequence.num_tokens, num_kv_heads, head_size).to(dtype=act_dtype)
    v_cur = torch.rand(subsequence.num_tokens, num_kv_heads, head_size).to(dtype=act_dtype) / high

    # Past KV are uncompressed fp16 token tensors of past_len.
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
    keep_order=False,
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

    # Primitive-facing flattened input shapes after staging are:
    # query: [batch_size_in_tokens, num_heads * head_size]
    # key:   [batch_size_in_tokens, num_kv_heads * head_size]
    # value: [batch_size_in_tokens, num_kv_heads * head_size]
    return query, key, value, past_lens, subsequence_begins, kvcache_table


def dequant_cache_per_token(cache: torch.Tensor, block_size: int, head_size: int) -> torch.Tensor:
    if cache.dtype != torch.uint8:
        raise ValueError("dequant_cache_per_token expects uint8 cache")

    data_size = block_size * head_size
    scale_size = block_size * 2
    data = cache[:, :, :data_size].reshape(cache.shape[0], cache.shape[1], block_size, head_size).to(dtype=torch.float16)
    dq_scale = cache[:, :, data_size:data_size + scale_size].reshape(cache.shape[0], cache.shape[1], block_size, 2).contiguous().view(dtype=torch.float16)
    zp = cache[:, :, data_size + scale_size:].reshape(cache.shape[0], cache.shape[1], block_size, 2).contiguous().view(dtype=torch.float16)

    return (data - zp) * dq_scale


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


def reference_attention(
    kern_attn_inputs: KernelInputs,
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    block_size: int,
    compressed_kvcache: bool,
    is_causal: bool,
) -> torch.Tensor:
    query = kern_attn_inputs["query"]
    past_lens = kern_attn_inputs["past_lens"]
    block_indices = kern_attn_inputs["block_indices"]
    key_cache = kern_attn_inputs["key_cache"]
    value_cache = kern_attn_inputs["value_cache"]
    assert isinstance(query, torch.Tensor)
    assert isinstance(past_lens, torch.Tensor)
    assert isinstance(block_indices, torch.Tensor)
    assert isinstance(key_cache, torch.Tensor)
    assert isinstance(value_cache, torch.Tensor)

    refs = []
    for sequence_index in range(int(past_lens.numel())):
        q_start, q_end, blk_start, blk_end = get_sequence_ranges(kern_attn_inputs, sequence_index)
        q_len = q_end - q_start
        past_len = int(past_lens[sequence_index].item())
        context_len = past_len + q_len

        q = query[q_start:q_end].reshape(q_len, num_heads, head_size).contiguous()
        physical_blocks = block_indices[blk_start:blk_end]
        key_cache_blocks = key_cache[physical_blocks].contiguous()
        value_cache_blocks = value_cache[physical_blocks].contiguous()

        key_context = KVCacheUpdater.recover_context_from_cache(
            key_cache_blocks,
            context_len,
            num_kv_heads,
            head_size,
            block_size,
            compressed_kvcache,
        )
        value_context = KVCacheUpdater.recover_context_from_cache(
            value_cache_blocks,
            context_len,
            num_kv_heads,
            head_size,
            block_size,
            compressed_kvcache,
        )
        attention_mask = get_attention_mask(
            q_len,
            context_len,
            num_heads,
            q.dtype,
            q.device,
            past_len=past_len,
            is_causal=is_causal,
        )
        ref = flash_attn_vlen_ref(q, key_context, value_context, [], is_causal, attention_mask)
        refs.append(ref.reshape(q_len, -1))

    return torch.cat(refs, dim=0)


def run_paged_attention_smoke_case(case: PagedAttentionTestCase, check_acc=True) -> KernelInputs:
    if case.key_cache_quant_mode != CacheQuantMode.BY_TOKEN:
        raise ValueError("Only CacheQuantMode.BY_TOKEN is supported in this test harness")
    if case.k_head_size != case.v_head_size:
        raise ValueError("k_head_size must equal v_head_size in this CM paged-attention harness")

    # Represent the process of qkv projection and cache update before executing the attention kernel, which is 
    # the same for both reference and tested implementation. The test harness focuses on validating the attention kernel itself, 
    # by ensuring it receives the same inputs in both reference and tested execution.

    subsequence_tensors = [
        create_subsequence_tensors(
            descriptor,
            case.num_heads,
            case.num_kv_heads,
            case.k_head_size,
        )
        for descriptor in case.subsequences
    ]

    query, key, value, past_lens, subsequence_begins, kvcache_table = create_paged_attention_inputs(
        case,
        subsequence_tensors,
    )

    runtime = PagedAttentionRuntime(
        case.num_heads,
        case.num_kv_heads,
        case.k_head_size,
        case.block_size,
        case.kv_cache_compression,
        True,
    )

    kern_attn_inputs = runtime.stage_kvcache_update(key, value, past_lens, subsequence_begins, kvcache_table)
    kern_attn_inputs["query"] = query
    attn_outputs = runtime.stage_attention(kern_attn_inputs)

    if check_acc:
        round_ref = reference_attention(
            kern_attn_inputs,
            num_heads=case.num_heads,
            num_kv_heads=case.num_kv_heads,
            head_size=case.k_head_size,
            block_size=case.block_size,
            compressed_kvcache=case.kv_cache_compression,
            is_causal=True,
        )
        check_close(round_ref, attn_outputs)
    else:
        assert torch.isfinite(attn_outputs).all().item()

    return kern_attn_inputs


def requires_decode_path(case: PagedAttentionTestCase) -> bool:
    return any(subsequence.past_len > 0 for subsequence in case.subsequences)


def assert_generate_stage_inputs(kern_attn_inputs: KernelInputs):
    query = kern_attn_inputs["query"]
    past_lens = kern_attn_inputs["past_lens"]
    subsequence_begins = kern_attn_inputs["subsequence_begins"]
    assert isinstance(query, torch.Tensor)
    assert isinstance(past_lens, torch.Tensor)
    assert isinstance(subsequence_begins, torch.Tensor)
    batch_size_in_tokens = int(query.shape[0])
    batch_size_in_sequences = int(past_lens.numel())
    assert batch_size_in_tokens == batch_size_in_sequences
    assert batch_size_in_tokens > 0
    assert torch.all(past_lens >= 1)
    token_counts = torch.diff(subsequence_begins)
    assert torch.all(token_counts == 1)


PREFILL_ONLY_SMOKE_CASES = (
    PagedAttentionTestCase(
        subsequences=(ss(10),),
    ),
    PagedAttentionTestCase(
        subsequences=(ss(36),),
    ),
    PagedAttentionTestCase(
        subsequences=(ss(10), ss(30)),
    ),
    PagedAttentionTestCase(
        subsequences=(ss(128), ss(256)),
    ),
    PagedAttentionTestCase(
        subsequences=(
            ss(10),
            ss(81),
            ss(129),
        ),
    ),
    PagedAttentionTestCase(
        subsequences=(ss(32),),
        num_heads=8,
        num_kv_heads=2,
        k_head_size=64,
        v_head_size=64,
    ),
    PagedAttentionTestCase(
        subsequences=(ss(32),),
        num_heads=2,
        num_kv_heads=2,
        k_head_size=32,
        v_head_size=32,
        block_size=32,
    ),
)


GENERATE_ONLY_SMOKE_CASES = (
    # Test cases with varying sequence lengths, all with past_len > 0 to ensure they are in the generate stage
    PagedAttentionTestCase(
        subsequences=(ss(1, 10),),
    ),
    PagedAttentionTestCase(
        subsequences=(ss(1, 34), ss(1, 515)),
    ),
    
    # Test cases with varying attention configurations, all with past_len > 0
    PagedAttentionTestCase(
        subsequences=(ss(1, 10),),
        k_head_size=32,
        v_head_size=32,
    ),
    PagedAttentionTestCase(
        subsequences=(ss(1, 34), ss(1, 515)),
        num_heads=8,
        num_kv_heads=2,
    ),
    
    # Test cases with varying block sizes, all with past_len > 0
    PagedAttentionTestCase(
        subsequences=(ss(1, 34),),
        block_size=32,
    ),
    PagedAttentionTestCase(
        subsequences=(ss(1, 34), ss(1, 515)),
        block_size=256,
    ),
    
    # Test cases with KV cache compression enabled
    PagedAttentionTestCase(
        subsequences=(ss(1, 10),),
        kv_cache_compression=True,
    ),
    PagedAttentionTestCase(
        subsequences=(ss(1, 34), ss(1, 515)),
        kv_cache_compression=True,
    ),
)


MIXED_ONLY_SMOKE_CASES = (
    PagedAttentionTestCase(
        subsequences=(
            ss(1, 34),
            ss(25),
            ss(10, 34),
        ),
    ),
    PagedAttentionTestCase(
        subsequences=(
            ss(1, 34),
            ss(25),
            ss(10, 34),
        ),
        k_head_size=32,
        v_head_size=32,
        block_size=32,
    ),
    PagedAttentionTestCase(
        subsequences=(
            ss(1, 34),
            ss(25),
            ss(10, 34),
        ),
        num_heads=8,
        num_kv_heads=2,
    ),
    PagedAttentionTestCase(
        subsequences=(
            ss(1, 34),
            ss(25),
            ss(10, 34),
        ),
        kv_cache_compression=True,
    ),
)


def make_smoke_case_id(case: PagedAttentionTestCase) -> str:
    seq_parts = []
    for subsequence in case.subsequences:
        if subsequence.past_len == 0:
            seq_parts.append(f"{subsequence.num_tokens}")
        else:
            seq_parts.append(f"{subsequence.num_tokens}x{subsequence.past_len}")

    seq_id = "_".join(seq_parts)
    return (
        f"{seq_id}"
        f"__h{case.num_heads}"
        f"_kv{case.num_kv_heads}"
        f"_khs{case.k_head_size}"
        f"_vhs{case.v_head_size}"
        f"_bls{case.block_size}"
        f"_cmpr{int(case.kv_cache_compression)}"
        f"_qm{case.key_cache_quant_mode.value}"
    )


@pytest.mark.parametrize("case", PREFILL_ONLY_SMOKE_CASES, ids=make_smoke_case_id)
def test_pa_smoke_paged_attention_prefill_only(case: PagedAttentionTestCase):
    print(f"{Colors.GREEN}[testcase] prefill_only id={make_smoke_case_id(case)}{Colors.END}")
    kern_attn_inputs = run_paged_attention_smoke_case(case, check_acc=True)
    max_context_len = int(kern_attn_inputs["max_context_len"])
    past_lens = kern_attn_inputs["past_lens"]
    query = kern_attn_inputs["query"]
    assert isinstance(past_lens, torch.Tensor)
    assert isinstance(query, torch.Tensor)

    assert max_context_len == max(subsequence.total_seq_len for subsequence in case.subsequences)
    assert past_lens.tolist() == [subsequence.past_len for subsequence in case.subsequences]

    assert all(subsequence.past_len == 0 for subsequence in case.subsequences)
    assert any(subsequence.num_tokens > 1 for subsequence in case.subsequences)
    assert int(query.shape[0]) != int(past_lens.numel())


@pytest.mark.parametrize("case", GENERATE_ONLY_SMOKE_CASES, ids=make_smoke_case_id)
def test_pa_smoke_paged_attention_generate_only(case: PagedAttentionTestCase):
    print(f"{Colors.GREEN}[testcase] generate_only id={make_smoke_case_id(case)}{Colors.END}")

    kern_attn_inputs = run_paged_attention_smoke_case(case, check_acc=False)
    max_context_len = int(kern_attn_inputs["max_context_len"])
    past_lens = kern_attn_inputs["past_lens"]
    assert isinstance(past_lens, torch.Tensor)

    assert max_context_len == max(subsequence.total_seq_len for subsequence in case.subsequences)
    assert past_lens.tolist() == [subsequence.past_len for subsequence in case.subsequences]
    assert_generate_stage_inputs(kern_attn_inputs)



@pytest.mark.parametrize("case", MIXED_ONLY_SMOKE_CASES, ids=make_smoke_case_id)
def test_pa_smoke_paged_attention_mixed_only(case: PagedAttentionTestCase):
    print(f"{Colors.GREEN}[testcase] mixed_only id={make_smoke_case_id(case)}{Colors.END}")

    kern_attn_inputs = run_paged_attention_smoke_case(case, check_acc=False)
    max_context_len = int(kern_attn_inputs["max_context_len"])
    past_lens = kern_attn_inputs["past_lens"]
    query = kern_attn_inputs["query"]
    assert isinstance(past_lens, torch.Tensor)
    assert isinstance(query, torch.Tensor)

    assert max_context_len == max(subsequence.total_seq_len for subsequence in case.subsequences)
    assert past_lens.tolist() == [subsequence.past_len for subsequence in case.subsequences]

    has_prefill = any(subsequence.past_len == 0 and subsequence.num_tokens > 1 for subsequence in case.subsequences)
    has_generate = any(subsequence.past_len >= 1 for subsequence in case.subsequences)
    assert has_prefill and has_generate
    assert int(query.shape[0]) != int(past_lens.numel())


# '''
# Usage:

# # `py_compile` to verify syntax correctness of the test file before running pytest.
# python -m py_compile test_pa_multiseq.py

# list all tests in the file without executing them, to verify that pytest can discover the tests correctly.
# python -m pytest --collect-only -q test_pa_multiseq.py | grep 'test_pa_smoke_paged_attention_generate_only'

# # `-q` for quieter pytest output, and `-k` to select the specific test to run.
# python -m pytest -q test_pa_multiseq.py -k 'test_pa_smoke_paged_attention_generate_only and 1x10'

# # `-s` to show captured print statement for debugging purposes.
# python -m pytest -s test_pa_multiseq.py -k 'generate_only or mixed_only'
# python -m pytest -s "test_pa_multiseq.py::test_pa_smoke_paged_attention_generate_only[1x10__h2_kv2_khs64_vhs64_bls16_cmpr1_qmBY_TOKEN]"

# '''

