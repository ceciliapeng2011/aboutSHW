"""
test_pa_multiseq.py

Purpose
-------
This module is the functional integration test harness for CM paged-attention
in multi-sequence scenarios. It validates that the OpenCL/CM kernel path and
the Python reference path produce consistent outputs across realistic batching
patterns.

What this file verifies
-----------------------
1) Prefill-only rounds
    - Multiple subsequences in one launch where all tokens are prompt tokens.

2) Generate-only rounds
    - Decode-like rounds with short token counts and non-zero past length.

3) Mixed rounds
    - Prompt and decode style subsequences scheduled together.

4) Routing behavior
    - Split-route and mixed-route execution produce numerically stable outputs
      and match the same reference attention computation.

5) Cache variants and kernel dispatch
    - FP16 KV cache and compressed KV cache modes.
        - Compression mode uses `KV_CACHE_COMPRESSION` with values: 0 (none),
            1 (by-token), 2 (by-channel).
    - Sparse/dense block configurations and dynamic vs optimized dispatch paths.

Implementation notes
--------------------
- `PaMultiTokenRunner` executes the CM kernel(s) and prepares kernel inputs.
- `PaSingleTokenRunner` executes the decode/single-token CM path used for
    generate-stage subsequences and mixed-route split mode.
- `PagedAttentionRunner` orchestrates cache updates, sequence mapping, and
  end-to-end round execution.
- `reference_attention()` computes a PyTorch reference for correctness checks.

In short, this file protects multi-sequence paged-attention correctness and
route consistency for the kernel implementation.

How this differs from `test_cb.py`
----------------------------------
- This file validates attention kernel correctness for a single assembled
    round (prefill/generate/mixed), including route selection and reference
    parity.
- `test_cb.py` validates continuous-batching scheduling policy and token
    accounting across many rounds, and then invokes runners from this file as
    execution backends.
"""

import functools
import os

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from clops import cl
from clops.utils import Colors
from pa_test_common import (
    DEFAULT_SUB_BLOCK_SIZE,
    DIV_UP,
    KV_CACHE_COMPRESSION_NONE,
    KV_CACHE_COMPRESSION_BY_CHANNEL,
    KV_CACHE_COMPRESSION_BY_TOKEN,
    KVCacheTable,
    KVCacheUpdater,
    KernelInputs,
    PagedAttentionTestCase,
    check_close,
    create_paged_attention_inputs,
    create_subsequence_tensors,
    get_attention_mask,
    get_cm_grf_width,
    get_sequence_ranges,
    normalize_kv_cache_compression,
    ss,
)

cl.profiling(True)
torch.manual_seed(0)
torch.set_printoptions(linewidth=1024)

CM_GRF_WIDTH = get_cm_grf_width()
print(f"{CM_GRF_WIDTH=}")


class PaMultiTokenRunner:
    def __init__(
        self,
        num_heads,
        num_kv_heads,
        head_size,
        block_sz,
        kv_cache_compression,
        sub_block_size=DEFAULT_SUB_BLOCK_SIZE,
        is_causal=True,
        sparse_block_size=1,
        enable_hybrid_dispatch=True,
    ):
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.block_sz = block_sz
        self.kv_cache_compression = normalize_kv_cache_compression(kv_cache_compression)
        self.sub_block_size = int(sub_block_size)
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
            f"{self.kv_cache_compression=} {self.sub_block_size=} {is_causal=} {sparse_block_size=} {enable_hybrid_dispatch=} "
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
            f" -DKV_CACHE_COMPRESSION={int(self.kv_cache_compression)}"
            f" -DSUB_BLOCK_SIZE={int(self.sub_block_size)}"
        )

        # Generic dynamic path (uses pa_runner local size to derive wg_seq_len in kernel).
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
        kv_cache_compression,
        is_causal,
        sub_block_size=DEFAULT_SUB_BLOCK_SIZE,
        sparse_block_size=1,
        enable_hybrid_dispatch=True,
    ):
        return PaMultiTokenRunner(
            num_heads,
            num_kv_heads,
            head_size,
            block_sz,
            kv_cache_compression,
            is_causal=is_causal,
            sub_block_size=sub_block_size,
            sparse_block_size=sparse_block_size,
            enable_hybrid_dispatch=enable_hybrid_dispatch,
        )

    def _format_cache_for_kernel(self, cache: torch.Tensor) -> torch.Tensor:
        # Never fallback to fp16 kernel path when compression is enabled.
        if self.kv_cache_compression != KV_CACHE_COMPRESSION_NONE:
            return cache.contiguous()
        return cache.reshape(cache.shape[0], self.num_kv_heads, -1).contiguous()

    @staticmethod
    def _build_wg_block_start_and_subseq_mapping(
        kern_attn_inputs: KernelInputs,
        selected_sequence_ids: torch.Tensor,
        wg_seq_len: int,
    ) -> tuple[torch.Tensor, int]:
        # Keep in sync with pa_multi_token.cm header notes.
        # Host prepares a packed int32 pair list consumed by kernel as:
        #   [block_start_pos_0, subsequence_id_0,
        #    block_start_pos_1, subsequence_id_1, ...]
        # For each selected subsequence, split its query range [q_start, q_end)
        # into ceil_div(subsequence_q_len, wg_seq_len) chunks and emit one pair
        # per chunk with block_start_pos = q_start + chunk_idx * wg_seq_len.
        blocked_q_starts_and_subseq_mapping: list[int] = []

        for sequence_id in selected_sequence_ids.tolist():
            sequence_index = int(sequence_id)
            q_start, q_end, _, _ = get_sequence_ranges(kern_attn_inputs, sequence_index)
            subsequence_q_len = q_end - q_start
            subseq_wg_count = DIV_UP(subsequence_q_len, wg_seq_len)
            for mapped_wg_id in range(subseq_wg_count):
                blocked_q_starts_and_subseq_mapping.append(q_start + mapped_wg_id * wg_seq_len)
                blocked_q_starts_and_subseq_mapping.append(sequence_index)

        wg_count = len(blocked_q_starts_and_subseq_mapping) // 2
        return torch.tensor(blocked_q_starts_and_subseq_mapping, dtype=torch.int32), wg_count

    def __call__(
        self,
        kern_attn_inputs: KernelInputs,
        out: torch.Tensor,
        prefill_seq_indices: list[int],
        n_repeats: int = 1,
    ) -> torch.Tensor:
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

        if len(prefill_seq_indices) == 0:
            raise ValueError("prefill_seq_indices must be non-empty")

        selected_sequence_ids = torch.tensor(prefill_seq_indices, dtype=torch.int32)
        prefill_seq_count = int(selected_sequence_ids.numel())

        batch_size_in_tokens = query.shape[0]
        expected_shape = (batch_size_in_tokens, self.num_heads * self.head_size)
        if tuple(out.shape) != expected_shape:
            raise ValueError(f"out shape mismatch: got {tuple(out.shape)}, expected {expected_shape}")
        if out.dtype != torch.float16:
            raise ValueError(f"out dtype mismatch: got {out.dtype}, expected torch.float16")
        output = out
        kv_dtype = torch.uint8 if self.kv_cache_compression != KV_CACHE_COMPRESSION_NONE else torch.float16

        cl.finish()

        for _ in range(n_repeats):
            q_tensor = query.reshape(batch_size_in_tokens, self.num_heads, self.head_size).contiguous()
            kernel_key_cache = self._format_cache_for_kernel(key_cache.contiguous())
            kernel_value_cache = self._format_cache_for_kernel(value_cache.contiguous())
            out_shape = (batch_size_in_tokens, self.num_heads, self.head_size)

            t_q = cl.tensor(q_tensor.to(torch.float16).detach().numpy())
            output_3d = output.reshape(batch_size_in_tokens, self.num_heads, self.head_size).contiguous()
            t_out = cl.tensor(output_3d.detach().numpy())
            t_key_cache = cl.tensor(kernel_key_cache.to(kv_dtype).detach().numpy())
            t_value_cache = cl.tensor(kernel_value_cache.to(kv_dtype).detach().numpy())
            t_past_lens = cl.tensor(past_lens.detach().numpy())
            t_block_indices = cl.tensor(block_indices.detach().numpy())
            t_block_indices_begins = cl.tensor(block_indices_begins.detach().numpy())
            t_subsequence_begins = cl.tensor(subsequence_begins.detach().numpy())

            q_step = self.q_step
            wg_size = self.max_wg_size
            wg_seq_len = wg_size * q_step

            blocked_q_starts_and_subseq_mapping, wg_count = self._build_wg_block_start_and_subseq_mapping(
                kern_attn_inputs,
                selected_sequence_ids,
                wg_seq_len,
            )
            if wg_count <= 0:
                raise ValueError("Invalid blocked query mapping: wg_count must be positive")

            t_blocked_q_starts_and_subseq_mapping = cl.tensor(blocked_q_starts_and_subseq_mapping.detach().numpy())
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
            print("[enqueue] prefill_seq_indices=", prefill_seq_indices)
            print("[enqueue] prefill_seq_count=", prefill_seq_count)
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
                t_blocked_q_starts_and_subseq_mapping,
                t_out,
                batch_size_in_tokens,
            )

            output_from_kernel = torch.from_numpy(t_out.numpy().reshape(batch_size_in_tokens, -1))
            out.copy_(output_from_kernel)
            output = out

        return output

class PaSingleTokenRunner:
    def __init__(
        self,
        num_heads: int,
        num_kv_heads: int,
        head_size: int,
        block_size: int,
        kv_cache_compression: int,
        sub_block_size: int = DEFAULT_SUB_BLOCK_SIZE,
    ):
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.block_size = block_size
        self.kv_cache_compression = normalize_kv_cache_compression(kv_cache_compression)
        self.sub_block_size = int(sub_block_size)
        if self.kv_cache_compression == KV_CACHE_COMPRESSION_BY_CHANNEL and self.block_size % self.sub_block_size != 0:
            raise ValueError(
                f"block_size ({self.block_size}) must be divisible by sub_block_size ({self.sub_block_size}) for mode-2"
            )

        self.cm_grf_width = get_cm_grf_width()
        self.xe_arch = 1 if self.cm_grf_width == 256 else 2
        self.kv_step = 8 if self.xe_arch == 1 else 16

        self.k_partition_block_num = 1
        self.kv_partition_size = int(self.block_size * self.k_partition_block_num)
        self.reduce_split_step = 8

        max_repeat_count = 8
        q_heads_per_kv_head = self.num_heads // self.num_kv_heads
        self.q_head_chunks_per_kv_head = (q_heads_per_kv_head + (max_repeat_count - 1)) // max_repeat_count
        self.q_head_chunk_size = self.num_heads // (self.num_kv_heads * self.q_head_chunks_per_kv_head)
        self.scale_factor = 1.0 / (self.head_size ** 0.5)

    @staticmethod
    @functools.cache
    def create_instance(
        num_heads: int,
        num_kv_heads: int,
        head_size: int,
        block_size: int,
        kv_cache_compression: int,
        sub_block_size: int = DEFAULT_SUB_BLOCK_SIZE,
    ):
        return PaSingleTokenRunner(
            num_heads,
            num_kv_heads,
            head_size,
            block_size,
            kv_cache_compression,
            sub_block_size,
        )

    @staticmethod
    @functools.cache
    def _create_kernels_cached(
        num_heads: int,
        num_kv_heads: int,
        head_size: int,
        kv_step: int,
        block_size: int,
        kv_partition_size: int,
        reduce_split_step: int,
        clean_unused_kvcache: int,
        kv_cache_compression: int,
        sub_block_size: int,
        xe_arch: int,
        q_head_chunks_per_kv_head: int,
        q_head_chunk_size: int,
        scale_factor: float,
    ):
        src = "\n".join(
            [
                '#include "pa_single_token.cm"',
                '#include "pa_single_token_finalization.cm"',
            ]
        )
        cwd = os.path.dirname(os.path.realpath(__file__))
        return cl.kernels(src, f'''-cmc -Qxcm_jit_option=""
                            -mCM_printregusage -mdump_asm -g2
                            -Qxcm_register_file_size=256 -I{cwd}
                            -DHEADS_NUM={num_heads} -DKV_HEADS_NUM={num_kv_heads} -DHEAD_SIZE={head_size}
                            -DQ_STEP=32 -DKV_STEP={kv_step}
                            -DWG_SIZE=1 -DKV_BLOCK_SIZE={block_size}
                            -DKV_PARTITION_SIZE={kv_partition_size} -DREDUCE_SPLIT_SIZE={reduce_split_step}
                            -DCLEAN_UNUSED_KVCACHE={clean_unused_kvcache}
                            -DKV_CACHE_COMPRESSION={kv_cache_compression}
                            -DSUB_BLOCK_SIZE={sub_block_size}
                            -DXE_ARCH={xe_arch}
                            -DQ_head_chunks_per_kv_head={q_head_chunks_per_kv_head}
                            -DQ_head_chunk_size={q_head_chunk_size}
                            -DSCALE_FACTOR={scale_factor}''')

    def _create_kernels(self):
        return self._create_kernels_cached(
            self.num_heads,
            self.num_kv_heads,
            self.head_size,
            self.kv_step,
            self.block_size,
            self.kv_partition_size,
            self.reduce_split_step,
            1,
            int(self.kv_cache_compression),
            int(self.sub_block_size),
            self.xe_arch,
            int(self.q_head_chunks_per_kv_head),
            int(self.q_head_chunk_size),
            self.scale_factor,
        )

    def __call__(
        self,
        kern_attn_inputs: KernelInputs,
        out: torch.Tensor,
        decode_seq_indices: list[int],
        n_repeats: int = 1,
    ) -> torch.Tensor:
        query = kern_attn_inputs["query"]
        past_lens_t = kern_attn_inputs["past_lens"]
        block_indices_t = kern_attn_inputs["block_indices"]
        block_indices_begins_t = kern_attn_inputs["block_indices_begins"]
        subsequence_begins_t = kern_attn_inputs["subsequence_begins"]
        key_cache_t = kern_attn_inputs["key_cache"]
        value_cache_t = kern_attn_inputs["value_cache"]

        assert isinstance(query, torch.Tensor)
        assert isinstance(past_lens_t, torch.Tensor)
        assert isinstance(block_indices_t, torch.Tensor)
        assert isinstance(block_indices_begins_t, torch.Tensor)
        assert isinstance(subsequence_begins_t, torch.Tensor)
        assert isinstance(key_cache_t, torch.Tensor)
        assert isinstance(value_cache_t, torch.Tensor)

        if len(decode_seq_indices) == 0:
            raise ValueError("decode_seq_indices must be non-empty")

        # Keep in sync with pa_single_token.cm notes.
        # selected_sequence_ids is a compact decode-subset -> original-sequence map:
        # kernel seq_idx indexes this array, then resolves orig_seq_idx to read
        # per-sequence metadata (past_lens/subsequence_begins/block_indices_begins).
        # selected_sequence_count equals len(selected_sequence_ids).
        selected_sequence_ids = torch.tensor(decode_seq_indices, dtype=torch.int32)
        decode_seq_count = int(selected_sequence_ids.numel())

        token_counts = torch.diff(subsequence_begins_t)
        if token_counts.numel() and decode_seq_count > 0:
            selected_token_counts = token_counts[selected_sequence_ids.to(dtype=torch.long)]
        else:
            selected_token_counts = token_counts
        if selected_token_counts.numel() and not torch.all(selected_token_counts == 1).item():
            raise ValueError("pa_single_token path expects num_tokens == 1 for every subsequence")

        batch_size = decode_seq_count
        max_context_len = int(kern_attn_inputs["max_context_len"])
        kv_partition_num = DIV_UP(max_context_len, self.kv_partition_size)
        q_tokens = query.reshape(int(query.shape[0]), self.num_heads, self.head_size).contiguous()
        output_tokens = int(query.shape[0])
        use_subset_execution = output_tokens != batch_size
        selected_token_indices = (subsequence_begins_t[selected_sequence_ids.to(dtype=torch.long) + 1] - 1).to(dtype=torch.long)

        expected_shape = (output_tokens, self.num_heads * self.head_size)
        if tuple(out.shape) != expected_shape:
            raise ValueError(f"out shape mismatch: got {tuple(out.shape)}, expected {expected_shape}")
        if out.dtype != torch.float16:
            raise ValueError(f"out dtype mismatch: got {out.dtype}, expected torch.float16")
        output = out

        kernels = self._create_kernels()
        gws = [batch_size, self.num_kv_heads * self.q_head_chunks_per_kv_head, kv_partition_num]
        lws = [1, 1, 1]
        gws_2 = [batch_size, self.num_heads, self.head_size // self.reduce_split_step]
        lws_2 = [1, 1, 1]

        for _ in range(n_repeats):
            t_q = cl.tensor(q_tokens.detach().numpy())
            t_k = cl.tensor(key_cache_t.detach().numpy())
            t_v = cl.tensor(value_cache_t.detach().numpy())
            t_past_lens = cl.tensor(past_lens_t.detach().numpy())
            t_block_indices = cl.tensor(block_indices_t.detach().numpy())
            t_block_indices_begins = cl.tensor(block_indices_begins_t.detach().numpy())
            t_subsequence_begins = cl.tensor(subsequence_begins_t.detach().numpy())
            t_selected_sequence_ids = cl.tensor(selected_sequence_ids.detach().numpy())
            t_out = cl.tensor([batch_size, self.num_heads, kv_partition_num, self.head_size], np.dtype(np.float32))
            t_out_final = cl.tensor([batch_size, 1, self.num_heads, self.head_size], np.dtype(np.float16))
            t_lse = cl.tensor([batch_size, self.num_heads, kv_partition_num], np.dtype(np.float32))

            print(f"{Colors.GREEN}[enqueue single] gws={gws} lws={lws}{Colors.END}")

            kernels.enqueue(
                "cm_sdpa_2nd",
                gws,
                lws,
                t_q,
                t_k,
                t_v,
                t_past_lens,
                t_block_indices,
                t_block_indices_begins,
                t_subsequence_begins,
                t_selected_sequence_ids,
                decode_seq_count,
                t_out,
                t_lse,
                1,
            )
            kernels.enqueue(
                "cm_sdpa_2nd_reduce",
                gws_2,
                lws_2,
                t_out,
                t_out_final,
                t_lse,
                kv_partition_num,
            )
            cl.finish()
            output_compact = torch.from_numpy(t_out_final.numpy().reshape(batch_size, -1))
            if use_subset_execution:
                out[selected_token_indices] = output_compact
            else:
                out.copy_(output_compact)
            output = out

        assert torch.isfinite(output).all().item()
        return output

class PagedAttentionRunner:
    @staticmethod
    def _create_multi_token_runner(
        num_heads: int,
        num_kv_heads: int,
        head_size: int,
        block_size: int,
        kv_cache_compression: int,
        sub_block_size: int,
        is_causal: bool,
    ) -> PaMultiTokenRunner:
        return PaMultiTokenRunner.create_instance(
            num_heads,
            num_kv_heads,
            head_size,
            block_size,
            kv_cache_compression,
            is_causal,
            sub_block_size=sub_block_size,
        )

    @staticmethod
    def _create_single_token_runner(
        num_heads: int,
        num_kv_heads: int,
        head_size: int,
        block_size: int,
        kv_cache_compression: int,
        sub_block_size: int,
    ) -> PaSingleTokenRunner:
        return PaSingleTokenRunner.create_instance(
            num_heads,
            num_kv_heads,
            head_size,
            block_size,
            kv_cache_compression,
            sub_block_size,
        )

    def __init__(
        self,
        num_heads: int,
        num_kv_heads: int,
        head_size: int,
        block_size: int,
        kv_cache_compression: int,
        sub_block_size: int = DEFAULT_SUB_BLOCK_SIZE,
        is_causal: bool = True,
    ):
        self.kvcache_updater = KVCacheUpdater(
            num_heads,
            num_kv_heads,
            head_size,
            block_size,
            kv_cache_compression,
            sub_block_size,
            is_causal,
        )
        self.multi_token_runner = self._create_multi_token_runner(
            num_heads,
            num_kv_heads,
            head_size,
            block_size,
            kv_cache_compression,
            sub_block_size,
            is_causal,
        )
        self.single_token_runner = self._create_single_token_runner(
            num_heads,
            num_kv_heads,
            head_size,
            block_size,
            kv_cache_compression,
            sub_block_size,
        )

    def _prepare_kern_attn_inputs(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        past_lens: list[int],
        subsequence_begins: list[int],
        kvcache_table: KVCacheTable,
        query: torch.Tensor,
    ) -> KernelInputs:
        kern_attn_inputs = self.kvcache_update(
            key,
            value,
            past_lens,
            subsequence_begins,
            kvcache_table,
        )
        kern_attn_inputs["query"] = query

        past_lens_t = kern_attn_inputs["past_lens"]
        if not isinstance(past_lens_t, torch.Tensor):
            raise TypeError("past_lens must be a torch.Tensor")
        return kern_attn_inputs

    @staticmethod
    def _get_max_context_len_for_sequences(
        kern_attn_inputs: KernelInputs,
        sequence_indices: list[int],
    ) -> int:
        past_lens_t = kern_attn_inputs["past_lens"]
        if not isinstance(past_lens_t, torch.Tensor):
            raise TypeError("past_lens must be a torch.Tensor")
        if len(sequence_indices) == 0:
            raise ValueError("sequence_indices must be non-empty")

        selected_sequence_ids = torch.tensor(sequence_indices, dtype=torch.int32)
        batch_size_in_sequences = int(past_lens_t.numel())
        if int(selected_sequence_ids.min().item()) < 0 or int(selected_sequence_ids.max().item()) >= batch_size_in_sequences:
            raise ValueError("sequence_indices contains out-of-range sequence ids")

        max_context_len = 0
        for sequence_index in selected_sequence_ids.tolist():
            q_start, q_end, _, _ = get_sequence_ranges(kern_attn_inputs, int(sequence_index))
            q_len = q_end - q_start
            past_len = int(past_lens_t[int(sequence_index)].item())
            context_len = past_len + q_len
            if context_len > max_context_len:
                max_context_len = context_len
        return max_context_len

    @staticmethod
    def _get_route_mode(
        query: torch.Tensor,
        past_lens_t: torch.Tensor,
        subsequence_begins_t: torch.Tensor,
        mixed_route_mode: str,
    ) -> tuple[str, list[int], list[int]]:
        if mixed_route_mode not in ("multi", "split"):
            raise ValueError(f"Unsupported mixed_route_mode={mixed_route_mode}. Expected 'multi' or 'split'.")
        
        decode_seq_indices, prefill_seq_indices = PagedAttentionRunner._sequence_groups_by_q_len(subsequence_begins_t)
        assert len(decode_seq_indices) + len(prefill_seq_indices) == int(past_lens_t.numel()), "All sequences must be classified into either decode or prefill groups"
        assert len(decode_seq_indices) > 0 or len(prefill_seq_indices) > 0, "At least one sequence should be present for attention computation"

        use_single_token_path = len(prefill_seq_indices) == 0
        if use_single_token_path:
            return "single", decode_seq_indices, prefill_seq_indices

        if mixed_route_mode == "split" and len(decode_seq_indices) > 0 and len(prefill_seq_indices) > 0:
            return "mixed_route_split", decode_seq_indices, prefill_seq_indices

        return "multi", decode_seq_indices, prefill_seq_indices

    @staticmethod
    def _sequence_groups_by_q_len(subsequence_begins_t: torch.Tensor) -> tuple[list[int], list[int]]:
        token_counts = torch.diff(subsequence_begins_t)
        decode_seq_indices: list[int] = []
        prefill_seq_indices: list[int] = []
        for sequence_index, token_count in enumerate(token_counts.tolist()):
            if int(token_count) == 1:
                decode_seq_indices.append(sequence_index)
            else:
                prefill_seq_indices.append(sequence_index)
        return decode_seq_indices, prefill_seq_indices

    def _run_mixed_route_split(
        self,
        kern_attn_inputs: KernelInputs,
        out: torch.Tensor,
        decode_seq_indices: list[int],
        prefill_seq_indices: list[int],
    ) -> torch.Tensor:
        assert len(decode_seq_indices) > 0 and len(prefill_seq_indices) > 0, (
            "mixed_route_split requires both decode and prefill sequences; route gating should enforce this"
        )

        self.single_token_attention(kern_attn_inputs, out=out, decode_seq_indices=decode_seq_indices)

        self.multi_token_attention(kern_attn_inputs, out=out, prefill_seq_indices=prefill_seq_indices)
        return out

    def kvcache_update(
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

    def multi_token_attention(
        self,
        kern_attn_inputs: KernelInputs,
        out: torch.Tensor,
        prefill_seq_indices: list[int],
    ) -> torch.Tensor:
        kern_attn_inputs["max_context_len"] = self._get_max_context_len_for_sequences(
            kern_attn_inputs,
            prefill_seq_indices,
        )
        attn_outputs = self.multi_token_runner(kern_attn_inputs, out=out, prefill_seq_indices=prefill_seq_indices)
        return attn_outputs

    def single_token_attention(
        self,
        kern_attn_inputs: KernelInputs,
        out: torch.Tensor,
        decode_seq_indices: list[int],
    ) -> torch.Tensor:
        kern_attn_inputs["max_context_len"] = self._get_max_context_len_for_sequences(
            kern_attn_inputs,
            decode_seq_indices,
        )
        attn_outputs = self.single_token_runner(kern_attn_inputs, out=out, decode_seq_indices=decode_seq_indices)
        return attn_outputs

    def run(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        past_lens: list[int],
        subsequence_begins: list[int],
        kvcache_table: KVCacheTable,
        query: torch.Tensor,
        mixed_route_mode: str = "multi",
    ) -> tuple[KernelInputs, torch.Tensor]:
        kern_attn_inputs = self._prepare_kern_attn_inputs(
            key,
            value,
            past_lens,
            subsequence_begins,
            kvcache_table,
            query,
        )

        past_lens_t = kern_attn_inputs["past_lens"]
        subsequence_begins_t = kern_attn_inputs["subsequence_begins"]
        assert isinstance(past_lens_t, torch.Tensor)
        assert isinstance(subsequence_begins_t, torch.Tensor)

        output = torch.zeros(
            int(query.shape[0]),
            self.multi_token_runner.num_heads * self.multi_token_runner.head_size,
            dtype=torch.float16,
        )

        route_mode, decode_seq_indices, prefill_seq_indices = self._get_route_mode(
            query,
            past_lens_t,
            subsequence_begins_t,
            mixed_route_mode,
        )

        if route_mode == "mixed_route_split":
            attn_outputs = self._run_mixed_route_split(
                kern_attn_inputs,
                out=output,
                decode_seq_indices=decode_seq_indices,
                prefill_seq_indices=prefill_seq_indices,
            )
            return kern_attn_inputs, attn_outputs

        if route_mode == "single":
            attn_outputs = self.single_token_attention(
                kern_attn_inputs,
                out=output,
                decode_seq_indices=decode_seq_indices,
            )
        else:
            if len(decode_seq_indices) > 0:
                prefill_seq_indices = decode_seq_indices + prefill_seq_indices
            attn_outputs = self.multi_token_attention(
                kern_attn_inputs,
                out=output,
                prefill_seq_indices=prefill_seq_indices,
            )
        return kern_attn_inputs, attn_outputs

    def reference_attention(self, kern_attn_inputs: KernelInputs) -> torch.Tensor:
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

            q = query[q_start:q_end].reshape(q_len, self.multi_token_runner.num_heads, self.multi_token_runner.head_size).contiguous()
            physical_blocks = block_indices[blk_start:blk_end]
            key_cache_blocks = key_cache[physical_blocks].contiguous()
            value_cache_blocks = value_cache[physical_blocks].contiguous()

            key_context = KVCacheUpdater.recover_context_from_cache(
                key_cache_blocks,
                context_len,
                self.multi_token_runner.num_kv_heads,
                self.multi_token_runner.head_size,
                self.multi_token_runner.block_sz,
                self.multi_token_runner.kv_cache_compression,
                cache_kind="key",
                sub_block_size=self.multi_token_runner.sub_block_size,
            )
            value_context = KVCacheUpdater.recover_context_from_cache(
                value_cache_blocks,
                context_len,
                self.multi_token_runner.num_kv_heads,
                self.multi_token_runner.head_size,
                self.multi_token_runner.block_sz,
                self.multi_token_runner.kv_cache_compression,
                cache_kind="value",
                sub_block_size=self.multi_token_runner.sub_block_size,
            )

            attention_mask = get_attention_mask(
                q_len,
                context_len,
                self.multi_token_runner.num_heads,
                q.dtype,
                q.device,
                past_len=past_len,
                is_causal=self.multi_token_runner.is_causal,
            )

            ref = F.scaled_dot_product_attention(
                q.transpose(0, 1).unsqueeze(0).to(torch.float16),
                key_context.transpose(0, 1).unsqueeze(0).to(torch.float16),
                value_context.transpose(0, 1).unsqueeze(0).to(torch.float16),
                attn_mask=attention_mask,
                dropout_p=0.0,
                enable_gqa=(self.multi_token_runner.num_kv_heads != self.multi_token_runner.num_heads),
            )
            refs.append(ref.squeeze(0).transpose(0, 1).reshape(q_len, -1).to(q.dtype))

        return torch.cat(refs, dim=0)


def run_paged_attention_smoke_case(
    case: PagedAttentionTestCase,
    check_acc=True,
    mixed_route_mode: str | None = None,
) -> KernelInputs:
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

    pa_runner = PagedAttentionRunner(
        case.num_heads,
        case.num_kv_heads,
        case.k_head_size,
        case.block_size,
        case.kv_cache_compression,
        case.sub_block_size,
        True,
    )

    kern_attn_inputs, attn_outputs = pa_runner.run(
        key,
        value,
        past_lens,
        subsequence_begins,
        kvcache_table,
        query,
        mixed_route_mode="multi" if mixed_route_mode is None else mixed_route_mode,
    )

    if check_acc:
        round_ref = pa_runner.reference_attention(kern_attn_inputs)
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


SMOKE_KV_CACHE_COMPRESSION_MODES = (
    KV_CACHE_COMPRESSION_NONE,
    KV_CACHE_COMPRESSION_BY_TOKEN,
    KV_CACHE_COMPRESSION_BY_CHANNEL,
)


def _with_kv_cache_compression_modes(
    cases: tuple[PagedAttentionTestCase, ...],
    modes: tuple[int, ...] = SMOKE_KV_CACHE_COMPRESSION_MODES,
) -> tuple[PagedAttentionTestCase, ...]:
    expanded_cases: list[PagedAttentionTestCase] = []
    for case in cases:
        for mode in modes:
            expanded_cases.append(
                PagedAttentionTestCase(
                    subsequences=case.subsequences,
                    num_heads=case.num_heads,
                    num_kv_heads=case.num_kv_heads,
                    k_head_size=case.k_head_size,
                    v_head_size=case.v_head_size,
                    block_size=case.block_size,
                    sub_block_size=case.sub_block_size,
                    kv_cache_compression=mode,
                )
            )
    return tuple(expanded_cases)


_PREFILL_ONLY_SMOKE_BASE_CASES = (
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
    # Align with smoke_cm_xattention/basic long prefill shapes (indices 15/18/27).
    # kv_cache_compression is expanded by _with_kv_cache_compression_modes(...).
    PagedAttentionTestCase(
        subsequences=(ss(2048),),
        num_heads=2,
        num_kv_heads=2,
        k_head_size=64,
        v_head_size=64,
        block_size=256,
    ),
    PagedAttentionTestCase(
        subsequences=(ss(2048),),
        num_heads=4,
        num_kv_heads=2,
        k_head_size=64,
        v_head_size=64,
        block_size=256,
    ),
)

PREFILL_ONLY_SMOKE_CASES = _with_kv_cache_compression_modes(_PREFILL_ONLY_SMOKE_BASE_CASES)


_GENERATE_ONLY_SMOKE_BASE_CASES = (
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
)

GENERATE_ONLY_SMOKE_CASES = _with_kv_cache_compression_modes(_GENERATE_ONLY_SMOKE_BASE_CASES)

_MIXED_ONLY_SMOKE_BASE_CASES = (
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
        block_size=256,
    ),
)

MIXED_ONLY_SMOKE_CASES = _with_kv_cache_compression_modes(_MIXED_ONLY_SMOKE_BASE_CASES)


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
        f"_sbls{case.sub_block_size}"
        f"_cmpr{int(case.kv_cache_compression)}"
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

    kern_attn_inputs = run_paged_attention_smoke_case(case, check_acc=True)
    max_context_len = int(kern_attn_inputs["max_context_len"])
    past_lens = kern_attn_inputs["past_lens"]
    assert isinstance(past_lens, torch.Tensor)

    assert max_context_len == max(subsequence.total_seq_len for subsequence in case.subsequences)
    assert past_lens.tolist() == [subsequence.past_len for subsequence in case.subsequences]
    assert_generate_stage_inputs(kern_attn_inputs)



@pytest.mark.parametrize("mixed_route_mode", ("multi", "split"), ids=("route_multi", "route_split"))
@pytest.mark.parametrize("case", MIXED_ONLY_SMOKE_CASES, ids=make_smoke_case_id)
def test_pa_smoke_paged_attention_mixed_only_route_matches_reference(
    case: PagedAttentionTestCase,
    mixed_route_mode: str,
):
    print(
        f"{Colors.GREEN}[testcase] mixed_only case={make_smoke_case_id(case)} route={mixed_route_mode}{Colors.END}"
    )

    kern_attn_inputs = run_paged_attention_smoke_case(
        case,
        check_acc=True,
        mixed_route_mode=mixed_route_mode,
    )
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


# Usage:
#   python -m py_compile test_pa_multiseq.py
#   python -m pytest --collect-only -q test_pa_multiseq.py | grep 'test_pa_smoke_paged_attention_mixed_only_route_matches_reference'
#   python -m pytest -q test_pa_multiseq.py -k 'test_pa_smoke_paged_attention_generate_only and 1x10'
#   python -m pytest -s test_pa_multiseq.py -k 'generate_only or mixed_only'
#   python -m pytest -s -q test_pa_multiseq.py -k 'test_pa_smoke_paged_attention_mixed_only_route_matches_reference and route_split'
#   python -m pytest -s -q test_pa_multiseq.py -k 'test_pa_smoke_paged_attention_mixed_only_route_matches_reference and route_multi'
#   timeout 120s python -m pytest -q test_pa_multiseq.py -vv
#   timeout 120s python -m pytest -q test_pa_multiseq.py -vv -k 'cmpr0 and (generate_only or mixed_only)'
#
# Notes:
#   - Mixed routing is selected explicitly by the parametrized `mixed_route_mode` test argument.
#   - Use `-k 'route_multi'` or `-k 'route_split'` to run just one mixed routing mode.

