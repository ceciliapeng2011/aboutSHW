import functools
import os

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from clops import cl
from clops.utils import Colors
from pa_test_common import (
    CacheQuantMode,
    DIV_UP,
    KVCacheTable,
    KVCacheUpdater,
    KernelInputs,
    PagedAttentionTestCase,
    check_close,
    create_paged_attention_inputs,
    create_subsequence_tensors,
    flash_attn_vlen_ref,
    get_attention_mask,
    get_cm_grf_width,
    get_sequence_ranges,
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
        compressed_kvcache,
        is_causal,
        sparse_block_size=1,
        enable_hybrid_dispatch=True,
    ):
        return PaMultiTokenRunner(
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


class PaSingleTokenRunner:
    def __init__(
        self,
        num_heads: int,
        num_kv_heads: int,
        head_size: int,
        block_size: int,
        kv_cache_compression: bool,
    ):
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.block_size = block_size
        self.kv_cache_compression = kv_cache_compression
        self.kvcache_quantization_by_token = 1

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
        kv_cache_compression: bool,
    ):
        return PaSingleTokenRunner(
            num_heads,
            num_kv_heads,
            head_size,
            block_size,
            kv_cache_compression,
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
        kv_cache_compression_by_token: int,
        xe_arch: int,
        q_head_chunks_per_kv_head: int,
        q_head_chunk_size: int,
        scale_factor: float,
    ):
        src = r'''#include "pa_single_token.cm"'''
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
                            -DKV_CACHE_COMPRESSION_BY_TOKEN={kv_cache_compression_by_token}
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
            self.kvcache_quantization_by_token,
            self.xe_arch,
            int(self.q_head_chunks_per_kv_head),
            int(self.q_head_chunk_size),
            self.scale_factor,
        )

    def __call__(self, kern_attn_inputs: KernelInputs, n_repeats: int = 1) -> torch.Tensor:
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

        batch_size = int(past_lens_t.numel())
        if int(query.shape[0]) != batch_size:
            raise ValueError("pa_single_token path expects one query token per sequence")

        token_counts = torch.diff(subsequence_begins_t)
        if token_counts.numel() and not torch.all(token_counts == 1).item():
            raise ValueError("pa_single_token path expects num_tokens == 1 for every subsequence")

        max_context_len = int(kern_attn_inputs["max_context_len"])
        kv_partition_num = DIV_UP(max_context_len, self.kv_partition_size)
        q_tokens = query.reshape(batch_size, self.num_heads, self.head_size).contiguous()

        output = torch.zeros(batch_size, self.num_heads * self.head_size, dtype=torch.float16)

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
                t_subsequence_begins,
                kv_partition_num,
            )
            cl.finish()
            output = torch.from_numpy(t_out_final.numpy().reshape(batch_size, -1))

        assert torch.isfinite(output).all().item()
        return output

class PagedAttentionRunner:
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
        self.multi_token_runner = PaMultiTokenRunner.create_instance(
            num_heads,
            num_kv_heads,
            head_size,
            block_size,
            kv_cache_compression,
            is_causal,
        )
        self.single_token_runner = PaSingleTokenRunner.create_instance(
            num_heads,
            num_kv_heads,
            head_size,
            block_size,
            kv_cache_compression,
        )

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
    ) -> torch.Tensor:
        attn_outputs = self.multi_token_runner(kern_attn_inputs)
        return attn_outputs

    def single_token_attention(
        self,
        kern_attn_inputs: KernelInputs,
    ) -> torch.Tensor:
        attn_outputs = self.single_token_runner(kern_attn_inputs)
        return attn_outputs

    def run(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        past_lens: list[int],
        subsequence_begins: list[int],
        kvcache_table: KVCacheTable,
        query: torch.Tensor,
    ) -> tuple[KernelInputs, torch.Tensor]:
        kern_attn_inputs = self.kvcache_update(
            key,
            value,
            past_lens,
            subsequence_begins,
            kvcache_table,
        )
        kern_attn_inputs["query"] = query

        past_lens_t = kern_attn_inputs["past_lens"]
        subsequence_begins_t = kern_attn_inputs["subsequence_begins"]
        assert isinstance(past_lens_t, torch.Tensor)
        assert isinstance(subsequence_begins_t, torch.Tensor)

        batch_size_in_tokens = int(query.shape[0])
        batch_size_in_sequences = int(past_lens_t.numel())
        token_counts = torch.diff(subsequence_begins_t)
        use_single_token_path = (
            batch_size_in_tokens == batch_size_in_sequences
            and (token_counts.numel() == 0 or torch.all(token_counts == 1).item())
        )

        if use_single_token_path:
            attn_outputs = self.single_token_attention(kern_attn_inputs)
        else:
            attn_outputs = self.multi_token_attention(kern_attn_inputs)
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
                self.multi_token_runner.compressed_kvcache,
            )
            value_context = KVCacheUpdater.recover_context_from_cache(
                value_cache_blocks,
                context_len,
                self.multi_token_runner.num_kv_heads,
                self.multi_token_runner.head_size,
                self.multi_token_runner.block_sz,
                self.multi_token_runner.compressed_kvcache,
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
            ref = flash_attn_vlen_ref(
                q,
                key_context,
                value_context,
                [],
                self.multi_token_runner.is_causal,
                attention_mask,
            )
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

    pa_runner = PagedAttentionRunner(
        case.num_heads,
        case.num_kv_heads,
        case.k_head_size,
        case.block_size,
        case.kv_cache_compression,
        True,
    )

    kern_attn_inputs, attn_outputs = pa_runner.run(
        key,
        value,
        past_lens,
        subsequence_begins,
        kvcache_table,
        query,
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

