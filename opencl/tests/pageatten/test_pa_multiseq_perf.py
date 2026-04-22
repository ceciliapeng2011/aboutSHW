import atexit
import csv
import os
import functools
import time
from dataclasses import dataclass

import numpy as np
import pytest
import torch

from clops import cl
from clops.utils import Colors
from pa_test_common import (
    DIV_UP,
    DEFAULT_SUB_BLOCK_SIZE,
    KVCacheTable,
    KernelInputs,
    PagedAttentionTestCase,
    create_paged_attention_inputs,
    create_subsequence_tensors,
    get_sequence_ranges,
    normalize_kv_cache_compression,
    ss,
)
from test_pa_multiseq import (
    MIXED_ONLY_SMOKE_CASES,
    PagedAttentionRunner,
    PaMultiTokenRunner,
    PaSingleTokenRunner,
    make_smoke_case_id,
)


torch.manual_seed(0)

TRUNK_SIZE = 4096
MAX_PROMPT_LEN = 32768
MIXED_COMPARE_SUMMARIES: dict[str, list[dict[str, str | float]]] = {
    "mixed_smoke": [],
    "mixed_only": [],
}
MIXED_COMPARE_CSV_PATH = "mixed_compare_summary.csv"


def _record_mixed_compare_summary(
    perf_type: str,
    perf_case_name: str,
    multi_stats: dict[str, torch.Tensor | str | float],
    split_stats: dict[str, torch.Tensor | str | float],
    speedup_gpu: float,
    speedup_host: float,
):
    if perf_type not in MIXED_COMPARE_SUMMARIES:
        raise ValueError(f"Unsupported mixed compare summary perf_type={perf_type}")

    if perf_type == "mixed_smoke":
        summary_name = perf_case_name.removeprefix("mixed_smoke_")
    else:
        summary_name = perf_case_name.removeprefix("mixed_only_")

    MIXED_COMPARE_SUMMARIES[perf_type].append(
        {
            "perf_type": perf_type,
            "name": summary_name,
            "multi_gpu_ms": float(multi_stats["avg_gpu_ms"]),
            "split_gpu_ms": float(split_stats["avg_gpu_ms"]),
            "multi_host_ms": float(multi_stats["avg_host_ms"]),
            "split_host_ms": float(split_stats["avg_host_ms"]),
            "speedup_gpu": speedup_gpu,
            "speedup_host": speedup_host,
        }
    )


@atexit.register
def _print_mixed_compare_summaries():
    fieldnames = [
        "perf_type",
        "name",
        "multi_gpu_ms",
        "split_gpu_ms",
        "speedup_gpu",
        "multi_host_ms",
        "split_host_ms",
        "speedup_host",
    ]
    combined_rows = [
        row
        for perf_type in ("mixed_smoke", "mixed_only")
        for row in MIXED_COMPARE_SUMMARIES[perf_type]
    ]
    if combined_rows:
        with open(MIXED_COMPARE_CSV_PATH, "w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(combined_rows)

    for perf_type, summary_rows in MIXED_COMPARE_SUMMARIES.items():
        if not summary_rows:
            continue

        print(f"{Colors.GREEN}[{perf_type} summary]{Colors.END}")
        print("cfg | multi_gpu | split_gpu | gpu_ratio | multi_host | split_host | host_ratio")
        for entry in sorted(summary_rows, key=lambda item: str(item["name"])):
            print(
                f"{entry['name']} | "
                f"{entry['multi_gpu_ms']:.3f} | {entry['split_gpu_ms']:.3f} | {entry['speedup_gpu']:.3f}x | "
                f"{entry['multi_host_ms']:.3f} | {entry['split_host_ms']:.3f} | {entry['speedup_host']:.3f}x"
            )
        print(f"csv | {MIXED_COMPARE_CSV_PATH}")

        print(f"{Colors.GREEN}[{perf_type} summary by gpu_ratio]{Colors.END}")
        print("cfg | gpu_ratio | multi_gpu | split_gpu | host_ratio")
        for entry in sorted(summary_rows, key=lambda item: float(item["speedup_gpu"]), reverse=True):
            print(
                f"{entry['name']} | {entry['speedup_gpu']:.3f}x | "
                f"{entry['multi_gpu_ms']:.3f} | {entry['split_gpu_ms']:.3f} | {entry['speedup_host']:.3f}x"
            )


class PaMultiTokenPerfRunner(PaMultiTokenRunner):
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
        return PaMultiTokenPerfRunner(
            num_heads,
            num_kv_heads,
            head_size,
            block_sz,
            kv_cache_compression,
            is_causal,
            sub_block_size=sub_block_size,
            sparse_block_size=sparse_block_size,
            enable_hybrid_dispatch=enable_hybrid_dispatch,
        )

    def run_perf(
        self,
        kern_attn_inputs: KernelInputs,
        prefill_seq_indices: list[int],
        out: torch.Tensor | None = None,
        n_warmup: int = 5,
        n_iters: int = 20,
    ) -> tuple[torch.Tensor, list[int], float, str]:
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
        kv_dtype = torch.uint8 if self.kv_cache_compression != 0 else torch.float16
        use_subset_execution = prefill_seq_count != int(past_lens.numel())

        q_tensor = query.reshape(batch_size_in_tokens, self.num_heads, self.head_size).contiguous()
        kernel_key_cache = self._format_cache_for_kernel(key_cache.contiguous())
        kernel_value_cache = self._format_cache_for_kernel(value_cache.contiguous())
        out_shape = (batch_size_in_tokens, self.num_heads, self.head_size)

        t_q = cl.tensor(q_tensor.to(torch.float16).detach().numpy())
        if out is not None:
            expected_shape = (batch_size_in_tokens, self.num_heads * self.head_size)
            if tuple(out.shape) != expected_shape:
                raise ValueError(f"out shape mismatch: got {tuple(out.shape)}, expected {expected_shape}")
            if out.dtype != torch.float16:
                raise ValueError(f"out dtype mismatch: got {out.dtype}, expected torch.float16")
            out_3d = out.reshape(batch_size_in_tokens, self.num_heads, self.head_size).contiguous()
            t_out = cl.tensor(out_3d.detach().numpy())
        else:
            if use_subset_execution:
                t_out = cl.tensor(np.zeros(out_shape, dtype=np.float16))
            else:
                t_out = cl.tensor(list(out_shape), np.dtype(np.float16))
        t_key_cache = cl.tensor(kernel_key_cache.to(kv_dtype).detach().numpy())
        t_value_cache = cl.tensor(kernel_value_cache.to(kv_dtype).detach().numpy())
        t_past_lens = cl.tensor(past_lens.detach().numpy())
        t_block_indices = cl.tensor(block_indices.detach().numpy())
        t_block_indices_begins = cl.tensor(block_indices_begins.detach().numpy())
        t_subsequence_begins = cl.tensor(subsequence_begins.detach().numpy())
        t_selected_sequence_ids = cl.tensor(selected_sequence_ids.detach().numpy())

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

        cl.finish()

        for _ in range(n_warmup):
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
        cl.finish()

        t0 = time.perf_counter()
        for _ in range(n_iters):
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
        gpu_latency_ns = cl.finish()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        output = torch.from_numpy(t_out.numpy().reshape(batch_size_in_tokens, -1))
        if out is not None:
            out.copy_(output)
            output = out
        assert torch.isfinite(output).all().item()
        return output, gpu_latency_ns, elapsed_ms, dispatch_mode


class PaSingleTokenPerfRunner(PaSingleTokenRunner):
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
        return PaSingleTokenPerfRunner(
            num_heads,
            num_kv_heads,
            head_size,
            block_size,
            kv_cache_compression,
            sub_block_size,
        )

    def run_perf(
        self,
        kern_attn_inputs: KernelInputs,
        decode_seq_indices: list[int],
        out: torch.Tensor | None = None,
        n_warmup: int = 5,
        n_iters: int = 20,
    ) -> tuple[torch.Tensor, list[int], float]:
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

        kernels = self._create_kernels()
        gws = [batch_size, self.num_kv_heads * self.q_head_chunks_per_kv_head, kv_partition_num]
        lws = [1, 1, 1]
        gws_2 = [batch_size, self.num_heads, self.head_size // self.reduce_split_step]
        lws_2 = [1, 1, 1]

        t_q = cl.tensor(q_tokens.detach().numpy())
        t_k = cl.tensor(key_cache_t.detach().numpy())
        t_v = cl.tensor(value_cache_t.detach().numpy())
        t_past_lens = cl.tensor(past_lens_t.detach().numpy())
        t_block_indices = cl.tensor(block_indices_t.detach().numpy())
        t_block_indices_begins = cl.tensor(block_indices_begins_t.detach().numpy())
        t_subsequence_begins = cl.tensor(subsequence_begins_t.detach().numpy())
        t_selected_sequence_ids = cl.tensor(selected_sequence_ids.detach().numpy())
        t_out = cl.tensor([batch_size, self.num_heads, kv_partition_num, self.head_size], np.dtype(np.float32))
        if out is not None:
            expected_shape = (output_tokens, self.num_heads * self.head_size)
            if tuple(out.shape) != expected_shape:
                raise ValueError(f"out shape mismatch: got {tuple(out.shape)}, expected {expected_shape}")
            if out.dtype != torch.float16:
                raise ValueError(f"out dtype mismatch: got {out.dtype}, expected torch.float16")
            t_out_final = cl.tensor([batch_size, 1, self.num_heads, self.head_size], np.dtype(np.float16))
        else:
            t_out_final = cl.tensor([batch_size, 1, self.num_heads, self.head_size], np.dtype(np.float16))
        t_lse = cl.tensor([batch_size, self.num_heads, kv_partition_num], np.dtype(np.float32))

        cl.finish()

        for _ in range(n_warmup):
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

        t0 = time.perf_counter()
        for _ in range(n_iters):
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
        gpu_latency_ns = cl.finish()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        output_compact = torch.from_numpy(t_out_final.numpy().reshape(batch_size, -1))
        if out is not None:
            if use_subset_execution:
                out[selected_token_indices] = output_compact
            else:
                out.copy_(output_compact)
            output = out
        else:
            if use_subset_execution:
                output = torch.zeros(output_tokens, self.num_heads * self.head_size, dtype=torch.float16)
                output[selected_token_indices] = output_compact
            else:
                output = output_compact
        assert torch.isfinite(output).all().item()
        return output, gpu_latency_ns, elapsed_ms


class PagedAttentionPerfRunner(PagedAttentionRunner):
    @staticmethod
    def _create_multi_token_runner(
        num_heads: int,
        num_kv_heads: int,
        head_size: int,
        block_size: int,
        kv_cache_compression: int,
        sub_block_size: int,
        is_causal: bool,
    ) -> PaMultiTokenPerfRunner:
        return PaMultiTokenPerfRunner.create_instance(
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
    ) -> PaSingleTokenPerfRunner:
        return PaSingleTokenPerfRunner.create_instance(
            num_heads,
            num_kv_heads,
            head_size,
            block_size,
            kv_cache_compression,
            sub_block_size,
        )

    @staticmethod
    def _group_gpu_ns_per_iter(gpu_latency_ns: list[int], kernels_per_iter: int, n_iters: int) -> list[int]:
        if kernels_per_iter <= 0:
            raise ValueError("kernels_per_iter must be positive")
        if len(gpu_latency_ns) != n_iters * kernels_per_iter:
            raise ValueError(
                f"Unexpected gpu latency sample count: got {len(gpu_latency_ns)}, expected {n_iters * kernels_per_iter}"
            )
        grouped = []
        for iter_index in range(n_iters):
            start = iter_index * kernels_per_iter
            end = start + kernels_per_iter
            grouped.append(int(sum(gpu_latency_ns[start:end])))
        return grouped

    def _run_perf_mixed_route_split(
        self,
        kern_attn_inputs: KernelInputs,
        decode_seq_indices: list[int],
        prefill_seq_indices: list[int],
        shared_out: torch.Tensor,
        n_warmup: int,
        n_iters: int,
    ) -> tuple[torch.Tensor, list[int], float, str, int]:
        query = kern_attn_inputs["query"]
        assert isinstance(query, torch.Tensor)
        assert len(decode_seq_indices) > 0 and len(prefill_seq_indices) > 0, (
            "mixed_route_split requires both decode and prefill sequences; route gating should enforce this"
        )

        elapsed_ms_total = 0.0

        _, decode_gpu_ns, decode_elapsed_ms = self.single_token_runner.run_perf(
            kern_attn_inputs,
            decode_seq_indices=decode_seq_indices,
            out=shared_out,
            n_warmup=n_warmup,
            n_iters=n_iters,
        )
        elapsed_ms_total += decode_elapsed_ms

        _, prefill_gpu_ns, prefill_elapsed_ms, _ = self.multi_token_runner.run_perf(
            kern_attn_inputs,
            prefill_seq_indices=prefill_seq_indices,
            out=shared_out,
            n_warmup=n_warmup,
            n_iters=n_iters,
        )
        elapsed_ms_total += prefill_elapsed_ms

        decode_iter_ns = self._group_gpu_ns_per_iter(decode_gpu_ns, kernels_per_iter=2, n_iters=n_iters)
        prefill_iter_ns = self._group_gpu_ns_per_iter(prefill_gpu_ns, kernels_per_iter=1, n_iters=n_iters)
        total_iter_ns = [decode_iter_ns[i] + prefill_iter_ns[i] for i in range(n_iters)]
        output = shared_out
        return output, total_iter_ns, elapsed_ms_total, "mixed_route_split/decode+prefill_indices/single+multi", 1

    def run_perf(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        past_lens: list[int],
        subsequence_begins: list[int],
        kvcache_table: KVCacheTable,
        query: torch.Tensor,
        n_warmup: int = 5,
        n_iters: int = 20,
        mixed_route_mode: str = "multi",
    ) -> tuple[KernelInputs, torch.Tensor, list[int], float, str, int]:
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

        output_tokens = int(query.shape[0])
        shared_out = torch.zeros(
            output_tokens,
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
            attn_outputs, gpu_latency_ns, elapsed_ms, dispatch_path, kernels_per_iter = self._run_perf_mixed_route_split(
                kern_attn_inputs,
                decode_seq_indices=decode_seq_indices,
                prefill_seq_indices=prefill_seq_indices,
                shared_out=shared_out,
                n_warmup=n_warmup,
                n_iters=n_iters,
            )
            return kern_attn_inputs, attn_outputs, gpu_latency_ns, elapsed_ms, dispatch_path, kernels_per_iter

        if route_mode == "single":
            kern_attn_inputs["max_context_len"] = self._get_max_context_len_for_sequences(
                kern_attn_inputs,
                decode_seq_indices,
            )
            attn_outputs, gpu_latency_ns, elapsed_ms = self.single_token_runner.run_perf(
                kern_attn_inputs,
                decode_seq_indices=decode_seq_indices,
                out=shared_out,
                n_warmup=n_warmup,
                n_iters=n_iters,
            )
            return kern_attn_inputs, attn_outputs, gpu_latency_ns, elapsed_ms, "single_token", 2

        if len(decode_seq_indices) > 0:
            prefill_seq_indices = decode_seq_indices + prefill_seq_indices
        kern_attn_inputs["max_context_len"] = self._get_max_context_len_for_sequences(
            kern_attn_inputs,
            prefill_seq_indices,
        )
        attn_outputs, gpu_latency_ns, elapsed_ms, dispatch_mode = self.multi_token_runner.run_perf(
            kern_attn_inputs,
            prefill_seq_indices=prefill_seq_indices,
            out=shared_out,
            n_warmup=n_warmup,
            n_iters=n_iters,
        )
        return kern_attn_inputs, attn_outputs, gpu_latency_ns, elapsed_ms, f"multi_token/{dispatch_mode}", 1


@dataclass(frozen=True)
class PerfCase:
    name: str
    perf_type: str
    case: PagedAttentionTestCase
    warmup: int = 1
    iters: int = 5


def _validate_perf_case(case: PagedAttentionTestCase):
    assert int(case.kv_cache_compression) in (0, 1, 2)
    assert case.k_head_size == case.v_head_size


def _validate_qwen3_8b_perf_case(case: PagedAttentionTestCase):
    assert int(case.kv_cache_compression) in (0, 1, 2)
    assert case.k_head_size == 128 and case.v_head_size == 128
    assert case.num_heads == 32 and case.num_kv_heads == 8
    assert all(s.num_tokens <= TRUNK_SIZE for s in case.subsequences)
    assert max(s.total_seq_len for s in case.subsequences) <= MAX_PROMPT_LEN


def _prepare_case(perf_case: PerfCase):
    case = perf_case.case
    _validate_perf_case(case)

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

    runner = PagedAttentionPerfRunner(
        case.num_heads,
        case.num_kv_heads,
        case.k_head_size,
        case.block_size,
        case.kv_cache_compression,
        case.sub_block_size,
        True,
    )

    return runner, query, key, value, past_lens, subsequence_begins, kvcache_table


def _prepare_qwen3_8b_case(perf_case: PerfCase):
    _validate_qwen3_8b_perf_case(perf_case.case)
    return _prepare_case(perf_case)


def _run_one_mode(
    runner: PagedAttentionPerfRunner,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    past_lens,
    subsequence_begins,
    kvcache_table,
    warmup: int,
    iters: int,
    mixed_route_mode: str,
):
    _, out, gpu_latency_ns, elapsed_ms, dispatch_path, kernels_per_iter = runner.run_perf(
        key,
        value,
        past_lens,
        subsequence_begins,
        kvcache_table,
        query,
        n_warmup=warmup,
        n_iters=iters,
        mixed_route_mode=mixed_route_mode,
    )
    assert torch.isfinite(out).all().item()

    avg_host_ms = elapsed_ms / iters
    if len(gpu_latency_ns) % kernels_per_iter != 0:
        raise AssertionError("Unexpected GPU profiling sample count")
    iter_gpu_ms = [
        sum(gpu_latency_ns[index:index + kernels_per_iter]) * 1e-6
        for index in range(0, len(gpu_latency_ns), kernels_per_iter)
    ]
    avg_gpu_ms = sum(iter_gpu_ms) / len(iter_gpu_ms)
    min_gpu_ms = min(iter_gpu_ms)
    max_gpu_ms = max(iter_gpu_ms)

    return {
        "out": out,
        "dispatch_path": dispatch_path,
        "avg_host_ms": avg_host_ms,
        "avg_gpu_ms": avg_gpu_ms,
        "min_gpu_ms": min_gpu_ms,
        "max_gpu_ms": max_gpu_ms,
    }


def _make_perf_case(
    perf_type: str,
    case_tag: str,
    subsequences: tuple,
    block_size: int,
    kv_cache_compression: int,
) -> PerfCase:
    cmpr = int(normalize_kv_cache_compression(kv_cache_compression))
    return PerfCase(
        name=f"{perf_type}_{case_tag}_bs{block_size}_cmpr{cmpr}",
        perf_type=perf_type,
        case=PagedAttentionTestCase(
            subsequences=subsequences,
            num_heads=32,
            num_kv_heads=8,
            k_head_size=128,
            v_head_size=128,
            block_size=block_size,
            kv_cache_compression=kv_cache_compression,
        ),
    )


PERF_SUBSEQUENCES = {
    "prefill_only": (
        ("single_ctx4k", (ss(4096, 0),)),
        ("single_ctx32k", (ss(4096, 28672),)),
        ("two_seq_ctx32k", (ss(2048, 30720), ss(2048, 30720))),
    ),
    "generate_only": (
        ("single_ctx32k", (ss(1, 32767),)),
        ("two_seq_ctx16k", (ss(1, 16383), ss(1, 16383))),
        ("three_seq_mixed_ctx", (ss(1, 8191), ss(1, 16383), ss(1, 32767))),
    ),
    "mixed_only": (
        ("trunk4k_plus_gen32k", (ss(4096, 24576), ss(1, 32767))),
        ("half_trunk_plus_gen", (ss(2048, 30720), ss(1, 32767))),
        ("two_prefill_plus_gen", (ss(1024, 31744), ss(1024, 31744), ss(1, 32767))),
    ),
}


MIXED_SMOKE_PERF_CASES = tuple(
    PerfCase(
        name=f"mixed_smoke_{make_smoke_case_id(case)}",
        perf_type="mixed_smoke",
        case=case,
        warmup=5,
        iters=20,
    )
    for case in MIXED_ONLY_SMOKE_CASES
)


QWEN3_8B_PERF_CASES = tuple(
    _make_perf_case(perf_type, case_tag, subsequences, block_size, kv_cache_compression)
    for perf_type in ("prefill_only", "generate_only", "mixed_only")
    for case_tag, subsequences in PERF_SUBSEQUENCES[perf_type]
    for block_size in (16, 256)
    for kv_cache_compression in (0, 1, 2)
)


def _perf_case_id(perf_case: PerfCase) -> str:
    return perf_case.name


@pytest.mark.perf
@pytest.mark.parametrize("perf_case", QWEN3_8B_PERF_CASES, ids=_perf_case_id)
def test_pa_multiseq_perf_qwen3_8b(perf_case: PerfCase):
    if os.getenv("PA_PERF", "0") != "1":
        pytest.skip("Set PA_PERF=1 to enable long-context perf tests")
    runner, query, key, value, past_lens, subsequence_begins, kvcache_table = _prepare_qwen3_8b_case(perf_case)
    _run_case_perf_with_prepared_inputs(
        perf_case,
        runner,
        query,
        key,
        value,
        past_lens,
        subsequence_begins,
        kvcache_table,
    )


def _run_case_perf_with_prepared_inputs(
    perf_case: PerfCase,
    runner: PagedAttentionPerfRunner,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    past_lens,
    subsequence_begins,
    kvcache_table,
):
    q_tokens = int(query.shape[0])
    max_ctx = max(s.total_seq_len for s in perf_case.case.subsequences)
    mixed_route_mode = "multi"

    stats = _run_one_mode(
        runner,
        query,
        key,
        value,
        past_lens,
        subsequence_begins,
        kvcache_table,
        perf_case.warmup,
        perf_case.iters,
        mixed_route_mode=mixed_route_mode,
    )

    out = stats["out"]
    dispatch_path = stats["dispatch_path"]
    avg_host_ms = stats["avg_host_ms"]
    avg_gpu_ms = stats["avg_gpu_ms"]
    min_gpu_ms = stats["min_gpu_ms"]
    max_gpu_ms = stats["max_gpu_ms"]

    q_tok_per_s = q_tokens / (avg_host_ms * 1e-3)

    print(
        f"{Colors.GREEN}[perf] type={perf_case.perf_type} name={perf_case.name} "
        f"route={mixed_route_mode} path={dispatch_path} iters={perf_case.iters} q_tokens={q_tokens} max_ctx={max_ctx} "
        f"host_avg_ms={avg_host_ms:.3f} gpu_avg_ms={avg_gpu_ms:.3f} "
        f"gpu_min_ms={min_gpu_ms:.3f} gpu_max_ms={max_gpu_ms:.3f} q_tok_s={q_tok_per_s:.1f}{Colors.END}"
    )

    if os.getenv("PA_MIXED_COMPARE", "0") == "1" and perf_case.perf_type in ("mixed_only", "mixed_smoke"):
        multi_stats = _run_one_mode(
            runner,
            query,
            key,
            value,
            past_lens,
            subsequence_begins,
            kvcache_table,
            perf_case.warmup,
            perf_case.iters,
            mixed_route_mode="multi",
        )
        split_stats = _run_one_mode(
            runner,
            query,
            key,
            value,
            past_lens,
            subsequence_begins,
            kvcache_table,
            perf_case.warmup,
            perf_case.iters,
            mixed_route_mode="split",
        )

        torch.testing.assert_close(multi_stats["out"], split_stats["out"], atol=5e-2, rtol=2e-1)

        speedup_gpu = multi_stats["avg_gpu_ms"] / split_stats["avg_gpu_ms"] if split_stats["avg_gpu_ms"] > 0 else float("inf")
        speedup_host = multi_stats["avg_host_ms"] / split_stats["avg_host_ms"] if split_stats["avg_host_ms"] > 0 else float("inf")
        print(
            f"{Colors.GREEN}[mixed route compare] name={perf_case.name} "
            f"route_multi(path={multi_stats['dispatch_path']}) gpu_ms={multi_stats['avg_gpu_ms']:.3f} host_ms={multi_stats['avg_host_ms']:.3f} | "
            f"route_split(path={split_stats['dispatch_path']}) gpu_ms={split_stats['avg_gpu_ms']:.3f} host_ms={split_stats['avg_host_ms']:.3f} | "
            f"speedup_gpu={speedup_gpu:.3f}x speedup_host={speedup_host:.3f}x{Colors.END}"
        )
        if perf_case.perf_type in ("mixed_smoke", "mixed_only"):
            _record_mixed_compare_summary(
                perf_case.perf_type,
                perf_case.name,
                multi_stats,
                split_stats,
                speedup_gpu,
                speedup_host,
            )


def _run_case_perf(perf_case: PerfCase):
    runner, query, key, value, past_lens, subsequence_begins, kvcache_table = _prepare_case(perf_case)
    _run_case_perf_with_prepared_inputs(
        perf_case,
        runner,
        query,
        key,
        value,
        past_lens,
        subsequence_begins,
        kvcache_table,
    )


@pytest.mark.perf
@pytest.mark.parametrize("perf_case", MIXED_SMOKE_PERF_CASES, ids=lambda perf_case: perf_case.name)
def test_pa_multiseq_perf_mixed_smoke_tradeoff(perf_case: PerfCase):
    if os.getenv("PA_PERF", "0") != "1":
        pytest.skip("Set PA_PERF=1 to enable perf tests")
    _run_case_perf(perf_case)


# Usage:
#   PA_PERF=1 timeout 120s python -m pytest -s -q test_pa_multiseq_perf.py -m perf
#   PA_PERF=1 PA_MIXED_COMPARE=1 timeout 120s python -m pytest -s -q test_pa_multiseq_perf.py -k 'mixed_only'
#   PA_PERF=1 PA_MIXED_COMPARE=1 timeout 120s python -m pytest -s -q test_pa_multiseq_perf.py -k 'mixed_smoke'
#
# Notes:
#   - Mixed routing mode selection is internal and explicit in the perf helpers; there is no routing env override.
#   - `mixed_smoke` reuses the short mixed cases from the smoke suite to study split-vs-multi tradeoffs on short prompts/contexts.
#   - `PA_MIXED_COMPARE=1` enables the extra multi-vs-split perf comparison printout for mixed perf cases.
