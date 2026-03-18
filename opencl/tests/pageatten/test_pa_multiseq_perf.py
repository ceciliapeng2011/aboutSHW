import os
from dataclasses import dataclass

import pytest
import torch

from clops import cl
from clops.utils import Colors
from pa_test_common import (
    CacheQuantMode,
    PagedAttentionTestCase,
    create_paged_attention_inputs,
    create_subsequence_tensors,
    ss,
)
from test_pa_multiseq import PagedAttentionRunner


torch.manual_seed(0)

TRUNK_SIZE = 4096
MAX_PROMPT_LEN = 32768


@dataclass(frozen=True)
class PerfCase:
    name: str
    perf_type: str
    case: PagedAttentionTestCase
    warmup: int = 1
    iters: int = 5


def _validate_perf_case(case: PagedAttentionTestCase):
    assert case.key_cache_quant_mode == CacheQuantMode.BY_TOKEN
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

    runner = PagedAttentionRunner(
        case.num_heads,
        case.num_kv_heads,
        case.k_head_size,
        case.block_size,
        case.kv_cache_compression,
        True,
    )

    return runner, query, key, value, past_lens, subsequence_begins, kvcache_table


def _run_case_perf(perf_case: PerfCase):
    runner, query, key, value, past_lens, subsequence_begins, kvcache_table = _prepare_case(perf_case)

    _, out, gpu_latency_ns, elapsed_ms, dispatch_path, kernels_per_iter = runner.run_perf(
        key,
        value,
        past_lens,
        subsequence_begins,
        kvcache_table,
        query,
        n_warmup=perf_case.warmup,
        n_iters=perf_case.iters,
    )
    assert torch.isfinite(out).all().item()

    avg_host_ms = elapsed_ms / perf_case.iters
    q_tokens = int(query.shape[0])
    q_tok_per_s = (q_tokens * perf_case.iters * 1000.0) / elapsed_ms
    max_ctx = max(s.total_seq_len for s in perf_case.case.subsequences)

    if len(gpu_latency_ns) % kernels_per_iter != 0:
        raise AssertionError("Unexpected GPU profiling sample count")
    iter_gpu_ms = [
        sum(gpu_latency_ns[index:index + kernels_per_iter]) * 1e-6
        for index in range(0, len(gpu_latency_ns), kernels_per_iter)
    ]
    avg_gpu_ms = sum(iter_gpu_ms) / len(iter_gpu_ms)
    min_gpu_ms = min(iter_gpu_ms)
    max_gpu_ms = max(iter_gpu_ms)

    print(
        f"{Colors.GREEN}[perf] type={perf_case.perf_type} name={perf_case.name} "
        f"path={dispatch_path} iters={perf_case.iters} q_tokens={q_tokens} max_ctx={max_ctx} "
        f"host_avg_ms={avg_host_ms:.3f} gpu_avg_ms={avg_gpu_ms:.3f} "
        f"gpu_min_ms={min_gpu_ms:.3f} gpu_max_ms={max_gpu_ms:.3f} q_tok_s={q_tok_per_s:.1f}{Colors.END}"
    )


def _make_perf_case(
    perf_type: str,
    case_tag: str,
    subsequences: tuple,
    block_size: int,
    kv_cache_compression: bool,
) -> PerfCase:

    cmpr = 1 if kv_cache_compression else 0
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


QWEN3_8B_PERF_CASES = tuple(
    _make_perf_case(perf_type, case_tag, subsequences, block_size, kv_cache_compression)
    for perf_type in ("prefill_only", "generate_only", "mixed_only")
    for case_tag, subsequences in PERF_SUBSEQUENCES[perf_type]
    for block_size in (16, 256)
    for kv_cache_compression in (False, True)
)


def _perf_case_id(perf_case: PerfCase) -> str:
    return perf_case.name


@pytest.mark.perf
@pytest.mark.parametrize("perf_case", QWEN3_8B_PERF_CASES, ids=_perf_case_id)
def test_pa_multiseq_perf_qwen3_8b(perf_case: PerfCase):
    if os.getenv("PA_PERF", "0") != "1":
        pytest.skip("Set PA_PERF=1 to enable long-context perf tests")
    _run_case_perf(perf_case)


# Usage:
#   PA_PERF=1 timeout 1800s python -m pytest -s -q test_pa_multiseq_perf.py -m perf
