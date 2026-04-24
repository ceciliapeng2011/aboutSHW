"""
test_cb.py

Purpose
-------
This module tests continuous batching (CB) scheduling behavior on top of the
paged-attention runners. It validates that session-level prompt/decode traffic
is split into legal per-round plans and that each round is executable and
correct.

What this file verifies
-----------------------
1) Scheduler correctness
    - `ContinuousBatchingScheduler` builds non-empty rounds that obey
      `max_num_batched_tokens` and optional per-subsequence limits.
    - Total token accounting is exact: all prompt tokens plus decode tokens are
      scheduled exactly once.

2) Dynamic split-fuse policy
    - Decode-ready sequences can be fused with prompt chunks in the same round.
    - Round plans remain capacity-safe and valid for kernel execution.

3) Prompt-first policy
    - Prompt-only rounds are preferred before decode rounds when configured.

4) End-to-end execution checks
    - Accuracy path: CB rounds match reference attention numerically.
    - Perf path: host/GPU timing is collected and dispatch path is reported.

Compression mode note
---------------------
`kv_cache_compression` is treated as integer mode (`KV_CACHE_COMPRESSION`):
0 (none), 1 (by-token), 2 (by-channel).

Implementation notes
--------------------
- `SessionDescriptor` describes one user/session request shape.
- `RoundPlan` represents one scheduled iteration and can execute itself via
  `PagedAttentionRunner` or `PagedAttentionPerfRunner`.

In short, this file protects the scheduling and runtime behavior required for
continuous batching workloads (Qwen3-like long prompt + decode patterns).

How this differs from `test_pa_multiseq.py`
-------------------------------------------
- This file focuses on continuous-batching orchestration: round planning,
    scheduling constraints, prompt/decode policy behavior, and end-to-end
    execution across rounds.
- `test_pa_multiseq.py` focuses on per-round attention kernel correctness
    (multi-token, single-token, mixed-route) and reference parity.
"""

import torch
import pytest
import time
from dataclasses import dataclass

from pa_test_common import (
    KV_CACHE_COMPRESSION_NONE,
    KV_CACHE_COMPRESSION_BY_TOKEN,
    PagedAttentionTestCase,
    SubsequenceDescriptor,
    check_close,
    create_paged_attention_inputs,
    create_subsequence_tensors,
)
from test_pa_multiseq import PagedAttentionRunner
from test_pa_multiseq_perf import PagedAttentionPerfRunner


torch.manual_seed(0)

@dataclass(frozen=True)
class SessionDescriptor:
    num_input_tokens: int
    num_output_tokens: int

    def __post_init__(self):
        if self.num_input_tokens < 1:
            raise ValueError(f"num_input_tokens must be >= 1, got {self.num_input_tokens}")
        if self.num_output_tokens < 1:
            raise ValueError(f"num_output_tokens must be >= 1, got {self.num_output_tokens}")
        if self.num_output_tokens > 2:
            # Current harness models at most one decode iteration after prefill.
            raise ValueError("num_output_tokens must be <= 2 in this harness")


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

    def to_test_case(self, template: PagedAttentionTestCase) -> PagedAttentionTestCase:
        return PagedAttentionTestCase(
            subsequences=self.subsequences,
            num_heads=template.num_heads,
            num_kv_heads=template.num_kv_heads,
            k_head_size=template.k_head_size,
            v_head_size=template.v_head_size,
            block_size=template.block_size,
            sub_block_size=template.sub_block_size,
            kv_cache_compression=template.kv_cache_compression,
        )

    def create_runner_inputs(
        self,
        template: PagedAttentionTestCase,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int], list[int], object]:
        case = self.to_test_case(template)
        subsequence_tensors = [
            create_subsequence_tensors(
                descriptor,
                case.num_heads,
                case.num_kv_heads,
                case.k_head_size,
            )
            for descriptor in case.subsequences
        ]
        return create_paged_attention_inputs(case, subsequence_tensors)

    def run(
        self,
        runner,
        template: PagedAttentionTestCase,
        mixed_route_mode: str = "split",
        n_warmup: int = 1,
        n_iters: int = 3,
    ) -> tuple[dict[str, object], torch.Tensor]:
        query, key, value, past_lens, subsequence_begins, kvcache_table = self.create_runner_inputs(template)

        if isinstance(runner, PagedAttentionPerfRunner):
            (
                kern_attn_inputs,
                attn_outputs,
                gpu_latency_ns,
                elapsed_ms,
                dispatch_path,
                kernels_per_iter,
            ) = runner.run_perf(
                key,
                value,
                past_lens,
                subsequence_begins,
                kvcache_table,
                query,
                n_warmup=n_warmup,
                n_iters=n_iters,
                mixed_route_mode=mixed_route_mode,
            )

            iter_gpu_ms = [
                sum(gpu_latency_ns[i:i + kernels_per_iter]) * 1e-6
                for i in range(0, len(gpu_latency_ns), kernels_per_iter)
            ]
            perf_meta: dict[str, object] = {
                "kern_attn_inputs": kern_attn_inputs,
                "host_avg_ms": elapsed_ms / n_iters,
                "gpu_avg_ms": (sum(iter_gpu_ms) / len(iter_gpu_ms)) if iter_gpu_ms else 0.0,
                "dispatch_path": dispatch_path,
            }
            return perf_meta, attn_outputs

        t0 = time.perf_counter()
        kern_attn_inputs, attn_outputs = runner.run(
            key,
            value,
            past_lens,
            subsequence_begins,
            kvcache_table,
            query,
            mixed_route_mode=mixed_route_mode,
        )
        meta: dict[str, object] = {
            "kern_attn_inputs": kern_attn_inputs,
            "host_ms": (time.perf_counter() - t0) * 1000.0,
        }
        return meta, attn_outputs


class ContinuousBatchingScheduler:
    def __init__(self, max_num_batched_tokens: int, max_subsequence_tokens: int | None = None, dynamic_split_fuse: bool = True):
        if max_num_batched_tokens <= 0:
            raise ValueError("max_num_batched_tokens must be positive")
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_subsequence_tokens = max_subsequence_tokens or max_num_batched_tokens
        self.dynamic_split_fuse = dynamic_split_fuse

    def schedule(self, sessions: list[SessionDescriptor]) -> list[RoundPlan]:
        remaining_input_tokens = [session.num_input_tokens for session in sessions]
        # num_output_tokens includes prefill completion. Decode count is (num_output_tokens - 1).
        remaining_decode_tokens = [max(session.num_output_tokens - 1, 0) for session in sessions]
        emitted_tokens = [0 for _ in sessions]
        round_plans: list[RoundPlan] = []

        while any((remaining_input_tokens[i] > 0 or remaining_decode_tokens[i] > 0) for i in range(len(sessions))):
            capacity = self.max_num_batched_tokens
            round_sequence_ids: list[int] = []
            round_subsequences: list[SubsequenceDescriptor] = []

            if self.dynamic_split_fuse:
                # Decode first: one token per decode-ready sequence.
                for sequence_id in range(len(sessions)):
                    if capacity == 0:
                        break
                    if remaining_input_tokens[sequence_id] == 0 and remaining_decode_tokens[sequence_id] > 0:
                        round_sequence_ids.append(sequence_id)
                        round_subsequences.append(
                            SubsequenceDescriptor(num_tokens=1, past_len=emitted_tokens[sequence_id])
                        )
                        emitted_tokens[sequence_id] += 1
                        remaining_decode_tokens[sequence_id] -= 1
                        capacity -= 1

                # Then prompt chunks.
                for sequence_id in range(len(sessions)):
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
                # Prompt phase first (whole prompt if fits), then decode phase.
                has_prompt_left = any(remaining > 0 for remaining in remaining_input_tokens)
                if has_prompt_left:
                    for sequence_id in range(len(sessions)):
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
                    for sequence_id in range(len(sessions)):
                        if capacity == 0:
                            break
                        if remaining_decode_tokens[sequence_id] <= 0:
                            continue
                        round_sequence_ids.append(sequence_id)
                        round_subsequences.append(
                            SubsequenceDescriptor(num_tokens=1, past_len=emitted_tokens[sequence_id])
                        )
                        emitted_tokens[sequence_id] += 1
                        remaining_decode_tokens[sequence_id] -= 1
                        capacity -= 1

            if not round_subsequences:
                raise RuntimeError("Failed to build a non-empty continuous batching round")

            round_plans.append(RoundPlan(tuple(round_sequence_ids), tuple(round_subsequences)))

        return round_plans


def _default_case_template(
    block_size: int = 16,
    kv_cache_compression: int = KV_CACHE_COMPRESSION_NONE,
) -> PagedAttentionTestCase:
    # Qwen3-8B-like metadata; subsequence content is replaced per-round.
    return PagedAttentionTestCase(
        subsequences=(SubsequenceDescriptor(num_tokens=1, past_len=0),),
        num_heads=32,
        num_kv_heads=8,
        k_head_size=128,
        v_head_size=128,
        block_size=block_size,
        kv_cache_compression=kv_cache_compression,
    )


def _create_accuracy_runner(template: PagedAttentionTestCase) -> PagedAttentionRunner:
    return PagedAttentionRunner(
        template.num_heads,
        template.num_kv_heads,
        template.k_head_size,
        template.block_size,
        template.kv_cache_compression,
        True,
    )


def _create_perf_runner(template: PagedAttentionTestCase) -> PagedAttentionPerfRunner:
    return PagedAttentionPerfRunner(
        template.num_heads,
        template.num_kv_heads,
        template.k_head_size,
        template.block_size,
        template.kv_cache_compression,
        True,
    )


ACCURACY_SESSION_GROUPS = (
    [
        SessionDescriptor(num_input_tokens=32, num_output_tokens=2),
        SessionDescriptor(num_input_tokens=25, num_output_tokens=1),
        SessionDescriptor(num_input_tokens=10, num_output_tokens=2),
    ],
    [
        SessionDescriptor(num_input_tokens=8, num_output_tokens=2),
        SessionDescriptor(num_input_tokens=6, num_output_tokens=2),
        SessionDescriptor(num_input_tokens=4, num_output_tokens=1),
    ],
)

MODEL_CONFIGS = (
    (16, KV_CACHE_COMPRESSION_NONE),
    (16, KV_CACHE_COMPRESSION_BY_TOKEN),
    (256, KV_CACHE_COMPRESSION_NONE),
    (256, KV_CACHE_COMPRESSION_BY_TOKEN),
)


@pytest.mark.parametrize("sessions", ACCURACY_SESSION_GROUPS)
@pytest.mark.parametrize("block_size,kv_cache_compression", MODEL_CONFIGS)
def test_cb_accuracy_short_prompts_against_reference(
    sessions: list[SessionDescriptor],
    block_size: int,
    kv_cache_compression: int,
):
    scheduler = ContinuousBatchingScheduler(max_num_batched_tokens=64, dynamic_split_fuse=True)
    round_plans = scheduler.schedule(sessions)
    assert len(round_plans) > 0

    template = _default_case_template(
        block_size=block_size,
        kv_cache_compression=kv_cache_compression,
    )
    runner = _create_accuracy_runner(template)

    for round_plan in round_plans:
        run_meta, attn_outputs = round_plan.run(runner, template, mixed_route_mode="split")
        kern_attn_inputs = run_meta["kern_attn_inputs"]
        assert isinstance(kern_attn_inputs, dict)
        query = kern_attn_inputs["query"]
        assert isinstance(query, torch.Tensor)
        assert int(query.shape[0]) == round_plan.batch_size_in_tokens
        assert attn_outputs.shape[0] == round_plan.batch_size_in_tokens
        assert torch.isfinite(attn_outputs).all().item()
        ref = runner.reference_attention(kern_attn_inputs)
        check_close(ref, attn_outputs)


@pytest.mark.parametrize(
    "sessions",
    [
        [
            SessionDescriptor(num_input_tokens=32768, num_output_tokens=1),
        ],
        [
            SessionDescriptor(num_input_tokens=32768, num_output_tokens=1000),
        ],
        [
            SessionDescriptor(num_input_tokens=16384, num_output_tokens=100),
            SessionDescriptor(num_input_tokens=16384, num_output_tokens=2),
        ],
        [
            SessionDescriptor(num_input_tokens=4096, num_output_tokens=100),
            SessionDescriptor(num_input_tokens=4096, num_output_tokens=100),
            SessionDescriptor(num_input_tokens=4096, num_output_tokens=100),
            SessionDescriptor(num_input_tokens=4096, num_output_tokens=100),
            SessionDescriptor(num_input_tokens=4096, num_output_tokens=100),
            SessionDescriptor(num_input_tokens=4096, num_output_tokens=100),
            SessionDescriptor(num_input_tokens=4096, num_output_tokens=100),
            SessionDescriptor(num_input_tokens=4096, num_output_tokens=100),
        ],
    ],
)
@pytest.mark.parametrize("block_size,kv_cache_compression", MODEL_CONFIGS)
def test_cb_perf_qwen3_8b_long_prompts_dynamic_split_fuse(
    sessions: list[SessionDescriptor],
    block_size: int,
    kv_cache_compression: int,
):
    scheduler = ContinuousBatchingScheduler(max_num_batched_tokens=4096, dynamic_split_fuse=True)
    round_plans = scheduler.schedule(sessions)

    assert len(round_plans) > 0
    assert all(plan.batch_size_in_tokens <= 4096 for plan in round_plans)

    # Validate full token accounting: all prompt tokens + decode tokens are scheduled exactly once.
    expected_total_tokens = sum(session.num_input_tokens + (session.num_output_tokens - 1) for session in sessions)
    actual_total_tokens = sum(plan.batch_size_in_tokens for plan in round_plans)
    assert actual_total_tokens == expected_total_tokens

    template = _default_case_template(
        block_size=block_size,
        kv_cache_compression=kv_cache_compression,
    )
    runner = _create_perf_runner(template)

    for round_id, round_plan in enumerate(round_plans):
        run_meta, attn_outputs = round_plan.run(
            runner,
            template,
            mixed_route_mode="split",
            n_warmup=1,
            n_iters=2,
        )
        assert torch.isfinite(attn_outputs).all().item()
        host_avg_ms = float(run_meta["host_avg_ms"])
        gpu_avg_ms = float(run_meta["gpu_avg_ms"])
        dispatch_path = str(run_meta["dispatch_path"])
        print(
            f"[cb perf] round={round_id} tokens={round_plan.batch_size_in_tokens} "
            f"seqs={round_plan.batch_size_in_sequences} host_avg_ms={host_avg_ms:.3f} "
            f"gpu_avg_ms={gpu_avg_ms:.3f} path={dispatch_path}"
        )
        assert host_avg_ms > 0
        assert gpu_avg_ms > 0


@pytest.mark.parametrize("block_size,kv_cache_compression", MODEL_CONFIGS)
def test_cb_perf_qwen3_8b_prompt_first_mode_valid_case(
    block_size: int,
    kv_cache_compression: int,
):
    sessions = [
        SessionDescriptor(num_input_tokens=4096, num_output_tokens=2),
        SessionDescriptor(num_input_tokens=2048, num_output_tokens=2),
        SessionDescriptor(num_input_tokens=1024, num_output_tokens=1),
    ]
    scheduler = ContinuousBatchingScheduler(
        max_num_batched_tokens=4096,
        max_subsequence_tokens=4096,
        dynamic_split_fuse=False,
    )
    round_plans = scheduler.schedule(sessions)

    assert len(round_plans) > 0
    assert all(plan.batch_size_in_tokens <= 4096 for plan in round_plans)

    expected_total_tokens = sum(session.num_input_tokens + (session.num_output_tokens - 1) for session in sessions)
    actual_total_tokens = sum(plan.batch_size_in_tokens for plan in round_plans)
    assert actual_total_tokens == expected_total_tokens

    template = _default_case_template(
        block_size=block_size,
        kv_cache_compression=kv_cache_compression,
    )
    runner = _create_perf_runner(template)
    for round_id, round_plan in enumerate(round_plans):
        run_meta, attn_outputs = round_plan.run(
            runner,
            template,
            mixed_route_mode="split",
            n_warmup=1,
            n_iters=2,
        )
        assert torch.isfinite(attn_outputs).all().item()
        host_avg_ms = float(run_meta["host_avg_ms"])
        gpu_avg_ms = float(run_meta["gpu_avg_ms"])
        dispatch_path = str(run_meta["dispatch_path"])
        print(
            f"[cb perf] prompt_first round={round_id} tokens={round_plan.batch_size_in_tokens} "
            f"seqs={round_plan.batch_size_in_sequences} host_avg_ms={host_avg_ms:.3f} "
            f"gpu_avg_ms={gpu_avg_ms:.3f} path={dispatch_path}"
        )
        assert host_avg_ms > 0
        assert gpu_avg_ms > 0

