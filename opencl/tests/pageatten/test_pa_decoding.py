import functools
import os
from dataclasses import dataclass

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from clops import cl
from kv_cache_quant_utils import (
    DEFAULT_SUB_BLOCK_SIZE,
    dequant_per_channel as _dequant_per_channel,
    dequant_per_token as _dequant_per_token,
    quant_per_channel as _quant_per_channel,
    quant_per_token as _quant_per_token,
)


cl.profiling(True)
torch.manual_seed(0)
torch.set_printoptions(linewidth=1024)

def get_cm_grf_width() -> int:
    cm_kernels = cl.kernels(
        r'''
        extern "C" _GENX_MAIN_ void cm_get_grf_width(int * info [[type("svmptr_t")]]) {
            info[0] = CM_GRF_WIDTH;
        }''',
        "-cmc",
    )
    t_info = cl.tensor([2], np.dtype(np.int32))
    cm_kernels.enqueue("cm_get_grf_width", [1], [1], t_info)
    return int(t_info.numpy()[0])


def _check_close(actual: torch.Tensor, expected: torch.Tensor, atol: float = 1e-2, rtol: float = 1e-3) -> None:
    if torch.allclose(actual, expected, atol=atol, rtol=rtol):
        return
    close_mask = torch.isclose(actual, expected, atol=atol, rtol=rtol)
    bad = torch.where(~close_mask)
    raise AssertionError(
        "Tensor mismatch\n"
        f"indices={bad}\n"
        f"actual={actual[bad]}\n"
        f"expected={expected[bad]}"
    )


@dataclass(frozen=True)
class DecodingCase:
    num_heads: int = 32
    num_kv_heads: int = 8
    head_size: int = 128
    block_size: int = 256
    sub_block_size: int = DEFAULT_SUB_BLOCK_SIZE
    kv_len: int = 4097
    kv_cache_compression: int = 0


class PaSingleTokenRunner:
    def __init__(
        self,
        num_heads: int,
        num_kv_heads: int,
        head_size: int,
        block_size: int,
        sub_block_size: int,
        kv_cache_compression: int,
    ):
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.block_size = block_size
        self.sub_block_size = sub_block_size
        self.kv_cache_compression = kv_cache_compression

        if self.block_size % self.sub_block_size != 0:
            raise ValueError(
                f"block_size ({self.block_size}) must be divisible by sub_block_size ({self.sub_block_size})"
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
        self.scale_factor = 1.0 / (self.head_size**0.5)

    @staticmethod
    @functools.cache
    def create_instance(
        num_heads: int,
        num_kv_heads: int,
        head_size: int,
        block_size: int,
        sub_block_size: int,
        kv_cache_compression: int,
    ):
        return PaSingleTokenRunner(
            num_heads,
            num_kv_heads,
            head_size,
            block_size,
            sub_block_size,
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
        sub_block_size: int,
        kv_partition_size: int,
        reduce_split_step: int,
        clean_unused_kvcache: int,
        kv_cache_compression: int,
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
        return cl.kernels(
            src,
            f'''-cmc -Qxcm_jit_option=""
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
                        -DSCALE_FACTOR={scale_factor}''',
        )

    def _create_kernels(self):
        return self._create_kernels_cached(
            self.num_heads,
            self.num_kv_heads,
            self.head_size,
            self.kv_step,
            self.block_size,
            self.sub_block_size,
            self.kv_partition_size,
            self.reduce_split_step,
            1,
            self.kv_cache_compression,
            self.xe_arch,
            int(self.q_head_chunks_per_kv_head),
            int(self.q_head_chunk_size),
            self.scale_factor,
        )

    @staticmethod
    def _validate_single_subsequence(subsequence_begins: torch.Tensor) -> None:
        if int(subsequence_begins.numel()) != 2:
            raise ValueError("single subsequence only: subsequence_begins must have len == 2")
        if int((subsequence_begins[1] - subsequence_begins[0]).item()) != 1:
            raise ValueError("single subsequence only: num_tokens must be 1")

    def __call__(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        past_lens: torch.Tensor,
        block_indices: torch.Tensor,
        block_indices_begins: torch.Tensor,
        subsequence_begins: torch.Tensor,
        out: torch.Tensor,
        n_repeats: int = 1,
    ) -> torch.Tensor:
        self._validate_single_subsequence(subsequence_begins)

        if out.dtype != torch.float16:
            raise ValueError(f"out dtype mismatch: got {out.dtype}, expected torch.float16")

        kernels = self._create_kernels()

        batch_size = int(query.shape[0])
        max_context_len = int(past_lens.max().item()) + 1
        kv_partition_num = (max_context_len + self.kv_partition_size - 1) // self.kv_partition_size

        gws = [batch_size, self.num_kv_heads * self.q_head_chunks_per_kv_head, kv_partition_num]
        lws = [1, 1, 1]
        gws_2 = [batch_size, self.num_heads, self.head_size // self.reduce_split_step]
        lws_2 = [1, 1, 1]

        for _ in range(n_repeats):
            t_q = cl.tensor(query.detach().numpy())
            t_k = cl.tensor(key_cache.contiguous().detach().numpy())
            t_v = cl.tensor(value_cache.contiguous().detach().numpy())
            t_past_lens = cl.tensor(past_lens.detach().numpy())
            t_block_indices = cl.tensor(block_indices.detach().numpy())
            t_block_indices_begins = cl.tensor(block_indices_begins.detach().numpy())
            t_subsequence_begins = cl.tensor(subsequence_begins.detach().numpy())
            t_out = cl.tensor([batch_size, self.num_heads, kv_partition_num, self.head_size], np.dtype(np.float32))
            t_out_final = cl.tensor(out.contiguous().detach().numpy())
            t_lse = cl.tensor([batch_size, self.num_heads, kv_partition_num], np.dtype(np.float32))

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
                kv_partition_num,
            )

            cl.finish()
            out.copy_(torch.from_numpy(t_out_final.numpy()))

        return out


def _build_single_subsequence_inputs(case: DecodingCase):
    batch = 1
    q_len = 1
    low, high = -127, 128

    new_kv_len = (case.kv_len + case.block_size - 1) // case.block_size * case.block_size
    total_blk_num = new_kv_len // case.block_size

    # BLHS
    q_blhs = torch.randint(low, high, [batch, q_len, case.num_heads, case.head_size], dtype=torch.int32).to(torch.float16) / high
    k_blhs = torch.randint(low, high, [batch, new_kv_len, case.num_kv_heads, case.head_size], dtype=torch.int32).to(torch.float16) / high
    v_blhs = torch.randint(low, high, [batch, new_kv_len, case.num_kv_heads, case.head_size], dtype=torch.int32).to(torch.float16) / high

    # BHLS
    q_bhls = q_blhs.transpose(1, 2).contiguous()
    k_bhls = k_blhs.transpose(1, 2).contiguous()
    v_bhls = v_blhs.transpose(1, 2).contiguous()

    # Keep old script semantics for unused cache area.
    k_bhls.view(torch.uint16)[:, :, case.kv_len:, :] = 0
    v_bhls.view(torch.uint16)[:, :, case.kv_len:, :] = 0xFE00

    # [B,H,L,S] -> [L,H,S] -> [blk,blk_sz,H,S] -> [blk,H,blk_sz,S]
    k_blocks = k_bhls[0].transpose(0, 1).reshape(total_blk_num, case.block_size, case.num_kv_heads, case.head_size).transpose(1, 2).contiguous()
    v_blocks = v_bhls[0].transpose(0, 1).reshape(total_blk_num, case.block_size, case.num_kv_heads, case.head_size).transpose(1, 2).contiguous()

    if case.kv_cache_compression == 1:
        k_cache = _quant_per_token(k_blocks)
        k_ref_blocks = _dequant_per_token(k_cache, case.head_size, case.block_size)
        v_cache = _quant_per_token(v_blocks)
        v_ref_blocks = _dequant_per_token(v_cache, case.head_size, case.block_size)
    elif case.kv_cache_compression == 2:
        k_cache = _quant_per_channel(k_blocks, case.sub_block_size)
        k_ref_blocks = _dequant_per_channel(k_cache, case.head_size, case.block_size, case.sub_block_size)
        v_cache = _quant_per_token(v_blocks)
        v_ref_blocks = _dequant_per_token(v_cache, case.head_size, case.block_size)
    else:
        k_cache = k_blocks
        v_cache = v_blocks
        k_ref_blocks = k_blocks
        v_ref_blocks = v_blocks

    block_indices = torch.randperm(total_blk_num, dtype=torch.int32)
    key_cache = torch.empty_like(k_cache)
    value_cache = torch.empty_like(v_cache)
    key_cache[block_indices.to(dtype=torch.long)] = k_cache
    value_cache[block_indices.to(dtype=torch.long)] = v_cache

    # Reference in logical order.
    k_ref_bhls = k_ref_blocks.transpose(1, 2).reshape(batch, new_kv_len, case.num_kv_heads, case.head_size).transpose(1, 2).contiguous()
    v_ref_bhls = v_ref_blocks.transpose(1, 2).reshape(batch, new_kv_len, case.num_kv_heads, case.head_size).transpose(1, 2).contiguous()

    attention_mask = torch.zeros([batch, 1, q_len, case.kv_len], dtype=torch.float16)
    expected = F.scaled_dot_product_attention(
        q_bhls,
        k_ref_bhls[:, :, : case.kv_len, :],
        v_ref_bhls[:, :, : case.kv_len, :],
        attention_mask,
        dropout_p=0.0,
        enable_gqa=(case.num_heads > case.num_kv_heads),
    ).transpose(1, 2).contiguous()

    past_lens = torch.tensor([case.kv_len - 1], dtype=torch.int32)
    block_indices_begins = torch.tensor([0, total_blk_num], dtype=torch.int32)
    subsequence_begins = torch.tensor([0, 1], dtype=torch.int32)

    return {
        "query": q_bhls,
        "key_cache": key_cache,
        "value_cache": value_cache,
        "past_lens": past_lens,
        "block_indices": block_indices,
        "block_indices_begins": block_indices_begins,
        "subsequence_begins": subsequence_begins,
        "expected": expected,
    }


_COMPRESSION_NAMES = {0: "fp16", 1: "by_token", 2: "by_channel"}

def _case_id(case: DecodingCase) -> str:
    return (
        f"1x{case.kv_len}"
        f"_h{case.num_heads}"
        f"_kv{case.num_kv_heads}"
        f"_hs{case.head_size}"
        f"_bls{case.block_size}"
        f"_sbls{case.sub_block_size}"
        f"_cmpr{_COMPRESSION_NAMES.get(case.kv_cache_compression, case.kv_cache_compression)}"
    )


GENERATE_ONLY_SINGLE_SUBSEQ_CASES = (
    DecodingCase(kv_len=129, kv_cache_compression=0),
    DecodingCase(kv_len=129, kv_cache_compression=1),
    DecodingCase(kv_len=129, kv_cache_compression=2),
    DecodingCase(kv_len=1025, kv_cache_compression=0),
    DecodingCase(kv_len=1025, kv_cache_compression=1),
    DecodingCase(kv_len=1025, kv_cache_compression=2),
    DecodingCase(num_heads=8, num_kv_heads=2, head_size=64, block_size=256, kv_len=513, kv_cache_compression=0),
    DecodingCase(num_heads=8, num_kv_heads=2, head_size=64, block_size=256, kv_len=513, kv_cache_compression=1),
    DecodingCase(num_heads=8, num_kv_heads=2, head_size=64, block_size=256, kv_len=513, kv_cache_compression=2),
)


@pytest.mark.parametrize("case", GENERATE_ONLY_SINGLE_SUBSEQ_CASES, ids=_case_id)
def test_pa_smoke_paged_attention_generate_only(case: DecodingCase):
    data = _build_single_subsequence_inputs(case)

    runner = PaSingleTokenRunner.create_instance(
        case.num_heads,
        case.num_kv_heads,
        case.head_size,
        case.block_size,
        case.sub_block_size,
        case.kv_cache_compression,
    )

    output = torch.empty([1, 1, case.num_heads, case.head_size], dtype=torch.float16)
    output = runner(
        data["query"],
        data["key_cache"],
        data["value_cache"],
        data["past_lens"],
        data["block_indices"],
        data["block_indices_begins"],
        data["subsequence_begins"],
        output,
        n_repeats=1,
    )

    assert torch.isfinite(output).all().item()
    tol = (2e-2, 2e-2) if case.kv_cache_compression > 0 else (1e-2, 1e-3)
    _check_close(output, data["expected"], atol=tol[0], rtol=tol[1])


def _run_bandwidth_measurement(case: DecodingCase, loop_cnt: int = 100, warmup: int = 5) -> dict[str, float]:
    data = _build_single_subsequence_inputs(case)
    runner = PaSingleTokenRunner.create_instance(
        case.num_heads,
        case.num_kv_heads,
        case.head_size,
        case.block_size,
        case.sub_block_size,
        case.kv_cache_compression,
    )
    kernels = runner._create_kernels()

    query = data["query"]
    key_cache = data["key_cache"]
    value_cache = data["value_cache"]
    past_lens = data["past_lens"]
    block_indices = data["block_indices"]
    block_indices_begins = data["block_indices_begins"]
    subsequence_begins = data["subsequence_begins"]

    batch = int(query.shape[0])
    max_context_len = int(past_lens.max().item()) + 1
    kv_partition_num = (max_context_len + runner.kv_partition_size - 1) // runner.kv_partition_size
    gws = [batch, runner.num_kv_heads * runner.q_head_chunks_per_kv_head, kv_partition_num]
    lws = [1, 1, 1]
    gws_2 = [batch, runner.num_heads, runner.head_size // runner.reduce_split_step]
    lws_2 = [1, 1, 1]

    all_layers = []
    mem_size = 0
    while len(all_layers) < loop_cnt and mem_size < 8e9:
        t_q = cl.tensor(query.detach().numpy())
        t_k = cl.tensor(key_cache.contiguous().detach().numpy())
        t_v = cl.tensor(value_cache.contiguous().detach().numpy())
        t_out = cl.tensor([batch, runner.num_heads, kv_partition_num, runner.head_size], np.dtype(np.float32))
        t_out_final = cl.tensor([batch, 1, runner.num_heads, runner.head_size], np.dtype(np.float16))
        all_layers.append((t_q, t_k, t_v, t_out, t_out_final))

        mem_size += query.numel() * query.element_size()
        mem_size += key_cache.numel() * key_cache.element_size()
        mem_size += value_cache.numel() * value_cache.element_size()

    if len(all_layers) == 0:
        raise RuntimeError("Failed to allocate perf input layers")

    t_past_lens = cl.tensor(past_lens.detach().numpy())
    t_block_indices = cl.tensor(block_indices.detach().numpy())
    t_block_indices_begins = cl.tensor(block_indices_begins.detach().numpy())
    t_subsequence_begins = cl.tensor(subsequence_begins.detach().numpy())
    t_lse = cl.tensor([batch, runner.num_heads, kv_partition_num], np.dtype(np.float32))

    # Clear any previously recorded profiling events (e.g., cm_get_grf_width).
    cl.finish()

    for i in range(loop_cnt):
        j = i % len(all_layers)
        t_q, t_k, t_v, t_out, t_out_final = all_layers[j]
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
            kv_partition_num,
        )

    latency = cl.finish()

    # Keep bandwidth size computation identical to pa_2nd_token.py (uses padded KV length).
    new_kv_len = int(key_cache.shape[0] * case.block_size)
    if case.kv_cache_compression > 0:
        kvcache_size = new_kv_len * case.num_kv_heads * case.head_size * 1 * 2
    else:
        kvcache_size = new_kv_len * case.num_kv_heads * case.head_size * 2 * 2
    intermedia_size = batch * case.num_heads * kv_partition_num * (case.head_size + 1) * 4

    # latency format: [cm_sdpa_2nd, cm_sdpa_2nd_reduce, ...] for this measured loop only.
    expected_event_count = 2 * loop_cnt
    if len(latency) < expected_event_count:
        raise RuntimeError(f"Expected at least {expected_event_count} events, got {len(latency)}")

    cm_sdpa_2nd_total_time = 0.0
    cm_sdpa_2nd_reduce_total_time = 0.0
    num_runs = 0

    for pair_idx in range(loop_cnt):
        kv_ns = float(latency[2 * pair_idx])
        inter_ns = float(latency[2 * pair_idx + 1])
        if kv_ns <= 0 or inter_ns <= 0:
            continue
        if pair_idx < warmup:
            continue

        cm_sdpa_2nd_total_time += kv_ns
        cm_sdpa_2nd_reduce_total_time += inter_ns
        num_runs += 1

    if num_runs <= 0 or cm_sdpa_2nd_total_time <= 0 or cm_sdpa_2nd_reduce_total_time <= 0:
        raise RuntimeError("Invalid perf timing accumulation")

    return {
        "num_runs": float(num_runs),
        "cm_sdpa_2nd_bw_gbs": float(kvcache_size * num_runs / cm_sdpa_2nd_total_time),
        "cm_sdpa_2nd_reduce_bw_gbs": float(intermedia_size * num_runs / cm_sdpa_2nd_reduce_total_time),
        "cm_sdpa_2nd_ms": float(cm_sdpa_2nd_total_time * 1e-6 / num_runs),
        "cm_sdpa_2nd_reduce_ms": float(cm_sdpa_2nd_reduce_total_time * 1e-6 / num_runs),
    }


def test_pa_perf_bandwidth_generate_single_subsequence_default_params():
    if os.environ.get("RUN_PA_PERF", "0") != "1":
        pytest.skip("Set RUN_PA_PERF=1 to enable bandwidth perf test")

    # Same core parameters as pa_2nd_token.py defaults.
    case = DecodingCase(
        num_heads=32,
        num_kv_heads=8,
        head_size=128,
        block_size=256,
        kv_len=32769,
        kv_cache_compression=0,
    )
    perf = _run_bandwidth_measurement(case, loop_cnt=100, warmup=5)

    print(
        "[perf] "
        f"cm_sdpa_2nd_bw={perf['cm_sdpa_2nd_bw_gbs']:.3f} GB/s, "
        f"cm_sdpa_2nd_reduce_bw={perf['cm_sdpa_2nd_reduce_bw_gbs']:.3f} GB/s, "
        f"cm_sdpa_2nd_ms={perf['cm_sdpa_2nd_ms']:.3f}, "
        f"cm_sdpa_2nd_reduce_ms={perf['cm_sdpa_2nd_reduce_ms']:.3f}"
    )

    assert perf["cm_sdpa_2nd_bw_gbs"] > 0.0
    assert perf["cm_sdpa_2nd_reduce_bw_gbs"] > 0.0

# Usage:
#   python -m py_compile test_pa_decoding.py
#   python -m pytest --collect-only -q test_pa_decoding.py
#   timeout 120s python -m pytest -s -q test_pa_decoding.py -vv
#   timeout 120s python -m pytest -s -q test_pa_decoding.py -vv -k 'generate_only and (cmprby_token or cmprby_channel)'