"""Perf sweep: SDPA-CM fp16/int8 x dense/sparse(bs128,d0.3)/sparse(bs256,d0.3) x ctx(16k,32k) x
{single-source, dual-source}.

Mirrors pageatten/bench_pa_dense_vs_sparse.py's sweep shape for the SDPA-CM prefill kernel
(cm_sdpa_opt_prefill.cm) instead of PA's paged-cache kernel.

Fixed shape: num_heads=32, num_kv_heads=8, head_size=128 (Qwen3-8B GQA config).

"int8" here is the same affine uint8 (KV_CACHE_COMPRESSION=2, BY_CHANNEL) scheme
pageatten/kv_cache_quant_utils.py implements -- see int8_kv_ref.py / test_sdpa_prefill_int8.py.

topology="single" runs the single-source causal kernel (cm_sdpa_opt_prefill.cm) over one
contiguous ctx_len-token K/V, same as before. topology="dual" runs the dual-source cache+current
kernel (cm_sdpa_opt_prefill_dualsource.cm, see bench_sdpa_dualsource.py) with the ctx_len split
into a CUR_CHUNK-token "current" segment (this step's new K/V) plus a (ctx_len - CUR_CHUNK)-token
"cache" segment (the accumulated cache) -- the realistic streaming-decode shape this kernel family
targets. Dual-source is non-causal only (no causal mode exists for this kernel family) and always
needs an explicit block mask (no bare-dense path), so its "dense" sparsity point uses
DUAL_DENSE_BLOCK with density=1.0 (an all-True mask) instead of single-source's literal
no-mask kernel.

Usage:
    HW_PEAK_FLOPS=30 python bench_sdpa_dense_vs_sparse.py
    SDPA_BENCH_CSV=my_results.csv python bench_sdpa_dense_vs_sparse.py
"""
import csv
import os

from clops.utils import Colors
from bench_sdpa import (
    bench_prefill,
    bench_prefill_int8,
    bench_prefill_block_mask_synthetic,
    bench_prefill_dual_fp16,
    bench_prefill_dual_int8cache,
)

NUM_HEADS = 32
NUM_KV_HEADS = 8
HEAD_SIZE = 128
IS_CAUSAL = True  # single-source only; dual-source has no causal mode.

TOPOLOGIES = ("single", "dual")
DTYPES = ("fp16", "int8")  # "int8" -> KV_CACHE_COMPRESSION=2, see module docstring

SPARSITY_CONFIGS = (
    ("dense", 1, 1.0),
    ("sparse_bs128_d0.3", 128, 0.3),
    ("sparse_bs256_d0.3", 256, 0.3),
)
DUAL_DENSE_BLOCK = 128  # block size used for dual-source's "dense" (density=1.0) sweep point.

CONTEXT_LENS = (16 * 1024, 32 * 1024)
CUR_CHUNK = 2048  # dual-source "current" (new-tokens) segment size; cache = ctx_len - CUR_CHUNK.

CSV_PATH = os.environ.get("SDPA_BENCH_CSV", "bench_sdpa_dense_vs_sparse.csv")
EMPTY_CACHE_CSV_PATH = os.environ.get(
    "SDPA_BENCH_EMPTY_CACHE_CSV", "bench_sdpa_empty_cache_parity.csv"
)


def _run_one(seq_len, dtype_name, topology, sparse_block_sz, density):
    common = dict(num_heads=NUM_HEADS, num_kv_heads=NUM_KV_HEADS, head_size=HEAD_SIZE)
    if topology == "single":
        common["is_causal"] = IS_CAUSAL
        if dtype_name == "fp16":
            if sparse_block_sz == 1:
                return bench_prefill(seq_len, **common)
            return bench_prefill_block_mask_synthetic(
                seq_len, sparse_block_size=sparse_block_sz, density=density, **common
            )
        # dtype_name == "int8" -> KV_CACHE_COMPRESSION=2 kernel; bench_prefill_int8 handles both
        # dense (block=1) and sparse (block in (128, 256)) through the same
        # sparse_block_size/density args.
        return bench_prefill_int8(
            seq_len, sparse_block_size=sparse_block_sz, density=density, **common
        )

    # topology == "dual": split ctx_len into cache (cache) + current (new chunk); non-causal only.
    kv_len_cache = seq_len - CUR_CHUNK
    dual_block = sparse_block_sz if sparse_block_sz > 1 else DUAL_DENSE_BLOCK
    if dtype_name == "fp16":
        return bench_prefill_dual_fp16(
            kv_len_cache, CUR_CHUNK, sparse_block_size=dual_block, density=density, **common
        )
    # dtype_name == "int8" -> KV_CACHE_COMPRESSION=2 kernel (cache quantized, current stays half).
    return bench_prefill_dual_int8cache(
        kv_len_cache, CUR_CHUNK, sparse_block_size=dual_block, density=density, **common
    )


def run():
    results = []
    for seq_len in CONTEXT_LENS:
        for topology in TOPOLOGIES:
            for dtype_name in DTYPES:
                for sparsity_name, sparse_block_sz, density in SPARSITY_CONFIGS:
                    print(
                        f"{Colors.GREEN}[bench] ctx={seq_len} topology={topology} dtype={dtype_name} "
                        f"sparsity={sparsity_name}{Colors.END}"
                    )
                    avg_ms, utilization = _run_one(seq_len, dtype_name, topology, sparse_block_sz, density)
                    results.append(
                        {
                            "ctx_len": seq_len,
                            "topology": topology,
                            "dtype": dtype_name,
                            "sparsity": sparsity_name,
                            "avg_ms": round(avg_ms, 2),
                            "utilization_pct": round(utilization, 2),
                        }
                    )

    with open(CSV_PATH, "w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["ctx_len", "topology", "dtype", "sparsity", "avg_ms", "utilization_pct"],
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\n{Colors.GREEN}[summary] heads={NUM_HEADS}/{NUM_KV_HEADS} hd={HEAD_SIZE} "
          f"causal(single-only)={IS_CAUSAL} cur_chunk(dual-only)={CUR_CHUNK}{Colors.END}")
    print(f"{'ctx_len':>8} | {'topology':>8} | {'dtype':>5} | {'sparsity':>20} | {'avg_ms':>10} | {'util%':>7}")
    for row in results:
        print(
            f"{row['ctx_len']:>8} | {row['topology']:>8} | {row['dtype']:>5} | {row['sparsity']:>20} | "
            f"{row['avg_ms']:>10.3f} | {row['utilization_pct']:>6.1f}%"
        )
    print(f"csv | {CSV_PATH}")


def run_empty_cache_parity():
    """Dual-source with an empty cache (kv_len_cache=0) must perform the same as single-source
    non-causal at the same total kv_len -- the barrier-free sdpa_kernel_lsc_prefetch_2src path
    (cm_sdpa_common.hpp) degrades its cache call to a no-op in that case, so the only real work left
    is current's own sdpa_accumulate_kv_source_prefetch call, which should cost the same as calling
    sdpa_kernel_lsc_prefetch directly on the same shape. Covers both fp16 and int8 (int8-cache stays on
    the unchanged SLM path regardless of cache length, so its ratio is not expected to move the way
    fp16's does -- it's included here for reference, not as the core claim). This is the performance
    half of the mirroring check -- test_sdpa_prefill_block_mask_dualsource.py's
    test_empty_cache_chunk0/test_empty_cache_sparse_bs128/test_empty_cache_sparse_bs256 are the
    correctness half of the same claim."""
    print(f"\n{Colors.GREEN}[empty-cache parity] dual-source(kv_len_cache=0) vs single-source(is_causal=False), "
          f"same total kv_len -- should be ~1.0x{Colors.END}")
    results = []
    for seq_len in CONTEXT_LENS:
        for dtype_name in DTYPES:
            for sparsity_name, sparse_block_sz, density in SPARSITY_CONFIGS:
                common = dict(num_heads=NUM_HEADS, num_kv_heads=NUM_KV_HEADS, head_size=HEAD_SIZE)
                dual_block = sparse_block_sz if sparse_block_sz > 1 else DUAL_DENSE_BLOCK
                if dtype_name == "fp16":
                    if sparse_block_sz == 1:
                        single_ms, single_util = bench_prefill(seq_len, is_causal=False, **common)
                    else:
                        single_ms, single_util = bench_prefill_block_mask_synthetic(
                            seq_len, sparse_block_size=sparse_block_sz, density=density,
                            is_causal=False, **common
                        )
                    dual_ms, dual_util = bench_prefill_dual_fp16(
                        0, seq_len, sparse_block_size=dual_block, density=density, **common
                    )
                else:
                    single_ms, single_util = bench_prefill_int8(
                        seq_len, sparse_block_size=sparse_block_sz, density=density,
                        is_causal=False, **common
                    )
                    dual_ms, dual_util = bench_prefill_dual_int8cache(
                        0, seq_len, sparse_block_size=dual_block, density=density, **common
                    )
                results.append(
                    {
                        "ctx_len": seq_len,
                        "dtype": dtype_name,
                        "sparsity": sparsity_name,
                        "single_ms": round(single_ms, 2),
                        "single_util_pct": round(single_util, 2),
                        "dual_ms": round(dual_ms, 2),
                        "dual_util_pct": round(dual_util, 2),
                        "ratio": round(dual_ms / single_ms, 3),
                    }
                )

    with open(EMPTY_CACHE_CSV_PATH, "w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "ctx_len",
                "dtype",
                "sparsity",
                "single_ms",
                "single_util_pct",
                "dual_ms",
                "dual_util_pct",
                "ratio",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"{'ctx_len':>8} | {'dtype':>5} | {'sparsity':>20} | {'single_ms':>10} | {'single_util%':>12} | "
          f"{'dual_ms':>10} | {'dual_util%':>10} | {'ratio':>6}")
    for row in results:
        print(f"{row['ctx_len']:>8} | {row['dtype']:>5} | {row['sparsity']:>20} | {row['single_ms']:>10.3f} | "
              f"{row['single_util_pct']:>11.1f}% | {row['dual_ms']:>10.3f} | "
              f"{row['dual_util_pct']:>9.1f}% | {row['ratio']:>5.3f}x")
    print(f"csv | {EMPTY_CACHE_CSV_PATH}")


if __name__ == "__main__":
    from clops import cl
    cl.profiling(True)
    run()
    run_empty_cache_parity()
