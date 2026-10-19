"""Perf sweep: PA fp16/int8 x dense/sparse(bs128,d0.3)/sparse(bs256,d0.3) x ctx(16k,32k).

Fixed shape: num_heads=32, num_kv_heads=8, head_size=128 (Qwen3-8B GQA config).
KV-cache paging: block_sz=256, trunk_sz=4096 (16 blocks/trunk), matching the
Qwen3-8B perf conventions used elsewhere in this directory (see smoke_perf_test
in test_pa.py). "int8" uses KV_CACHE_COMPRESSION_BY_TOKEN.

Usage:
    HW_PEAK_FLOPS=30 HW_PEAK_BW_GBS=100 python bench_pa_dense_vs_sparse.py
    PA_BENCH_CSV=my_results.csv python bench_pa_dense_vs_sparse.py
"""
import csv
import os

from clops.utils import Colors
from test_pa import (
    KV_CACHE_COMPRESSION_BY_TOKEN,
    KV_CACHE_COMPRESSION_NONE,
    test_page_attn_causal_batch1,
)
from kv_cache_quant_utils import DEFAULT_SUB_BLOCK_SIZE

NUM_HEADS = 32
NUM_KV_HEADS = 8
HEAD_SIZE = 128

BLOCK_SZ = 256
BLOCKS_PER_TRUNK = 16
TRUNK_SZ = BLOCK_SZ * BLOCKS_PER_TRUNK

DTYPES = (
    ("fp16", KV_CACHE_COMPRESSION_NONE),
    ("int8", KV_CACHE_COMPRESSION_BY_TOKEN),
)

SPARSITY_CONFIGS = (
    ("dense", 1, 1.0),
    ("sparse_bs128_d0.3", 128, 0.3),
    ("sparse_bs256_d0.3", 256, 0.3),
)

CONTEXT_LENS = (16 * 1024, 32 * 1024)

CSV_PATH = os.environ.get("PA_BENCH_CSV", "bench_pa_dense_vs_sparse.csv")


def run():
    results = []
    for seq_len in CONTEXT_LENS:
        for dtype_name, kv_mode in DTYPES:
            for sparsity_name, sparse_block_sz, density in SPARSITY_CONFIGS:
                print(
                    f"{Colors.GREEN}[bench] ctx={seq_len} dtype={dtype_name} "
                    f"sparsity={sparsity_name}{Colors.END}"
                )
                result = test_page_attn_causal_batch1(
                    seq_len,
                    num_heads=NUM_HEADS,
                    num_kv_heads=NUM_KV_HEADS,
                    head_size=HEAD_SIZE,
                    block_sz=BLOCK_SZ,
                    trunk_sz=TRUNK_SZ,
                    compressed_kvcache=kv_mode,
                    sub_block_sz=DEFAULT_SUB_BLOCK_SIZE,
                    sparse_block_sz=sparse_block_sz,
                    density=density,
                    check_acc=False,
                )
                avg_ms, utilization = result if result is not None else (float("nan"), float("nan"))
                results.append(
                    {
                        "ctx_len": seq_len,
                        "dtype": dtype_name,
                        "sparsity": sparsity_name,
                        "avg_ms": avg_ms,
                        "utilization_pct": utilization,
                    }
                )

    with open(CSV_PATH, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["ctx_len", "dtype", "sparsity", "avg_ms", "utilization_pct"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\n{Colors.GREEN}[summary] heads={NUM_HEADS}/{NUM_KV_HEADS} hd={HEAD_SIZE} "
          f"block_sz={BLOCK_SZ} trunk_sz={TRUNK_SZ}{Colors.END}")
    print(f"{'ctx_len':>8} | {'dtype':>5} | {'sparsity':>20} | {'avg_ms':>10} | {'util%':>7}")
    for row in results:
        print(
            f"{row['ctx_len']:>8} | {row['dtype']:>5} | {row['sparsity']:>20} | "
            f"{row['avg_ms']:>10.3f} | {row['utilization_pct']:>6.1f}%"
        )
    print(f"csv | {CSV_PATH}")


if __name__ == "__main__":
    run()
