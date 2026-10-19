"""Perf sweep: SDPA-CM fp16/fp8 x dense/sparse(bs128,d0.3)/sparse(bs256,d0.3) x ctx(16k,32k).

Mirrors pageatten/bench_pa_dense_vs_sparse.py's sweep shape for the SDPA-CM prefill kernel
(cm_sdpa_opt_prefill.cm) instead of PA's paged-cache kernel.

Fixed shape: num_heads=32, num_kv_heads=8, head_size=128 (Qwen3-8B GQA config), causal prefill.

sdpa_cm has no separate int8-quantized-KV mode (unlike pageatten's KV_CACHE_COMPRESSION_BY_TOKEN).
Its only sub-fp16 K/V representation is fp8e4m3 ("hf8", KV_FP8E4M3=1, see hf8_e4m3_ref.py /
test_sdpa_prefill_fp8.py) -- 1 byte/elem, same on-the-wire footprint as an int8 quant would have.
"int8" in this sweep is therefore mapped to fp8e4m3; see bench_sdpa.py's own "fp16 vs
KV_FP8E4M3=1" framing for the same convention.

Usage:
    HW_PEAK_FLOPS=30 python bench_sdpa_dense_vs_sparse.py
    SDPA_BENCH_CSV=my_results.csv python bench_sdpa_dense_vs_sparse.py
"""
import csv
import os

from clops.utils import Colors
from bench_sdpa import (
    bench_prefill,
    bench_prefill_fp8,
    bench_prefill_block_mask_synthetic,
)

NUM_HEADS = 32
NUM_KV_HEADS = 8
HEAD_SIZE = 128
IS_CAUSAL = True

DTYPES = ("fp16", "int8")  # "int8" -> fp8e4m3 (KV_FP8E4M3=1), see module docstring

SPARSITY_CONFIGS = (
    ("dense", 1, 1.0),
    ("sparse_bs128_d0.3", 128, 0.3),
    ("sparse_bs256_d0.3", 256, 0.3),
)

CONTEXT_LENS = (16 * 1024, 32 * 1024)

CSV_PATH = os.environ.get("SDPA_BENCH_CSV", "bench_sdpa_dense_vs_sparse.csv")


def _run_one(seq_len, dtype_name, sparse_block_sz, density):
    common = dict(
        num_heads=NUM_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        head_size=HEAD_SIZE,
        is_causal=IS_CAUSAL,
    )
    if dtype_name == "fp16":
        if sparse_block_sz == 1:
            return bench_prefill(seq_len, **common)
        return bench_prefill_block_mask_synthetic(
            seq_len, sparse_block_size=sparse_block_sz, density=density, **common
        )
    # dtype_name == "int8" -> fp8e4m3 kernel; bench_prefill_fp8 handles both dense (block=1)
    # and sparse (block in (128, 256)) through the same sparse_block_size/density args.
    return bench_prefill_fp8(
        seq_len, sparse_block_size=sparse_block_sz, density=density, **common
    )


def run():
    results = []
    for seq_len in CONTEXT_LENS:
        for dtype_name in DTYPES:
            for sparsity_name, sparse_block_sz, density in SPARSITY_CONFIGS:
                print(
                    f"{Colors.GREEN}[bench] ctx={seq_len} dtype={dtype_name} "
                    f"sparsity={sparsity_name}{Colors.END}"
                )
                avg_ms, utilization = _run_one(seq_len, dtype_name, sparse_block_sz, density)
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
          f"causal={IS_CAUSAL} (int8 == fp8e4m3){Colors.END}")
    print(f"{'ctx_len':>8} | {'dtype':>5} | {'sparsity':>20} | {'avg_ms':>10} | {'util%':>7}")
    for row in results:
        print(
            f"{row['ctx_len']:>8} | {row['dtype']:>5} | {row['sparsity']:>20} | "
            f"{row['avg_ms']:>10.3f} | {row['utilization_pct']:>6.1f}%"
        )
    print(f"csv | {CSV_PATH}")


if __name__ == "__main__":
    from clops import cl
    cl.profiling(True)
    run()
