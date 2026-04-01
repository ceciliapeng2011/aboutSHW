#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 <log_file> [output_csv]" >&2
  echo "Example: $0 log" >&2
  exit 1
fi

LOG_FILE="$1"
if [[ ! -f "$LOG_FILE" ]]; then
  echo "Error: log file not found: $LOG_FILE" >&2
  exit 1
fi

if [[ $# -eq 2 ]]; then
  OUTPUT_CSV="$2"
else
  base="$(basename "$LOG_FILE")"
  dir="$(dirname "$LOG_FILE")"
  OUTPUT_CSV="$dir/${base%.*}_summary.csv"
  if [[ "$base" != *.* ]]; then
    OUTPUT_CSV="$dir/${base}_summary.csv"
  fi
fi

python3 - "$LOG_FILE" "$OUTPUT_CSV" <<'PY'
from pathlib import Path
import csv
import re
import sys

log_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])

text = log_path.read_text(errors="replace")
pattern = re.compile(
  r'test_pa_decoding\.py::test_pa_perf_bandwidth_generate_single_subsequence_default_params\['
  r'(?P<block_size>\d+)-(?P<kv_len>\d+)-(?P<kv_cache_compression>True|False)-(?P<kv_cache_quant_mode>by_token|by_channel)\] '
  r'\[perf\] cm_sdpa_2nd_bw=(?P<cm_sdpa_2nd_bw_gbs>[0-9.]+) GB/s, '
    r'cm_sdpa_2nd_reduce_bw=(?P<cm_sdpa_2nd_reduce_bw_gbs>[0-9.]+) GB/s, '
    r'cm_sdpa_2nd_ms=(?P<cm_sdpa_2nd_ms>[0-9.]+), '
    r'cm_sdpa_2nd_reduce_ms=(?P<cm_sdpa_2nd_reduce_ms>[0-9.]+)'
)

rows = [m.groupdict() for m in pattern.finditer(text)]
if not rows:
    raise SystemExit("No perf rows parsed from log")

for row in rows:
  if row["kv_cache_compression"] == "False":
    row["kv_cache_quant_mode"] = "-"

quant_mode_order = {
  "-": 0,
  "by_channel": 1,
  "by_token": 2,
}

rows.sort(
    key=lambda r: (
        int(r["kv_len"]),
        0 if r["kv_cache_compression"] == "False" else 1,
    quant_mode_order[r["kv_cache_quant_mode"]],
        int(r["block_size"]),
    )
)

fieldnames = [
    "kv_len",
    "kv_cache_compression",
    "kv_cache_quant_mode",
    "cm_sdpa_2nd_bw_gbs",
    "cm_sdpa_2nd_reduce_bw_gbs",
    "cm_sdpa_2nd_ms",
    "cm_sdpa_2nd_reduce_ms",
    "block_size",
]

out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows({k: row[k] for k in fieldnames} for row in rows)

print(f"Wrote {len(rows)} rows to {out_path}")
PY
