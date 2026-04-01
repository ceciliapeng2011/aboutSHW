#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 <benchmark_log_file> [output_csv]" >&2
  echo "Example: $0 log2" >&2
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
  OUTPUT_CSV="$dir/${base%.*}_benchmark_summary.csv"
  if [[ "$base" != *.* ]]; then
    OUTPUT_CSV="$dir/${base}_benchmark_summary.csv"
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

# Expected printed pair:
#   kv_len= 2048  standard=0.210ms  turboquant=3.454ms  speedup=0.061x
#     std_bw(k/reduce)=21.137/11.223 GB/s  tq_bw(k/reduce)=0.941/8.530 GB/s
perf_line = re.compile(
    r'^\s*kv_len=\s*(?P<kv_len>\d+)\s+'
    r'standard=(?P<standard_ms>[0-9.]+)ms\s+'
    r'turboquant=(?P<turboquant_ms>[0-9.]+)ms\s+'
    r'speedup=(?P<speedup>[0-9.]+)x\s*$',
    re.MULTILINE,
)
bw_line = re.compile(
    r'^\s*std_bw\(k/reduce\)=(?P<standard_k_bw_gbs>[0-9.]+)/(?P<standard_reduce_bw_gbs>[0-9.]+) GB/s\s+'
    r'tq_bw\(k/reduce\)=(?P<turboquant_k_bw_gbs>[0-9.]+)/(?P<turboquant_reduce_bw_gbs>[0-9.]+) GB/s\s*$',
    re.MULTILINE,
)

perf_matches = list(perf_line.finditer(text))
bw_matches = list(bw_line.finditer(text))

if not perf_matches:
    raise SystemExit("No benchmark perf rows parsed from log")
if len(perf_matches) != len(bw_matches):
    raise SystemExit(
        f"Parsed mismatched perf/bw lines: perf={len(perf_matches)} bw={len(bw_matches)}"
    )

rows = []
for pm, bm in zip(perf_matches, bw_matches):
    row = pm.groupdict()
    row.update(bm.groupdict())
    rows.append(row)

rows.sort(key=lambda r: int(r["kv_len"]))

fieldnames = [
    "kv_len",
    "standard_ms",
    "turboquant_ms",
    "speedup",
    "standard_k_bw_gbs",
    "standard_reduce_bw_gbs",
    "turboquant_k_bw_gbs",
    "turboquant_reduce_bw_gbs",
]

out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {len(rows)} rows to {out_path}")
PY
