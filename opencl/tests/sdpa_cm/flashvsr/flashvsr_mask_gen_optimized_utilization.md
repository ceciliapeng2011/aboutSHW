# FlashVSR draft-mask generation: utilization findings and Stage C optimization

Follow-up to the radix-select/score-softmax-fusion pass (see git history for
`flashvsr_draft_topk.cm`/`flashvsr_draft_gemm.cm`, since renamed/split upstream into
`draft_block_mask_topk_mask.cm`/`draft_block_mask_pool.cm`/`draft_block_mask_score_softmax.cm`).
After that pass, two
utilization hotspots remained, at very different scales:

1. **`cm_sdpa_opt_prefill` (sparse) itself** -- 73-80% of the total critical
   path at production density (26.5ms of 36.2ms steady, 60.8ms of 75.8ms
   warm-up), only 3.5-4.7% of compute roofline utilized. At FlashVSR's
   ~0.3-0.4% mask density, per-block dispatch/skip overhead dominates over
   useful work per visited block. **Out of scope** for this effort -- the SDPA
   kernel itself was explicitly excluded from the plan that scoped this work.
2. **`flashvsr_score_softmax`'s QK^T dot-product loop** (Stage C from the
   original optimization plan) -- roughly tied with `flashvsr_topk_mask`
   (3.4-5.9ms vs. 3.4-5.8ms at production density) but running at only ~0.5%
   of compute roofline: a fully serial per-thread loop over up to
   `num_k_windows` pooled dot products. **In scope** -- addressed below.

## Diagnosis

The loop (`flashvsr_draft_gemm.cm`'s `flashvsr_score_softmax`, Pass 1) does,
per `m`: a 128-wide block load of `pooled_k[m]`, a multiply + horizontal
`cm_sum` reduction, a scalar mask lookup/branch, a store, and a running-max
update. At only 0.5% of compute roofline, the dot product itself isn't the
cost -- the loop is dominated by **per-iteration scalar control overhead**
(loop increment/compare, modulo for `m_spatial`, mask branch), paid once per
K-row with almost nothing to amortize it against.

### Attempt 1: manual load prefetch (reverted, no effect)

Hypothesized the loop was memory-latency-bound and tried software-pipelining
the `pooled_k` load one iteration ahead (load `m+1` before reducing `m`).
Result: no measurable change (450x450: 5.93ms vs. 5.90ms baseline; 900x900:
25-28ms vs. 25.2ms baseline, both within run-to-run noise), plus +5 registers
(49->54) for nothing. Reverted. Conclusion: load latency isn't what's being
paid here -- the loop has no independent work to overlap it with regardless
of when the load is issued.

### Attempt 2: 4x loop unroll (kept)

Unrolled Pass 1 by 4, processing `m..m+3` per iteration with independent
load/multiply/reduce/mask/store/max chains (remainder handled by a trailing
scalar loop for `num_k_windows` not divisible by 4). This amortizes the
per-iteration scalar overhead across 4 K-rows instead of paying it once per
row. Correctness unchanged -- mask/`raw_scores` writes are per-`m`
independent and `row_max` is commutative, so reordering across the unrolled
group is safe. `test_flashvsr_mask.py` (5/5) passes unchanged.

## Results

Median of 3 runs each, `bench_flashvsr_mask.py`:

| num_q=num_k | baseline | 4x-unrolled | speedup |
|---|---|---|---|
| 64 | 0.117 ms | 0.091 ms | 1.29x |
| 150 (FlashVSR steady) | 0.687 ms | 0.536 ms | 1.28x |
| 450 (FlashVSR warm-up) | 5.90 ms | 4.57 ms | 1.29x |
| 600 | 10.82 ms | 8.03 ms | 1.35x |
| 900 | ~24-30 ms (noisy) | ~18-20 ms | ~1.3-1.4x |

Roofline utilization: 0.5% -> 0.68% of XMX peak GFLOPS. Registers: 49 -> 137
(2.8x) -- a real cost, but didn't visibly hurt at any tested window count
(900-window case improved as much as the smaller ones), so occupancy loss
isn't biting yet at these chunk sizes. Worth re-checking if this kernel is
ever pushed to much larger `num_k_windows` or run alongside other
register-hungry kernels on the same EUs.

On FlashVSR's actual production chunk shapes specifically: steady-state
(150x150) drops from 0.687ms to 0.536ms, warm-up (450x450) drops from 5.90ms
to 4.57ms per call -- both a ~22-24% latency reduction, consistent across
repeated runs.

## What's left

- `cm_sdpa_opt_prefill` remains the dominant cost by a wide margin (see item
  1 above) -- the real lever, but out of scope here; would need a separate
  decision to open that up.
- Within mask-gen scope: `flashvsr_topk_mask`'s radix-select passes are now
  the other roughly-tied hotspot; no further work done on it in this pass.
- Unroll factor (4) wasn't tuned against 2 or 8 -- 4 was chosen as a
  reasonable first cut given the register-cost/latency-hiding tradeoff
  observed. Could be revisited if register pressure becomes a problem at
  larger chunk sizes.
