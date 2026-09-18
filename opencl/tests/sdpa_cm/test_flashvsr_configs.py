"""
Accuracy tests at FlashVSR v1.1's real attention configs (see
/home/ceciliapeng/repos/reports/flashvsr_v1.1_full_inference_pipeline.md), reusing sdpa_cm's
existing dense/sparse prefill infra unchanged -- no kernel/adapter code needed since the kernel
already supports arbitrary q_len != kv_len, MHA (num_kv_heads == num_heads), and is_causal=False.

FlashVSR's three attention sites, from the report:
- DiT self-attention: MHA, num_heads=12, head_dim=128, block-sparse (window=128 tokens, matching
  this repo's own SPARSE_BLOCK_SIZE=128 XAttention granularity). The kernel itself is invoked with
  is_causal=False -- temporal causality is entirely handled by a Python-side FIFO KV-cache outside
  the attention op (report Sec 3.3/8), not by the op's own causal mask.
- DiT cross-attention: MHA, num_heads=12, head_dim=128, dense (no sparsity), is_causal=False,
  kv_len=512 (fixed text-context length) vs. a much larger q_len (video-chunk token count).
- VAE bottleneck attention: single head, head_size=384 -- OUT OF SCOPE. head_size=384 overflows
  flashattn/cm_sdpa_common.hpp's shared rO accumulator matrix past CM's 16KB hard limit (same wall
  PA hit at head_size=256, fixed there via an rO_lo/rO_hi split never ported to sdpa_cm's shared
  kernel). Skipped per explicit user decision rather than porting that fix now.

Real per-chunk token counts, for the requested real case (320x480 portrait source, upsampled by
FlashVSR's fixed 4x scale to a 1280x1920 target -- exact multiples of 128, no flooring loss;
VAE 8x spatial downsample -> latent 160x240 -> patchify(2x2) -> post-patchify token grid
h*w = 80*120 = 9600): warm-up chunk (f=6) = 57600 tokens, steady-state chunk (f=2) = 19200 tokens
(see plan doc's "FlashVSR-config" section for the full derivation). Those exact sizes are used for
perf benchmarking (bench_sdpa.py's "FlashVSR configs" section, GPU-only, no reference needed) but
are infeasible for an accuracy test here: PyTorch's reference SDPA materializes the full
[q_len, kv_len] fp32 attention matrix per head (e.g. 57600^2*4 bytes/head ~= 13.3GB), so these
tests use proportionally smaller, CPU-feasible stand-ins of the same shape family (same head
config, same q_len:kv_len ratio) instead of the literal production sizes.
"""
import torch

from clops import cl

import test_sdpa_prefill as base
import test_sdpa_prefill_xattn as xattn_mod
import bench_sdpa as bench

torch.manual_seed(7)

NUM_HEADS = 12       # FlashVSR DiT: dim=1536, num_heads=12, head_dim=128 -- MHA, not GQA.
NUM_KV_HEADS = 12
HEAD_SIZE = 128
SPARSE_BLOCK_SIZE = 128  # FlashVSR's WindowPartition3D window == 128 tokens (2x8x8).
S_CTX = 512               # FlashVSR's fixed text cross-attention context length.

# Scaled-down stand-ins for the real 19200 (steady) / 57600 (warm-up) chunk token counts --
# see module docstring for why the literal sizes aren't accuracy-testable on CPU.
STEADY_CHUNK_TEST = 2048
WARMUP_CHUNK_TEST = 4096


def test_self_attention_dense_steady():
    # Self-attention kernel call itself is always non-causal in FlashVSR (see module docstring).
    q, k, v = base.make_qkv(batch=1, num_heads=NUM_HEADS, num_kv_heads=NUM_KV_HEADS,
                             q_len=STEADY_CHUNK_TEST, kv_len=STEADY_CHUNK_TEST, head_size=HEAD_SIZE)
    out = base.run_prefill(q, k, v, is_causal=False)
    ref = base.ref_sdpa(q, k, v, is_causal=False)
    base.check_close(out, ref)


def test_self_attention_dense_warmup():
    q, k, v = base.make_qkv(batch=1, num_heads=NUM_HEADS, num_kv_heads=NUM_KV_HEADS,
                             q_len=WARMUP_CHUNK_TEST, kv_len=WARMUP_CHUNK_TEST, head_size=HEAD_SIZE)
    out = base.run_prefill(q, k, v, is_causal=False)
    ref = base.ref_sdpa(q, k, v, is_causal=False)
    base.check_close(out, ref)


def test_self_attention_sparse_steady():
    # FlashVSR's real sparsity (11x11 local-window neighborhood + query-adaptive top-k, via
    # block_sparse_attn_cuda.fwd_block) isn't ported here -- that's a CUDA-side mask-generation
    # algorithm from a different repo, out of scope for this sdpa_cm accuracy test. Instead this
    # validates that the kernel correctly honors an arbitrary block mask at FlashVSR's exact block
    # granularity (128 tokens) and non-causal setting, using bench_sdpa's synthetic density-sweep
    # mask generator at a representative (illustrative, not derived) density.
    q, k, v = base.make_qkv(batch=1, num_heads=NUM_HEADS, num_kv_heads=NUM_KV_HEADS,
                             q_len=STEADY_CHUNK_TEST, kv_len=STEADY_CHUNK_TEST, head_size=HEAD_SIZE)
    mask, density = bench.fake_block_mask(NUM_HEADS, STEADY_CHUNK_TEST, SPARSE_BLOCK_SIZE,
                                           density=0.3, is_causal=False)
    out = xattn_mod.run_prefill_sparse(q, k, v, is_causal=False, sparse_block_size=SPARSE_BLOCK_SIZE,
                                        block_mask=mask)
    ref = xattn_mod.ref_sdpa_sparse(q, k, v, is_causal=False, mask=mask,
                                     sparse_block_size=SPARSE_BLOCK_SIZE)
    base.check_close(out, ref)
    print(f"[test_self_attention_sparse_steady] effective_density={density:.3f}")


def test_cross_attention():
    # DiT cross-attention: dense, non-causal, small fixed-length K/V (persistent text-embedding
    # cache) against a much longer Q (video-chunk tokens) -- already supported since make_qkv/
    # run_prefill take q_len/kv_len independently.
    q, k, v = base.make_qkv(batch=1, num_heads=NUM_HEADS, num_kv_heads=NUM_KV_HEADS,
                             q_len=STEADY_CHUNK_TEST, kv_len=S_CTX, head_size=HEAD_SIZE)
    out = base.run_prefill(q, k, v, is_causal=False)
    ref = base.ref_sdpa(q, k, v, is_causal=False)
    base.check_close(out, ref)


if __name__ == "__main__":
    cl.profiling(True)
    test_self_attention_dense_steady()
    print("test_self_attention_dense_steady PASSED")
    test_self_attention_dense_warmup()
    print("test_self_attention_dense_warmup PASSED")
    test_self_attention_sparse_steady()
    print("test_self_attention_sparse_steady PASSED")
    test_cross_attention()
    print("test_cross_attention PASSED")
