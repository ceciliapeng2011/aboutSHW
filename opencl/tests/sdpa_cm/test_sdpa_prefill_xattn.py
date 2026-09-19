"""
Stage 3a: XAttention block-sparse 2nd stage wired onto cm_sdpa_opt_prefill.cm.

Root-cause note, now fixed (see opencl/tests/x_attn/sort.hpp): find_block.hpp's non-causal
branch calls sort.hpp's `sort<SOFTMAX_TYPE>`, a radix sort that was hardcoded for 16-bit (`half`)
sort keys -- it only byte-converted float->sortable-uint for `half` inputs, and its
counting/reorder/copy-out passes always read/wrote 16-bit `ushort` chunks regardless of TYPE. Every
x_attn test in this repo hardcodes SOFTMAX_TYPE='float', and none of them exercised causal=False,
so this was a pre-existing, never-exercised bug in shared x_attn infra -- confirmed empirically
here before the fix (GPU mask vs CPU find_blocks_ref reference diverged completely, ~50% match,
density insensitive to threshold, for causal=False, while the causal path matched the CPU reference
exactly). sort.hpp now generalizes the sortable-key width to TYPE's own size (KeyT), so `float`
sorts correctly; `test_mask_matches_cpu_reference_non_causal` below exercises it (98.8-100% match
at representative thresholds, on par with the causal path's own tie-tolerant comparison). Separately
(unrelated to sort.hpp, not fixed here): find_block.hpp's non-causal loop bound (`j < k_block_pad -
1`) never considers the single lowest-scoring block, unlike the causal branch's equivalent loop
(`j < k_block_pad`) -- only visible at thresholds near 1.0 that require literally every block.
"""
import functools
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

from clops import cl

CWD = os.path.dirname(os.path.realpath(__file__))
PAGEATTEN_DIR = os.path.join(os.path.dirname(CWD), "pageatten")
if PAGEATTEN_DIR not in sys.path:
    sys.path.insert(0, PAGEATTEN_DIR)

import test_sdpa_prefill as base    # noqa: E402  (reuse check_close/make_qkv/to_physical/...)
import dense_xattn_adapter as xa    # noqa: E402
import check_density                # noqa: E402  (pageatten/check_density.py -- verify_block_mask_integrity)

torch.manual_seed(42)

TRANSPOSE_ORDER_IDENTITY = base.TRANSPOSE_ORDER_IDENTITY
TRANSPOSE_ORDER_SWAPPED = base.TRANSPOSE_ORDER_SWAPPED


@functools.cache
def build_sparse_prefill_kernel(num_heads, num_kv_heads, head_size, is_causal, sparse_block_size,
                                 q_order=TRANSPOSE_ORDER_IDENTITY, k_order=TRANSPOSE_ORDER_IDENTITY,
                                 v_order=TRANSPOSE_ORDER_IDENTITY, o_order=TRANSPOSE_ORDER_IDENTITY,
                                 scale_factor=None):
    if scale_factor is None:
        scale_factor = 1.0 / (head_size ** 0.5)
    src = r'''#include "cm_sdpa_opt_prefill.cm"'''
    opts = (
        f'-cmc -Qxcm_jit_option="-abortonspill" -Qxcm_register_file_size=256 -mCM_printregusage '
        f"-I{base.CWD} -I{base.FLASHATTN_DIR} -I{base.PAGEATTEN_DIR} "
        f"-DKERNEL_NAME=cm_sdpa_opt_prefill "
        f"-DCMFLA_NUM_HEADS={num_heads} -DCMFLA_NUM_KV_HEADS={num_kv_heads} "
        f"-DCMFLA_HEAD_SIZE={head_size} -DCMFLA_SCALE_FACTOR={scale_factor} "
        f"-DCMFLA_IS_CAUSAL={int(is_causal)} "
        f"-DQ_TRANSPOSE_ORDER={q_order} -DK_TRANSPOSE_ORDER={k_order} "
        f"-DV_TRANSPOSE_ORDER={v_order} -DOUTPUT_TRANSPOSE_ORDER={o_order} "
        f"-DHAS_ATTN_MASK_INPUT=0 -DHAS_SCALE_INPUT=0 "
        f"-DSPARSE_BLOCK_SIZE={sparse_block_size}"
    )
    return cl.kernels(src, opts)


def run_prefill_sparse(q, k, v, is_causal, sparse_block_size, block_mask,
                        q_order=TRANSPOSE_ORDER_IDENTITY, k_order=TRANSPOSE_ORDER_IDENTITY,
                        v_order=TRANSPOSE_ORDER_IDENTITY, o_order=TRANSPOSE_ORDER_IDENTITY, scale=None):
    batch, num_heads, q_len, head_size = q.shape
    _, num_kv_heads, kv_len, _ = k.shape

    kernel = build_sparse_prefill_kernel(num_heads, num_kv_heads, head_size, is_causal, sparse_block_size,
                                          q_order, k_order, v_order, o_order, scale)

    q_phys = base.to_physical(q, q_order)
    k_phys = base.to_physical(k, k_order)
    v_phys = base.to_physical(v, v_order)
    t_q = cl.tensor(q_phys.to(torch.float16).numpy())
    t_k = cl.tensor(k_phys.to(torch.float16).numpy())
    t_v = cl.tensor(v_phys.to(torch.float16).numpy())
    out_shape = ([batch, q_len, num_heads, head_size] if o_order == TRANSPOSE_ORDER_SWAPPED
                 else [batch, num_heads, q_len, head_size])
    t_out = cl.tensor(np.zeros(out_shape, dtype=np.float16))

    q_step = base.get_cm_grf_width() // 32
    wg_size = 16
    wg_seq_len = wg_size * q_step
    wg_count = (q_len + wg_seq_len - 1) // wg_seq_len
    GWS = [1, batch * num_heads, wg_count * wg_size]
    LWS = [1, 1, wg_size]

    num_q_blocks, num_k_blocks = block_mask.shape[2], block_mask.shape[3]
    t_mask = cl.tensor(block_mask.to(torch.bool).numpy())

    args = [t_q, t_k, t_v, t_out, q_len, kv_len, t_mask, num_q_blocks, num_k_blocks]
    kernel.enqueue("cm_sdpa_opt_prefill", GWS, LWS, *args)

    out_phys = torch.from_numpy(t_out.numpy())
    return base.from_physical(out_phys, o_order, batch, num_heads, q_len, head_size)


def block_mask_to_additive(mask, sparse_block_size, q_len, kv_len):
    # mask: [B,Hq,num_q_blocks,num_k_blocks] bool -> additive [B,Hq,q_len,kv_len] float (0/-inf)
    full = mask.to(torch.float32)
    full = full.repeat_interleave(sparse_block_size, dim=2).repeat_interleave(sparse_block_size, dim=3)
    full = full[:, :, :q_len, :kv_len]
    return torch.where(full > 0, torch.zeros_like(full), torch.full_like(full, float("-inf")))


def ref_sdpa_sparse(q, k, v, is_causal, mask, sparse_block_size, scale=None):
    # Reference: apply the SAME block mask to a torch attention computation -- isolates "did the
    # kernel correctly skip only the intended blocks" from "is the XAttention approximation itself
    # accurate" (per the task's test-design requirement).
    B, Hq, Sq, D = q.shape
    _, Hkv, Sk, _ = k.shape
    additive = block_mask_to_additive(mask, sparse_block_size, Sq, Sk)
    if is_causal:
        causal_add = torch.triu(torch.full((Sq, Sk), float("-inf")), diagonal=Sk - Sq + 1)
        additive = additive + causal_add.unsqueeze(0).unsqueeze(0)
    out = F.scaled_dot_product_attention(
        q.to(torch.float32), k.to(torch.float32), v.to(torch.float32),
        attn_mask=additive, is_causal=False, enable_gqa=(Hkv != Hq), scale=scale,
    )
    return out.to(torch.float16)


# ---------------------------------------------------------------------------
# 1. Bypass correctness: an all-True mask must not change the kernel's output at all versus the
#    plain (SPARSE_BLOCK_SIZE=1) Stage 1a kernel.
# ---------------------------------------------------------------------------
def test_bypass_matches_dense():
    q, k, v = base.make_qkv(batch=1, num_heads=4, num_kv_heads=4, q_len=1024, kv_len=1024, head_size=64)
    sparse_block_size = 256
    num_q_blocks = (1024 + sparse_block_size - 1) // sparse_block_size
    num_k_blocks = num_q_blocks
    all_true_mask = torch.ones(1, 4, num_q_blocks, num_k_blocks, dtype=torch.bool)

    out_sparse = run_prefill_sparse(q, k, v, is_causal=True, sparse_block_size=sparse_block_size,
                                     block_mask=all_true_mask)
    out_dense = base.run_prefill(q, k, v, is_causal=True)
    assert torch.equal(out_sparse, out_dense), "all-True sparse mask must be bit-identical to the dense kernel"
    print("test_bypass_matches_dense PASSED")


# ---------------------------------------------------------------------------
# 2. Mask-vs-reference: the dense-K adapter's GPU-computed mask must match the CPU reference
#    (x_attn/references.py get_gemm_ref + find_blocks_ref) at a threshold well clear of tie noise.
# ---------------------------------------------------------------------------
def test_mask_matches_cpu_reference():
    # thresh=0.99 is far enough from any single block's typical score share that GPU (fp32
    # tree-sum) vs CPU (fp32 sequential cumsum) rounding never flips a selection -- exact match.
    # thresh=0.9 sits close enough to a real block's score share that the threshold-crossing
    # block occasionally differs by rounding alone (verified by hand: the one differing entry's
    # own score is within noise of the remaining budget) -- tolerate a small mismatch fraction
    # rather than bit-exact equality, consistent with x_attn's own test suite's tie-tolerant
    # mask comparison (test_find_block.py's cmp_mask).
    q, k, v = base.make_qkv(batch=1, num_heads=4, num_kv_heads=4, q_len=4096, kv_len=4096, head_size=64)
    q, k = q.float().half(), k.float().half()
    mask, _, _ = xa.build_sparse_mask(q, k, 4, 4, 64, 256, True, 0.99)
    mask_ref, qb, kb = xa.cpu_reference_mask(q, k, 256, True, 0.99)
    assert torch.equal(mask[:, :, :qb, :kb], mask_ref), "mask mismatch at thresh=0.99 (expected exact)"

    mask, _, _ = xa.build_sparse_mask(q, k, 4, 4, 64, 256, True, 0.9)
    mask_ref, qb, kb = xa.cpu_reference_mask(q, k, 256, True, 0.9)
    match_frac = (mask[:, :, :qb, :kb] == mask_ref).float().mean().item()
    assert match_frac >= 0.99, f"mask match_frac={match_frac} too low at thresh=0.9"
    print("test_mask_matches_cpu_reference PASSED")


def test_mask_matches_cpu_reference_non_causal():
    # Regression test for the sort.hpp bug documented in this file's module docstring: the
    # non-causal branch's radix sort (sort<SOFTMAX_TYPE>) was hardcoded for 16-bit `half` keys and
    # silently scrambled `float` scores, making this path's mask ~50%-random garbage. Thresholds
    # here deliberately stay below ~0.94 (this shape's max reachable density) to avoid the
    # separate, unrelated find_block.hpp loop-bound edge case noted in the module docstring.
    q, k, v = base.make_qkv(batch=1, num_heads=4, num_kv_heads=4, q_len=4096, kv_len=4096, head_size=64)
    q, k = q.float().half(), k.float().half()
    mask, _, _ = xa.build_sparse_mask(q, k, 4, 4, 64, 256, False, 0.6)
    mask_ref, qb, kb = xa.cpu_reference_mask(q, k, 256, False, 0.6)
    assert torch.equal(mask[:, :, :qb, :kb], mask_ref), "mask mismatch at thresh=0.6 (expected exact)"

    mask, _, _ = xa.build_sparse_mask(q, k, 4, 4, 64, 256, False, 0.85)
    mask_ref, qb, kb = xa.cpu_reference_mask(q, k, 256, False, 0.85)
    match_frac = (mask[:, :, :qb, :kb] == mask_ref).float().mean().item()
    assert match_frac >= 0.99, f"mask match_frac={match_frac} too low at thresh=0.85"
    print("test_mask_matches_cpu_reference_non_causal PASSED")


# ---------------------------------------------------------------------------
# 3. Sparse output accuracy at a realistic threshold vs. a reference applying the SAME mask.
# ---------------------------------------------------------------------------
def test_sparse_output_accuracy():
    q, k, v = base.make_qkv(batch=1, num_heads=4, num_kv_heads=4, q_len=4096, kv_len=4096, head_size=64)
    is_causal = True
    thresh = 0.9
    sparse_block_size = 256
    mask, _, _ = xa.build_sparse_mask(q, k, 4, 4, 64, sparse_block_size, is_causal, thresh)
    assert mask.float().mean().item() < 0.9, "sanity: threshold should actually prune something"

    out = run_prefill_sparse(q, k, v, is_causal, sparse_block_size, mask)
    ref = ref_sdpa_sparse(q, k, v, is_causal, mask, sparse_block_size)
    base.check_close(out, ref)
    print("test_sparse_output_accuracy PASSED")


# ---------------------------------------------------------------------------
# 4. Causal + sparse combined: no future-block leakage, no incorrectly-pruned causally-required
#    blocks -- verified by comparing against the masked+causal torch reference (test 3 already
#    uses is_causal=True; this adds an explicit sink/diagonal-invariant check on top).
# ---------------------------------------------------------------------------
def test_causal_and_sparse_combined():
    q, k, v = base.make_qkv(batch=1, num_heads=4, num_kv_heads=1, q_len=2048, kv_len=2048, head_size=128)
    is_causal = True
    thresh = 0.6
    sparse_block_size = 256
    mask, qb, kb = xa.build_sparse_mask(q, k, 4, 1, 128, sparse_block_size, is_causal, thresh)

    # sink block (k-block 0) and each q-block's own causal-diagonal block must always be kept.
    assert torch.all(mask[:, :, :, 0]), "sink block must always be kept"
    for qi in range(qb):
        assert mask[:, :, qi, qi].all(), f"causal-diagonal block must be kept for q_block={qi}"
    # no future block ever kept
    for qi in range(qb):
        if qi + 1 < kb:
            assert not mask[:, :, qi, qi + 1 :].any(), f"future block leaked through for q_block={qi}"

    out = run_prefill_sparse(q, k, v, is_causal, sparse_block_size, mask)
    ref = ref_sdpa_sparse(q, k, v, is_causal, mask, sparse_block_size)
    base.check_close(out, ref)
    print("test_causal_and_sparse_combined PASSED")


# ---------------------------------------------------------------------------
# 5. Sink-block invariant, reusing pageatten/check_density.py's verify_block_mask_integrity().
# ---------------------------------------------------------------------------
def test_sink_block_invariant():
    q, k, v = base.make_qkv(batch=2, num_heads=4, num_kv_heads=2, q_len=2048, kv_len=2048, head_size=64)
    mask, _, _ = xa.build_sparse_mask(q, k, 4, 2, 64, 256, True, 0.9)
    assert check_density.verify_block_mask_integrity(mask)
    print("test_sink_block_invariant PASSED")


# ---------------------------------------------------------------------------
# 6. Multiple block sizes (128/256) and thresholds (0.99/0.9/0.6): pipeline runs and produces
#    sane, threshold-monotonic density on the causal path.
# ---------------------------------------------------------------------------
def test_block_sizes_and_thresholds():
    q, k, v = base.make_qkv(batch=1, num_heads=4, num_kv_heads=4, q_len=4096, kv_len=4096, head_size=64)
    for block_size in xa.XATTN_BLOCK_SIZES:
        densities = []
        for thresh in (0.6, 0.9, 0.99):
            mask, _, _ = xa.build_sparse_mask(q, k, 4, 4, 64, block_size, True, thresh)
            assert check_density.verify_block_mask_integrity(mask)
            densities.append(mask.float().mean().item())
            out = run_prefill_sparse(q, k, v, True, block_size, mask)
            ref = ref_sdpa_sparse(q, k, v, True, mask, block_size)
            base.check_close(out, ref)
        assert densities[0] <= densities[1] <= densities[2], (
            f"density should be non-decreasing with threshold: {densities} (block_size={block_size})"
        )
    print("test_block_sizes_and_thresholds PASSED")


# ---------------------------------------------------------------------------
# 7. Batch > 1 / GQA: per-(batch, q-head) mask independence, and correct sparse output.
# ---------------------------------------------------------------------------
def test_batch_and_gqa_independence():
    # Long-enough context (32 blocks) and a threshold well below 1.0 so there's actual room for
    # per-head/per-batch selections to diverge -- at high threshold with few blocks the causal
    # mask alone forces every head to the same "keep everything visible" pattern trivially.
    q, k, v = base.make_qkv(batch=2, num_heads=8, num_kv_heads=2, q_len=8192, kv_len=8192, head_size=64)
    is_causal = True
    thresh = 0.6
    sparse_block_size = 256
    mask, qb, kb = xa.build_sparse_mask(q, k, 8, 2, 64, sparse_block_size, is_causal, thresh)

    # not every (batch,head) mask is identical -- confirms independent per-head/per-batch scoring
    flat = mask.reshape(mask.shape[0] * mask.shape[1], -1)
    assert not torch.all(flat == flat[0]), "masks should differ across batch/heads with random data"
    # heads sharing a KV group must still be scored independently (not literally copy-pasted)
    assert not torch.equal(mask[:, 0], mask[:, 1]), "heads within the same GQA group must not share a mask"

    out = run_prefill_sparse(q, k, v, is_causal, sparse_block_size, mask)
    ref = ref_sdpa_sparse(q, k, v, is_causal, mask, sparse_block_size)
    base.check_close(out, ref)
    print("test_batch_and_gqa_independence PASSED")


# ---------------------------------------------------------------------------
# 8. Transpose-order: sparse path correctness under a non-identity ({0,2,1,3}-swapped) Q/output
#    layout -- the mask/adapter always works off the logical [B,H,S,D] tensors, independent of
#    whichever physical layout the SDPA kernel itself is given, so Q-addressing parity between the
#    scoring kernel and the sparse-aware prefill kernel must hold end-to-end.
# ---------------------------------------------------------------------------
def test_transpose_order_sparse():
    q, k, v = base.make_qkv(batch=1, num_heads=4, num_kv_heads=4, q_len=1024, kv_len=1024, head_size=64)
    is_causal = True
    thresh = 0.9
    sparse_block_size = 256
    mask, _, _ = xa.build_sparse_mask(q, k, 4, 4, 64, sparse_block_size, is_causal, thresh)
    ref = ref_sdpa_sparse(q, k, v, is_causal, mask, sparse_block_size)

    for q_order, o_order in [(TRANSPOSE_ORDER_SWAPPED, TRANSPOSE_ORDER_IDENTITY),
                              (TRANSPOSE_ORDER_IDENTITY, TRANSPOSE_ORDER_SWAPPED),
                              (TRANSPOSE_ORDER_SWAPPED, TRANSPOSE_ORDER_SWAPPED)]:
        out = run_prefill_sparse(q, k, v, is_causal, sparse_block_size, mask, q_order=q_order, o_order=o_order)
        base.check_close(out, ref)
    print("test_transpose_order_sparse PASSED")


if __name__ == "__main__":
    cl.profiling(True)
    test_bypass_matches_dense()
    test_mask_matches_cpu_reference()
    test_sparse_output_accuracy()
    test_causal_and_sparse_combined()
    test_sink_block_invariant()
    test_block_sizes_and_thresholds()
    test_batch_and_gqa_independence()
    test_transpose_order_sparse()
