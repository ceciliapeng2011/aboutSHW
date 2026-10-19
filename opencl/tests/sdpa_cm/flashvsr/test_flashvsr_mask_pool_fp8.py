"""
fp8e4m3-packed-input correctness tests for draft_block_mask_pool.cm's K-pooling stage
(K_FP8E4M3 macro) -- Stage 1 of DraftBlockMask. Q is never quantized in this project (only ring
K/V ever is), so this fp8 input path only ever matters for K-pooling; there is no Q counterpart to
test. Stages 2/3 (draft_block_mask_score_softmax.cm, draft_block_mask_topk_mask.cm) only ever
consume the already-pooled half output of this stage regardless of how it was produced, so they
need no fp8-specific test coverage of their own.

Two tests:
  1. test_hf8_lut_decode_pool_copy_full_byte_range -- decode-sanity check of the LUT-gather decode
     function draft_block_mask_pool.cm duplicates from cm_sdpa_common.hpp (NOT reused via
     #include there -- see K_FP8E4M3's header comment in draft_block_mask_pool.cm for why), against
     the full 0-255 byte domain. Mirrors sdpa_cm/test_hf8_e4m3_decode.py's own full-byte-range
     decode-sanity test but exercises THIS file's copy of the function, so a future edit that
     silently diverges the duplicate from the original is caught here directly rather than only by
     luck via the end-to-end test below.
  2. test_pool_fp8_matches_half_bit_exact_* -- the actual Stage-1 K-pooling correctness test:
     host-encodes a random windowed K tensor to fp8e4m3 bytes, runs the new K_FP8E4M3=1 kernel
     variant directly on those raw bytes (via flashvsr_mask_adapter.pool_k_fp8e4m3), and compares
     against running the EXISTING (unmodified) half kernel on the host-decoded-back-to-half version
     of the SAME bytes -- bit-exact (atol=0, rtol=0), not an approximate-quantization-error check,
     matching this project's established fp8 testing methodology (see test_sdpa_prefill_fp8_ring.py).
"""
import functools
import os
import sys

import numpy as np
import torch

from clops import cl

CWD = os.path.dirname(os.path.realpath(__file__))
SDPA_CM_DIR = os.path.dirname(CWD)
if SDPA_CM_DIR not in sys.path:
    sys.path.insert(0, SDPA_CM_DIR)

from hf8_e4m3_ref import build_hf8_decode_lut_f16, decode_hf8_to_f16, encode_f16_to_hf8

import flashvsr_mask_adapter as gpu_mod

torch.manual_seed(17)


@functools.cache
def _build_pool_hf8_decode_sanity_kernel():
    """Compiles draft_block_mask_pool.cm with K_FP8E4M3=1 (dummy tiny NUM_HEADS/HEAD_DIM/
    WINDOW_TOKENS -- the compiled KERNEL_NAME entry point itself is never invoked here) purely to
    pull its local hf8_e4m3_decode_to_f16_lut duplicate into scope, then calls that duplicate
    directly from a second kernel over the full 0-255 byte range -- same methodology as
    sdpa_cm/test_hf8_e4m3_decode.py's test_hf8_e4m3_decode_lut_full_byte_range, but targeting this
    file's own copy of the function instead of cm_sdpa_common.hpp's."""
    src = r'''
#define K_FP8E4M3 1
#define NUM_HEADS 1
#define HEAD_DIM 4
#define WINDOW_TOKENS 1
#define KERNEL_NAME flashvsr_pool_fp8_unused
#include "draft_block_mask_pool.cm"

extern "C" _GENX_MAIN_ void pool_hf8_decode_test(uchar* in_bytes [[type("svmptr_t")]],
                                                  half* lut [[type("svmptr_t")]],
                                                  half* out_half [[type("svmptr_t")]]) {
    vector<uchar, 256> raw;
    cm_svm_block_read(reinterpret_cast<svmptr_t>(in_bytes), raw);
    vector<half, 256> out = hf8_e4m3_decode_to_f16_lut<256>(raw.select_all(), reinterpret_cast<svmptr_t>(lut));
    cm_svm_block_write(reinterpret_cast<svmptr_t>(out_half), out);
}
'''
    opts = f'-cmc -Qxcm_jit_option="-abortonspill" -Qxcm_register_file_size=256 -I{CWD}'
    return cl.kernels(src, opts)


def test_hf8_lut_decode_pool_copy_full_byte_range():
    kernel = _build_pool_hf8_decode_sanity_kernel()

    bytes_in = np.arange(256, dtype=np.uint8)
    lut = build_hf8_decode_lut_f16()
    t_in = cl.tensor(bytes_in)
    t_lut = cl.tensor(lut)
    t_out = cl.tensor(np.zeros(256, dtype=np.float16))
    kernel.enqueue("pool_hf8_decode_test", [1], [1], t_in, t_lut, t_out)

    gpu_out = t_out.numpy()
    ref_out = decode_hf8_to_f16(bytes_in)

    # Bit-exact (NaN-safe: compares raw bit patterns, so hf8's single NaN encoding decoding to the
    # same fp16 NaN bit pattern on both sides is a match, not a "nan != nan" false mismatch).
    gpu_bits = gpu_out.view(np.uint16)
    ref_bits = ref_out.view(np.uint16)
    mismatches = np.where(gpu_bits != ref_bits)[0]
    assert len(mismatches) == 0, (
        f"{len(mismatches)}/256 byte values decoded differently than the reference: "
        f"{[(int(i), hex(int(gpu_bits[i])), hex(int(ref_bits[i]))) for i in mismatches[:10]]}"
    )
    print("test_hf8_lut_decode_pool_copy_full_byte_range PASSED")


def _make_windowed_k_fp8(num_windows, window_tokens, nheads, head_dim, seed):
    g = torch.Generator().manual_seed(seed)
    d_full = nheads * head_dim
    k_w = torch.randn(num_windows, window_tokens, d_full, generator=g, dtype=torch.float32).to(torch.float16)
    k_w_np = k_w.contiguous().numpy()
    fp8_bytes = encode_f16_to_hf8(k_w_np)
    # The oracle for the fp8 kernel path is NOT the original random half data -- it's what the
    # EXISTING half kernel produces from the host-decoded-back-to-half version of the SAME
    # (lossy-encoded) bytes, matching this project's established fp8 testing methodology (encode
    # once, compare both consumers of the identical byte stream against each other).
    k_w_roundtrip_half = decode_hf8_to_f16(fp8_bytes)
    return fp8_bytes, k_w_roundtrip_half


def _run_pool_fp8_vs_half(num_windows, window_tokens, nheads, head_dim, seed):
    fp8_bytes, half_roundtrip = _make_windowed_k_fp8(num_windows, window_tokens, nheads, head_dim, seed)

    t_fp8_out = gpu_mod.pool_k_fp8e4m3(fp8_bytes, num_windows, nheads, head_dim, window_tokens)

    half_kernel = gpu_mod._get_pool_kernel(nheads, head_dim, window_tokens)
    t_half_out = gpu_mod._pool(half_kernel, half_roundtrip, num_windows, nheads, head_dim)

    return t_fp8_out.numpy(), t_half_out.numpy()


def _assert_bit_exact(fp8_out, half_out, label):
    fp8_bits = fp8_out.view(np.uint16)
    half_bits = half_out.view(np.uint16)
    mismatches = np.where(fp8_bits != half_bits)[0]
    assert len(mismatches) == 0, (
        f"{label}: {len(mismatches)}/{fp8_bits.size} pooled elements mismatched (first few: "
        f"{[(int(i), hex(int(fp8_bits[i])), hex(int(half_bits[i]))) for i in mismatches[:10]]})"
    )
    print(f"{label} PASSED ({fp8_bits.size} elements, atol=0 rtol=0 bit-exact)")


def test_pool_fp8_matches_half_bit_exact_small():
    fp8_out, half_out = _run_pool_fp8_vs_half(num_windows=5, window_tokens=8, nheads=3, head_dim=16, seed=21)
    _assert_bit_exact(fp8_out, half_out, "test_pool_fp8_matches_half_bit_exact_small")


def test_pool_fp8_matches_half_bit_exact_flashvsr_config():
    # FlashVSR real production values: head_dim=128, nheads=12, window_tokens=128.
    fp8_out, half_out = _run_pool_fp8_vs_half(num_windows=6, window_tokens=128, nheads=12, head_dim=128, seed=22)
    _assert_bit_exact(fp8_out, half_out, "test_pool_fp8_matches_half_bit_exact_flashvsr_config")


if __name__ == "__main__":
    test_hf8_lut_decode_pool_copy_full_byte_range()
    test_pool_fp8_matches_half_bit_exact_small()
    test_pool_fp8_matches_half_bit_exact_flashvsr_config()
    print("ALL PASSED")
