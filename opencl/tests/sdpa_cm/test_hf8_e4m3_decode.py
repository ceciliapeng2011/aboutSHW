"""
Isolated unit test for cm_sdpa_common.hpp's hf8_e4m3_decode_to_f16 (the fp8e4m3->fp16 in-register
decode used by the KV_RING_FP8E4M3 ring path, see cm_sdpa_opt_prefill_multisource.cm) against the
full 0-255 byte range, independent of the full attention kernel -- isolates decode-function bugs
from V-path VNNI-repacking-layout bugs (see test_sdpa_prefill_fp8_ring.py for the full-kernel
test).

Oracle: hf8_e4m3_ref.decode_hf8_to_f16, a pure-numpy port of the exact same
_intel_convert_hf8_to_f16 reference algorithm cm_sdpa_common.hpp ports into CM (both independently
verified to match torch.float8_e4m3fn's decode bit-for-bit, see hf8_e4m3_ref.py's module
docstring / __main__ sanity check).

test_hf8_e4m3_decode_lut_full_byte_range below is the equivalent isolated unit test for the
LUT-gather decode alternative, hf8_e4m3_decode_to_f16_lut (see KV_FP8_DECODE_LUT in
cm_sdpa_opt_prefill.cm / test_sdpa_prefill_fp8.py's KV_FP8_DECODE_LUT=1 full-kernel tests) --
same full-byte-range, bit-exact methodology, oracle built from hf8_e4m3_ref.build_hf8_decode_lut_f16
instead.
"""
import functools
import os

import numpy as np

from clops import cl

import test_sdpa_prefill as base
from hf8_e4m3_ref import decode_hf8_to_f16, build_hf8_decode_lut_f16


@functools.cache
def build_hf8_decode_test_kernel():
    src = r'''
#include "cm_sdpa_common.hpp"
extern "C" _GENX_MAIN_ void hf8_decode_test(uchar* in_bytes [[type("svmptr_t")]],
                                             half* out_half [[type("svmptr_t")]],
                                             int n) {
    vector<uchar, 256> raw;
    cm_svm_block_read(reinterpret_cast<svmptr_t>(in_bytes), raw);
    vector<half, 256> out = hf8_e4m3_decode_to_f16<256>(raw.select_all());
    cm_svm_block_write(reinterpret_cast<svmptr_t>(out_half), out);
}
'''
    opts = (
        f'-cmc -Qxcm_jit_option="-abortonspill" -Qxcm_register_file_size=256 '
        f"-I{base.CWD} -I{base.FLASHATTN_DIR} -I{base.PAGEATTEN_DIR} -DCMFLA_SCALE_FACTOR=1.0"
    )
    return cl.kernels(src, opts)


def test_hf8_e4m3_decode_full_byte_range():
    kernel = build_hf8_decode_test_kernel()

    bytes_in = np.arange(256, dtype=np.uint8)
    t_in = cl.tensor(bytes_in)
    t_out = cl.tensor(np.zeros(256, dtype=np.float16))
    kernel.enqueue("hf8_decode_test", [1], [1], t_in, t_out, 256)

    gpu_out = t_out.numpy()
    ref_out = decode_hf8_to_f16(bytes_in)

    # Bit-exact compare (NaN-safe: compares raw bit patterns, so hf8's single NaN encoding
    # (exp==0xF, mant==0x7) decoding to the same fp16 NaN bit pattern on both sides is a match,
    # not a "nan != nan" false mismatch).
    gpu_bits = gpu_out.view(np.uint16)
    ref_bits = ref_out.view(np.uint16)
    mismatches = np.where(gpu_bits != ref_bits)[0]
    assert len(mismatches) == 0, (
        f"{len(mismatches)}/256 byte values decoded differently than the reference: "
        f"{[(int(i), hex(int(gpu_bits[i])), hex(int(ref_bits[i]))) for i in mismatches[:10]]}"
    )
    print("test_hf8_e4m3_decode_full_byte_range PASSED")


@functools.cache
def build_hf8_lut_decode_test_kernel():
    src = r'''
#include "cm_sdpa_common.hpp"
extern "C" _GENX_MAIN_ void hf8_lut_decode_test(uchar* in_bytes [[type("svmptr_t")]],
                                                 half* lut [[type("svmptr_t")]],
                                                 half* out_half [[type("svmptr_t")]],
                                                 int n) {
    vector<uchar, 256> raw;
    cm_svm_block_read(reinterpret_cast<svmptr_t>(in_bytes), raw);
    vector<half, 256> out = hf8_e4m3_decode_to_f16_lut<256>(raw.select_all(), reinterpret_cast<svmptr_t>(lut));
    cm_svm_block_write(reinterpret_cast<svmptr_t>(out_half), out);
}
'''
    opts = (
        f'-cmc -Qxcm_jit_option="-abortonspill" -Qxcm_register_file_size=256 '
        f"-I{base.CWD} -I{base.FLASHATTN_DIR} -I{base.PAGEATTEN_DIR} -DCMFLA_SCALE_FACTOR=1.0"
    )
    return cl.kernels(src, opts)


def test_hf8_e4m3_decode_lut_full_byte_range():
    """LUT-gather counterpart of test_hf8_e4m3_decode_full_byte_range above: same full 0-255 byte
    domain, same bit-exact (NaN-safe) methodology, but exercising hf8_e4m3_decode_to_f16_lut (a
    cm_ptr_load<half,...,DataSize::U16> gather against a 256-entry half LUT) instead of the
    bit-manipulation decode -- see KV_FP8_DECODE_LUT in cm_sdpa_opt_prefill.cm for where this path
    is wired into the full attention kernel."""
    kernel = build_hf8_lut_decode_test_kernel()

    bytes_in = np.arange(256, dtype=np.uint8)
    lut = build_hf8_decode_lut_f16()
    t_in = cl.tensor(bytes_in)
    t_lut = cl.tensor(lut)
    t_out = cl.tensor(np.zeros(256, dtype=np.float16))
    kernel.enqueue("hf8_lut_decode_test", [1], [1], t_in, t_lut, t_out, 256)

    gpu_out = t_out.numpy()
    ref_out = decode_hf8_to_f16(bytes_in)

    gpu_bits = gpu_out.view(np.uint16)
    ref_bits = ref_out.view(np.uint16)
    mismatches = np.where(gpu_bits != ref_bits)[0]
    assert len(mismatches) == 0, (
        f"{len(mismatches)}/256 byte values decoded differently than the reference: "
        f"{[(int(i), hex(int(gpu_bits[i])), hex(int(ref_bits[i]))) for i in mismatches[:10]]}"
    )
    print("test_hf8_e4m3_decode_lut_full_byte_range PASSED")


if __name__ == "__main__":
    test_hf8_e4m3_decode_full_byte_range()
    test_hf8_e4m3_decode_lut_full_byte_range()
    print("ALL PASSED")
