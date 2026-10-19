"""
Host-side (numpy) reference for the fp8e4m3 ("hf8") <-> fp16 bit algorithms used by
cm_sdpa_common.hpp's hf8_e4m3_decode_to_f16 (CM in-register decode for the fp8-packed ring K/V --
see KV_RING_FP8E4M3 in cm_sdpa_opt_prefill_multisource.cm).

Ported directly (bit-for-bit, not via any third-party fp8 dtype) from the real OpenVINO intel_gpu
plugin's XeTLA-derived scalar reference, _intel_convert_hf8_to_f16 (decode) and
_f16_to_hf8_universal (encode), in
thirdparty/openvino/src/plugins/intel_gpu/src/kernel_selector/cl_kernels/include/f8_utils.cl
(read-only reference; this is an independent re-implementation, not an include/link).

decode_hf8_to_f16 is the ground truth the CM kernel's hf8_e4m3_decode_to_f16 must match
bit-for-bit (see test_hf8_e4m3_decode.py). encode_f16_to_hf8 exists only to build the fp8 test
fixture from random fp16 inputs -- it does not need to be an independently-certified encoder,
just internally consistent with decode_hf8_to_f16 (both ported from the same spec), since the
attention-kernel test's oracle is "encode then decode" vs. the GPU kernel's own in-register
decode of the SAME encoded bytes (see test_sdpa_prefill_fp8_ring.py).
"""
import numpy as np


def encode_f16_to_hf8(val_f16: np.ndarray, saturate: bool = True) -> np.ndarray:
    """Port of _f16_to_hf8_universal. val_f16: np.float16 array. Returns np.uint8 array."""
    val_f16 = np.asarray(val_f16, dtype=np.float16)
    src = val_f16.view(np.uint16).astype(np.int64)

    src_exp_size, src_mant_size = 5, 10
    src_exp_bias = (1 << (src_exp_size - 1)) - 1  # 15
    src_mant_mask = (1 << src_mant_size) - 1      # 0x3FF
    max_exp_unbiased, min_exp_unbiased, exp_bias = 8, -6, 7
    mant_size = 3
    nan_val, max_val = 0x7F, 0x7E

    src_sign = (src >> (src_exp_size + src_mant_size)) & 0x1
    src_exp = (src >> src_mant_size) & ((1 << src_exp_size) - 1)
    src_exp_unbiased = src_exp - src_exp_bias
    src_mant = src & src_mant_mask

    is_src_inf_nan = src_exp == 0x1F
    is_overflow = (src_exp_unbiased > max_exp_unbiased) | (
        (src_exp_unbiased == max_exp_unbiased) & (src_mant > 0x0340))
    is_zero = src_exp_unbiased < (min_exp_unbiased - mant_size)
    is_denorm = (src_exp_unbiased < min_exp_unbiased) & (~is_zero)

    tail_size = src_mant_size - mant_size  # 7

    # ---- normal-case path (computed for every lane, masked in below) ----
    sticky_n = (src_mant & ((1 << (tail_size - 1)) - 1)) != 0
    lsb_n = (src_mant & (1 << tail_size)) != 0
    rnd_n = (src_mant & (1 << (tail_size - 1))) != 0
    carry_n = (lsb_n & rnd_n) | (rnd_n & sticky_n)
    src_m_n = (src_mant >> tail_size) + carry_n.astype(np.int64)
    src_e_n = src_exp_unbiased + exp_bias
    dst_val_normal = (src_e_n << mant_size) + src_m_n

    # ---- denormal-case path ----
    src_m_d = src_mant | 0x0400
    shift_out_bit = min_exp_unbiased - src_exp_unbiased
    # Only valid (and only used) where is_denorm (shift_out_bit > 0 there); clip elsewhere to
    # keep the shift amount in-range for the vectorized computation (result is discarded there).
    shift_clipped = np.clip(shift_out_bit, 0, 15)
    mask1 = (np.int64(1) << shift_clipped) - 1
    sticky_d = (src_m_d & mask1) != 0
    src_m_d2 = src_m_d >> shift_clipped
    sticky_d = sticky_d | ((src_m_d2 & ((1 << (tail_size - 1)) - 1)) != 0)
    lsb_d = (src_m_d2 & (1 << tail_size)) != 0
    rnd_d = (src_m_d2 & (1 << (tail_size - 1))) != 0
    carry_d = (lsb_d & rnd_d) | (rnd_d & sticky_d)
    dst_val_denorm = (src_m_d2 >> tail_size) + carry_d.astype(np.int64)

    dst_val = np.where(is_denorm, dst_val_denorm, dst_val_normal)
    dst_val = np.where(is_zero, 0, dst_val)
    dst_val = np.where(is_overflow, (max_val if saturate else nan_val), dst_val)
    dst_val = np.where(is_src_inf_nan, nan_val, dst_val)

    result = ((src_sign << (4 + mant_size)) | (dst_val & 0xFF)).astype(np.uint8)
    return result


def decode_hf8_to_f16(raw_u8: np.ndarray) -> np.ndarray:
    """Port of _intel_convert_hf8_to_f16. raw_u8: np.uint8 array. Returns np.float16 array."""
    raw = np.asarray(raw_u8, dtype=np.uint8).astype(np.uint16)

    exp_bias = 7
    sign = (raw >> 7) & 1
    exp = (raw >> 3) & 0xF
    mant = raw & 0x7

    is_nan = (exp == 0xF) & (mant == 0x7)
    is_zero = (exp == 0) & (mant == 0)
    is_denorm = (exp == 0) & (mant != 0)

    lz_count = np.where(mant > 3, 0, np.where(mant > 1, 1, 2)).astype(np.uint16)

    dst_exp_normal = (exp - exp_bias + 15).astype(np.uint16)
    dst_exp_denorm = (exp - exp_bias + 15 - lz_count).astype(np.uint16)
    dst_exp = np.where(is_denorm, dst_exp_denorm, dst_exp_normal).astype(np.uint16)

    dst_mant_denorm = ((mant << (lz_count + 1)) & 0x7).astype(np.uint16)
    dst_mant = np.where(is_denorm, dst_mant_denorm, mant).astype(np.uint16)

    dst_val = ((dst_exp << 10) | (dst_mant << 7)).astype(np.uint16)
    dst_val = np.where(is_zero, 0, dst_val).astype(np.uint16)
    dst_val = np.where(is_nan, 0x7FFF, dst_val).astype(np.uint16)

    bits = ((sign << 15) | dst_val).astype(np.uint16)
    return bits.view(np.float16)


def build_hf8_decode_lut_f16() -> np.ndarray:
    """256-entry half lookup table for cm_sdpa_common.hpp's hf8_e4m3_decode_to_f16_lut (the
    LUT-gather fp8e4m3->fp16 decode, an additional path alongside the bit-manipulation
    hf8_e4m3_decode_to_f16 -- see KV_FP8E4M3 in cm_sdpa_opt_prefill.cm, always LUT-decode via
    sdpa_kernel_lsc_fp8). fp8e4m3 has only 256 possible byte values, so table[byte] ==
    decode_hf8_to_f16(byte) for every byte 0..255 covers the whole domain exactly.

    Laid out as a plain np.float16[256] array (NOT packed into uint32) -- a direct compile+run
    probe against this toolchain's /cm/builtin/cm/cm_lsc.h (not just this repo's own prior gather
    usages, which only ever gather at uint/4-byte granularity) found that cm_ptr_load supports a
    true 2-byte-granularity (DataSize::U16) per-lane-offset gather directly against a half*
    pointer, verified bit-exact on real hardware. That made the uint32-packed/extract-low-half
    layout originally expected to be necessary unneeded."""
    return decode_hf8_to_f16(np.arange(256, dtype=np.uint8))


if __name__ == "__main__":
    # Quick sanity round-trip of a few hand-picked values.
    xs = np.array([0.0, 1.0, -1.0, 2.0, 0.5, 0.125, 448.0, -448.0, 240.0, 1e-3, -1e-3],
                  dtype=np.float16)
    bytes_ = encode_f16_to_hf8(xs)
    back = decode_hf8_to_f16(bytes_)
    for x, b, y in zip(xs, bytes_, back):
        print(f"{float(x):>10.5f} -> byte=0x{b:02x} -> {float(y):>10.5f}")
