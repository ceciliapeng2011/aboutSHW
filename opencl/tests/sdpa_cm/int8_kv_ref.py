"""
Host-side (torch) reference for the int8 (uint8, affine scale+zero-point) KV-cache quantization
used by cm_int8_kv_decode.hpp's int8_decode_by_channel/int8_decode_by_token -- the SAME affine
quantization opencl/tests/pageatten/kv_cache_quant_utils.py already implements and tests.

This module does not reimplement the quant/dequant math -- it thinly wraps
kv_cache_quant_utils.quant_per_channel_parts/quant_per_token_parts (round-half-to-even, [0,255]
clamp, and the small-qrange guard are all theirs) and reshapes their packed-blob-oriented output
into three separate flat tensors (data, scale, zero-point) -- see this file's own module docstring
in cm_int8_kv_decode.hpp for why sdpa_cm uses three flat tensors instead of pageatten's single
interleaved blob (no on-device cache-update kernel here, so scale/zp never need to travel
physically attached to the data).
"""
import os
import sys

import torch

PAGEATTEN_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "pageatten")
if PAGEATTEN_DIR not in sys.path:
    sys.path.insert(0, PAGEATTEN_DIR)

from kv_cache_quant_utils import DEFAULT_SUB_BLOCK_SIZE, quant_per_channel_parts, quant_per_token_parts

__all__ = [
    "DEFAULT_SUB_BLOCK_SIZE",
    "encode_k_by_channel",
    "dequant_k_by_channel",
    "encode_by_token",
    "dequant_by_token",
]


def encode_k_by_channel(k: torch.Tensor, sub_block_size: int = DEFAULT_SUB_BLOCK_SIZE):
    """k: [B, H, kv_len, D] half. Returns (k_u8, k_scale, k_zp):
    k_u8   [B, H, kv_len, D] uint8
    k_scale, k_zp  [B, H, kv_len // sub_block_size, D] half -- one (scale, zp) pair per
                   (sub_block_size-token sub-block, channel), matching
                   cm_int8_kv_decode.hpp's int8_decode_by_channel.
    """
    B, H, S, D = k.shape
    assert S % sub_block_size == 0, f"kv_len ({S}) must be divisible by sub_block_size ({sub_block_size})"
    num_sub_blocks = S // sub_block_size
    k_sub = k.reshape(B, H, num_sub_blocks, sub_block_size, D)
    k_u8_flat, scale_bytes, zp_bytes = quant_per_channel_parts(k_sub, num_sub_blocks, 0)
    k_u8 = k_u8_flat.reshape(B, H, S, D)
    k_scale = scale_bytes.view(dtype=torch.half).reshape(B, H, num_sub_blocks, D)
    k_zp = zp_bytes.view(dtype=torch.half).reshape(B, H, num_sub_blocks, D)
    return k_u8, k_scale, k_zp


def dequant_k_by_channel(k_u8: torch.Tensor, k_scale: torch.Tensor, k_zp: torch.Tensor,
                          sub_block_size: int = DEFAULT_SUB_BLOCK_SIZE) -> torch.Tensor:
    """Inverse of encode_k_by_channel. Returns [B, H, kv_len, D] half."""
    B, H, S, D = k_u8.shape
    num_sub_blocks = S // sub_block_size
    k_f = k_u8.to(torch.half).reshape(B, H, num_sub_blocks, sub_block_size, D)
    scale_r = k_scale.reshape(B, H, num_sub_blocks, 1, D)
    zp_r = k_zp.reshape(B, H, num_sub_blocks, 1, D)
    return ((k_f - zp_r) * scale_r).reshape(B, H, S, D).to(torch.half)


def encode_by_token(v: torch.Tensor):
    """v: [B, H, kv_len, D] half. Returns (v_u8, v_scale, v_zp):
    v_u8  [B, H, kv_len, D] uint8
    v_scale, v_zp  [B, H, kv_len] half -- one (scale, zp) pair per KV token, matching
                   cm_int8_kv_decode.hpp's int8_decode_by_token. Used for V always, and for K when
                   BY_TOKEN granularity is selected instead of encode_k_by_channel.
    """
    B, H, S, D = v.shape
    v_u8_flat, scale_bytes, zp_bytes = quant_per_token_parts(v)
    v_u8 = v_u8_flat.reshape(B, H, S, D)
    v_scale = scale_bytes.view(dtype=torch.half).reshape(B, H, S)
    v_zp = zp_bytes.view(dtype=torch.half).reshape(B, H, S)
    return v_u8, v_scale, v_zp


def dequant_by_token(v_u8: torch.Tensor, v_scale: torch.Tensor, v_zp: torch.Tensor) -> torch.Tensor:
    """Inverse of encode_by_token. Returns [B, H, kv_len, D] half."""
    B, H, S, D = v_u8.shape
    v_f = v_u8.to(torch.half)
    scale_r = v_scale.reshape(B, H, S, 1)
    zp_r = v_zp.reshape(B, H, S, 1)
    return ((v_f - zp_r) * scale_r).to(torch.half)


if __name__ == "__main__":
    torch.manual_seed(0)
    k = (torch.randn(1, 2, 32, 8) * 3).to(torch.half)
    k_u8, k_scale, k_zp = encode_k_by_channel(k, sub_block_size=16)
    k_rt = dequant_k_by_channel(k_u8, k_scale, k_zp, sub_block_size=16)
    print("by-channel max abs err:", (k.to(torch.float32) - k_rt.to(torch.float32)).abs().max().item())

    v = (torch.randn(1, 2, 32, 8) * 3).to(torch.half)
    v_u8, v_scale, v_zp = encode_by_token(v)
    v_rt = dequant_by_token(v_u8, v_scale, v_zp)
    print("by-token max abs err:", (v.to(torch.float32) - v_rt.to(torch.float32)).abs().max().item())
