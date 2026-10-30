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
    "TRANSPOSE_ORDER_IDENTITY",
    "TRANSPOSE_ORDER_SWAPPED",
    "encode_k_by_channel",
    "dequant_k_by_channel",
    "encode_by_token",
    "dequant_by_token",
    "encode_cache_by_token",
    "dequant_cache_by_token",
    "encode_cache_k_by_channel",
    "dequant_cache_k_by_channel",
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


TRANSPOSE_ORDER_IDENTITY = 0  # [B, H, S, D]  -- OCL SDPA default order
TRANSPOSE_ORDER_SWAPPED = 1   # [B, S, H, D]  -- {0,2,1,3} heads<->seq swap fused in


def _to_physical_region_a(u8: torch.Tensor, order: int) -> torch.Tensor:
    """u8: [B, Hkv, S, D] uint8. SWAPPED -> token-major/heads-interleaved [B, S, Hkv, D] (the real
    primitive's own convention, see cm_sdpa_opt_prefill_dualsource.cm's module docstring);
    IDENTITY -> left as [B, Hkv, S, D] (heads-major, matching the plain fp16 path's own
    K_TRANSPOSE_ORDER/V_TRANSPOSE_ORDER==IDENTITY layout). Mirrors test_sdpa_prefill.py's
    to_physical, inlined here to keep this module test-infra-independent."""
    if order == TRANSPOSE_ORDER_SWAPPED:
        return u8.transpose(1, 2).contiguous()
    return u8.contiguous()


def _from_physical_region_a(region_a_flat: torch.Tensor, order: int, B: int, Hkv: int, S: int, D: int) -> torch.Tensor:
    """Inverse of _to_physical_region_a. region_a_flat: [B, S*Hkv*D] uint8. Returns [B, Hkv, S, D]."""
    if order == TRANSPOSE_ORDER_SWAPPED:
        return region_a_flat.reshape(B, S, Hkv, D).transpose(1, 2).contiguous()
    return region_a_flat.reshape(B, Hkv, S, D).contiguous()


def encode_cache_by_token(v: torch.Tensor, order: int = TRANSPOSE_ORDER_SWAPPED) -> torch.Tensor:
    """BY_TOKEN quantization for the region-split cache layout `cm_sdpa_opt_prefill.cm`,
    `cm_sdpa_opt_prefill_dualsource.cm`, and `kv_cache_write.cm` all share -- the real OpenVINO
    BlockSparseAttention primitive's design (see those files' module docstrings). The buffer is
    split into two contiguous regions instead of interleaving a widened per-token entry: region A
    (`[kv_len, num_kv_heads, D]` when order==SWAPPED, byte-identical to the plain fp16 path's own
    K_TRANSPOSE_ORDER/V_TRANSPOSE_ORDER convention) holds every token's quantized data; region B
    (`[num_kv_heads, kv_len]` packed (scale, zp) half-pairs, head-major, TOKEN-CONTIGUOUS within a
    head -- always this ONE layout regardless of `order`, matching kv_cache_write.cm's own
    order-agnostic region B) is appended immediately after region A.

    v: [B, Hkv, kv_len, D] half. Returns packed [B, kv_len*Hkv*D + Hkv*kv_len*4] uint8 (flat,
    per-batch region A then region B).
    """
    B, Hkv, S, D = v.shape
    v_u8, v_scale, v_zp = encode_by_token(v)
    region_a = _to_physical_region_a(v_u8, order).reshape(B, S * Hkv * D)
    # Region B: head-major, token-contiguous, interleaved (scale, zp) half-pairs.
    scale_bytes = v_scale.contiguous().view(torch.uint8).reshape(B, Hkv, S, 2)
    zp_bytes = v_zp.contiguous().view(torch.uint8).reshape(B, Hkv, S, 2)
    region_b = torch.empty(B, Hkv, S, 4, dtype=torch.uint8)
    region_b[..., 0:2] = scale_bytes
    region_b[..., 2:4] = zp_bytes
    region_b = region_b.reshape(B, Hkv * S * 4)
    return torch.cat([region_a, region_b], dim=-1)


def dequant_cache_by_token(packed: torch.Tensor, kv_len: int, head_size: int, num_kv_heads: int,
                            order: int = TRANSPOSE_ORDER_SWAPPED) -> torch.Tensor:
    """Inverse of encode_cache_by_token. packed: [B, kv_len*Hkv*D + Hkv*kv_len*4] uint8. Returns
    [B, Hkv, kv_len, D] half."""
    B = packed.shape[0]
    D, Hkv, S = head_size, num_kv_heads, kv_len
    region_a_size = S * Hkv * D
    v_u8 = _from_physical_region_a(packed[:, :region_a_size], order, B, Hkv, S, D)
    region_b = packed[:, region_a_size:].reshape(B, Hkv, S, 4)
    v_scale = region_b[..., 0:2].contiguous().view(torch.half).reshape(B, Hkv, S)
    v_zp = region_b[..., 2:4].contiguous().view(torch.half).reshape(B, Hkv, S)
    return dequant_by_token(v_u8, v_scale, v_zp)


def encode_cache_k_by_channel(k: torch.Tensor, group_size: int = DEFAULT_SUB_BLOCK_SIZE,
                               order: int = TRANSPOSE_ORDER_SWAPPED) -> torch.Tensor:
    """BY_CHANNEL quantization for the region-split cache layout (K only -- V is always
    `encode_cache_by_token`), matching the real OpenVINO BlockSparseAttention primitive's design.
    Region A is the same `[kv_len, num_kv_heads, D]`-when-SWAPPED layout `encode_cache_by_token`
    uses; region B is `[num_kv_heads, kv_len/group_size, D]` packed (scale, zp) half-pairs,
    head-major then GROUP-major then CHANNEL-contiguous within a group (always this ONE layout
    regardless of `order`, same as `encode_cache_by_token`'s region B).

    k: [B, Hkv, kv_len, D] half. Returns packed [B, kv_len*Hkv*D + Hkv*(kv_len/group_size)*D*4]
    uint8.
    """
    B, Hkv, S, D = k.shape
    k_u8, k_scale, k_zp = encode_k_by_channel(k, sub_block_size=group_size)
    num_groups = S // group_size
    region_a = _to_physical_region_a(k_u8, order).reshape(B, S * Hkv * D)
    scale_bytes = k_scale.contiguous().view(torch.uint8).reshape(B, Hkv, num_groups, D, 2)
    zp_bytes = k_zp.contiguous().view(torch.uint8).reshape(B, Hkv, num_groups, D, 2)
    region_b = torch.empty(B, Hkv, num_groups, D, 4, dtype=torch.uint8)
    region_b[..., 0:2] = scale_bytes
    region_b[..., 2:4] = zp_bytes
    region_b = region_b.reshape(B, Hkv * num_groups * D * 4)
    return torch.cat([region_a, region_b], dim=-1)


def dequant_cache_k_by_channel(packed: torch.Tensor, kv_len: int, head_size: int, num_kv_heads: int,
                                group_size: int = DEFAULT_SUB_BLOCK_SIZE,
                                order: int = TRANSPOSE_ORDER_SWAPPED) -> torch.Tensor:
    """Inverse of encode_cache_k_by_channel. packed:
    [B, kv_len*Hkv*D + Hkv*(kv_len/group_size)*D*4] uint8. Returns [B, Hkv, kv_len, D] half."""
    B = packed.shape[0]
    D, Hkv, S = head_size, num_kv_heads, kv_len
    num_groups = S // group_size
    region_a_size = S * Hkv * D
    k_u8 = _from_physical_region_a(packed[:, :region_a_size], order, B, Hkv, S, D)
    region_b = packed[:, region_a_size:].reshape(B, Hkv, num_groups, D, 4)
    k_scale = region_b[..., 0:2].contiguous().view(torch.half).reshape(B, Hkv, num_groups, D)
    k_zp = region_b[..., 2:4].contiguous().view(torch.half).reshape(B, Hkv, num_groups, D)
    return dequant_k_by_channel(k_u8, k_scale, k_zp, sub_block_size=group_size)


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

    v_packed = encode_cache_by_token(v)
    v_cache_rt = dequant_cache_by_token(v_packed, kv_len=32, head_size=8, num_kv_heads=2)
    print("cache by-token (region-split) max abs err:",
          (v.to(torch.float32) - v_cache_rt.to(torch.float32)).abs().max().item())

    k_packed = encode_cache_k_by_channel(k, group_size=16)
    k_cache_rt = dequant_cache_k_by_channel(k_packed, kv_len=32, head_size=8, num_kv_heads=2, group_size=16)
    print("cache by-channel (region-split) max abs err:",
          (k.to(torch.float32) - k_cache_rt.to(torch.float32)).abs().max().item())
