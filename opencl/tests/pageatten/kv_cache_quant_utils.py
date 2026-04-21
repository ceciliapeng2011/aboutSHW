import torch


DEFAULT_SUB_BLOCK_SIZE = 16


def round_to_even(tensor: torch.Tensor) -> torch.Tensor:
    rounded = torch.floor(tensor + 0.5)
    adjustment = (rounded % 2 != 0) & (torch.abs(tensor - rounded) == 0.5000)
    adjustment = adjustment | (rounded > 255)
    result = rounded - adjustment.to(rounded.dtype)
    return torch.clamp(result, min=0, max=255)


def quant_per_token(kv: torch.Tensor) -> torch.Tensor:
    kv_u8, dq_scale_u8, kv_zp_u8 = quant_per_token_parts(kv)
    return torch.concat((kv_u8, dq_scale_u8, kv_zp_u8), dim=-1)


def quant_per_token_parts(kv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    blk_num, kv_heads, *_ = kv.shape
    kv_max = kv.amax(dim=-1, keepdim=True).to(dtype=torch.float)
    kv_min = kv.amin(dim=-1, keepdim=True).to(dtype=torch.float)
    qrange = (kv_max - kv_min).to(dtype=torch.float)

    u8_max = torch.tensor(255.0, dtype=torch.float)
    u8_min = torch.tensor(0.0, dtype=torch.float)
    u8_range = (u8_max - u8_min).to(dtype=torch.float)

    kv_scale = (u8_range / qrange).to(dtype=torch.float)
    zero_mask = qrange == 0
    if zero_mask.any():
        kv_scale = torch.where(zero_mask, torch.ones_like(kv_scale), kv_scale)
    kv_scale_div = (1.0 / kv_scale).to(dtype=torch.float)
    kv_zp = ((0.0 - kv_min) * kv_scale + u8_min).to(dtype=torch.float)

    kv_u8 = round_to_even(kv * kv_scale + kv_zp).to(dtype=torch.uint8).reshape(blk_num, kv_heads, -1)
    dq_scale = kv_scale_div.to(dtype=torch.half).view(dtype=torch.uint8).reshape(blk_num, kv_heads, -1)
    kv_zp = kv_zp.to(dtype=torch.half).view(dtype=torch.uint8).reshape(blk_num, kv_heads, -1)
    return kv_u8, dq_scale, kv_zp


def dequant_per_token(kv: torch.Tensor, head_size: int, blk_size: int) -> torch.Tensor:
    blk_num, kv_head_num, _ = kv.shape
    kv_u8 = kv[:, :, : head_size * blk_size].to(dtype=torch.float16).reshape(blk_num, kv_head_num, blk_size, head_size)
    kv_scale = kv[:, :, head_size * blk_size : (head_size * blk_size + blk_size * 2)].view(dtype=torch.float16).reshape(
        blk_num,
        kv_head_num,
        blk_size,
        1,
    )
    kv_zp = kv[:, :, (head_size * blk_size + blk_size * 2) : (head_size * blk_size + blk_size * 4)].view(
        dtype=torch.float16
    ).reshape(
        blk_num,
        kv_head_num,
        blk_size,
        1,
    )
    return (kv_u8 - kv_zp) * kv_scale


def quant_per_channel(kv: torch.Tensor, sub_block_size: int = DEFAULT_SUB_BLOCK_SIZE) -> torch.Tensor:
    blk_num, kv_heads, blk_size, head_size = kv.shape
    if blk_size % sub_block_size != 0:
        raise ValueError(f"blk_size ({blk_size}) must be divisible by sub_block_size ({sub_block_size})")

    num_sub_blocks = blk_size // sub_block_size
    kv_sub = kv.reshape(blk_num, kv_heads, num_sub_blocks, sub_block_size, head_size)
    kv_u8, dq_scale_u8, kv_zp_u8 = quant_per_channel_parts(kv_sub, num_sub_blocks, 0)
    return torch.concat((kv_u8, dq_scale_u8, kv_zp_u8), dim=-1)


def quant_per_channel_parts(
    kv: torch.Tensor,
    tail_sub_block: int,
    tail_token: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    blk_num, kv_heads, num_sub_blocks, sub_block_size, head_size = kv.shape
    mask = torch.ones_like(kv, dtype=torch.bool)
    if tail_token:
        mask[:, :, tail_sub_block:tail_sub_block + 1, tail_token:, :] = False

    kv_max = torch.where(mask, kv, torch.tensor(float("-inf"), dtype=torch.float16)).amax(dim=3, keepdim=True)
    kv_min = torch.where(mask, kv, torch.tensor(float("inf"), dtype=torch.float16)).amin(dim=3, keepdim=True)
    qrange = kv_max - kv_min

    u8_max = torch.tensor(255.0, dtype=torch.float)
    u8_min = torch.tensor(0.0, dtype=torch.float)
    u8_range = u8_max - u8_min

    kv_scale = u8_range / qrange.to(dtype=torch.float)
    zero_mask = qrange == 0
    if zero_mask.any():
        kv_scale = torch.where(zero_mask, torch.ones_like(kv_scale), kv_scale)

    kv_scale_div = (1.0 / kv_scale).to(dtype=torch.half)
    kv_zp = ((0.0 - kv_min) * kv_scale + u8_min).to(dtype=torch.half)

    kv_u8 = round_to_even((kv * kv_scale).to(dtype=torch.half) + kv_zp).to(dtype=torch.uint8)
    kv_u8 = kv_u8.reshape(blk_num, kv_heads, -1)
    dq_scale = kv_scale_div.view(dtype=torch.uint8).reshape(blk_num, kv_heads, num_sub_blocks * head_size * 2)
    kv_zp = kv_zp.view(dtype=torch.uint8).reshape(blk_num, kv_heads, num_sub_blocks * head_size * 2)
    return kv_u8, dq_scale, kv_zp


def dequant_per_channel(
    kv: torch.Tensor,
    head_size: int,
    blk_size: int,
    sub_block_size: int = DEFAULT_SUB_BLOCK_SIZE,
) -> torch.Tensor:
    blk_num, kv_head_num, _ = kv.shape
    if blk_size % sub_block_size != 0:
        raise ValueError(f"blk_size ({blk_size}) must be divisible by sub_block_size ({sub_block_size})")

    num_sub_blocks = blk_size // sub_block_size
    quantized_bytes = head_size * blk_size
    metadata_bytes = num_sub_blocks * head_size * 2

    kv_u8 = kv[:, :, : head_size * blk_size].to(dtype=torch.float16).reshape(blk_num, kv_head_num, blk_size, head_size)
    kv_scale = kv[:, :, quantized_bytes : (quantized_bytes + metadata_bytes)].view(dtype=torch.float16).reshape(
        blk_num,
        kv_head_num,
        num_sub_blocks,
        1,
        head_size,
    )
    kv_zp = kv[:, :, (quantized_bytes + metadata_bytes) : (quantized_bytes + metadata_bytes * 2)].view(
        dtype=torch.float16
    ).reshape(
        blk_num,
        kv_head_num,
        num_sub_blocks,
        1,
        head_size,
    )

    kv_u8 = kv_u8.reshape(blk_num, kv_head_num, num_sub_blocks, sub_block_size, head_size)
    return ((kv_u8 - kv_zp) * kv_scale).reshape(blk_num, kv_head_num, blk_size, head_size)