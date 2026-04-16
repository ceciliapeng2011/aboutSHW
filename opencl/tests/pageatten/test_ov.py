from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from turboquant_cm import CompressedKVCache_Update_CM


def get_tensor(name: str, dtype=np.float16) -> torch.Tensor:
    with open(name, "rb") as f:
        data = f.read()
    np_data = np.frombuffer(data, dtype=dtype).copy()
    return torch.from_numpy(np_data)


def _find_one(base: Path, pattern: str) -> Path:
    cands = sorted(base.glob(pattern))
    if not cands:
        raise FileNotFoundError(f"No file matches pattern: {pattern} under {base}")
    if len(cands) > 1:
        raise RuntimeError(f"Multiple files match pattern {pattern}: {cands}")
    return cands[0]


def _find_optional(base: Path, pattern: str) -> Path | None:
    cands = sorted(base.glob(pattern))
    if not cands:
        return None
    if len(cands) > 1:
        raise RuntimeError(f"Multiple files match pattern {pattern}: {cands}")
    return cands[0]


def _infer_mode_and_bits(k_head_size: int, v_head_size: int, k_token_bytes: int, v_token_bytes: int) -> tuple[int, str]:
    bits = None
    for b in range(1, 9):
        if ((k_head_size * b + 7) // 8) + 2 == k_token_bytes:
            bits = b
            break
    if bits is None:
        raise ValueError(
            f"Key cache token_bytes={k_token_bytes} does not match TurboQuant layout for head_size={k_head_size}."
        )

    by_token_bytes = v_head_size + 4
    turboquant_v_bytes = ((v_head_size * bits + 7) // 8) + 2
    if v_token_bytes == by_token_bytes:
        value_mode = "by_token"
    elif v_token_bytes == turboquant_v_bytes:
        value_mode = "turboquant"
    else:
        raise ValueError(
            f"Unsupported value cache token_bytes={v_token_bytes}; expected {by_token_bytes} (by_token) "
            f"or {turboquant_v_bytes} (turboquant)."
        )

    return bits, value_mode


def _reshape_tokens_with_possible_padding(x: torch.Tensor, batch_tokens: int, expected_width: int) -> torch.Tensor:
    if x.numel() == batch_tokens * expected_width:
        return x.reshape(batch_tokens, expected_width)
    if x.numel() % batch_tokens != 0:
        raise ValueError("Input dump size is incompatible with subsequence metadata")
    row_width = x.numel() // batch_tokens
    if row_width < expected_width:
        raise ValueError(f"Input row width {row_width} is smaller than expected {expected_width}")
    # Some dumps store extra concatenated planes per token row; keep the trailing logical chunk.
    return x.reshape(batch_tokens, row_width)[:, row_width - expected_width :].contiguous()


def _assert_mismatch_ratio_le(a: torch.Tensor, b: torch.Tensor, max_ratio: float, name: str) -> None:
    if a.shape != b.shape:
        raise AssertionError(f"{name} shape mismatch: {a.shape} vs {b.shape}")
    diff = (a != b).sum().item()
    total = a.numel()
    ratio = diff / max(total, 1)
    assert ratio <= max_ratio, f"{name} mismatch ratio {ratio:.6f} exceeds {max_ratio:.6f} ({diff}/{total})"


def _iter_update_positions(
    past_lens: torch.Tensor,
    subsequence_begins: torch.Tensor,
    block_indices: torch.Tensor,
    block_indices_begins: torch.Tensor,
    block_size: int,
):
    n_seqs = int(subsequence_begins.numel()) - 1
    for seq in range(n_seqs):
        tok_begin = int(subsequence_begins[seq].item())
        tok_end = int(subsequence_begins[seq + 1].item())
        n_tok = tok_end - tok_begin
        past = int(past_lens[seq].item())
        blk_begin = int(block_indices_begins[seq].item())
        for ti in range(n_tok):
            token_idx = tok_begin + ti
            pos = past + ti
            blk = int(block_indices[blk_begin + (pos // block_size)].item())
            off = pos % block_size
            yield seq, token_idx, pos, blk, off


def _iter_sequence_positions(
    seq: int,
    positions: range,
    block_indices: torch.Tensor,
    block_indices_begins: torch.Tensor,
    block_size: int,
):
    blk_begin = int(block_indices_begins[seq].item())
    for pos in positions:
        blk = int(block_indices[blk_begin + (pos // block_size)].item())
        off = pos % block_size
        yield pos, blk, off


def _unpack_4bit_indices(packed_u8: torch.Tensor, head_size: int) -> torch.Tensor:
    lo = packed_u8 & 0x0F
    hi = (packed_u8 >> 4) & 0x0F
    out = torch.empty((*packed_u8.shape[:-1], head_size), dtype=torch.uint8)
    out[..., 0::2] = lo
    out[..., 1::2] = hi
    return out


def _decode_key_token_fp16(cache: torch.Tensor, blk: int, off: int, updater: CompressedKVCache_Update_CM) -> torch.Tensor:
    # Layout is [block_size * packed_bytes][block_size * fp16_norm] (not per-token interleaved).
    packed_s = slice(off * updater.k_packed_bytes, (off + 1) * updater.k_packed_bytes)
    norms_base = updater.block_size * updater.k_packed_bytes
    norm_s = slice(norms_base + off * 2, norms_base + (off + 1) * 2)
    packed = cache[blk, :, packed_s].contiguous()
    norm_u8 = cache[blk, :, norm_s].contiguous()
    idx = _unpack_4bit_indices(packed, updater.k_head_size)
    norms = norm_u8.view(torch.float16).reshape(-1)
    return updater.tq_k.ref.dequantize({"idx": idx, "norms": norms}).to(torch.float16)


def _decode_value_token_fp16(cache: torch.Tensor, blk: int, off: int, updater: CompressedKVCache_Update_CM) -> torch.Tensor:
    if updater.value_cache_mode:
        start = off * updater.v_token_bytes
        end = (off + 1) * updater.v_token_bytes
        tok = cache[blk, :, start:end].contiguous()
        packed = tok[:, :updater.v_packed_bytes]
        norm_u8 = tok[:, updater.v_packed_bytes : updater.v_packed_bytes + 2].contiguous()
        idx = _unpack_4bit_indices(packed, updater.v_head_size)
        norms = norm_u8.view(torch.float16).reshape(-1)
        return updater.tq_v.ref.dequantize({"idx": idx, "norms": norms}).to(torch.float16)

    data_bytes = updater.block_size * updater.v_head_size
    scale_base = data_bytes
    zp_base = data_bytes + updater.block_size * 2

    data_s = slice(off * updater.v_head_size, (off + 1) * updater.v_head_size)
    sc_s = slice(scale_base + off * 2, scale_base + (off + 1) * 2)
    zp_s = slice(zp_base + off * 2, zp_base + (off + 1) * 2)

    q_u8 = cache[blk, :, data_s].to(torch.float32)
    dq_scale = cache[blk, :, sc_s].contiguous().view(torch.float16).reshape(-1, 1).to(torch.float32)
    zp = cache[blk, :, zp_s].contiguous().view(torch.float16).reshape(-1, 1).to(torch.float32)
    return ((q_u8 - zp) * dq_scale).to(torch.float16)


def _cache_vs_reference_cosines(
    key_cache_u8: torch.Tensor,
    value_cache_u8: torch.Tensor,
    update_slots: list[tuple[int, int]],
    key_gt_f: torch.Tensor,
    value_gt_f: torch.Tensor,
    updater: CompressedKVCache_Update_CM,
) -> tuple[float, float]:
    k_dec = []
    v_dec = []
    for blk, off in update_slots:
        k_dec.append(_decode_key_token_fp16(key_cache_u8, blk, off, updater).reshape(-1).float())
        v_dec.append(_decode_value_token_fp16(value_cache_u8, blk, off, updater).reshape(-1).float())

    k_dec_f = torch.cat(k_dec)
    v_dec_f = torch.cat(v_dec)

    k_dec_f = torch.nan_to_num(k_dec_f, nan=0.0, posinf=0.0, neginf=0.0)
    v_dec_f = torch.nan_to_num(v_dec_f, nan=0.0, posinf=0.0, neginf=0.0)
    k_gt_f = torch.nan_to_num(key_gt_f, nan=0.0, posinf=0.0, neginf=0.0)
    v_gt_f = torch.nan_to_num(value_gt_f, nan=0.0, posinf=0.0, neginf=0.0)

    k_cos = F.cosine_similarity(k_dec_f, k_gt_f, dim=0, eps=1e-8).item()
    v_cos = F.cosine_similarity(v_dec_f, v_gt_f, dim=0, eps=1e-8).item()
    return k_cos, v_cos


def _build_reference_kv(
    key_current: torch.Tensor,
    value_current: torch.Tensor,
    key_cache_input: torch.Tensor,
    value_cache_input: torch.Tensor,
    past_lens: torch.Tensor,
    subsequence_begins: torch.Tensor,
    block_indices: torch.Tensor,
    block_indices_begins: torch.Tensor,
    updater: CompressedKVCache_Update_CM,
) -> tuple[list[tuple[int, int]], torch.Tensor, torch.Tensor]:
    if int(past_lens.max().item()) == 0:
        key_ref = key_current
        value_ref = value_current
        ref_subsequence_begins = subsequence_begins
    else:
        n_seqs = int(subsequence_begins.numel()) - 1
        key_rows: list[torch.Tensor] = []
        value_rows: list[torch.Tensor] = []
        ref_begins = [0]

        for seq in range(n_seqs):
            cur_begin = int(subsequence_begins[seq].item())
            cur_end = int(subsequence_begins[seq + 1].item())
            cur_n = cur_end - cur_begin
            past = int(past_lens[seq].item())

            for _pos, blk, off in _iter_sequence_positions(
                seq, range(past), block_indices, block_indices_begins, updater.block_size
            ):
                key_rows.append(_decode_key_token_fp16(key_cache_input, blk, off, updater).reshape(-1).to(torch.float16))
                value_rows.append(_decode_value_token_fp16(value_cache_input, blk, off, updater).reshape(-1).to(torch.float16))

            for ti in range(cur_n):
                key_rows.append(key_current[cur_begin + ti].reshape(-1).to(torch.float16))
                value_rows.append(value_current[cur_begin + ti].reshape(-1).to(torch.float16))

            ref_begins.append(ref_begins[-1] + past + cur_n)

        key_ref = torch.stack(key_rows, dim=0).contiguous()
        value_ref = torch.stack(value_rows, dim=0).contiguous()
        ref_subsequence_begins = torch.tensor(ref_begins, dtype=torch.int32)

    # Build per-update references once so both output caches can reuse them.
    update_slots: list[tuple[int, int]] = []
    key_gt: list[torch.Tensor] = []
    value_gt: list[torch.Tensor] = []
    for seq, _tok, pos, blk, off in _iter_update_positions(
        past_lens, subsequence_begins, block_indices, block_indices_begins, updater.block_size
    ):
        update_slots.append((blk, off))
        ref_row = int(ref_subsequence_begins[seq].item()) + pos
        key_gt.append(key_ref[ref_row].reshape(-1).float())
        value_gt.append(value_ref[ref_row].reshape(-1).float())

    key_gt_f = torch.cat(key_gt)
    value_gt_f = torch.cat(value_gt)
    return update_slots, key_gt_f, value_gt_f


def test_kvcache_update():
    base = Path("/home/ceciliapeng/openvino/dump_debug_bin_PagedAttentionExtension_28853")
    pa_node_name = "PagedAttentionExtension_28853"
    network_tag = "program1_network1_0_"
    if not base.exists():
        pytest.skip(f"Dump directory not found: {base}")

    key_file = _find_one(base, f"{network_tag}*_{pa_node_name}_src1__f16*__bfyx.bin")
    value_file = _find_one(base, f"{network_tag}*_{pa_node_name}_src2__f16*__bfyx.bin")

    key_cache_in_file = _find_one(base, f"{network_tag}*_{pa_node_name}_src3__i8*__bfyx.bin")
    value_cache_in_file = _find_one(base, f"{network_tag}*_{pa_node_name}_src4__i8*__bfyx.bin")

    updated_key_cache_from_ov_file = _find_one(base, f"{network_tag}*_{pa_node_name}_updated_src_3__i8*__bfyx.bin")
    updated_value_cache_from_ov_file = _find_one(base, f"{network_tag}*_{pa_node_name}_updated_src_4__i8*__bfyx.bin")

    past_lens_file = _find_one(base, f"{network_tag}*_{pa_node_name}_src5__i32*__bfyx.bin")
    subseq_file = _find_one(base, f"{network_tag}*_{pa_node_name}_src6__i32*__bfyx.bin")
    block_indices_file = _find_one(base, f"{network_tag}*_{pa_node_name}_src7__i32*__bfyx.bin")
    block_begins_file = _find_one(base, f"{network_tag}*_{pa_node_name}_src8__i32*__bfyx.bin")

    key = get_tensor(str(key_file), np.float16)
    value = get_tensor(str(value_file), np.float16)

    key_cache_input = get_tensor(str(key_cache_in_file), np.int8)
    value_cache_input = get_tensor(str(value_cache_in_file), np.int8)

    past_lens = get_tensor(str(past_lens_file), np.int32).reshape(-1).to(torch.int32)
    subsequence_begins = get_tensor(str(subseq_file), np.int32).reshape(-1).to(torch.int32)
    block_indices = get_tensor(str(block_indices_file), np.int32).reshape(-1).to(torch.int32)
    block_indices_begins = get_tensor(str(block_begins_file), np.int32).reshape(-1).to(torch.int32)

    batch_tokens = int(subsequence_begins[-1].item())
    if batch_tokens <= 0:
        raise ValueError("Invalid subsequence_begins: empty token span")

    token_shape_k = key_cache_in_file.name.split("__")[-2]
    token_shape_v = value_cache_in_file.name.split("__")[-2]
    _, num_kv_heads_k, block_size_k, k_token_bytes = [int(x) for x in token_shape_k.split("_")]
    _, num_kv_heads_v, block_size_v, v_token_bytes = [int(x) for x in token_shape_v.split("_")]
    if num_kv_heads_k != num_kv_heads_v or block_size_k != block_size_v:
        raise ValueError("Key/value cache dump shapes mismatch")

    num_kv_heads = num_kv_heads_k
    kv_block_size = block_size_k

    if key.numel() % batch_tokens != 0:
        raise ValueError("Input key dump size is incompatible with subsequence metadata")
    key_row_width = key.numel() // batch_tokens
    if key_row_width % num_kv_heads != 0:
        raise ValueError("Input key width is not divisible by num_kv_heads")
    k_head_size = key_row_width // num_kv_heads

    # infer v_head_size from value token layout first (supports this dump where src2 row width is padded)
    v_head_size_guess = v_token_bytes - 4
    bits, value_cache_mode = _infer_mode_and_bits(k_head_size, v_head_size_guess, k_token_bytes, v_token_bytes)
    if value_cache_mode == "by_token":
        v_head_size = v_head_size_guess
    else:
        if bits <= 0:
            raise ValueError("Invalid inferred bits")
        if (v_token_bytes - 2) * 8 % bits != 0:
            raise ValueError("Cannot infer v_head_size for turboquant value mode")
        v_head_size = ((v_token_bytes - 2) * 8) // bits

    print(
        f"bits={bits}, value_cache_mode={value_cache_mode}, v_head_size={v_head_size}, "
        f"k_head_size={k_head_size}, num_kv_heads_k={num_kv_heads_k}, "
        f"block_size_k={block_size_k}, k_token_bytes={k_token_bytes}"
    )

    key = _reshape_tokens_with_possible_padding(key, batch_tokens, num_kv_heads * k_head_size).to(torch.float16)
    value = _reshape_tokens_with_possible_padding(value, batch_tokens, num_kv_heads * v_head_size).to(torch.float16)

    key_cache = (
        key_cache_input.reshape(-1, num_kv_heads, kv_block_size, k_token_bytes)
        .to(torch.uint8)
        .reshape(-1, num_kv_heads, kv_block_size * k_token_bytes)
        .contiguous()
    )
    value_cache = (
        value_cache_input.reshape(-1, num_kv_heads, kv_block_size, v_token_bytes)
        .to(torch.uint8)
        .reshape(-1, num_kv_heads, kv_block_size * v_token_bytes)
        .contiguous()
    )
    
    print(f"Input key shape: {key.shape}, value shape: {value.shape}")
    print(f"Input key_cache shape: {key_cache.shape}, value_cache shape: {value_cache.shape}")
    print(f"Past lens: {past_lens.tolist()}")

    updater = CompressedKVCache_Update_CM(
        num_kv_heads=num_kv_heads,
        k_head_size=k_head_size,
        v_head_size=v_head_size,
        block_size=kv_block_size,
        bits=bits,
        value_cache_mode=value_cache_mode,
    )

    out_key_cache, out_value_cache = updater(
        key,
        value,
        key_cache,
        value_cache,
        past_lens.tolist(),
        subsequence_begins.tolist(),
        block_indices.tolist(),
        block_indices_begins.tolist(),
        n_repeats=1,
    )

    updated_key_cache_from_ov = (
        get_tensor(str(updated_key_cache_from_ov_file), np.int8)
        .reshape(-1, num_kv_heads, kv_block_size, k_token_bytes)
        .to(torch.uint8)
        .reshape(-1, num_kv_heads, kv_block_size * k_token_bytes)
        .contiguous()
    )
    updated_value_cache_from_ov = (
        get_tensor(str(updated_value_cache_from_ov_file), np.int8)
        .reshape(-1, num_kv_heads, kv_block_size, v_token_bytes)
        .to(torch.uint8)
        .reshape(-1, num_kv_heads, kv_block_size * v_token_bytes)
        .contiguous()
    )

    update_slots, key_gt_f, value_gt_f = _build_reference_kv(
        key,
        value,
        key_cache,
        value_cache,
        past_lens,
        subsequence_begins,
        block_indices,
        block_indices_begins,
        updater,
    )

    out_k_cos, out_v_cos = _cache_vs_reference_cosines(
        out_key_cache,
        out_value_cache,
        update_slots,
        key_gt_f,
        value_gt_f,
        updater,
    )
    ov_k_cos, ov_v_cos = _cache_vs_reference_cosines(
        updated_key_cache_from_ov,
        updated_value_cache_from_ov,
        update_slots,
        key_gt_f,
        value_gt_f,
        updater,
    )
    print(f"cos(out_key_cache vs ref_fp16)={out_k_cos:.6f}, cos(out_value_cache vs ref_fp16)={out_v_cos:.6f}")
    print(
        f"cos(updated_key_cache_from_ov vs ref_fp16)={ov_k_cos:.6f}, "
        f"cos(updated_value_cache_from_ov vs ref_fp16)={ov_v_cos:.6f}"
    )
    assert out_k_cos > 0.90, f"Low cosine similarity for key cache: {out_k_cos:.6f}"
    assert out_v_cos > 0.90, f"Low cosine similarity for value cache: {out_v_cos:.6f}"
    assert ov_k_cos > 0.90, f"Low cosine similarity for updated key cache from OV: {ov_k_cos:.6f}"
    assert ov_v_cos > 0.90, f"Low cosine similarity for updated value cache from OV: {ov_v_cos:.6f}"    

    assert out_key_cache.dtype == torch.uint8
    assert out_value_cache.dtype == torch.uint8
    assert out_key_cache.shape == updated_key_cache_from_ov.shape
    assert out_value_cache.shape == updated_value_cache_from_ov.shape

    if k_token_bytes > 2:
        k_data_bytes = kv_block_size * (k_token_bytes - 2)
        _assert_mismatch_ratio_le(
            out_key_cache[:, :, :k_data_bytes],
            updated_key_cache_from_ov[:, :, :k_data_bytes],
            max_ratio=2e-3,
            name="key_cache_data",
        )
        out_k_norm = out_key_cache[:, :, k_data_bytes:].view(torch.float16).float()
        exp_k_norm = updated_key_cache_from_ov[:, :, k_data_bytes:].view(torch.float16).float()
        assert torch.allclose(out_k_norm, exp_k_norm, atol=5e-3, rtol=2e-2)

    if value_cache_mode == "by_token":
        v_data_bytes = kv_block_size * v_head_size
        _assert_mismatch_ratio_le(
            out_value_cache[:, :, :v_data_bytes],
            updated_value_cache_from_ov[:, :, :v_data_bytes],
            max_ratio=2e-3,
            name="value_cache_data",
        )
        out_v_meta = out_value_cache[:, :, v_data_bytes:].view(torch.float16).float()
        exp_v_meta = updated_value_cache_from_ov[:, :, v_data_bytes:].view(torch.float16).float()
        assert torch.allclose(out_v_meta, exp_v_meta, atol=5e-3, rtol=2e-2)
    else:
        v_data_bytes = kv_block_size * (((v_head_size * bits + 7) // 8))
        _assert_mismatch_ratio_le(
            out_value_cache[:, :, :v_data_bytes],
            updated_value_cache_from_ov[:, :, :v_data_bytes],
            max_ratio=2e-3,
            name="value_cache_data",
        )
        out_v_norm = out_value_cache[:, :, v_data_bytes:].view(torch.float16).float()
        exp_v_norm = updated_value_cache_from_ov[:, :, v_data_bytes:].view(torch.float16).float()
        assert torch.allclose(out_v_norm, exp_v_norm, atol=5e-3, rtol=2e-2)
        
# Usage:        
# - python -m pytest -s -q test_ov.py -vv        