import torch
import os
import pytest

from clops import compare
from turboquant_cm import TurboQuantMSE_CM, CompressedKVCache_Update_CM


def _build_pa_metadata(num_tokens, past_lens, block_size):
    batch = len(num_tokens)
    subsequence_begins = [0]
    block_indices_begins = [0]
    for i in range(batch):
        subsequence_begins.append(subsequence_begins[-1] + num_tokens[i])
        required_blocks = (num_tokens[i] + past_lens[i] + block_size - 1) // block_size
        block_indices_begins.append(block_indices_begins[-1] + required_blocks)

    num_blocks = block_indices_begins[-1]
    block_indices = torch.arange(num_blocks, dtype=torch.int32)
    block_indices = block_indices[torch.randperm(num_blocks)]

    return subsequence_begins, block_indices.tolist(), block_indices_begins


def _pack_norm_to_u8(norm_half: torch.Tensor) -> torch.Tensor:
    return norm_half.view(torch.uint8)


def _pack_idx_bits(idx: torch.Tensor, bits: int) -> torch.Tensor:
    idx = idx.to(torch.int32).reshape(-1)
    n = idx.numel()
    out_bytes = (n * bits + 7) // 8
    out = torch.zeros(out_bytes, dtype=torch.uint8)
    mask = (1 << bits) - 1
    bit_pos = 0
    for i in range(n):
        v = int(idx[i].item()) & mask
        byte_pos = bit_pos >> 3
        shift = bit_pos & 7
        out[byte_pos] = torch.tensor(int(out[byte_pos]) | ((v << shift) & 0xFF), dtype=torch.uint8)
        if shift + bits > 8:
            out[byte_pos + 1] = torch.tensor(int(out[byte_pos + 1]) | (v >> (8 - shift)), dtype=torch.uint8)
        bit_pos += bits
    return out


def _unpack_idx_bits(packed: torch.Tensor, n_idx: int, bits: int) -> torch.Tensor:
    packed = packed.to(torch.int32).reshape(-1)
    out = torch.zeros(n_idx, dtype=torch.long)
    mask = (1 << bits) - 1
    bit_pos = 0
    for i in range(n_idx):
        byte_pos = bit_pos >> 3
        shift = bit_pos & 7
        v = (int(packed[byte_pos].item()) >> shift) & mask
        if shift + bits > 8:
            v_hi = int(packed[byte_pos + 1].item()) << (8 - shift)
            v = (v | v_hi) & mask
        out[i] = v
        bit_pos += bits
    return out


def _reference_update_tq(
    key_cache,
    value_cache,
    key,
    value,
    past_lens,
    subsequence_begins,
    block_indices,
    block_indices_begins,
    tq_key,
    tq_val,
    num_kv_heads,
    k_head_size,
    v_head_size,
    block_size,
    bits,
):
    out_k = key_cache.clone()
    out_v = value_cache.clone()

    out_k_flat = out_k.reshape(out_k.shape[0], out_k.shape[1], -1)
    out_v_flat = out_v.reshape(out_v.shape[0], out_v.shape[1], -1)

    # Use CM quantization outputs directly for exact parity with kernel path.
    # tq_key/tq_val are single-head helpers (num_kv_heads=1), so flatten heads.
    k_flat = key.view(key.shape[0], num_kv_heads, k_head_size).reshape(-1, k_head_size).float()
    v_flat = value.view(value.shape[0], num_kv_heads, v_head_size).reshape(-1, v_head_size).float()
    k_q_flat = tq_key.quantize(k_flat)
    v_q_flat = tq_val.quantize(v_flat)

    k_q_all = {
        "idx": k_q_flat["idx"].view(key.shape[0], num_kv_heads, k_head_size),
        "norms": k_q_flat["norms"].view(key.shape[0], num_kv_heads),
    }
    v_q_all = {
        "idx": v_q_flat["idx"].view(value.shape[0], num_kv_heads, v_head_size),
        "norms": v_q_flat["norms"].view(value.shape[0], num_kv_heads),
    }

    batch = len(past_lens)
    for seq in range(batch):
        seq_tokens = subsequence_begins[seq + 1] - subsequence_begins[seq]
        for t in range(seq_tokens):
            token_idx = subsequence_begins[seq] + t
            cur_block_idx = (past_lens[seq] + t) // block_size
            token_pos = (past_lens[seq] + t) % block_size
            block_pos = block_indices[block_indices_begins[seq] + cur_block_idx]

            for h in range(num_kv_heads):
                # key
                k_packed_bytes = (k_head_size * bits + 7) // 8
                v_packed_bytes = (v_head_size * bits + 7) // 8

                k_idx = _pack_idx_bits(k_q_all["idx"][token_idx, h], bits)
                out_k_flat[block_pos, h, token_pos * k_packed_bytes:(token_pos + 1) * k_packed_bytes] = k_idx

                k_norm_u8 = _pack_norm_to_u8(k_q_all["norms"][token_idx, h:h + 1].half())
                k_norm_base = block_size * k_packed_bytes + token_pos * 2
                out_k_flat[block_pos, h, k_norm_base:k_norm_base + 2] = k_norm_u8.reshape(-1)

                # value
                v_idx = _pack_idx_bits(v_q_all["idx"][token_idx, h], bits)
                out_v_flat[block_pos, h, token_pos * v_packed_bytes:(token_pos + 1) * v_packed_bytes] = v_idx

                v_norm_u8 = _pack_norm_to_u8(v_q_all["norms"][token_idx, h:h + 1].half())
                v_norm_base = block_size * v_packed_bytes + token_pos * 2
                out_v_flat[block_pos, h, v_norm_base:v_norm_base + 2] = v_norm_u8.reshape(-1)

    return out_k, out_v


def _reconstruct_updated_tokens_from_cache(
    cache,
    past_lens,
    subsequence_begins,
    block_indices,
    block_indices_begins,
    num_kv_heads,
    head_size,
    block_size,
    bits,
    tq_ref,
):
    packed_bytes = (head_size * bits + 7) // 8
    flat = cache.reshape(cache.shape[0], cache.shape[1], -1)

    rows = []
    batch = len(past_lens)
    for seq in range(batch):
        seq_tokens = subsequence_begins[seq + 1] - subsequence_begins[seq]
        for t in range(seq_tokens):
            cur_block_idx = (past_lens[seq] + t) // block_size
            token_pos = (past_lens[seq] + t) % block_size
            block_pos = block_indices[block_indices_begins[seq] + cur_block_idx]

            row_heads = []
            for h in range(num_kv_heads):
                start = token_pos * packed_bytes
                end = start + packed_bytes
                idx_packed = flat[block_pos, h, start:end]
                idx = _unpack_idx_bits(idx_packed, head_size, bits)

                norm_off = block_size * packed_bytes + token_pos * 2
                norm = flat[block_pos, h, norm_off:norm_off + 2].view(torch.float16)[0].item()

                x_rot = tq_ref.centroids[idx].float()
                x_unit = x_rot @ tq_ref.Q.float()
                row_heads.append(x_unit * float(norm))
            rows.append(torch.stack(row_heads, dim=0))

    return torch.stack(rows, dim=0)


def _assert_algorithm_level_match(ref_rec: torch.Tensor, out_rec: torch.Tensor, bits: int, label: str):
    ref = ref_rec.float().reshape(-1, ref_rec.shape[-1])
    out = out_rec.float().reshape(-1, out_rec.shape[-1])
    diff = (ref - out).abs()

    mae = diff.mean().item()
    max_abs = diff.max().item()
    cos = torch.nn.functional.cosine_similarity(ref, out, dim=-1).mean().item()

    max_abs_limit = {2: 0.60, 3: 0.36, 4: 0.30}.get(bits, 0.60)
    assert mae < 1.5e-2, f"{label}: MAE too large: {mae}"
    assert cos > 0.995, f"{label}: cosine similarity too low: {cos}"
    assert max_abs < max_abs_limit, f"{label}: max abs diff too large: {max_abs} >= {max_abs_limit}"


def test_turboquant_mse_cm_quant_dequant():
    torch.manual_seed(7)

    n_tokens = 11
    num_kv_heads = 2
    head_size = 16
    bits = 4

    x = torch.randn(n_tokens, num_kv_heads * head_size, dtype=torch.float16)

    tq = TurboQuantMSE_CM(head_size=head_size, num_kv_heads=num_kv_heads, bits=bits)

    q_cm = tq.quantize(x)
    x_cm = tq.dequantize(q_cm)

    q_ref = tq.quantize_reference(x)
    x_ref = tq.dequantize_reference(q_ref)

    compare(q_ref["idx"].numpy(), q_cm["idx"].numpy(), 0)
    compare(q_ref["norms"].numpy(), q_cm["norms"].numpy(), 2e-3)
    compare(x_ref.numpy(), x_cm.numpy(), 3e-2)


def test_compressed_kv_cache_update_cm():
    torch.manual_seed(11)

    num_tokens = [7, 4, 3]
    past_lens = [5, 1, 6]
    num_kv_heads = 2
    k_head_size = 16
    v_head_size = 16
    block_size = 8
    bits = 4

    subsequence_begins, block_indices, block_indices_begins = _build_pa_metadata(num_tokens, past_lens, block_size)

    batch_tokens = sum(num_tokens)
    key = torch.randn(batch_tokens, num_kv_heads * k_head_size, dtype=torch.float16)
    value = torch.randn(batch_tokens, num_kv_heads * v_head_size, dtype=torch.float16)

    num_blocks = len(block_indices)
    k_packed_bytes = (k_head_size * bits + 7) // 8
    v_packed_bytes = (v_head_size * bits + 7) // 8
    key_cache = torch.zeros(num_blocks, num_kv_heads, block_size * (k_packed_bytes + 2), dtype=torch.uint8)
    value_cache = torch.zeros(num_blocks, num_kv_heads, block_size * (v_packed_bytes + 2), dtype=torch.uint8)

    updater = CompressedKVCache_Update_CM(
        num_kv_heads=num_kv_heads,
        k_head_size=k_head_size,
        v_head_size=v_head_size,
        block_size=block_size,
        bits=bits,
    )

    out_k, out_v = updater(
        key,
        value,
        key_cache,
        value_cache,
        past_lens,
        subsequence_begins,
        block_indices,
        block_indices_begins,
    )

    ref_k, ref_v = _reference_update_tq(
        key_cache,
        value_cache,
        key,
        value,
        past_lens,
        subsequence_begins,
        block_indices,
        block_indices_begins,
        updater.tq_k.ref,
        updater.tq_v.ref,
        num_kv_heads,
        k_head_size,
        v_head_size,
        block_size,
        bits,
    )

    out_k_rec = _reconstruct_updated_tokens_from_cache(
        out_k, past_lens, subsequence_begins, block_indices, block_indices_begins,
        num_kv_heads, k_head_size, block_size, bits, updater.tq_k.ref)
    ref_k_rec = _reconstruct_updated_tokens_from_cache(
        ref_k, past_lens, subsequence_begins, block_indices, block_indices_begins,
        num_kv_heads, k_head_size, block_size, bits, updater.tq_k.ref)
    out_v_rec = _reconstruct_updated_tokens_from_cache(
        out_v, past_lens, subsequence_begins, block_indices, block_indices_begins,
        num_kv_heads, v_head_size, block_size, bits, updater.tq_v.ref)
    ref_v_rec = _reconstruct_updated_tokens_from_cache(
        ref_v, past_lens, subsequence_begins, block_indices, block_indices_begins,
        num_kv_heads, v_head_size, block_size, bits, updater.tq_v.ref)

    _assert_algorithm_level_match(ref_k_rec, out_k_rec, bits, "k")
    _assert_algorithm_level_match(ref_v_rec, out_v_rec, bits, "v")


def _run_compressed_kv_case(num_tokens,
                            past_lens,
                            num_kv_heads,
                            k_head_size,
                            v_head_size,
                            block_size,
                            bits=4,
                            check_perf=False):
    subsequence_begins, block_indices, block_indices_begins = _build_pa_metadata(num_tokens, past_lens, block_size)

    batch_tokens = sum(num_tokens)
    key = torch.randn(batch_tokens, num_kv_heads * k_head_size, dtype=torch.float16)
    value = torch.randn(batch_tokens, num_kv_heads * v_head_size, dtype=torch.float16)

    num_blocks = len(block_indices)
    k_packed_bytes = (k_head_size * bits + 7) // 8
    v_packed_bytes = (v_head_size * bits + 7) // 8
    key_cache = torch.zeros(num_blocks, num_kv_heads, block_size * (k_packed_bytes + 2), dtype=torch.uint8)
    value_cache = torch.zeros(num_blocks, num_kv_heads, block_size * (v_packed_bytes + 2), dtype=torch.uint8)

    updater = CompressedKVCache_Update_CM(
        num_kv_heads=num_kv_heads,
        k_head_size=k_head_size,
        v_head_size=v_head_size,
        block_size=block_size,
        bits=bits,
    )

    n_repeats = 20 if check_perf else 1
    out_k, out_v = updater(
        key,
        value,
        key_cache,
        value_cache,
        past_lens,
        subsequence_begins,
        block_indices,
        block_indices_begins,
        n_repeats=n_repeats,
    )
    
    check_acc = not check_perf
    if check_acc:
        ref_k, ref_v = _reference_update_tq(
            key_cache,
            value_cache,
            key,
            value,
            past_lens,
            subsequence_begins,
            block_indices,
            block_indices_begins,
            updater.tq_k.ref,
            updater.tq_v.ref,
            num_kv_heads,
            k_head_size,
            v_head_size,
            block_size,
            bits,
        )

        out_k_rec = _reconstruct_updated_tokens_from_cache(
            out_k, past_lens, subsequence_begins, block_indices, block_indices_begins,
            num_kv_heads, k_head_size, block_size, bits, updater.tq_k.ref)
        ref_k_rec = _reconstruct_updated_tokens_from_cache(
            ref_k, past_lens, subsequence_begins, block_indices, block_indices_begins,
            num_kv_heads, k_head_size, block_size, bits, updater.tq_k.ref)
        out_v_rec = _reconstruct_updated_tokens_from_cache(
            out_v, past_lens, subsequence_begins, block_indices, block_indices_begins,
            num_kv_heads, v_head_size, block_size, bits, updater.tq_v.ref)
        ref_v_rec = _reconstruct_updated_tokens_from_cache(
            ref_v, past_lens, subsequence_begins, block_indices, block_indices_begins,
            num_kv_heads, v_head_size, block_size, bits, updater.tq_v.ref)

        _assert_algorithm_level_match(ref_k_rec, out_k_rec, bits, "k")
        _assert_algorithm_level_match(ref_v_rec, out_v_rec, bits, "v")


@pytest.mark.parametrize("n_bits", [2, 3, 4])
@pytest.mark.parametrize(
    "num_tokens,past_lens,num_kv_heads,k_head_size,v_head_size,block_size",
    [
        ([16], [0], 8, 128, 128, 256),
        ([16], [0], 8, 96, 96, 256),
        ([16], [0], 8, 48, 48, 256),
        ([16], [0], 8, 48, 96, 256),
        ([16], [4 * 1024], 8, 128, 128, 256),
        ([16], [0], 1, 16, 16, 16),
    ],
)
def test_compressed_kv_cache_update_cm_parameterized(
    n_bits,
    num_tokens,
    past_lens,
    num_kv_heads,
    k_head_size,
    v_head_size,
    block_size,
):
    """Parameterized like test_kvcache_update.py (KV_CACHE_COMPRESSION_PER_TOKEN=False path only)."""
    torch.manual_seed(11)
    _run_compressed_kv_case(
        num_tokens=num_tokens,
        past_lens=past_lens,
        num_kv_heads=num_kv_heads,
        k_head_size=k_head_size,
        v_head_size=v_head_size,
        block_size=block_size,
        bits=n_bits,
        check_perf=False,
    )


@pytest.mark.skipif(os.getenv("RUN_PERF_TESTS") != "1", reason="Set RUN_PERF_TESTS=1 to enable perf tests")
@pytest.mark.parametrize("n_bits", [2, 3, 4])
@pytest.mark.parametrize(
    "num_tokens,past_lens,num_kv_heads,k_head_size,v_head_size,block_size",
    [
        ([32 * 1024], [0], 8, 128, 128, 256),
        ([32 * 1024], [0], 8, 96, 96, 256),
        ([32 * 1024], [0], 8, 48, 48, 256),
        ([32 * 1024], [0], 8, 48, 96, 256),
        ([32 * 1024], [4 * 1024], 8, 128, 128, 256),
        ([32 * 1024], [0], 1, 16, 16, 16),
    ],
)
def test_compressed_kv_cache_update_cm_perf(
    n_bits,
    num_tokens,
    past_lens,
    num_kv_heads,
    k_head_size,
    v_head_size,
    block_size,
):
    """Perf-only path: accuracy disabled, kernel prints TPUT_* internally."""
    torch.manual_seed(11)
    _run_compressed_kv_case(
        num_tokens=num_tokens,
        past_lens=past_lens,
        num_kv_heads=num_kv_heads,
        k_head_size=k_head_size,
        v_head_size=v_head_size,
        block_size=block_size,
        bits=n_bits,
        check_perf=True,
    )


def _compressed_size_bytes_like_core(q: dict, bits: int) -> int:
    idx_bits = q["idx"].numel() * bits
    norm_bytes = q["norms"].numel() * 2
    return idx_bits // 8 + norm_bytes


def test_turboquant_mse_cm_self_test_like():
    """Self-test similar to turboquant_core.self_test for TurboQuantMSE_CM."""
    torch.manual_seed(123)
    d, n = 256, 64
    x = torch.randn(n, d, dtype=torch.float32)

    prev_mse = None
    prev_cos = None
    for bits in [2, 3, 4]:
        tq = TurboQuantMSE_CM.create_instance(head_size=d, num_kv_heads=1, bits=bits, rotation_seed=0)
        q = tq.quantize(x)
        x_hat = tq.dequantize(q).float()

        mse = ((x - x_hat) ** 2).mean().item()
        cos = torch.nn.functional.cosine_similarity(x, x_hat, dim=-1).mean().item()

        pairs = 32
        ip_true = (x[:pairs].unsqueeze(1) * x[pairs:2 * pairs].unsqueeze(0)).sum(-1)
        ip_est = (x_hat[:pairs].unsqueeze(1) * x_hat[pairs:2 * pairs].unsqueeze(0)).sum(-1)
        ip_corr = torch.corrcoef(torch.stack([ip_true.flatten(), ip_est.flatten()]))[0, 1].item()

        orig_bytes = x.numel() * 4
        comp_bytes = _compressed_size_bytes_like_core(q, bits)

        print(f"TurboQuantMSE_CM  d={d}  bits={bits}  n={n}")
        print(f"  MSE:                {mse:.6f}")
        print(f"  Mean cosine sim:    {cos:.4f}")
        print(f"  Inner-product corr: {ip_corr:.4f}")
        print(f"  Size: {orig_bytes:,} -> {comp_bytes:,} bytes  ({orig_bytes / comp_bytes:.1f}x)")
        print(f"  dtype: {x.dtype} -> {q['idx'].dtype} + {q['norms'].dtype}")
        print()

        assert comp_bytes < orig_bytes
        assert mse == mse and cos == cos and ip_corr == ip_corr  # not NaN

        if prev_mse is not None:
            assert mse < prev_mse + 1e-6
            assert cos > prev_cos - 1e-6
        prev_mse = mse
        prev_cos = cos


def test_compressed_kv_cache_update_cm_self_test_like():
    """Self-test style quality check for CompressedKVCache_Update_CM."""
    torch.manual_seed(321)

    num_tokens = [64]
    past_lens = [0]
    num_kv_heads = 8
    k_head_size = 128
    v_head_size = 128
    block_size = 256

    subsequence_begins, block_indices, block_indices_begins = _build_pa_metadata(num_tokens, past_lens, block_size)
    batch_tokens = sum(num_tokens)

    key = torch.randn(batch_tokens, num_kv_heads * k_head_size, dtype=torch.float16)
    value = torch.randn(batch_tokens, num_kv_heads * v_head_size, dtype=torch.float16)

    key_ref = key.view(batch_tokens, num_kv_heads, k_head_size).float()
    value_ref = value.view(batch_tokens, num_kv_heads, v_head_size).float()

    prev_k_mse = None
    prev_v_mse = None
    prev_k_cos = None
    prev_v_cos = None

    for bits in [2, 3, 4]:
        k_packed_bytes = (k_head_size * bits + 7) // 8
        v_packed_bytes = (v_head_size * bits + 7) // 8
        num_blocks = len(block_indices)
        key_cache = torch.zeros(num_blocks, num_kv_heads, block_size * (k_packed_bytes + 2), dtype=torch.uint8)
        value_cache = torch.zeros(num_blocks, num_kv_heads, block_size * (v_packed_bytes + 2), dtype=torch.uint8)

        updater = CompressedKVCache_Update_CM(
            num_kv_heads=num_kv_heads,
            k_head_size=k_head_size,
            v_head_size=v_head_size,
            block_size=block_size,
            bits=bits,
        )

        out_k, out_v = updater(
            key,
            value,
            key_cache,
            value_cache,
            past_lens,
            subsequence_begins,
            block_indices,
            block_indices_begins,
        )

        k_rec = _reconstruct_updated_tokens_from_cache(
            out_k, past_lens, subsequence_begins, block_indices, block_indices_begins,
            num_kv_heads, k_head_size, block_size, bits, updater.tq_k.ref)
        v_rec = _reconstruct_updated_tokens_from_cache(
            out_v, past_lens, subsequence_begins, block_indices, block_indices_begins,
            num_kv_heads, v_head_size, block_size, bits, updater.tq_v.ref)

        k_mse = ((key_ref - k_rec) ** 2).mean().item()
        v_mse = ((value_ref - v_rec) ** 2).mean().item()
        k_cos = torch.nn.functional.cosine_similarity(
            key_ref.reshape(-1, k_head_size), k_rec.reshape(-1, k_head_size), dim=-1).mean().item()
        v_cos = torch.nn.functional.cosine_similarity(
            value_ref.reshape(-1, v_head_size), v_rec.reshape(-1, v_head_size), dim=-1).mean().item()

        key_orig_bytes = key.numel() * key.element_size()
        value_orig_bytes = value.numel() * value.element_size()
        key_comp_bytes = batch_tokens * num_kv_heads * (k_packed_bytes + 2)
        value_comp_bytes = batch_tokens * num_kv_heads * (v_packed_bytes + 2)

        print(f"CompressedKVCache_Update_CM  bits={bits}  tokens={batch_tokens}  heads={num_kv_heads}")
        print(f"  key   MSE={k_mse:.6f}  cos={k_cos:.4f}  size: {key_orig_bytes:,} -> {key_comp_bytes:,} ({key_orig_bytes / key_comp_bytes:.1f}x)")
        print(f"  value MSE={v_mse:.6f}  cos={v_cos:.4f}  size: {value_orig_bytes:,} -> {value_comp_bytes:,} ({value_orig_bytes / value_comp_bytes:.1f}x)")
        print()

        assert k_mse == k_mse and v_mse == v_mse and k_cos == k_cos and v_cos == v_cos
        assert key_comp_bytes < key_orig_bytes
        assert value_comp_bytes < value_orig_bytes

        if prev_k_mse is not None:
            assert k_mse < prev_k_mse + 1e-6
            assert v_mse < prev_v_mse + 1e-6
            assert k_cos > prev_k_cos - 1e-6
            assert v_cos > prev_v_cos - 1e-6

        prev_k_mse = k_mse
        prev_v_mse = v_mse
        prev_k_cos = k_cos
        prev_v_cos = v_cos