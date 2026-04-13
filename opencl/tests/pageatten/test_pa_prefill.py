import functools
import os

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from clops import cl
from turboquant_cm import TurboQuantMSE_CM


torch.manual_seed(0)


def get_cm_grf_width() -> int:
    cm_kernels = cl.kernels(
        r'''
        extern "C" _GENX_MAIN_ void cm_get_grf_width(int * info [[type("svmptr_t")]]) {
            info[0] = CM_GRF_WIDTH;
        }''',
        "-cmc",
    )
    t_info = cl.tensor([2], np.dtype(np.int32))
    cm_kernels.enqueue("cm_get_grf_width", [1], [1], t_info)
    return int(t_info.numpy()[0])


CM_GRF_WIDTH = get_cm_grf_width()
xe_arch = 1 if CM_GRF_WIDTH == 256 else 2


def quan_per_token(kv: torch.Tensor) -> torch.Tensor:
    blk_num, kv_heads, _, _ = kv.shape
    kv_max = kv.amax(dim=-1, keepdim=True)
    kv_min = kv.amin(dim=-1, keepdim=True)
    qrange = kv_max - kv_min

    intmax = 255.0
    intmin = 0.0
    intrange = intmax - intmin
    kv_scale = torch.zeros_like(qrange, dtype=torch.float16)
    kv_scale[qrange != 0] = (intrange / qrange[qrange != 0]).to(dtype=torch.float16)
    kv_zp = ((0.0 - kv_min) * kv_scale + intmin).to(dtype=torch.half)
    kv_u8 = torch.round(kv * kv_scale + kv_zp).clamp(intmin, intmax).to(dtype=torch.uint8).reshape(blk_num, kv_heads, -1)

    dq_scale = torch.zeros_like(kv_scale, dtype=torch.float16)
    dq_scale[kv_scale != 0] = (1.0 / kv_scale[kv_scale != 0]).to(dtype=torch.float16)
    dq_scale = dq_scale.view(dtype=torch.uint8).reshape(blk_num, kv_heads, -1)
    kv_zp_u8 = kv_zp.view(dtype=torch.uint8).reshape(blk_num, kv_heads, -1)
    return torch.concat((kv_u8, dq_scale, kv_zp_u8), dim=-1)


def dequant_per_token(kv: torch.Tensor, head_size: int, blk_size: int) -> torch.Tensor:
    blk_num, kv_head_num, _ = kv.shape
    kv_u8 = kv[:, :, : head_size * blk_size].to(dtype=torch.float16).reshape(blk_num, kv_head_num, blk_size, head_size)
    kv_scale = kv[:, :, head_size * blk_size:head_size * blk_size + blk_size * 2].view(dtype=torch.float16).reshape(
        blk_num, kv_head_num, blk_size, 1
    )
    kv_zp = kv[:, :, head_size * blk_size + blk_size * 2:head_size * blk_size + blk_size * 4].view(
        dtype=torch.float16
    ).reshape(blk_num, kv_head_num, blk_size, 1)

    out = torch.empty([blk_num, kv_head_num, blk_size, head_size], dtype=torch.float16)
    for m in range(blk_num):
        for n in range(kv_head_num):
            for i in range(blk_size):
                out[m, n, i, :] = (kv_u8[m, n, i, :] - kv_zp[m, n, i, 0]) * kv_scale[m, n, i, 0]
    return out


def turboquant_rotate_query(q: torch.Tensor, tq: TurboQuantMSE_CM) -> torch.Tensor:
    return torch.matmul(q.float(), tq.q_t.float()).to(dtype=q.dtype)


def turboquant_pack_key_blocks(k_blocks: torch.Tensor, tq: TurboQuantMSE_CM, bits: int = 4) -> torch.Tensor:
    if bits != 4:
        raise ValueError("TurboQuant prefill currently supports bits==4")
    blk_num, kv_heads, blk_size, head_size = k_blocks.shape
    packed_bytes = (head_size * bits + 7) // 8

    # Use CM quantization path to stay aligned with kernel-side table precision/behavior.
    k_tokens = k_blocks.transpose(1, 2).reshape(blk_num * blk_size, kv_heads * head_size).contiguous()
    q = tq.quantize(k_tokens)
    idx = q["idx"].reshape(blk_num, blk_size, kv_heads, head_size).transpose(1, 2).contiguous().to(torch.uint8)
    norms = q["norms"].reshape(blk_num, blk_size, kv_heads).transpose(1, 2).contiguous().to(torch.float16)

    packed = torch.zeros((blk_num, kv_heads, blk_size, packed_bytes), dtype=torch.uint8)
    packed[..., : head_size // 2] = (idx[..., 0::2] & 0x0F) | ((idx[..., 1::2] & 0x0F) << 4)

    packed_flat = packed.reshape(blk_num, kv_heads, blk_size * packed_bytes)
    norms_u8 = norms.view(torch.uint8).reshape(blk_num, kv_heads, blk_size * 2)
    return torch.concat((packed_flat, norms_u8), dim=-1)


def _check_close(actual: torch.Tensor, expected: torch.Tensor, atol: float = 1e-2, rtol: float = 1e-2) -> None:
    if torch.allclose(actual, expected, atol=atol, rtol=rtol, equal_nan=True):
        return
    close_mask = torch.isclose(actual, expected, atol=atol, rtol=rtol)
    bad = torch.where(~close_mask)
    raise AssertionError(
        "Tensor mismatch\n"
        f"indices={bad}\n"
        f"actual={actual[bad]}\n"
        f"expected={expected[bad]}"
    )


def _flash_ref(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    qh = q.transpose(0, 1)
    kh = k.transpose(0, 1)
    vh = v.transpose(0, 1)
    out = F.scaled_dot_product_attention(
        qh.unsqueeze(0).to(torch.float16),
        kh.unsqueeze(0).to(torch.float16),
        vh.unsqueeze(0).to(torch.float16),
        is_causal=True,
        dropout_p=0.0,
        enable_gqa=(k.shape[1] != q.shape[1]),
    )
    return out.squeeze(0).transpose(0, 1).to(q.dtype)


class page_atten_cm:
    """Extracted valid prefill branches from test_pa.py (no sparse/mask/test_ov/perf)."""

    def __init__(
        self,
        num_heads: int,
        num_kv_heads: int,
        head_size: int,
        block_sz: int,
        trunk_sz: int,
        compressed_kvcache: bool,
        is_causal: bool = True,
        sparse_block_sz: int = 1,
        turboquant_enabled: bool = False,
        turboquant_bits: int = 4,
    ):
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.block_sz = block_sz
        self.trunk_sz = trunk_sz
        self.compressed_kvcache = compressed_kvcache
        self.sparse_block_sz = sparse_block_sz
        self.turboquant_enabled = bool(turboquant_enabled)
        self.turboquant_bits = int(turboquant_bits)

        if sparse_block_sz != 1:
            raise ValueError("This extracted prefill runner only supports sparse_block_sz==1")
        if self.turboquant_enabled and xe_arch < 2:
            raise ValueError("TurboQuant prefill requires XE_ARCH>=2")
        if self.turboquant_enabled and self.sparse_block_sz > 1:
            raise ValueError("TurboQuant prefill requires sparse_block_sz==1")
        if self.turboquant_enabled and (not self.compressed_kvcache):
            raise ValueError("TurboQuant prefill requires KV_CACHE_COMPRESSION==True (compressed_kvcache=True)")

        wg_size = 16
        q_step = CM_GRF_WIDTH // 32
        self.wg_seq_len = wg_size * q_step

        if self.turboquant_enabled:
            src = r'''#include "pa_multi_token_turboquant.cm"'''
            self.kernel_name = "cm_pa_multi_token_turboquant"
            self.tq = TurboQuantMSE_CM.create_instance(
                head_size=head_size,
                num_kv_heads=num_kv_heads,
                bits=self.turboquant_bits,
            )
        else:
            src = r'''#include "pa_multi_token.cm"'''
            self.kernel_name = "cm_page_attention"
            self.tq = None

        cwd = os.path.dirname(os.path.realpath(__file__))
        scale_factor = 1.0 / (head_size ** 0.5)
        self.kernels = cl.kernels(
            src,
            (
                f'-cmc -Qxcm_jit_option="-abortonspill" -Qxcm_register_file_size=256 -mCM_printregusage -I{cwd}'
                f' -DKERNEL_NAME={self.kernel_name}'
                f" -DCMFLA_NUM_HEADS={num_heads}"
                f" -DCMFLA_NUM_KV_HEADS={num_kv_heads}"
                f" -DCMFLA_HEAD_SIZE={head_size}"
                f" -DCMFLA_SCALE_FACTOR={scale_factor}"
                f" -DCMFLA_IS_CAUSAL={int(is_causal)}"
                f" -DCMPA_BLOCK_SZ={self.block_sz}"
                f" -DSPARSE_BLOCK_SIZE=1"
                f" -DCMPA_WG_SEQ_LEN={int(self.wg_seq_len)}"
                f" -DCMPA_KVCACHE_U8={int(compressed_kvcache)}"
                f" -DTQ_BITS={int(self.turboquant_bits)}"
                f" -mdump_asm -g2"
            ),
        )

    def __call__(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        seq_len, _, head_size = q.shape
        assert head_size == self.head_size

        aligned_seq_len = seq_len
        if seq_len % self.block_sz != 0:
            pad = self.block_sz - seq_len % self.block_sz
            aligned_seq_len = seq_len + pad
            k = torch.nn.functional.pad(k, (0, 0, 0, 0, 0, pad), "constant", 1)
            v = torch.nn.functional.pad(v, (0, 0, 0, 0, 0, pad), "constant", 1)

        k_cache = k.reshape(aligned_seq_len // self.block_sz, self.block_sz, self.num_kv_heads, self.head_size).transpose(1, 2).contiguous()
        v_cache = v.reshape(aligned_seq_len // self.block_sz, self.block_sz, self.num_kv_heads, self.head_size).transpose(1, 2).contiguous()

        if self.compressed_kvcache:
            if self.turboquant_enabled:
                k_cache = turboquant_pack_key_blocks(k_cache, self.tq, self.turboquant_bits)
            else:
                k_cache = quan_per_token(k_cache)
            v_cache = quan_per_token(v_cache)
        else:
            k_cache = k_cache.reshape(aligned_seq_len // self.block_sz, self.num_kv_heads, -1)
            v_cache = v_cache.reshape(aligned_seq_len // self.block_sz, self.num_kv_heads, -1)

        out = torch.zeros(seq_len, self.num_heads, self.head_size, dtype=torch.float16)

        kv_dtype = torch.uint8 if self.compressed_kvcache else torch.half
        if self.compressed_kvcache:
            k_token_sz = ((head_size * self.turboquant_bits + 7) // 8 + 2) if self.turboquant_enabled else (head_size + 4)
            v_token_sz = head_size + 4
        else:
            k_token_sz = head_size
            v_token_sz = head_size

        block_num = aligned_seq_len // self.block_sz
        block_indices = torch.arange(block_num, dtype=torch.int32)
        sub_k = torch.zeros(block_num, self.num_kv_heads, self.block_sz * k_token_sz, dtype=kv_dtype)
        sub_v = torch.zeros(block_num, self.num_kv_heads, self.block_sz * v_token_sz, dtype=kv_dtype)
        for i in range(block_num):
            sub_k[block_indices[i], :] = k_cache[i, :]
            sub_v[block_indices[i], :] = v_cache[i, :]

        t_q = cl.tensor(q.to(torch.float16).detach().numpy())
        t_k = cl.tensor(sub_k.to(kv_dtype).detach().numpy())
        t_v = cl.tensor(sub_v.to(kv_dtype).detach().numpy())
        t_out = cl.tensor([seq_len, self.num_heads, self.head_size], np.dtype(np.float16))

        past_lens = torch.tensor([0], dtype=torch.int32)
        block_indices_begins = torch.tensor([0, block_num], dtype=torch.int32)
        subsequence_begins = torch.tensor([0, seq_len], dtype=torch.int32)

        t_block_indices = cl.tensor(block_indices.detach().numpy())
        t_past_lens = cl.tensor(past_lens.detach().numpy())
        t_block_indices_begins = cl.tensor(block_indices_begins.detach().numpy())
        t_subsequence_begins = cl.tensor(subsequence_begins.detach().numpy())

        wg_size = 16
        wg_count = (seq_len + self.wg_seq_len - 1) // self.wg_seq_len
        gws = [1, self.num_heads, int(wg_count * wg_size)]
        lws = [1, 1, wg_size]

        if self.turboquant_enabled:
            t_q_t = cl.tensor(self.tq.q_t.numpy())
            t_centroids = cl.tensor(self.tq.centroids.numpy())
            self.kernels.enqueue(
                self.kernel_name,
                gws,
                lws,
                t_q,
                t_k,
                t_v,
                t_past_lens,
                t_block_indices,
                t_block_indices_begins,
                t_subsequence_begins,
                t_out,
                t_q_t,
                t_centroids,
                seq_len,
            )
        else:
            self.kernels.enqueue(
                self.kernel_name,
                gws,
                lws,
                t_q,
                t_k,
                t_v,
                t_past_lens,
                t_block_indices,
                t_block_indices_begins,
                t_subsequence_begins,
                t_out,
                seq_len,
            )

        cl.finish()
        out.copy_(torch.from_numpy(t_out.numpy()))
        return out

    @staticmethod
    @functools.cache
    def create_instance(
        num_heads: int,
        num_kv_heads: int,
        head_size: int,
        block_sz: int,
        trunk_sz: int,
        compressed_kvcache: bool,
        is_causal: bool,
        sparse_block_sz: int,
        turboquant_enabled: bool = False,
        turboquant_bits: int = 4,
    ):
        return page_atten_cm(
            num_heads,
            num_kv_heads,
            head_size,
            block_sz,
            trunk_sz,
            compressed_kvcache,
            is_causal,
            sparse_block_sz,
            turboquant_enabled,
            turboquant_bits,
        )


def _dequant_per_token_seq(x: torch.Tensor, block_size: int, num_kv_heads: int, head_size: int) -> torch.Tensor:
    seq_len = int(x.shape[0])
    blk_num = (seq_len + block_size - 1) // block_size
    aligned_len = blk_num * block_size
    if aligned_len != seq_len:
        x = torch.nn.functional.pad(x, (0, 0, 0, 0, 0, aligned_len - seq_len), "constant", 1)

    blocks = x.reshape(blk_num, block_size, num_kv_heads, head_size).transpose(1, 2).contiguous()
    q = quan_per_token(blocks)
    dq = dequant_per_token(q, head_size, block_size)
    out = dq.transpose(1, 2).reshape(aligned_len, num_kv_heads, head_size)
    return out[:seq_len]


def _randn_fp16(shape: tuple[int, ...], scale: float = 0.4) -> torch.Tensor:
    return (torch.randn(*shape, dtype=torch.float32) * scale).clamp(-1.0, 1.0).to(torch.float16)


def _run_prefill_in_trunks(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    block_size: int,
    trunk_size: int,
    turboquant_enabled: bool,
) -> torch.Tensor:
    seq_len = int(q.shape[0])
    outs = []
    start = 0
    while start < seq_len:
        end = min(start + trunk_size, seq_len)
        page_atten_cm.create_instance.cache_clear()
        runner = page_atten_cm.create_instance(
            num_heads,
            num_kv_heads,
            head_size,
            block_size,
            trunk_size,
            True,
            True,
            1,
            turboquant_enabled,
            4,
        )
        out_prefix = runner(q[:end], k[:end], v[:end])
        outs.append(out_prefix[start:end])
        start = end
    return torch.cat(outs, dim=0)


@pytest.mark.parametrize(
    ("seq_len", "num_heads", "num_kv_heads", "head_size", "block_size"),
    [
        (64, 16, 16, 128, 16),
        (80, 8, 2, 64, 16),
    ],
)
def test_quant_by_token_against_flash_ref(
    seq_len: int,
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    block_size: int,
):
    q = torch.randint(-127, 128, [seq_len, num_heads, head_size], dtype=torch.int32).to(torch.float16) / 128.0
    k = torch.randint(-127, 128, [seq_len, num_kv_heads, head_size], dtype=torch.int32).to(torch.float16) / 128.0
    v = torch.randint(-127, 128, [seq_len, num_kv_heads, head_size], dtype=torch.int32).to(torch.float16) / 128.0

    page_atten_cm.create_instance.cache_clear()
    runner = page_atten_cm.create_instance(
        num_heads,
        num_kv_heads,
        head_size,
        block_size,
        seq_len,
        True,
        True,
        1,
        False,
        4,
    )
    out = runner(q, k, v)

    k_ref = _dequant_per_token_seq(k, block_size, num_kv_heads, head_size)
    v_ref = _dequant_per_token_seq(v, block_size, num_kv_heads, head_size)
    ref = _flash_ref(q, k_ref, v_ref)
    _check_close(out, ref, atol=3e-2, rtol=3e-2)


@pytest.mark.parametrize(
    ("seq_len", "num_heads", "num_kv_heads", "head_size", "block_size"),
    [
        (64, 16, 16, 128, 16),
        (80, 8, 2, 64, 16),
    ],
)
def test_turboquant_against_quant_by_token(
    seq_len: int,
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    block_size: int,
):
    if xe_arch < 2:
        pytest.skip("TurboQuant prefill requires XE_ARCH>=2")

    q = torch.randint(-127, 128, [seq_len, num_heads, head_size], dtype=torch.int32).to(torch.float16) / 128.0
    k = torch.randint(-127, 128, [seq_len, num_kv_heads, head_size], dtype=torch.int32).to(torch.float16) / 128.0
    v = torch.randint(-127, 128, [seq_len, num_kv_heads, head_size], dtype=torch.int32).to(torch.float16) / 128.0

    page_atten_cm.create_instance.cache_clear()
    qtoken_runner = page_atten_cm.create_instance(
        num_heads,
        num_kv_heads,
        head_size,
        block_size,
        seq_len,
        True,
        True,
        1,
        False,
        4,
    )
    turbo_runner = page_atten_cm.create_instance(
        num_heads,
        num_kv_heads,
        head_size,
        block_size,
        seq_len,
        True,
        True,
        1,
        True,
        4,
    )

    out_qtoken = qtoken_runner(q, k, v)
    out_turbo = turbo_runner(q, k, v)
    diff = (out_turbo - out_qtoken).abs().float()
    mae = diff.mean().item()
    cos = F.cosine_similarity(out_turbo.flatten().float(), out_qtoken.flatten().float(), dim=0).item()
    print(f"[tq_vs_qtoken] seq_len={seq_len} h={num_heads} kv={num_kv_heads} hs={head_size} bls={block_size} mae={mae:.6f} cos={cos:.6f}")
    assert mae < 0.05
    assert cos > 0.95


@pytest.mark.parametrize(
    ("seq_len", "num_heads", "num_kv_heads", "head_size", "block_size"),
    [
        (64, 16, 16, 128, 16),
        (80, 8, 2, 64, 16),
    ],
)
def test_turboquant_against_fp16(
    seq_len: int,
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    block_size: int,
):
    if xe_arch < 2:
        pytest.skip("TurboQuant prefill requires XE_ARCH>=2")

    q = _randn_fp16((seq_len, num_heads, head_size))
    k = _randn_fp16((seq_len, num_kv_heads, head_size))
    v = _randn_fp16((seq_len, num_kv_heads, head_size))

    page_atten_cm.create_instance.cache_clear()
    turbo_runner = page_atten_cm.create_instance(
        num_heads,
        num_kv_heads,
        head_size,
        block_size,
        seq_len,
        True,
        True,
        1,
        True,
        4,
    )

    out_turbo = turbo_runner(q, k, v)
    ref_fp16 = _flash_ref(q, k, v)

    diff = (out_turbo - ref_fp16).abs().float()
    mae = diff.mean().item()
    cos = F.cosine_similarity(out_turbo.flatten().float(), ref_fp16.flatten().float(), dim=0).item()
    print(f"[tq_vs_fp16] seq_len={seq_len} h={num_heads} kv={num_kv_heads} hs={head_size} bls={block_size} mae={mae:.6f} cos={cos:.6f}")
    assert mae < 0.02
    assert cos > 0.95


def test_turboquant_against_fp16_long_2k_trunk1k():
    if xe_arch < 2:
        pytest.skip("TurboQuant prefill requires XE_ARCH>=2")

    seq_len = 2048
    trunk_size = 1024
    num_heads = 2
    num_kv_heads = 2
    head_size = 32
    block_size = 16

    q = _randn_fp16((seq_len, num_heads, head_size))
    k = _randn_fp16((seq_len, num_kv_heads, head_size))
    v = _randn_fp16((seq_len, num_kv_heads, head_size))

    out_turbo_trunked = _run_prefill_in_trunks(
        q,
        k,
        v,
        num_heads,
        num_kv_heads,
        head_size,
        block_size,
        trunk_size,
        turboquant_enabled=True,
    )
    ref_fp16 = _flash_ref(q, k, v)

    diff = (out_turbo_trunked - ref_fp16).abs().float()
    mae = diff.mean().item()
    cos = F.cosine_similarity(out_turbo_trunked.flatten().float(), ref_fp16.flatten().float(), dim=0).item()
    print(f"[tq_vs_fp16_trunked] seq_len={seq_len} trunk={trunk_size} h={num_heads} kv={num_kv_heads} hs={head_size} bls={block_size} mae={mae:.6f} cos={cos:.6f}")
    assert mae < 0.02
    assert cos > 0.95
