import os
import math
import functools
import torch

from clops import cl
from clops.utils import Colors

enable_debug_prints = True

# ----------------------------------------------------------------------------
# CPU helpers (same math as turboquant_core.py)
# ----------------------------------------------------------------------------

def make_rotation_matrix(d: int, seed: int = 0) -> torch.Tensor:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    g = torch.randn(d, d, generator=gen, dtype=torch.float32)
    q, r = torch.linalg.qr(g)
    q = q * torch.sign(torch.diag(r)).unsqueeze(0)
    return q.contiguous()


def _beta_pdf_unnorm(x: torch.Tensor, d: int) -> torch.Tensor:
    alpha = (d - 1) / 2.0
    return torch.exp((alpha - 1) * torch.log(torch.clamp(1 - x * x, min=1e-30)))


def build_lloyd_max_codebook(d: int, bits: int, n_iter: int = 300, grid_size: int = 50000) -> torch.Tensor:
    n_levels = 1 << bits
    sigma = 1.0 / math.sqrt(d) if d > 1 else 0.5
    lo = max(-1.0 + 1e-7, -6 * sigma)
    hi = min(1.0 - 1e-7, 6 * sigma)
    grid = torch.linspace(lo, hi, grid_size, dtype=torch.float32)
    pdf = _beta_pdf_unnorm(grid, d)
    pdf = pdf / pdf.sum()

    cdf = pdf.cumsum(0)
    cdf = cdf / cdf[-1]
    targets = torch.linspace(1 / (2 * n_levels), 1 - 1 / (2 * n_levels), n_levels, dtype=torch.float32)
    centroid_idx = torch.searchsorted(cdf, targets).clamp(0, grid_size - 1)
    centroids = grid[centroid_idx]

    for _ in range(n_iter):
        dists = (grid.unsqueeze(1) - centroids.unsqueeze(0)).abs()
        assignments = dists.argmin(dim=1)
        new_centroids = torch.zeros_like(centroids)
        for i in range(n_levels):
            mask = assignments == i
            if mask.any():
                w = pdf[mask]
                new_centroids[i] = (grid[mask] * w).sum() / w.sum()
            else:
                new_centroids[i] = centroids[i]
        centroids = new_centroids

    return centroids.sort().values.contiguous()


class _TurboQuantCPURef:
    def __init__(self, d: int, bits: int, seed: int = 0):
        self.d = d
        self.bits = bits
        self.Q = make_rotation_matrix(d, seed)
        self.Q_T = self.Q.T.contiguous()
        self.centroids = build_lloyd_max_codebook(d, bits)
        self.boundaries = ((self.centroids[:-1] + self.centroids[1:]) / 2).contiguous()

    def quantize(self, x: torch.Tensor) -> dict:
        norms = x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        x_unit = x / norms
        x_rot = x_unit @ self.Q_T
        idx = torch.searchsorted(self.boundaries, x_rot.contiguous())
        return {
            "idx": idx.to(torch.uint8),
            "norms": norms.squeeze(-1).half(),
        }

    def dequantize(self, q: dict) -> torch.Tensor:
        x_rot = self.centroids[q["idx"].long()]
        x_unit = x_rot @ self.Q
        return x_unit * q["norms"].float().unsqueeze(-1)


# ----------------------------------------------------------------------------
# TurboQuantMSE_CM
# ----------------------------------------------------------------------------

class TurboQuantMSE_CM:
    def __init__(self, head_size: int, num_kv_heads: int = 1, bits: int = 4, rotation_seed: int = 0):
        assert head_size % 16 == 0
        assert bits <= 8

        self.head_size = head_size
        self.num_kv_heads = num_kv_heads
        self.bits = bits
        self.wg_size = 16

        self.ref = _TurboQuantCPURef(head_size, bits, rotation_seed)

        self.q_t = self.ref.Q_T.half().contiguous()
        self.q = self.ref.Q.half().contiguous()
        self.centroids = self.ref.centroids.half().contiguous()
        self.boundaries = self.ref.boundaries.half().contiguous()

        self._t_q_t = None
        self._t_q = None
        self._t_centroids = None
        self._t_boundaries = None

        src = r'''#include "turboquant_cm_ref.hpp"'''
        cwd = os.path.dirname(os.path.realpath(__file__))
        jit_option = '-abortonspill -noschedule '

        # Define all macros used by all kernels in the shared header.
        self.kernels = cl.kernels(
            src,
            (
                f' -cmc -Qxcm_jit_option="{jit_option}" -Qxcm_register_file_size=256 -mCM_printregusage -I{cwd}'
                f" -DWG_SIZE={self.wg_size}"
                f" -DKV_HEADS_NUM={num_kv_heads}"
                f" -DHEAD_SIZE={head_size}"
                f" -DTQ_BITS={bits}"
                f" -DK_HEAD_SIZE={head_size}"
                f" -DV_HEAD_SIZE={head_size}"
                f" -DPAGED_ATTENTION_BLOCK_SIZE=16"
                f" -mdump_asm -g2"
            ),
        )

    def _ensure_quant_table_tensors(self):
        if self._t_q_t is None:
            self._t_q_t = cl.tensor(self.q_t.numpy())
        if self._t_boundaries is None:
            self._t_boundaries = cl.tensor(self.boundaries.numpy())

    def _ensure_dequant_table_tensors(self):
        if self._t_q is None:
            self._t_q = cl.tensor(self.q.numpy())
        if self._t_centroids is None:
            self._t_centroids = cl.tensor(self.centroids.numpy())

    def _gws_lws(self, n_tokens: int):
        wg_count = (n_tokens + self.wg_size - 1) // self.wg_size
        return [1, self.num_kv_heads, int(wg_count * self.wg_size)], [1, 1, self.wg_size]

    def quantize_enqueue(self, t_x, t_idx, t_norms, x_pitch: int, x_offset: int, n_tokens: int):
        self._ensure_quant_table_tensors()
        gws, lws = self._gws_lws(n_tokens)
        self.kernels.enqueue(
            "turboquant_quantize",
            gws,
            lws,
            t_x,
            self._t_q_t,
            self._t_boundaries,
            t_idx,
            t_norms,
            x_pitch,
            x_offset,
            n_tokens,
        )

    def dequantize_enqueue(self, t_idx, t_norms, t_x, x_pitch: int, x_offset: int, n_tokens: int):
        self._ensure_dequant_table_tensors()
        gws, lws = self._gws_lws(n_tokens)
        self.kernels.enqueue(
            "turboquant_dequantize",
            gws,
            lws,
            t_idx,
            t_norms,
            self._t_q,
            self._t_centroids,
            t_x,
            x_pitch,
            x_offset,
            n_tokens,
        )

    def quantize(self, x: torch.Tensor) -> dict:
        assert x.dim() == 2
        assert x.shape[1] == self.num_kv_heads * self.head_size

        n_tokens = x.shape[0]
        x_pitch = x.stride()[0]

        x_h = x.to(torch.float16).contiguous()
        idx = torch.zeros(n_tokens, self.num_kv_heads, self.head_size, dtype=torch.uint8)
        norms = torch.zeros(n_tokens, self.num_kv_heads, dtype=torch.float16)

        t_x = cl.tensor(x_h.numpy())
        t_idx = cl.tensor(idx.numpy())
        t_norms = cl.tensor(norms.numpy())

        self.quantize_enqueue(t_x, t_idx, t_norms, x_pitch=x_pitch, x_offset=0, n_tokens=n_tokens)
        cl.finish()

        return {
            "idx": torch.tensor(t_idx.numpy(), dtype=torch.uint8),
            "norms": torch.tensor(t_norms.numpy(), dtype=torch.float16),
        }

    def dequantize(self, q: dict) -> torch.Tensor:
        idx = q["idx"].to(torch.uint8).contiguous()
        norms = q["norms"].to(torch.float16).contiguous()

        assert idx.dim() == 3
        n_tokens, n_heads, d = idx.shape
        assert n_heads == self.num_kv_heads and d == self.head_size
        assert list(norms.shape) == [n_tokens, n_heads]

        x = torch.zeros(n_tokens, self.num_kv_heads * self.head_size, dtype=torch.float16)

        t_idx = cl.tensor(idx.numpy())
        t_norms = cl.tensor(norms.numpy())
        t_x = cl.tensor(x.numpy())

        self.dequantize_enqueue(t_idx, t_norms, t_x, x_pitch=x.stride()[0], x_offset=0, n_tokens=n_tokens)
        cl.finish()

        return torch.tensor(t_x.numpy(), dtype=torch.float16)

    def quantize_reference(self, x: torch.Tensor) -> dict:
        x3 = x.view(x.shape[0], self.num_kv_heads, self.head_size).float()
        q = self.ref.quantize(x3)
        return {
            "idx": q["idx"].to(torch.uint8),
            "norms": q["norms"].half(),
        }

    def dequantize_reference(self, q: dict) -> torch.Tensor:
        out = self.ref.dequantize({"idx": q["idx"], "norms": q["norms"]})
        return out.reshape(out.shape[0], self.num_kv_heads * self.head_size).half()

    @staticmethod
    @functools.lru_cache(maxsize=None)
    def create_instance(head_size: int, num_kv_heads: int = 1, bits: int = 4, rotation_seed: int = 0):
        return TurboQuantMSE_CM(
            head_size=head_size,
            num_kv_heads=num_kv_heads,
            bits=bits,
            rotation_seed=rotation_seed,
        )


# ----------------------------------------------------------------------------
# CompressedKVCache_Update_CM (single-layer, PA tensor contract)
# ----------------------------------------------------------------------------

class CompressedKVCache_Update_CM:
    def __init__(self, num_kv_heads: int, k_head_size: int, v_head_size: int, block_size: int, bits: int = 4, rotation_seed: int = 0):
        assert k_head_size % 16 == 0 and v_head_size % 16 == 0

        self.num_kv_heads = num_kv_heads
        self.k_head_size = k_head_size
        self.v_head_size = v_head_size
        self.block_size = block_size
        self.bits = bits
        self.wg_size = 16
        self.k_packed_bytes = (k_head_size * bits + 7) // 8
        self.v_packed_bytes = (v_head_size * bits + 7) // 8
        self.k_token_bytes = self.k_packed_bytes + 2
        self.v_token_bytes = self.v_packed_bytes + 2

        # Reuse TurboQuantMSE_CM as helper providers for table construction.
        self.tq_k = TurboQuantMSE_CM.create_instance(k_head_size, 1, bits, rotation_seed)
        self.tq_v = TurboQuantMSE_CM.create_instance(v_head_size, 1, bits, rotation_seed)

        self.key_q_t = self.tq_k.q_t
        self.key_boundaries = self.tq_k.boundaries
        self.value_q_t = self.tq_v.q_t
        self.value_boundaries = self.tq_v.boundaries

        self._t_key_q_t = None
        self._t_key_boundaries = None
        self._t_value_q_t = None
        self._t_value_boundaries = None

        src = r'''#include "turboquant_cm_ref.hpp"'''
        cwd = os.path.dirname(os.path.realpath(__file__))
        jit_option = '-abortonspill -noschedule '

        self.kernels = cl.kernels(
            src,
            (
                f' -cmc -Qxcm_jit_option="{jit_option}" -Qxcm_register_file_size=256 -mCM_printregusage -I{cwd}'
                f" -DWG_SIZE={self.wg_size}"
                f" -DKV_HEADS_NUM={num_kv_heads}"
                f" -DHEAD_SIZE={k_head_size}"
                f" -DTQ_BITS={bits}"
                f" -DK_HEAD_SIZE={k_head_size}"
                f" -DV_HEAD_SIZE={v_head_size}"
                f" -DPAGED_ATTENTION_BLOCK_SIZE={block_size}"
                f" -mdump_asm -g2"
            ),
        )

    def _ensure_table_tensors(self):
        if self._t_key_q_t is None:
            self._t_key_q_t = cl.tensor(self.key_q_t.numpy())
        if self._t_key_boundaries is None:
            self._t_key_boundaries = cl.tensor(self.key_boundaries.numpy())
        if self._t_value_q_t is None:
            self._t_value_q_t = cl.tensor(self.value_q_t.numpy())
        if self._t_value_boundaries is None:
            self._t_value_boundaries = cl.tensor(self.value_boundaries.numpy())

    def __call__(self,
                 key,
                 value,
                 key_cache,
                 value_cache,
                 past_lens,
                 subsequence_begins,
                 block_indices,
                 block_indices_begins,
                 n_repeats: int = 1):
        n_tokens = key.shape[0]
        check_perf = n_repeats > 1
        self._ensure_table_tensors()

        t_key = cl.tensor(key.to(torch.float16).contiguous().numpy())
        t_value = cl.tensor(value.to(torch.float16).contiguous().numpy())

        t_key_cache = cl.tensor(key_cache.to(torch.uint8).contiguous().numpy())
        t_value_cache = cl.tensor(value_cache.to(torch.uint8).contiguous().numpy())

        t_past_lens = cl.tensor(torch.tensor(past_lens, dtype=torch.int32).numpy())
        t_subsequence_begins = cl.tensor(torch.tensor(subsequence_begins, dtype=torch.int32).numpy())
        t_block_indices = cl.tensor(torch.tensor(block_indices, dtype=torch.int32).numpy())
        t_block_indices_begins = cl.tensor(torch.tensor(block_indices_begins, dtype=torch.int32).numpy())

        wg_count = (n_tokens + self.wg_size - 1) // self.wg_size
        gws = [1, self.num_kv_heads, int(wg_count * self.wg_size)]
        lws = [1, 1, self.wg_size]
        
        # Warmup run (to exclude one-time setup overheads from timing).
        if check_perf:
            if enable_debug_prints:
                print(f'{Colors.GREEN}warmup compressed_kv_cache_update_tq {gws=} {lws=}{Colors.END}')
            self.kernels.enqueue(
                "compressed_kv_cache_update_tq",
                gws,
                lws,
                t_key,
                t_value,
                t_past_lens,
                t_block_indices,
                t_block_indices_begins,
                t_subsequence_begins,
                t_key_cache,
                t_value_cache,
                self._t_key_q_t,
                self._t_key_boundaries,
                self._t_value_q_t,
                self._t_value_boundaries,
                key.stride()[0],
                0,
                value.stride()[0],
                0,
                len(past_lens),
            )
            cl.finish()

        for i in range(n_repeats):
            if enable_debug_prints:
                print(f'{Colors.GREEN}calling compressed_kv_cache_update_tq {gws=} {lws=} at {i + 1}/{n_repeats}{Colors.END}')
            self.kernels.enqueue(
                "compressed_kv_cache_update_tq",
                gws,
                lws,
                t_key,
                t_value,
                t_past_lens,
                t_block_indices,
                t_block_indices_begins,
                t_subsequence_begins,
                t_key_cache,
                t_value_cache,
                self._t_key_q_t,
                self._t_key_boundaries,
                self._t_value_q_t,
                self._t_value_boundaries,
                key.stride()[0],
                0,
                value.stride()[0],
                0,
                len(past_lens),
            )
        
        # Wait for completion and optionally print performance info.
        if check_perf:
            ns = cl.finish()
            for k, time_opt in enumerate(ns):
                total_bytes = (
                    n_tokens * self.num_kv_heads *
                    (2 * (self.k_head_size + self.v_head_size) + self.k_token_bytes + self.v_token_bytes)
                )
                tput = total_bytes / time_opt
                print(f'(compressed_kv_cache_update_tq)TPUT_{k}:[{total_bytes*1e-6:.3f} MB] {time_opt*1e-3:.0f} us, {tput:,.2f} GB/s')
        else:
            cl.finish()

        return torch.tensor(t_key_cache.numpy(), dtype=torch.uint8), torch.tensor(t_value_cache.numpy(), dtype=torch.uint8)


@functools.lru_cache(maxsize=None)
def create_turboquant_mse_cm(head_size: int, num_kv_heads: int, bits: int, rotation_seed: int = 0):
    return TurboQuantMSE_CM.create_instance(
        head_size=head_size,
        num_kv_heads=num_kv_heads,
        bits=bits,
        rotation_seed=rotation_seed,
    )
