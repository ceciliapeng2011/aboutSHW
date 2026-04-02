import functools
import os
import sys
from dataclasses import dataclass

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from clops import cl
from turboquant_cm import TurboQuantMSE_CM, CompressedKVCache_Update_CM


cl.profiling(True)
torch.manual_seed(0)
torch.set_printoptions(linewidth=1024)


# Keep this aligned with -Qxcm_register_file_size in kernel build options.
# It is also used by OpenVINO-style q_chunking budget estimation.
PA_CM_REGISTER_FILE_SIZE = 256


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


def get_single_token_q_chunking_openvino_style(
    num_heads: int,
    num_kv_heads: int,
    kv_partition_size: int,
    xe_arch: int,
    kv_step: int,
) -> tuple[int, int]:
    # Match OpenVINO get_single_token_q_chunking() logic.
    max_repeat_count = 8
    reg_m = 1  # RepeatCount
    bytes_per_float = 4
    reg_n = 8 if xe_arch == 1 else 16

    q_heads_per_kv_head = num_heads // num_kv_heads
    kv_partition_step_num = kv_partition_size // kv_step
    rs_cols = reg_m * kv_partition_step_num * reg_n

    grf_bytes = 32 if xe_arch == 1 else 64
    budget_bytes = PA_CM_REGISTER_FILE_SIZE * grf_bytes - 1
    max_q_by_matrix = budget_bytes // (bytes_per_float * rs_cols)
    if max_q_by_matrix < 1:
        max_q_by_matrix = 1

    target_chunk = min(max_repeat_count, max_q_by_matrix)
    q_head_chunk_size = min(q_heads_per_kv_head, target_chunk)
    while q_head_chunk_size > 1 and (q_heads_per_kv_head % q_head_chunk_size) != 0:
        q_head_chunk_size -= 1
    q_head_chunks_per_kv_head = q_heads_per_kv_head // q_head_chunk_size
    return int(q_head_chunks_per_kv_head), int(q_head_chunk_size)


def get_default_kv_partition_size(block_size: int) -> int:
    # Match OpenVINO get_partition_size behavior used by single-token path:
    # if block size is legacy (<256), use 128; otherwise use 256.

    # It is an intriguing heuristic that seems to balance register usage and parallelism for typical block sizes, 
    # but the optimal choice may vary based on the specific workload and hardware characteristics.
    # Sweeping this parameter could be an interesting avenue for performance tuning.
    return 256 if block_size >= 256 else 128


def _check_close(actual: torch.Tensor, expected: torch.Tensor, atol: float = 1e-2, rtol: float = 1e-3) -> None:
    if torch.allclose(actual, expected, atol=atol, rtol=rtol):
        return
    close_mask = torch.isclose(actual, expected, atol=atol, rtol=rtol)
    bad = torch.where(~close_mask)
    raise AssertionError(
        "Tensor mismatch\n"
        f"indices={bad}\n"
        f"actual={actual[bad]}\n"
        f"expected={expected[bad]}"
    )


def _quant_per_token(kv: torch.Tensor) -> torch.Tensor:
    blk_num, kv_heads, blk_size, head_size = kv.shape
    kv_max = kv.amax(dim=3, keepdim=True)
    kv_min = kv.amin(dim=3, keepdim=True)
    qrange = kv_max - kv_min

    intmax = 255.0
    intmin = 0.0
    intrange = intmax - intmin

    kv_scale = torch.zeros(blk_num, kv_heads, blk_size, 1, dtype=torch.float16)
    kv_scale[qrange != 0] = (intrange / qrange[qrange != 0]).to(dtype=torch.float16)
    kv_zp = ((0.0 - kv_min) * kv_scale + intmin).to(dtype=torch.float16)

    kv_u8 = torch.round(kv * kv_scale + kv_zp).clamp(intmin, intmax).to(dtype=torch.uint8).reshape(blk_num, kv_heads, -1)

    dq_scale = torch.zeros(blk_num, kv_heads, blk_size, 1, dtype=torch.float16)
    dq_scale[kv_scale != 0] = (1.0 / kv_scale[kv_scale != 0]).to(dtype=torch.float16)
    dq_scale_u8 = dq_scale.view(dtype=torch.uint8).reshape(blk_num, kv_heads, -1)
    kv_zp_u8 = kv_zp.view(dtype=torch.uint8).reshape(blk_num, kv_heads, -1)

    return torch.concat((kv_u8, dq_scale_u8, kv_zp_u8), dim=-1)


def _dequant_per_token(kv: torch.Tensor, head_size: int, blk_size: int) -> torch.Tensor:
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


def _quant_per_channel(kv: torch.Tensor) -> torch.Tensor:
    blk_num, kv_heads, blk_size, head_size = kv.shape
    kv_max = kv.amax(dim=2, keepdim=True)
    kv_min = kv.amin(dim=2, keepdim=True)
    qrange = kv_max - kv_min

    intmax = 255.0
    intmin = 0.0
    intrange = intmax - intmin

    kv_scale = torch.zeros(blk_num, kv_heads, 1, head_size, dtype=torch.float16)
    kv_scale[qrange != 0] = (intrange / qrange[qrange != 0]).to(dtype=torch.float16)
    kv_zp = ((0.0 - kv_min) * kv_scale + intmin).to(dtype=torch.float16)

    kv_u8 = torch.round(kv * kv_scale + kv_zp).clamp(intmin, intmax).to(dtype=torch.uint8).reshape(blk_num, kv_heads, -1)

    dq_scale = torch.zeros(blk_num, kv_heads, 1, head_size, dtype=torch.float16)
    dq_scale[kv_scale != 0] = (1.0 / kv_scale[kv_scale != 0]).to(dtype=torch.float16)
    dq_scale_u8 = dq_scale.view(dtype=torch.uint8).reshape(blk_num, kv_heads, -1)
    kv_zp_u8 = kv_zp.view(dtype=torch.uint8).reshape(blk_num, kv_heads, -1)

    return torch.concat((kv_u8, dq_scale_u8, kv_zp_u8), dim=-1)


def _dequant_per_channel(kv: torch.Tensor, head_size: int, blk_size: int) -> torch.Tensor:
    blk_num, kv_head_num, _ = kv.shape
    kv_u8 = kv[:, :, : head_size * blk_size].to(dtype=torch.float16).reshape(blk_num, kv_head_num, blk_size, head_size)
    kv_scale = kv[:, :, head_size * blk_size : (head_size * blk_size + head_size * 2)].view(dtype=torch.float16).reshape(
        blk_num,
        kv_head_num,
        1,
        head_size,
    )
    kv_zp = kv[:, :, (head_size * blk_size + head_size * 2) : (head_size * blk_size + head_size * 4)].view(
        dtype=torch.float16
    ).reshape(
        blk_num,
        kv_head_num,
        1,
        head_size,
    )
    return (kv_u8 - kv_zp) * kv_scale


@dataclass(frozen=True)
class DecodingCase:
    num_heads: int = 32
    num_kv_heads: int = 8
    head_size: int = 128
    block_size: int = 256
    kv_len: int = 4097
    kv_cache_compression: bool = False
    kv_cache_quant_mode: str = "by_token"
    turboquant_enabled: bool = False
    turboquant_bits: int = 4


class PaSingleTokenRunner:
    def __init__(
        self,
        num_heads: int,
        num_kv_heads: int,
        head_size: int,
        block_size: int,
        kv_cache_compression: bool,
        kv_cache_quant_mode: str,
        turboquant_enabled: bool,
        turboquant_bits: int,
        kv_partition_size: int | None = None,
        reduce_split_step: int = 8,
    ):
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.block_size = block_size
        self.kv_cache_compression = kv_cache_compression
        self.kv_cache_quant_mode = kv_cache_quant_mode
        self.turboquant_enabled = turboquant_enabled
        self.turboquant_bits = turboquant_bits

        self.cm_grf_width = get_cm_grf_width()
        self.xe_arch = 1 if self.cm_grf_width == 256 else 2
        self.kv_step = 8 if self.xe_arch == 1 else 16

        self.use_turboquant_kernel = bool(self.turboquant_enabled and self.xe_arch >= 2)
        if self.turboquant_enabled and self.xe_arch == 1:
            raise ValueError("TurboQuant kernel requires XE_ARCH>=2")

        self.use_quant_bytoken_kernel = bool(
            (not self.use_turboquant_kernel)
            and self.kv_cache_compression
            and (self.kv_cache_quant_mode == "by_token")
        )

        if self.use_turboquant_kernel:
            self.sdpa_2nd_kernel_name = "cm_sdpa_2nd_turboquant"
        elif self.use_quant_bytoken_kernel:
            self.sdpa_2nd_kernel_name = "cm_sdpa_2nd_quant_bytoken"
        else:
            self.sdpa_2nd_kernel_name = "cm_sdpa_2nd"

        self.kv_partition_size = int(
            get_default_kv_partition_size(self.block_size) if kv_partition_size is None else kv_partition_size
        )
        if self.kv_partition_size <= 0:
            raise ValueError("kv_partition_size must be > 0")
        if self.kv_partition_size % self.block_size != 0:
            raise ValueError("kv_partition_size must be divisible by block_size")

        self.reduce_split_step = int(reduce_split_step)
        if self.reduce_split_step <= 0:
            raise ValueError("reduce_split_step must be > 0")
        if self.head_size % self.reduce_split_step != 0:
            raise ValueError("reduce_split_step must divide head_size")

        self.q_head_chunks_per_kv_head, self.q_head_chunk_size = get_single_token_q_chunking_openvino_style(
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            kv_partition_size=self.kv_partition_size,
            xe_arch=self.xe_arch,
            kv_step=self.kv_step,
        )
        self.scale_factor = 1.0 / (self.head_size**0.5)

        if self.use_turboquant_kernel:
            tq = TurboQuantMSE_CM.create_instance(head_size=self.head_size, num_kv_heads=1, bits=self.turboquant_bits)
            self._t_tq_q_t_host = torch.from_numpy(tq.q_t.numpy())
            self._t_tq_centroids = cl.tensor(tq.centroids.numpy())
        else:
            self._t_tq_q_t_host = None
            self._t_tq_centroids = None

    def _prepare_query_for_kernel(self, query: torch.Tensor) -> torch.Tensor:
        if not self.use_turboquant_kernel:
            return query
        if self._t_tq_q_t_host is None:
            raise RuntimeError("TurboQuant Q rotation matrix is not initialized")
        q_t = self._t_tq_q_t_host.to(dtype=torch.float32)
        return torch.matmul(query.float(), q_t).to(dtype=query.dtype)

    @staticmethod
    @functools.cache
    def create_instance(
        num_heads: int,
        num_kv_heads: int,
        head_size: int,
        block_size: int,
        kv_cache_compression: bool,
        kv_cache_quant_mode: str,
        turboquant_enabled: bool,
        turboquant_bits: int,
        kv_partition_size: int | None = None,
        reduce_split_step: int = 8,
    ):
        return PaSingleTokenRunner(
            num_heads,
            num_kv_heads,
            head_size,
            block_size,
            kv_cache_compression,
            kv_cache_quant_mode,
            turboquant_enabled,
            turboquant_bits,
            kv_partition_size,
            reduce_split_step,
        )

    @staticmethod
    @functools.cache
    def _create_kernels_cached(
        num_heads: int,
        num_kv_heads: int,
        head_size: int,
        kv_step: int,
        block_size: int,
        kv_partition_size: int,
        reduce_split_step: int,
        clean_unused_kvcache: int,
        kv_cache_compression: int,
        kv_cache_compression_by_token: int,
        kv_cache_turboquant: int,
        use_turboquant_kernel: int,
        use_quant_bytoken_kernel: int,
        tq_bits: int,
        xe_arch: int,
        q_head_chunks_per_kv_head: int,
        q_head_chunk_size: int,
        scale_factor: float,
    ):
        if use_turboquant_kernel:
            cm_src_file = "pa_single_token_turboquant.cm"
        elif use_quant_bytoken_kernel:
            cm_src_file = "pa_generate_quant_bytoken.cm"
        else:
            cm_src_file = "pa_single_token.cm"
        src = f'''
                #include "{cm_src_file}"
                #include "pa_single_token_finalization.cm"
                '''
        cwd = os.path.dirname(os.path.realpath(__file__))
        return cl.kernels(
            src,
            f'''-cmc -Qxcm_jit_option=""
                        -mCM_printregusage -mdump_asm -g2
                        -Qxcm_register_file_size={PA_CM_REGISTER_FILE_SIZE} -I{cwd}
                        -DHEADS_NUM={num_heads} -DKV_HEADS_NUM={num_kv_heads} -DHEAD_SIZE={head_size}
                        -DQ_STEP=32 -DKV_STEP={kv_step}
                        -DWG_SIZE=1 -DKV_BLOCK_SIZE={block_size}
                        -DKV_PARTITION_SIZE={kv_partition_size} -DREDUCE_SPLIT_SIZE={reduce_split_step}
                        -DCLEAN_UNUSED_KVCACHE={clean_unused_kvcache}
                        -DKV_CACHE_COMPRESSION={kv_cache_compression}
                        -DKV_CACHE_COMPRESSION_BY_TOKEN={kv_cache_compression_by_token}
                        -DKV_CACHE_TURBOQUANT={kv_cache_turboquant}
                        -DUSE_TURBOQUANT_KERNEL={use_turboquant_kernel}
                        -DTQ_BITS={tq_bits}
                        -DXE_ARCH={xe_arch}
                        -DQ_head_chunks_per_kv_head={q_head_chunks_per_kv_head}
                        -DQ_head_chunk_size={q_head_chunk_size}
                        -DSCALE_FACTOR={scale_factor}''',
        )

    def _create_kernels(self):
        # TurboQuant path consumes packed/compressed KV cache (packed K + fp16 norm metadata).
        kv_cache_compression_jit = int(self.kv_cache_compression or self.use_turboquant_kernel)
        kv_cache_by_token_jit = 1 if self.use_turboquant_kernel else int(self.kv_cache_quant_mode == "by_token")
        return self._create_kernels_cached(
            self.num_heads,
            self.num_kv_heads,
            self.head_size,
            self.kv_step,
            self.block_size,
            self.kv_partition_size,
            self.reduce_split_step,
            1,
            kv_cache_compression_jit,
            kv_cache_by_token_jit,
            int(self.turboquant_enabled),
            int(self.use_turboquant_kernel),
            int(self.use_quant_bytoken_kernel),
            int(self.turboquant_bits),
            self.xe_arch,
            int(self.q_head_chunks_per_kv_head),
            int(self.q_head_chunk_size),
            self.scale_factor,
        )

    @staticmethod
    def _validate_single_subsequence(subsequence_begins: torch.Tensor) -> None:
        if int(subsequence_begins.numel()) != 2:
            raise ValueError("single subsequence only: subsequence_begins must have len == 2")
        if int((subsequence_begins[1] - subsequence_begins[0]).item()) != 1:
            raise ValueError("single subsequence only: num_tokens must be 1")

    def __call__(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        past_lens: torch.Tensor,
        block_indices: torch.Tensor,
        block_indices_begins: torch.Tensor,
        subsequence_begins: torch.Tensor,
        out: torch.Tensor,
        n_repeats: int = 1,
    ) -> torch.Tensor:
        self._validate_single_subsequence(subsequence_begins)

        if out.dtype != torch.float16:
            raise ValueError(f"out dtype mismatch: got {out.dtype}, expected torch.float16")

        kernels = self._create_kernels()
        query_kernel = self._prepare_query_for_kernel(query)

        batch_size = int(query.shape[0])
        max_context_len = int(past_lens.max().item()) + 1
        kv_partition_num = (max_context_len + self.kv_partition_size - 1) // self.kv_partition_size

        gws = [batch_size, self.num_kv_heads * self.q_head_chunks_per_kv_head, kv_partition_num]
        lws = [1, 1, 1]
        gws_2 = [batch_size, self.num_heads, self.head_size // self.reduce_split_step]
        lws_2 = [1, 1, 1]

        for _ in range(n_repeats):
            t_q = cl.tensor(query_kernel.detach().numpy())
            t_k = cl.tensor(key_cache.contiguous().detach().numpy())
            t_v = cl.tensor(value_cache.contiguous().detach().numpy())
            t_past_lens = cl.tensor(past_lens.detach().numpy())
            t_block_indices = cl.tensor(block_indices.detach().numpy())
            t_block_indices_begins = cl.tensor(block_indices_begins.detach().numpy())
            t_subsequence_begins = cl.tensor(subsequence_begins.detach().numpy())
            t_out = cl.tensor([batch_size, self.num_heads, kv_partition_num, self.head_size], np.dtype(np.float32))
            t_out_final = cl.tensor(out.contiguous().detach().numpy())
            t_lse = cl.tensor([batch_size, self.num_heads, kv_partition_num], np.dtype(np.float32))

            if self.use_turboquant_kernel:
                kernels.enqueue(
                    self.sdpa_2nd_kernel_name,
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
                    t_lse,
                    self._t_tq_centroids,
                    1,
                )
            else:
                kernels.enqueue(
                    self.sdpa_2nd_kernel_name,
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
                    t_lse,
                    1,
                )
            kernels.enqueue(
                "cm_sdpa_2nd_reduce",
                gws_2,
                lws_2,
                t_out,
                t_out_final,
                t_lse,
                kv_partition_num,
            )

            cl.finish()
            out.copy_(torch.from_numpy(t_out_final.numpy()))

        return out


def _build_single_subsequence_inputs(case: DecodingCase):
    batch = 1
    q_len = 1
    low, high = -127, 128

    new_kv_len = (case.kv_len + case.block_size - 1) // case.block_size * case.block_size
    total_blk_num = new_kv_len // case.block_size

    # BLHS
    q_blhs = torch.randint(low, high, [batch, q_len, case.num_heads, case.head_size], dtype=torch.int32).to(torch.float16) / high
    k_blhs = torch.randint(low, high, [batch, new_kv_len, case.num_kv_heads, case.head_size], dtype=torch.int32).to(torch.float16) / high
    v_blhs = torch.randint(low, high, [batch, new_kv_len, case.num_kv_heads, case.head_size], dtype=torch.int32).to(torch.float16) / high

    # BHLS
    q_bhls = q_blhs.transpose(1, 2).contiguous()
    k_bhls = k_blhs.transpose(1, 2).contiguous()
    v_bhls = v_blhs.transpose(1, 2).contiguous()

    # Keep old script semantics for unused cache area.
    k_bhls.view(torch.uint16)[:, :, case.kv_len:, :] = 0
    v_bhls.view(torch.uint16)[:, :, case.kv_len:, :] = 0xFE00

    # [B,H,L,S] -> [L,H,S] -> [blk,blk_sz,H,S] -> [blk,H,blk_sz,S]
    k_blocks = k_bhls[0].transpose(0, 1).reshape(total_blk_num, case.block_size, case.num_kv_heads, case.head_size).transpose(1, 2).contiguous()
    v_blocks = v_bhls[0].transpose(0, 1).reshape(total_blk_num, case.block_size, case.num_kv_heads, case.head_size).transpose(1, 2).contiguous()

    if case.turboquant_enabled:
        tq = TurboQuantMSE_CM.create_instance(head_size=case.head_size, num_kv_heads=1, bits=case.turboquant_bits)
        tq_ref = tq.ref

        # Build compressed key cache with the CM updater implementation.
        k_packed_bytes = (case.head_size * case.turboquant_bits + 7) // 8
        key_cache_logical = torch.zeros(
            total_blk_num,
            case.num_kv_heads,
            case.block_size * (k_packed_bytes + 2),
            dtype=torch.uint8,
        )
        # updater also needs value cache argument; it is ignored for decode path here.
        value_cache_dummy = torch.zeros_like(key_cache_logical)

        updater = CompressedKVCache_Update_CM(
            num_kv_heads=case.num_kv_heads,
            k_head_size=case.head_size,
            v_head_size=case.head_size,
            block_size=case.block_size,
            bits=case.turboquant_bits,
        )

        key_tokens = k_bhls[0].transpose(0, 1).reshape(new_kv_len, case.num_kv_heads * case.head_size).contiguous()
        value_tokens = v_bhls[0].transpose(0, 1).reshape(new_kv_len, case.num_kv_heads * case.head_size).contiguous()
        _k_cache_updated, _ = updater(
            key_tokens,
            value_tokens,
            key_cache_logical,
            value_cache_dummy,
            [0],
            [0, new_kv_len],
            list(range(total_blk_num)),
            [0, total_blk_num],
            n_repeats=1,
        )
        k_cache = _k_cache_updated

        # Build rotated-domain key reference directly from packed cache bytes.
        k_packed_region = k_cache[:, :, : case.block_size * k_packed_bytes].reshape(
            total_blk_num,
            case.num_kv_heads,
            case.block_size,
            k_packed_bytes,
        )
        idx = torch.empty(
            total_blk_num,
            case.num_kv_heads,
            case.block_size,
            case.head_size,
            dtype=torch.long,
        )
        idx[:, :, :, 0::2] = (k_packed_region & 0x0F).reshape(total_blk_num, case.num_kv_heads, case.block_size, case.head_size // 2).long()
        idx[:, :, :, 1::2] = (k_packed_region >> 4).reshape(total_blk_num, case.num_kv_heads, case.block_size, case.head_size // 2).long()

        k_norms = (
            k_cache[:, :, case.block_size * k_packed_bytes : case.block_size * (k_packed_bytes + 2)]
            .view(torch.float16)
            .reshape(total_blk_num, case.num_kv_heads, case.block_size, 1)
            .float()
        )
        k_ref_blocks = (tq_ref.centroids[idx] * k_norms).to(torch.float16)

        # Reference attention is computed in rotated query/key domain.
        q_rot_blhs = torch.matmul(q_blhs.float(), tq_ref.Q_T.float()).to(torch.float16)

        # Value path follows by-token uint8 cache (+ fp16 scale/zp metadata).
        v_cache = _quant_per_token(v_blocks)
        v_ref_blocks = _dequant_per_token(v_cache, case.head_size, case.block_size)
    elif case.kv_cache_compression:
        if case.kv_cache_quant_mode == "by_token":
            k_cache = _quant_per_token(k_blocks)
            k_ref_blocks = _dequant_per_token(k_cache, case.head_size, case.block_size)
        elif case.kv_cache_quant_mode == "by_channel":
            k_cache = _quant_per_channel(k_blocks)
            k_ref_blocks = _dequant_per_channel(k_cache, case.head_size, case.block_size)
        else:
            raise ValueError(f"Unsupported kv_cache_quant_mode={case.kv_cache_quant_mode}")

        # value path supports per-token quantization in pa_2nd_token flow
        v_cache = _quant_per_token(v_blocks)
        v_ref_blocks = _dequant_per_token(v_cache, case.head_size, case.block_size)
    else:
        k_cache = k_blocks
        v_cache = v_blocks
        k_ref_blocks = k_blocks
        v_ref_blocks = v_blocks

    block_indices = torch.randperm(total_blk_num, dtype=torch.int32)
    if case.turboquant_enabled:
        key_cache = torch.empty_like(k_cache)
        value_cache = torch.empty_like(v_cache)
        key_cache[block_indices.to(dtype=torch.long)] = k_cache
        value_cache[block_indices.to(dtype=torch.long)] = v_cache
    else:
        key_cache = torch.empty_like(k_cache)
        value_cache = torch.empty_like(v_cache)
        key_cache[block_indices.to(dtype=torch.long)] = k_cache
        value_cache[block_indices.to(dtype=torch.long)] = v_cache

    # Reference in logical order.
    q_ref_bhls = q_bhls
    if case.turboquant_enabled:
        q_ref_bhls = q_rot_blhs.transpose(1, 2).contiguous()

    k_ref_bhls = k_ref_blocks.transpose(1, 2).reshape(batch, new_kv_len, case.num_kv_heads, case.head_size).transpose(1, 2).contiguous()
    v_ref_bhls = v_ref_blocks.transpose(1, 2).reshape(batch, new_kv_len, case.num_kv_heads, case.head_size).transpose(1, 2).contiguous()

    attention_mask = torch.zeros([batch, 1, q_len, case.kv_len], dtype=torch.float16)
    expected = F.scaled_dot_product_attention(
        q_ref_bhls,
        k_ref_bhls[:, :, : case.kv_len, :],
        v_ref_bhls[:, :, : case.kv_len, :],
        attention_mask,
        dropout_p=0.0,
        enable_gqa=(case.num_heads > case.num_kv_heads),
    ).transpose(1, 2).contiguous()

    past_lens = torch.tensor([case.kv_len - 1], dtype=torch.int32)
    block_indices_begins = torch.tensor([0, total_blk_num], dtype=torch.int32)
    subsequence_begins = torch.tensor([0, 1], dtype=torch.int32)

    return {
        "query": q_bhls,
        "key_cache": key_cache,
        "value_cache": value_cache,
        "past_lens": past_lens,
        "block_indices": block_indices,
        "block_indices_begins": block_indices_begins,
        "subsequence_begins": subsequence_begins,
        "expected": expected,
    }


def _case_id(case: DecodingCase) -> str:
    return (
        f"1x{case.kv_len}"
        f"_h{case.num_heads}"
        f"_kv{case.num_kv_heads}"
        f"_hs{case.head_size}"
        f"_bls{case.block_size}"
        f"_cmpr{int(case.kv_cache_compression)}"
        f"_qm{case.kv_cache_quant_mode}"
        f"_tq{int(case.turboquant_enabled)}"
        f"_tqb{case.turboquant_bits}"
    )


TEST_BLOCK_SIZES = (16, 256)


GENERATE_ONLY_SINGLE_SUBSEQ_CASES = (
    tuple(
        DecodingCase(kv_len=129, block_size=block_size, kv_cache_compression=False, kv_cache_quant_mode="by_token")
        for block_size in TEST_BLOCK_SIZES
    )
    + tuple(
        DecodingCase(kv_len=1025, block_size=block_size, kv_cache_compression=True, kv_cache_quant_mode="by_token")
        for block_size in TEST_BLOCK_SIZES
    )
    + tuple(
        DecodingCase(kv_len=1025, block_size=block_size, kv_cache_compression=True, kv_cache_quant_mode="by_channel")
        for block_size in TEST_BLOCK_SIZES
    )
    + tuple(
        DecodingCase(
            num_heads=8,
            num_kv_heads=2,
            head_size=64,
            block_size=block_size,
            kv_len=513,
            kv_cache_compression=False,
        )
        for block_size in TEST_BLOCK_SIZES
    )
)


@pytest.mark.parametrize("case", GENERATE_ONLY_SINGLE_SUBSEQ_CASES, ids=_case_id)
def test_pa_smoke_paged_attention_generate_only(case: DecodingCase):
    data = _build_single_subsequence_inputs(case)

    runner = PaSingleTokenRunner.create_instance(
        case.num_heads,
        case.num_kv_heads,
        case.head_size,
        case.block_size,
        case.kv_cache_compression,
        case.kv_cache_quant_mode,
        case.turboquant_enabled,
        case.turboquant_bits,
    )

    output = torch.empty([1, 1, case.num_heads, case.head_size], dtype=torch.float16)
    output = runner(
        data["query"],
        data["key_cache"],
        data["value_cache"],
        data["past_lens"],
        data["block_indices"],
        data["block_indices_begins"],
        data["subsequence_begins"],
        output,
        n_repeats=1,
    )

    assert torch.isfinite(output).all().item()
    tol = (3e-2, 3e-2) if (case.kv_cache_compression or case.turboquant_enabled) else (1e-2, 1e-3)
    _check_close(output, data["expected"], atol=tol[0], rtol=tol[1])


TURBOQUANT_SINGLE_SUBSEQ_CASES = (
    tuple(
        DecodingCase(
            kv_len=1025,
            block_size=block_size,
            kv_cache_compression=False,
            turboquant_enabled=True,
            turboquant_bits=4,
        )
        for block_size in TEST_BLOCK_SIZES
    )
    + tuple(
        DecodingCase(
            num_heads=8,
            num_kv_heads=2,
            head_size=64,
            block_size=block_size,
            kv_len=513,
            kv_cache_compression=False,
            turboquant_enabled=True,
            turboquant_bits=4,
        )
        for block_size in TEST_BLOCK_SIZES
    )
)


@pytest.mark.parametrize("case", TURBOQUANT_SINGLE_SUBSEQ_CASES, ids=_case_id)
def test_pa_smoke_paged_attention_generate_only_turboquant(case: DecodingCase):
    data = _build_single_subsequence_inputs(case)

    runner = PaSingleTokenRunner.create_instance(
        case.num_heads,
        case.num_kv_heads,
        case.head_size,
        case.block_size,
        case.kv_cache_compression,
        case.kv_cache_quant_mode,
        case.turboquant_enabled,
        case.turboquant_bits,
    )

    output = torch.empty([1, 1, case.num_heads, case.head_size], dtype=torch.float16)
    output = runner(
        data["query"],
        data["key_cache"],
        data["value_cache"],
        data["past_lens"],
        data["block_indices"],
        data["block_indices_begins"],
        data["subsequence_begins"],
        output,
        n_repeats=1,
    )

    assert torch.isfinite(output).all().item()
    _check_close(output, data["expected"], atol=5e-2, rtol=5e-2)


def test_pa_turboquant_xe1_raises(monkeypatch: pytest.MonkeyPatch):
    PaSingleTokenRunner.create_instance.cache_clear()
    monkeypatch.setattr(sys.modules[__name__], "get_cm_grf_width", lambda: 256)

    case = DecodingCase(kv_len=513, turboquant_enabled=True, kv_cache_compression=False)
    with pytest.raises(ValueError, match="XE_ARCH>=2"):
        PaSingleTokenRunner.create_instance(
            case.num_heads,
            case.num_kv_heads,
            case.head_size,
            case.block_size,
            case.kv_cache_compression,
            case.kv_cache_quant_mode,
            case.turboquant_enabled,
            case.turboquant_bits,
        )

    PaSingleTokenRunner.create_instance.cache_clear()


def _run_bandwidth_measurement(
    case: DecodingCase,
    loop_cnt: int = 100,
    warmup: int = 5,
    kv_partition_size: int | None = None,
    reduce_split_step: int = 8,
) -> dict[str, float]:
    data = _build_single_subsequence_inputs(case)
    runner = PaSingleTokenRunner.create_instance(
        case.num_heads,
        case.num_kv_heads,
        case.head_size,
        case.block_size,
        case.kv_cache_compression,
        case.kv_cache_quant_mode,
        case.turboquant_enabled,
        case.turboquant_bits,
        kv_partition_size,
        reduce_split_step,
    )
    kernels = runner._create_kernels()

    query = runner._prepare_query_for_kernel(data["query"])
    key_cache = data["key_cache"]
    value_cache = data["value_cache"]
    past_lens = data["past_lens"]
    block_indices = data["block_indices"]
    block_indices_begins = data["block_indices_begins"]
    subsequence_begins = data["subsequence_begins"]

    batch = int(query.shape[0])
    max_context_len = int(past_lens.max().item()) + 1
    kv_partition_num = (max_context_len + runner.kv_partition_size - 1) // runner.kv_partition_size
    gws = [batch, runner.num_kv_heads * runner.q_head_chunks_per_kv_head, kv_partition_num]
    lws = [1, 1, 1]
    gws_2 = [batch, runner.num_heads, runner.head_size // runner.reduce_split_step]
    lws_2 = [1, 1, 1]

    all_layers = []
    mem_size = 0
    while len(all_layers) < loop_cnt and mem_size < 8e9:
        t_q = cl.tensor(query.detach().numpy())
        t_k = cl.tensor(key_cache.contiguous().detach().numpy())
        t_v = cl.tensor(value_cache.contiguous().detach().numpy())
        t_out = cl.tensor([batch, runner.num_heads, kv_partition_num, runner.head_size], np.dtype(np.float32))
        t_out_final = cl.tensor([batch, 1, runner.num_heads, runner.head_size], np.dtype(np.float16))
        all_layers.append((t_q, t_k, t_v, t_out, t_out_final))

        mem_size += query.numel() * query.element_size()
        mem_size += key_cache.numel() * key_cache.element_size()
        mem_size += value_cache.numel() * value_cache.element_size()

    if len(all_layers) == 0:
        raise RuntimeError("Failed to allocate perf input layers")

    t_past_lens = cl.tensor(past_lens.detach().numpy())
    t_block_indices = cl.tensor(block_indices.detach().numpy())
    t_block_indices_begins = cl.tensor(block_indices_begins.detach().numpy())
    t_subsequence_begins = cl.tensor(subsequence_begins.detach().numpy())
    t_lse = cl.tensor([batch, runner.num_heads, kv_partition_num], np.dtype(np.float32))

    # Clear any previously recorded profiling events (e.g., cm_get_grf_width).
    cl.finish()

    for i in range(loop_cnt):
        j = i % len(all_layers)
        t_q, t_k, t_v, t_out, t_out_final = all_layers[j]
        if runner.use_turboquant_kernel:
            kernels.enqueue(
                runner.sdpa_2nd_kernel_name,
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
                t_lse,
                runner._t_tq_centroids,
                1,
            )
        else:
            kernels.enqueue(
                runner.sdpa_2nd_kernel_name,
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
                t_lse,
                1,
            )
        kernels.enqueue(
            "cm_sdpa_2nd_reduce",
            gws_2,
            lws_2,
            t_out,
            t_out_final,
            t_lse,
            kv_partition_num,
        )

    latency = cl.finish()

    # Keep bandwidth size computation identical to pa_2nd_token.py (uses padded KV length).
    new_kv_len = int(key_cache.shape[0] * case.block_size)
    if case.turboquant_enabled:
        k_packed_bytes = (case.head_size * case.turboquant_bits + 7) // 8
        # TurboQuant decode kernel consumes packed key (+fp16 norm) and
        # by-token quantized value (+fp16 scale/zp per token).
        kvcache_size = new_kv_len * case.num_kv_heads * (k_packed_bytes + 2 + case.head_size + 4)
    elif case.kv_cache_compression:
        kvcache_size = new_kv_len * case.num_kv_heads * case.head_size * 1 * 2
    else:
        kvcache_size = new_kv_len * case.num_kv_heads * case.head_size * 2 * 2
    intermedia_size = batch * case.num_heads * kv_partition_num * (case.head_size + 1) * 4

    # latency format: [cm_sdpa_2nd, cm_sdpa_2nd_reduce, ...] for this measured loop only.
    expected_event_count = 2 * loop_cnt
    if len(latency) < expected_event_count:
        raise RuntimeError(f"Expected at least {expected_event_count} events, got {len(latency)}")

    cm_sdpa_2nd_total_time = 0.0
    cm_sdpa_2nd_reduce_total_time = 0.0
    num_runs = 0

    for pair_idx in range(loop_cnt):
        kv_ns = float(latency[2 * pair_idx])
        inter_ns = float(latency[2 * pair_idx + 1])
        if kv_ns <= 0 or inter_ns <= 0:
            continue
        if pair_idx < warmup:
            continue

        cm_sdpa_2nd_total_time += kv_ns
        cm_sdpa_2nd_reduce_total_time += inter_ns
        num_runs += 1

    if num_runs <= 0 or cm_sdpa_2nd_total_time <= 0 or cm_sdpa_2nd_reduce_total_time <= 0:
        raise RuntimeError("Invalid perf timing accumulation")

    return {
        "num_runs": float(num_runs),
        "cm_sdpa_2nd_bw_gbs": float(kvcache_size * num_runs / cm_sdpa_2nd_total_time),
        "cm_sdpa_2nd_reduce_bw_gbs": float(intermedia_size * num_runs / cm_sdpa_2nd_reduce_total_time),
        "cm_sdpa_2nd_ms": float(cm_sdpa_2nd_total_time * 1e-6 / num_runs),
        "cm_sdpa_2nd_reduce_ms": float(cm_sdpa_2nd_reduce_total_time * 1e-6 / num_runs),
    }


@pytest.mark.parametrize(
    ("kv_cache_compression", "kv_cache_quant_mode"),
    [
        (False, "by_token"),
        (True, "by_token"),
        (True, "by_channel"),
    ],
)
@pytest.mark.parametrize("kv_len", [2048, 4096, 8192, 16384, 24576, 32768])
@pytest.mark.parametrize("block_size", [16, 256])
def test_pa_perf_bandwidth_generate_single_subsequence_default_params(
    kv_cache_compression: bool,
    kv_cache_quant_mode: str,
    kv_len: int,
    block_size: int,
):
    if os.environ.get("RUN_PA_PERF", "0") != "1":
        pytest.skip("Set RUN_PA_PERF=1 to enable bandwidth perf test")

    case = DecodingCase(
        num_heads=32,
        num_kv_heads=8,
        head_size=128,
        block_size=block_size,
        kv_len=kv_len,
        kv_cache_compression=kv_cache_compression,
        kv_cache_quant_mode=kv_cache_quant_mode,
    )
    perf = _run_bandwidth_measurement(case, loop_cnt=10, warmup=1)

    print(
        "[perf] "
        f"cm_sdpa_2nd_bw={perf['cm_sdpa_2nd_bw_gbs']:.3f} GB/s, "
        f"cm_sdpa_2nd_reduce_bw={perf['cm_sdpa_2nd_reduce_bw_gbs']:.3f} GB/s, "
        f"cm_sdpa_2nd_ms={perf['cm_sdpa_2nd_ms']:.3f}, "
        f"cm_sdpa_2nd_reduce_ms={perf['cm_sdpa_2nd_reduce_ms']:.3f}"
    )

    assert perf["cm_sdpa_2nd_bw_gbs"] > 0.0
    assert perf["cm_sdpa_2nd_reduce_bw_gbs"] > 0.0

# Compare decoding performance of standard compressed KV cache vs TurboQuant compressed KV cache with same bitwidth, 
# to understand TurboQuant decode overhead and potential speedup over standard quantization.
def _benchmark_decoding_standard_vs_turboquant(
    kv_lens: tuple[int, ...] = (2048,),
    loop_cnt: int = 20,
    warmup: int = 1,
    turboquant_bits: int = 4,
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    
    block_size = 16

    for kv_len in kv_lens:
        std_case = DecodingCase(
            num_heads=32,
            num_kv_heads=8,
            head_size=128,
            block_size=block_size,
            kv_len=kv_len,
            kv_cache_compression=True,
            kv_cache_quant_mode="by_token",
            turboquant_enabled=False,
            turboquant_bits=turboquant_bits,
        )
        tq_case = DecodingCase(
            num_heads=32,
            num_kv_heads=8,
            head_size=128,
            block_size=block_size,
            kv_len=kv_len,
            kv_cache_compression=True,
            kv_cache_quant_mode="by_token",
            turboquant_enabled=True,
            turboquant_bits=turboquant_bits,
        )

        std_perf = _run_bandwidth_measurement(std_case, loop_cnt=loop_cnt, warmup=warmup)
        tq_perf = _run_bandwidth_measurement(tq_case, loop_cnt=loop_cnt, warmup=warmup)

        std_total_ms = std_perf["cm_sdpa_2nd_ms"] + std_perf["cm_sdpa_2nd_reduce_ms"]
        tq_total_ms = tq_perf["cm_sdpa_2nd_ms"] + tq_perf["cm_sdpa_2nd_reduce_ms"]
        speedup = std_total_ms / tq_total_ms if tq_total_ms > 0 else 0.0

        rows.append(
            {
                "kv_len": float(kv_len),
                "standard_ms": float(std_total_ms),
                "turboquant_ms": float(tq_total_ms),
                "speedup": float(speedup),
                "standard_k_bw_gbs": float(std_perf["cm_sdpa_2nd_bw_gbs"]),
                "standard_reduce_bw_gbs": float(std_perf["cm_sdpa_2nd_reduce_bw_gbs"]),
                "turboquant_k_bw_gbs": float(tq_perf["cm_sdpa_2nd_bw_gbs"]),
                "turboquant_reduce_bw_gbs": float(tq_perf["cm_sdpa_2nd_reduce_bw_gbs"]),
            }
        )

    return rows

@pytest.mark.parametrize("bench_kv_len", [24576])
def test_pa_benchmark_turboquant_vs_standard_decoding(bench_kv_len: int):
    """Benchmark-style pytest similar to triton_attention.benchmark_fused_vs_standard."""
    if os.environ.get("RUN_PA_BENCH", "0") != "1":
        pytest.skip("Set RUN_PA_BENCH=1 to enable standard-vs-turboquant benchmark")

    rows = _benchmark_decoding_standard_vs_turboquant(
        kv_lens=(bench_kv_len,),
        loop_cnt=10,
        warmup=1,
        turboquant_bits=4,
    )

    print("[benchmark] decoding standard vs turboquant")
    for row in rows:
        print(
            f"  kv_len={int(row['kv_len']):5d}  "
            f"standard={row['standard_ms']:.3f}ms  "
            f"turboquant={row['turboquant_ms']:.3f}ms  "
            f"speedup={row['speedup']:.3f}x\n"
            f"    std_bw(k/reduce)={row['standard_k_bw_gbs']:.3f}/{row['standard_reduce_bw_gbs']:.3f} GB/s  "
            f"tq_bw(k/reduce)={row['turboquant_k_bw_gbs']:.3f}/{row['turboquant_reduce_bw_gbs']:.3f} GB/s"
        )

    assert len(rows) > 0
    for row in rows:
        assert row["standard_ms"] > 0.0
        assert row["turboquant_ms"] > 0.0
        assert np.isfinite(row["speedup"])
    # assert rows[0]["speedup"] > 1.0

# Sweep different kv_partition_size with fixed reduce_split_step to find good default config.
def _benchmark_partition_size_sweep(
    kv_len: int = 24576,
    loop_cnt: int = 10,
    warmup: int = 1,
    turboquant_enabled: bool = False,
    block_sizes: tuple[int, ...] = (16, 256),
    kv_partition_sizes: tuple[int, ...] = (16, 32, 64, 128, 256),
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    reduce_split_step = 8

    # Keep reduce_split_step constant and sweep kv_partition_size.
    for block_size in block_sizes:
        case = DecodingCase(
            num_heads=32,
            num_kv_heads=8,
            head_size=128,
            block_size=block_size,
            kv_len=kv_len,
            kv_cache_compression=True,
            kv_cache_quant_mode="by_token",
            turboquant_enabled=turboquant_enabled,
            turboquant_bits=4,
        )

        if case.head_size % reduce_split_step != 0:
            raise ValueError("reduce_split_step must divide head_size")

        for kv_partition_size in kv_partition_sizes:
            try:
                perf = _run_bandwidth_measurement(
                    case,
                    loop_cnt=loop_cnt,
                    warmup=warmup,
                    kv_partition_size=kv_partition_size,
                    reduce_split_step=reduce_split_step,
                )
            except Exception as e:
                print(
                    f"[sweep] skip block_size={block_size}, kv_partition_size={kv_partition_size}, "
                    f"fixed_reduce_split_step={reduce_split_step}: {str(e).splitlines()[0]}"
                )
                continue

            total_ms = perf["cm_sdpa_2nd_ms"] + perf["cm_sdpa_2nd_reduce_ms"]
            rows.append(
                {
                    "kv_len": float(kv_len),
                    "block_size": float(block_size),
                    "kv_partition_size": float(kv_partition_size),
                    "total_ms": float(total_ms),
                    "cm_sdpa_2nd_ms": float(perf["cm_sdpa_2nd_ms"]),
                    "cm_sdpa_2nd_reduce_ms": float(perf["cm_sdpa_2nd_reduce_ms"]),
                    "cm_sdpa_2nd_bw_gbs": float(perf["cm_sdpa_2nd_bw_gbs"]),
                    "cm_sdpa_2nd_reduce_bw_gbs": float(perf["cm_sdpa_2nd_reduce_bw_gbs"]),
                }
            )

    rows.sort(key=lambda x: x["total_ms"])
    return rows

@pytest.mark.parametrize("bench_kv_len", [24576])
@pytest.mark.parametrize(
    ("turboquant_enabled", "sweep_name"),
    [
        (False, "non-TurboQuant partition-size sweep"),
        (True, "TurboQuant partition-size sweep"),
    ],
)
@pytest.mark.parametrize(
    ("block_sizes", "block_sizes_name"),
    [
        ((16,), "block_size=16"),
        ((256,), "block_size=256"),
        ((16, 256), "block_size=16,256"),
    ],
)
def test_pa_sweep_partition_size(
    bench_kv_len: int,
    turboquant_enabled: bool,
    sweep_name: str,
    block_sizes: tuple[int, ...],
    block_sizes_name: str,
):
    if os.environ.get("RUN_PA_SWEEP", "0") != "1":
        pytest.skip("Set RUN_PA_SWEEP=1 to enable partition-size sweep")

    rows = _benchmark_partition_size_sweep(
        kv_len=bench_kv_len,
        loop_cnt=10,
        warmup=1,
        turboquant_enabled=turboquant_enabled,
        block_sizes=block_sizes,
    )

    print(f"[benchmark] {sweep_name} ({block_sizes_name})")
    for row in rows:
        print(
            f"  kv_len={int(row['kv_len']):5d}  "
            f"block_size={int(row['block_size'])}  "
            f"kv_partition_size={int(row['kv_partition_size'])}  "
            f"total={row['total_ms']:.3f}ms  "
            f"k/reduce={row['cm_sdpa_2nd_ms']:.3f}/{row['cm_sdpa_2nd_reduce_ms']:.3f}ms  "
            f"bw={row['cm_sdpa_2nd_bw_gbs']:.3f}/{row['cm_sdpa_2nd_reduce_bw_gbs']:.3f} GB/s"
        )

    assert len(rows) > 0
    for row in rows:
        assert row["total_ms"] > 0.0
        assert np.isfinite(row["total_ms"])


# Usages:
# - python -m pytest test_pa_decoding.py --collect-only -q
# - python -m pytest test_pa_decoding.py -vv
# - RUN_PA_PERF=1 python -m pytest -s test_pa_decoding.py -k 'test_pa_perf_bandwidth_generate_single_subsequence_default_params' -vv
# - RUN_PA_BENCH=1 python -m pytest -s test_pa_decoding.py -k 'test_pa_benchmark_turboquant_vs_standard_decoding' -vv
#
# - RUN_PA_SWEEP=1 python -m pytest -s test_pa_decoding.py -k 'test_pa_sweep_partition_size' -vv
# [benchmark] non-TurboQuant partition-size sweep (block_size=16,256)
#   kv_len=24576  block_size=16  kv_partition_size=256  total=0.576ms  k/reduce=0.525/0.051ms  bw=95.866/31.035 GB/s
#   kv_len=24576  block_size=16  kv_partition_size=128  total=0.626ms  k/reduce=0.536/0.090ms  bw=93.917/35.045 GB/s
#   kv_len=24576  block_size=256  kv_partition_size=256  total=0.659ms  k/reduce=0.583/0.076ms  bw=86.333/20.900 GB/s
#   kv_len=24576  block_size=16  kv_partition_size=64  total=0.740ms  k/reduce=0.566/0.175ms  bw=89.001/36.314 GB/s
#   kv_len=24576  block_size=16  kv_partition_size=32  total=0.953ms  k/reduce=0.611/0.342ms  bw=82.370/37.064 GB/s
#   kv_len=24576  block_size=16  kv_partition_size=16  total=1.381ms  k/reduce=0.718/0.663ms  bw=70.054/38.265 GB/s
# [benchmark] TurboQuant partition-size sweep (block_size=16,256)
#   kv_len=24576  block_size=16  kv_partition_size=64  total=2.478ms  k/reduce=2.326/0.152ms  bw=16.737/41.635 GB/s
#   kv_len=24576  block_size=16  kv_partition_size=256  total=2.533ms  k/reduce=2.489/0.044ms  bw=15.638/36.146 GB/s
#   kv_len=24576  block_size=16  kv_partition_size=128  total=2.536ms  k/reduce=2.457/0.079ms  bw=15.845/39.929 GB/s
#   kv_len=24576  block_size=256  kv_partition_size=256  total=2.575ms  k/reduce=2.530/0.045ms  bw=15.386/35.100 GB/s
#   kv_len=24576  block_size=16  kv_partition_size=32  total=2.862ms  k/reduce=2.534/0.328ms  bw=15.364/38.675 GB/s
#   kv_len=24576  block_size=16  kv_partition_size=16  total=3.208ms  k/reduce=2.542/0.666ms  bw=15.313/38.063 GB/s