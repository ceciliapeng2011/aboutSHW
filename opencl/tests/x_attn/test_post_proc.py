import os
import torch
import numpy as np

from clops import cl
from clops.utils import Colors
from clops import compare
xe_arch = 2
print(f"xe_arch: {xe_arch}")
def div_up(a, b):
    return (a + b - 1) // b
def rnd_up(a, b):
    return (a + b - 1) // b * b

#xe_arch: 1: xe, 2: xe2
def get_cm_grf_width():
    cm_kernels = cl.kernels(r'''
    extern "C" _GENX_MAIN_ void cm_get_grf_width(int * info [[type("svmptr_t")]]) {
        info[0] = CM_GRF_WIDTH;
    }''', f"-cmc")
    t_info = cl.tensor([2], np.dtype(np.int32))
    cm_kernels.enqueue("cm_get_grf_width", [1], [1], t_info)
    return t_info.numpy()[0]
CM_GRF_WIDTH = get_cm_grf_width()
PA_WG_SIZE = 256 if CM_GRF_WIDTH == 512 else 128

def test_post_proc(HQ = 1, BLOCK_SIZE = 128):
    assert PA_WG_SIZE % BLOCK_SIZE == 0, "PA WG size must be divisible by XATTN block size."
    MERGED_Q_NUM = PA_WG_SIZE // BLOCK_SIZE
    assert MERGED_Q_NUM == 1 or MERGED_Q_NUM == 2, "PA WG size must be no less than XATTN block size."

    STRIDE = 16
    def create_kernels():
        # kernel
        src = r'''#include "xattn_post_proc.hpp"'''
        cwd = os.path.dirname(os.path.realpath(__file__))
        print(f"compiling {cwd} ...")

        jit_option = '-abortonspill -noschedule '
        kernels = cl.kernels(src, f'''-cmc -Qxcm_jit_option="{jit_option}"
                            -mCM_printregusage -mdump_asm -g2
                            -Qxcm_register_file_size=256 -I{cwd}
                            -DSTRIDE={STRIDE} -DHQ={HQ}
                            -DBLOCK_SIZE={BLOCK_SIZE}
                            -DMERGED_Q_NUM={MERGED_Q_NUM}''')
        return kernels

    kernels = create_kernels()
    sum_per_token_in_block = BLOCK_SIZE // STRIDE
    qk = [
        (4096, 4096),
        (4096+1, 4096),
        (4096, 32768),
        (4096+1, 32768)
        # (612, 612)
    ]
    for q, k in qk:
        print(f"{Colors.GREEN}Test {q=} {k=} {BLOCK_SIZE=} ... {Colors.END}")
        q_stride_pad = q // STRIDE
        q_block_valid = q_stride_pad // sum_per_token_in_block
        q_block_pad = div_up(q, BLOCK_SIZE)
        k_block_pad = div_up(k, BLOCK_SIZE)
        merged_q_blocks = div_up(q_block_pad, MERGED_Q_NUM)
        org = np.random.randint(low=0, high=2, size=[1, HQ, q_block_pad, k_block_pad], dtype=np.int8)
        if q_block_valid != q_block_pad:
            org[:,:,-1,:] = 1

        t_mask = cl.tensor(org)
        t_merged_mask = cl.tensor(np.ones([1, HQ, merged_q_blocks, k_block_pad], dtype=np.int8) * 100)

        # Build single-entry metadata for the unified kernel
        meta_np = np.array([[
            0,              # [0]  SUBSEQ_Q_BEGIN
            q,              # [1]  SUBSEQ_Q_LEN
            0,              # [2]  M
            0,              # [3]  N
            q_stride_pad,   # [4]  Q_STRIDE_PAD
            0,              # [5]  N_KQ_GROUPS
            q_block_pad,    # [6]  Q_BLOCK_PAD
            k_block_pad,    # [7]  K_BLOCK_PAD
            0,              # [8]  CAUSAL_START
            0,              # [9]  Q_START_STRIDED
            0,              # [10] BUF_OFF_KQ_MAX
            0,              # [11] BUF_OFF_EXP_SUM
            0,              # [12] BUF_OFF_MASK
            0,              # [13] BUF_OFF_MASK_WG
            0,              # [14] BLOCK_IDX_BEGIN
            0,              # [15] WG_OFFSET
        ]], dtype=np.int32)
        t_meta = cl.tensor(meta_np)

        cl.finish()
        kernels.enqueue("post_proc_mask", [merged_q_blocks, HQ, 1], [1, 1, 1], t_mask, t_merged_mask, t_meta)
        ns = cl.finish()
        t_merged_mask_np = t_merged_mask.numpy()

        t_mask_np = t_mask.numpy()
        if MERGED_Q_NUM == 2:
            if q_block_valid != q_block_pad:
                assert np.all(t_mask_np[:,:,-(q_block_pad - q_block_valid):,:] == 1), f"new pad value must be one, {q=}, {k=}"
                org_pad = np.ones([1, HQ, q_block_pad + 1, k_block_pad], dtype=np.int8)
                org_pad[:,:,:-1,:] = org
                org = org_pad
            t_merged_mask_ref = org[:,:,0::2,:] | org[:,:,1::2,:]
        else:
            t_merged_mask_ref = org.copy()
        compare(t_merged_mask_ref, t_merged_mask_np)
        assert np.all(t_merged_mask_ref == t_merged_mask_np), f"merged mask not equal to ref, {q=} {k=}"

        for i, time_opt in enumerate(ns):
            print(f'(POSTPROC)TPUT_{i}: {time_opt*1e-3:,.0f} us')
    print(f'{Colors.GREEN}test_post_proc done.{Colors.END}')

# ============================================================================
# Multi-subsequence support
# ============================================================================

XATTN_META_STRIDE_PY = 16

def test_post_proc_multi_subseq(HQ=1, BLOCK_SIZE=128):
    assert PA_WG_SIZE % BLOCK_SIZE == 0
    MERGED_Q_NUM = PA_WG_SIZE // BLOCK_SIZE
    assert MERGED_Q_NUM == 1 or MERGED_Q_NUM == 2

    STRIDE = 16
    sum_per_token_in_block = BLOCK_SIZE // STRIDE

    src = r'''#include "xattn_post_proc.hpp"'''
    cwd = os.path.dirname(os.path.realpath(__file__))
    print(f"compiling multi-subseq post_proc {cwd} ...")

    jit_option = '-abortonspill -noschedule '
    kernels = cl.kernels(src, f'''-cmc -Qxcm_jit_option="{jit_option}"
                        -mCM_printregusage -mdump_asm -g2
                        -Qxcm_register_file_size=256 -I{cwd}
                        -DSTRIDE={STRIDE} -DHQ={HQ}
                        -DBLOCK_SIZE={BLOCK_SIZE}
                        -DMERGED_Q_NUM={MERGED_Q_NUM}
                        ''')

    cases = [
        ([4096], [4096], "1 subseq baseline"),
        ([4096, 4096], [4096, 4096], "2 subseqs equal"),
        ([4096, 32768], [4096, 32768], "2 subseqs unequal"),
        ([2048, 2048], [2048, 2048], "2 equal subseqs"),
        ([4096, 4096, 32768], [4096, 4096, 32768], "3 subseqs"),
    ]

    for q_lens, k_lens, desc in cases:
        print(f"{Colors.GREEN}Test multi-subseq post_proc {desc} BLOCK_SIZE={BLOCK_SIZE} ... {Colors.END}")
        num_subseqs = len(q_lens)

        meta_rows = []
        offset_mask = 0
        offset_mask_wg = 0
        max_merged_q_blocks = 0
        per_subseq = []

        for q, k in zip(q_lens, k_lens):
            q_stride_pad = q // STRIDE
            q_block_valid = q_stride_pad // sum_per_token_in_block
            q_block_pad = div_up(q, BLOCK_SIZE)
            k_block_pad = div_up(k, BLOCK_SIZE)
            merged_q_blocks = div_up(q_block_pad, MERGED_Q_NUM)
            max_merged_q_blocks = max(max_merged_q_blocks, merged_q_blocks)

            org = np.random.randint(low=0, high=2, size=[1, HQ, q_block_pad, k_block_pad], dtype=np.int8)
            if q_block_valid != q_block_pad:
                org[:, :, -1, :] = 1

            per_subseq.append({
                'org': org,
                'q_stride_pad': q_stride_pad,
                'q_block_pad': q_block_pad,
                'k_block_pad': k_block_pad,
                'q_block_valid': q_block_valid,
                'merged_q_blocks': merged_q_blocks,
            })

            meta_rows.append([
                0,              # [0]
                q,              # [1] SUBSEQ_Q_LEN
                0,              # [2] M
                0,              # [3] N
                q_stride_pad,   # [4] Q_STRIDE_PAD
                0,              # [5]
                q_block_pad,    # [6] Q_BLOCK_PAD
                k_block_pad,    # [7] K_BLOCK_PAD
                0,              # [8]
                0,              # [9]
                0,              # [10]
                0,              # [11]
                offset_mask,    # [12] BUF_OFF_MASK
                offset_mask_wg, # [13] BUF_OFF_MASK_WG
                0,              # [14]
                0,              # [15]
            ])

            offset_mask += HQ * q_block_pad * k_block_pad
            offset_mask_wg += HQ * merged_q_blocks * k_block_pad

        meta_np = np.array(meta_rows, dtype=np.int32)
        t_meta = cl.tensor(meta_np)

        # Build combined mask buffer
        combined_mask = np.zeros(offset_mask, dtype=np.int8)
        for i, s in enumerate(per_subseq):
            off = meta_rows[i][12]
            flat = s['org'].reshape(-1)
            combined_mask[off:off + flat.size] = flat

        t_mask = cl.tensor(combined_mask)
        t_merged_mask = cl.tensor(np.ones(offset_mask_wg, dtype=np.int8) * 100)

        GWS = [max_merged_q_blocks, HQ, num_subseqs]
        LWS = [1, 1, 1]

        cl.finish()
        kernels.enqueue("post_proc_mask", GWS, LWS, t_mask, t_merged_mask, t_meta)
        cl.finish()

        # Validate per-subsequence
        merged_all = t_merged_mask.numpy()
        for i, s in enumerate(per_subseq):
            off_wg = meta_rows[i][13]
            merged_q_blocks = s['merged_q_blocks']
            k_block_pad = s['k_block_pad']
            q_block_pad = s['q_block_pad']
            q_block_valid = s['q_block_valid']
            org = s['org']

            merged_subseq = np.zeros([1, HQ, merged_q_blocks, k_block_pad], dtype=np.int8)
            for h in range(HQ):
                start = off_wg + h * merged_q_blocks * k_block_pad
                merged_subseq[0, h] = merged_all[start:start + merged_q_blocks * k_block_pad].reshape(merged_q_blocks, k_block_pad)

            # Reference
            if MERGED_Q_NUM == 2:
                if q_block_valid != q_block_pad:
                    org_pad = np.ones([1, HQ, q_block_pad + 1, k_block_pad], dtype=np.int8)
                    org_pad[:, :, :-1, :] = org
                    ref = org_pad[:, :, 0::2, :] | org_pad[:, :, 1::2, :]
                else:
                    ref = org[:, :, 0::2, :] | org[:, :, 1::2, :]
            else:
                ref = org.copy()

            assert np.all(ref == merged_subseq), f"multi-subseq post_proc failed for subseq {i}"
            print(f'{Colors.GREEN}  subseq[{i}] post_proc passed{Colors.END}')

        print(f'{Colors.GREEN}multi-subseq post_proc "{desc}" passed{Colors.END}')


def test_post_proc_multi_subseq_with_decode(HQ=1, BLOCK_SIZE=128):
    """Test post_proc with decode+prefill mixing under both routing strategies.

    post_proc is safe with decode subseqs (operates only on mask data, no softmax
    buffers), so both split-route and multi-route are tested.
    For decode subseqs: q_len < STRIDE → q_block_pad=1, q_stride_pad=0.
    In multi-route, the decode subseq produces a trivial 1-row mask (all ones).
    """
    assert PA_WG_SIZE % BLOCK_SIZE == 0
    MERGED_Q_NUM = PA_WG_SIZE // BLOCK_SIZE
    assert MERGED_Q_NUM == 1 or MERGED_Q_NUM == 2

    STRIDE = 16
    sum_per_token_in_block = BLOCK_SIZE // STRIDE

    src = r'''#include "xattn_post_proc.hpp"'''
    cwd = os.path.dirname(os.path.realpath(__file__))
    print(f"compiling decode+prefill post_proc {cwd} ...")

    jit_option = '-abortonspill -noschedule '
    kernels = cl.kernels(src, f'''-cmc -Qxcm_jit_option="{jit_option}"
                        -mCM_printregusage -mdump_asm -g2
                        -Qxcm_register_file_size=256 -I{cwd}
                        -DSTRIDE={STRIDE} -DHQ={HQ}
                        -DBLOCK_SIZE={BLOCK_SIZE}
                        -DMERGED_Q_NUM={MERGED_Q_NUM}
                        ''')

    cases = [
        ([(1, 4096), (4096, 4096)],
         "1 decode + 1 prefill"),
        ([(1, 8192), (4096, 4096), (32768, 32768)],
         "1 decode + 2 prefills"),
        ([(4096, 4096), (1, 16384), (32768, 32768)],
         "prefill, decode, prefill (interleaved)"),
        ([(1, 2048), (1, 4096), (4096, 4096), (32768, 32768)],
         "2 decodes + 2 prefills"),
        ([(1, 4096), (1, 8192)],
         "all decode (zero prefill WGs)"),
    ]

    for subseq_specs, desc in cases:
        q_lens = [ql for ql, _ in subseq_specs]
        kv_lens = [kvl for _, kvl in subseq_specs]

        is_decode = [ql < STRIDE for ql in q_lens]
        prefill_indices = [i for i, d in enumerate(is_decode) if not d]

        for route_mode in ["split", "multi"]:
            if route_mode == "split" and len(prefill_indices) == 0:
                continue

            label = f'{route_mode}-route: "{desc}"'
            print(f"{Colors.GREEN}Test post_proc decode+prefill ({label}) BLOCK_SIZE={BLOCK_SIZE} ... {Colors.END}")

            if route_mode == "split":
                meta_indices = prefill_indices
            else:
                meta_indices = list(range(len(q_lens)))

            num_subseqs = len(meta_indices)

            meta_rows = []
            offset_mask = 0
            offset_mask_wg = 0
            max_merged_q_blocks = 0
            per_subseq = []

            for idx in meta_indices:
                q = q_lens[idx]
                k = kv_lens[idx]
                q_stride_pad = q // STRIDE
                q_block_valid = q_stride_pad // sum_per_token_in_block
                q_block_pad = div_up(q, BLOCK_SIZE)
                k_block_pad = div_up(k, BLOCK_SIZE)
                merged_q_blocks = div_up(q_block_pad, MERGED_Q_NUM)
                max_merged_q_blocks = max(max_merged_q_blocks, merged_q_blocks)

                if q >= STRIDE:
                    org = np.random.randint(low=0, high=2, size=[1, HQ, q_block_pad, k_block_pad], dtype=np.int8)
                    if q_block_valid != q_block_pad:
                        org[:, :, -1, :] = 1
                else:
                    org = np.ones([1, HQ, q_block_pad, k_block_pad], dtype=np.int8)

                per_subseq.append({
                    'org': org,
                    'q_stride_pad': q_stride_pad,
                    'q_block_pad': q_block_pad,
                    'k_block_pad': k_block_pad,
                    'q_block_valid': q_block_valid,
                    'merged_q_blocks': merged_q_blocks,
                    'orig_idx': idx,
                })

                meta_rows.append([
                    0,              # [0]
                    q,              # [1] SUBSEQ_Q_LEN
                    0,              # [2] M
                    0,              # [3] N
                    q_stride_pad,   # [4] Q_STRIDE_PAD
                    0,              # [5]
                    q_block_pad,    # [6] Q_BLOCK_PAD
                    k_block_pad,    # [7] K_BLOCK_PAD
                    0,              # [8]
                    0,              # [9]
                    0,              # [10]
                    0,              # [11]
                    offset_mask,    # [12] BUF_OFF_MASK
                    offset_mask_wg, # [13] BUF_OFF_MASK_WG
                    0,              # [14]
                    0,              # [15]
                ])

                offset_mask += HQ * q_block_pad * k_block_pad
                offset_mask_wg += HQ * merged_q_blocks * k_block_pad

            if max_merged_q_blocks == 0:
                print(f'{Colors.GREEN}  {label}: no work — passed{Colors.END}')
                continue

            meta_np = np.array(meta_rows, dtype=np.int32)
            t_meta = cl.tensor(meta_np)

            combined_mask = np.zeros(offset_mask, dtype=np.int8)
            for i, s in enumerate(per_subseq):
                off = meta_rows[i][12]
                flat = s['org'].reshape(-1)
                combined_mask[off:off + flat.size] = flat

            t_mask = cl.tensor(combined_mask)
            t_merged_mask = cl.tensor(np.ones(offset_mask_wg, dtype=np.int8) * 100)

            GWS = [max_merged_q_blocks, HQ, num_subseqs]
            LWS = [1, 1, 1]

            cl.finish()
            kernels.enqueue("post_proc_mask", GWS, LWS, t_mask, t_merged_mask, t_meta)
            cl.finish()

            merged_all = t_merged_mask.numpy()
            for i, s in enumerate(per_subseq):
                off_wg = meta_rows[i][13]
                merged_q_blocks = s['merged_q_blocks']
                k_block_pad = s['k_block_pad']
                q_block_pad = s['q_block_pad']
                q_block_valid = s['q_block_valid']
                org = s['org']

                merged_subseq = np.zeros([1, HQ, merged_q_blocks, k_block_pad], dtype=np.int8)
                for h in range(HQ):
                    start = off_wg + h * merged_q_blocks * k_block_pad
                    merged_subseq[0, h] = merged_all[start:start + merged_q_blocks * k_block_pad].reshape(merged_q_blocks, k_block_pad)

                if MERGED_Q_NUM == 2:
                    if q_block_valid != q_block_pad:
                        org_pad = np.ones([1, HQ, q_block_pad + 1, k_block_pad], dtype=np.int8)
                        org_pad[:, :, :-1, :] = org
                        ref = org_pad[:, :, 0::2, :] | org_pad[:, :, 1::2, :]
                    else:
                        ref = org[:, :, 0::2, :] | org[:, :, 1::2, :]
                else:
                    ref = org.copy()

                assert np.all(ref == merged_subseq), f"decode+prefill post_proc failed for subseq {i} (orig {s['orig_idx']})"
                print(f'{Colors.GREEN}  subseq[{i}] (orig {s["orig_idx"]}): post_proc passed{Colors.END}')

            print(f'{Colors.GREEN}post_proc decode+prefill ({label}) passed{Colors.END}')


def main():
    # Default configuration
    test_post_proc(BLOCK_SIZE = 128)
    if xe_arch == 2:
        test_post_proc(BLOCK_SIZE = 256)

    # phi-3-mini-128k-instruct (head_size=96, num_heads=4)
    test_post_proc(HQ=4, BLOCK_SIZE=128)
    if xe_arch == 2:
        test_post_proc(HQ=4, BLOCK_SIZE=256)

    # minicpm4 (head_ratio 16:1, num_heads=32)
    test_post_proc(HQ=32, BLOCK_SIZE=128)
    if xe_arch == 2:
        test_post_proc(HQ=32, BLOCK_SIZE=256)

def main_multi():
    # Default configuration
    test_post_proc_multi_subseq(BLOCK_SIZE=128)
    if xe_arch == 2:
        test_post_proc_multi_subseq(BLOCK_SIZE=256)

    # phi-3-mini-128k-instruct
    test_post_proc_multi_subseq(HQ=4, BLOCK_SIZE=128)
    if xe_arch == 2:
        test_post_proc_multi_subseq(HQ=4, BLOCK_SIZE=256)

    # minicpm4
    test_post_proc_multi_subseq(HQ=32, BLOCK_SIZE=128)
    if xe_arch == 2:
        test_post_proc_multi_subseq(HQ=32, BLOCK_SIZE=256)

def main_multi_with_decode():
    # Default configuration
    test_post_proc_multi_subseq_with_decode(BLOCK_SIZE=128)
    if xe_arch == 2:
        test_post_proc_multi_subseq_with_decode(BLOCK_SIZE=256)

    # phi-3-mini-128k-instruct
    test_post_proc_multi_subseq_with_decode(HQ=4, BLOCK_SIZE=128)
    if xe_arch == 2:
        test_post_proc_multi_subseq_with_decode(HQ=4, BLOCK_SIZE=256)

    # minicpm4
    test_post_proc_multi_subseq_with_decode(HQ=32, BLOCK_SIZE=128)
    if xe_arch == 2:
        test_post_proc_multi_subseq_with_decode(HQ=32, BLOCK_SIZE=256)

if __name__ == "__main__":
    torch.manual_seed(3)
    torch.set_printoptions(linewidth=1024)

    cl.profiling(True)

    import sys
    if '--multi-subseq-decode' in sys.argv:
        main_multi_with_decode()
    elif '--multi-subseq' in sys.argv:
        main_multi()
    else:
        main()