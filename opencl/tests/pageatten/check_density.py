import os
import re
import torch
import numpy as np
from collections import defaultdict
from clops.utils import Colors

torch.set_printoptions(linewidth=1024, precision=2)
np.set_printoptions(precision=2, suppress=True)

def get_tensor(name, dtype=np.float16):
    with open(name, 'rb') as f:
        data = f.read()
        np_data = np.frombuffer(data, dtype=dtype).copy()
        return torch.from_numpy(np_data)

def count_false_percentage(mask):
    B, H, NQ, NL = mask.shape
    tril_mask = torch.tril(torch.ones((NQ, NL), dtype=torch.bool, device=mask.device))
    expanded_tril = tril_mask.unsqueeze(0).unsqueeze(0).expand(B, H, -1, -1)
    # Count elements in the tril region
    tril_elements = torch.sum(expanded_tril).item()
    # Count False elements in the tril region
    false_in_tril = torch.sum(~mask & expanded_tril).item()
    # Calculate percentage
    if tril_elements > 0:
        false_percentage = (false_in_tril / tril_elements) * 100
    else:
        false_percentage = 0.0
    return false_percentage

def check_cuda_density(base, dump_name):
    base = '/home/ceciliapeng/OCL/xattn-cuda/'
    mask = torch.from_numpy(np.load(os.path.join(base, dump_name)))
    # print(f'{mask.shape=}')
    density = 100.0 - count_false_percentage(mask)

    B, H, NQ, NL = mask.shape
    densities = np.array([100.0 - count_false_percentage(mask[:, h, :, :].reshape(B, 1, NQ, NL)) for h in range (H)])
    print(f'densities over heads {densities}')
    densities = np.array([100.0 - count_false_percentage(mask[b, :, :, :].reshape(1, H, NQ, NL)) for b in range (B)])
    print(f'densities over layers {densities}')

    return density

def paired_adjacent_row_diff_pct(block_mask: torch.Tensor):
    """
    For each (L,H), compare non-overlapping adjacent row pairs along Q:
      (0,1), (2,3), (4,5), ...

    For each pair, compute the percentage of columns (K) that differ.

    Returns:
      pct_per_pair: (L, H, num_pairs) float tensor in [0,100]
      mean_pct:     (L, H) mean across pairs
      max_pct:      (L, H) max across pairs
    """
    assert block_mask.dtype == torch.bool, "block_mask must be torch.bool"
    L, H, Q, K = block_mask.shape

    # Number of full pairs
    num_pairs = Q // 2
    if num_pairs == 0:
        pct_per_pair = torch.zeros((L, H, 0), dtype=torch.float32, device=block_mask.device)
        mean_pct = torch.zeros((L, H), dtype=torch.float32, device=block_mask.device)
        max_pct  = torch.zeros((L, H), dtype=torch.float32, device=block_mask.device)
        return pct_per_pair, mean_pct, max_pct

    # Use only the even-length prefix (drop last row if Q is odd)
    Q_even = num_pairs * 2

    # Compare (0,1), (2,3), ... -> shapes (L,H,num_pairs,K)
    diff = block_mask[:, :, 0:Q_even:2, :] ^ block_mask[:, :, 1:Q_even:2, :]

    # Percentage of differing columns per pair: mean over K -> (L,H,num_pairs)
    pct_per_pair = diff.float().mean(dim=3) * 100.0

    # Summaries per (L,H)
    mean_pct = pct_per_pair.mean(dim=2)
    max_pct  = pct_per_pair.max(dim=2).values

    return pct_per_pair, mean_pct, max_pct

def save_block_mask_csv_per_layer_head(block_mask: torch.Tensor, out_dir: str):
    assert block_mask.dtype == torch.bool
    assert block_mask.device.type == "cpu"

    L, H, Q, K = block_mask.shape
    os.makedirs(out_dir, exist_ok=True)

    bm = block_mask.numpy().astype(np.uint8)  # 0/1

    for l in range(L):
        for h in range(H):
            path = os.path.join(out_dir, f"block_mask_L{l:02d}_H{h:02d}.csv")
            np.savetxt(path, bm[l, h], fmt="%d", delimiter=",")
  
def check_ov_density(
    base_dir,
    xattn_block_size=128,
    prompt_len=32*1024,
    pa_trunk_size=4*1024,
    num_layers=36,
    num_heads=32
):
    # Extract node IDs and KV lens
    node_re = re.compile(r"PagedAttentionExtension_(\d+)")
    kv_len_re = re.compile(r"__boolean__(\d+)_")

    # node -> { kv_len -> filename }
    node_to_kvlen_to_file = defaultdict(dict)

    for fn in os.listdir(base_dir):
        if not fn.startswith("xattn_internals_4__pagedattentionextension"):
            continue
        if not fn.endswith(".bin"):
            continue

        m_node = node_re.search(fn)
        m_kv_len = kv_len_re.search(fn)

        if m_node and m_kv_len:
            node = int(m_node.group(1))
            kv_len = int(m_kv_len.group(1))

            # Store filename for this (node, kv_len)
            # If the same (node, kv_len) appears again, this will overwrite with the latest.
            node_to_kvlen_to_file[node][kv_len] = fn

    # Build: node -> sorted list of kv_lens
    node_to_sorted_kv_lens = {
        node: sorted(kvlen_to_file.keys())
        for node, kvlen_to_file in node_to_kvlen_to_file.items()
    }
    node_to_sorted_kv_lens = dict(sorted(node_to_sorted_kv_lens.items(), key=lambda x: x[0]))
    # print(node_to_sorted_kv_lens)

    # Count kv_lens per node
    value_counts = {node: len(kv_list) for node, kv_list in node_to_sorted_kv_lens.items()}
    counts_set = set(value_counts.values())
    chunk_cnt = next(iter(counts_set), 0)
    # print(value_counts)

    assert len(counts_set) == 1, "Not all nodes have the same number of kv_lens!"

    # Load block mask of PA
    num_q_blocks = prompt_len // xattn_block_size
    num_k_blocks = prompt_len // xattn_block_size
    num_iters = (prompt_len + pa_trunk_size - 1) // pa_trunk_size
    # print(f'{num_iters=}, {num_q_blocks=}, {num_k_blocks=}')

    assert num_layers == len(node_to_sorted_kv_lens), "num_layers mismatch!"
    assert num_iters == chunk_cnt, "num_iters mismatch!"

    block_mask = torch.zeros([num_layers, num_heads, num_q_blocks, num_k_blocks],
                             device='cpu', dtype=torch.bool)

    # Print kv_len + filename
    # print("\nKV lens count per node (with index):")
    for idx, (node, kv_list) in enumerate(node_to_sorted_kv_lens.items()):
        # print(f"Node index {idx} - Node {node}: {len(kv_list)}")
        for i, kv_len in enumerate(kv_list):
            filename = node_to_kvlen_to_file[node][kv_len]
            # print(f"Chunk index {i} - {kv_len} - {filename}")

            layer_mask = get_tensor(os.path.join(base_dir, filename), np.uint8)
            cur_q_blocks = pa_trunk_size // xattn_block_size
            cur_k_blocks = (i + 1) * pa_trunk_size // xattn_block_size 
            # print(f'iter {i}, layer {idx} {filename} {cur_q_blocks=}, {cur_k_blocks=}')
            layer_mask = layer_mask.reshape([num_heads, cur_q_blocks, cur_k_blocks])
            block_mask[idx, :, i * cur_q_blocks : (i + 1) * cur_q_blocks, : cur_k_blocks] = layer_mask
    
    # extra checks     
    # save_block_mask_csv_per_layer_head(block_mask, os.path.join(base_dir, "block_mask_csv"))

    if xattn_block_size == 128:
        pct_per_pair, mean_pct, max_pct = paired_adjacent_row_diff_pct(block_mask)
        L, H, Q, K = block_mask.shape
        print(f"block_mask shape={block_mask.shape}, num_pairs={Q//2} (dropping last row if Q is odd)")
        for l in range(L):
            for h in range(H):
                print(f"L{l:02d} H{h:02d} | mean={mean_pct[l,h].item():.2f}% max={max_pct[l,h].item():.2f}%")
                # If you want per-pair detail:
                # for p in range(pct_per_pair.shape[2]):
                #     q0, q1 = 2*p, 2*p + 1
                #     print(f"  pair rows ({q0},{q1}) : {pct_per_pair[l,h,p].item():.2f}%")

    if True:
        B, H, NQ, NL = block_mask.shape
        densities = np.array([100.0 - count_false_percentage(block_mask[:, h, :, :].reshape(B, 1, NQ, NL)) for h in range (H)])
        print(f'densities over heads {densities}')
        densities = np.array([100.0 - count_false_percentage(block_mask[b, :, :, :].reshape(1, H, NQ, NL)) for b in range (B)])
        print(f'densities over layers {densities}')

    return 100.0 - count_false_percentage(block_mask)

if __name__ == "__main__":
    density = check_ov_density('/home/ceciliapeng/toolbox/linux/xattn_thresh0.9/dump_xattn_mask_bs128', xattn_block_size=128)
    print(f'{Colors.BLUE}=============== density of OV with thresh 0.9 block size 128: {density:.2f}%  ==============={Colors.END}\n')

    density = check_ov_density('/home/ceciliapeng/toolbox/linux/xattn_thresh0.9/dump_xattn_mask_bs256', xattn_block_size=256)
    print(f'{Colors.BLUE}=============== density of OV with thresh 0.9 block size 256: {density:.2f}%  ==============={Colors.END}\n')
    
    density = check_ov_density('/home/ceciliapeng/toolbox/linux/xattn_thresh0.6/dump_xattn_mask_bs128', xattn_block_size=128)
    print(f'{Colors.BLUE}=============== density of OV with thresh 0.6 block size 128: {density:.2f}%  ==============={Colors.END}\n')

    density = check_ov_density('/home/ceciliapeng/toolbox/linux/xattn_thresh0.6/dump_xattn_mask_bs256', xattn_block_size=256)
    print(f'{Colors.BLUE}=============== density of OV with thresh 0.6 block size 256: {density:.2f}%  ==============={Colors.END}\n')

    # def get_all_last_subdir_paths(base_path):
    #     last_subdirs = []
    #     for root, dirs, files in os.walk(base_path):
    #         if not dirs:  # Leaf directory
    #             last_subdirs.append(root)
    #     return last_subdirs

    # # check density of all subdirs
    # base_path = "/home/ceciliapeng/openvino/dump_block_mask"
    # last_subdir_paths = get_all_last_subdir_paths(base_path)
    # print("Full paths of last subdirectories:")
    # for path in last_subdir_paths:
    #     print(f'{path}')
    #     if "32k" in path:
    #         density = check_ov_density(path, prompt_len = 32*1024, pa_trunk_size = 4*1024)
    #         print(f'{Colors.YELLOW}=============== density of OV with {path}: {density:.2f}%  ==============={Colors.END}\n')
    #     elif "64k" in path:
    #         density = check_ov_density(path, prompt_len = 64*1024, pa_trunk_size = 4*1024)
    #         print(f'{Colors.BLUE}=============== density of OV with {path}: {density:.2f}%  ==============={Colors.END}\n')