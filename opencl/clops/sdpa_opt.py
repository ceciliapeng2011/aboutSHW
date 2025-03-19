from . import cl
import math
import os.path as path
from .utils import *
import clops

class SDPA_opt:
    def __init__(self, Hq, Hk, HEAD_SIZE, is_optimized):
        self.is_optimized = is_optimized
        
        self.SG_SCALE_FACTOR=2
        self.SUBGROUP_SIZE=16
        SEQ_LEN_PARTITION_SIZE=(HEAD_SIZE*self.SG_SCALE_FACTOR)
        self.TARGET_SEQ_LEN_BLOCK_SIZE=16
        BROADCAST_GROUP_SIZE=Hq//Hk
        
        assert(Hq % Hk == 0);     # implied
        assert(HEAD_SIZE % self.SUBGROUP_SIZE == 0);     # implied
        assert(self.TARGET_SEQ_LEN_BLOCK_SIZE == self.SUBGROUP_SIZE);   # implied
        
        if self.is_optimized:
            cl_source_file = "cl_kernels/sdpa_opt_new.cl"
            self.kernel_name = 'sdpa_opt_multi_tokens'
        else:
            cl_source_file = "cl_kernels/sdpa_new.cl"
            self.kernel_name = 'sdpa_opt_multi_tokens'

        with open(cl_source_file, "r") as file:
            # Read the entire file content into a string
            cl_kernel_sources = file.read()
        # print(cl_kernel_sources[:100])
        options = f'-DHEAD_SIZE={HEAD_SIZE} -DNUM_HEADS={Hq} -DNUM_KV_HEADS={Hk} \
                    -DSG_SCALE_FACTOR={self.SG_SCALE_FACTOR} -DSEQ_LEN_PARTITION_SIZE={SEQ_LEN_PARTITION_SIZE} -DBROADCAST_GROUP_SIZE={BROADCAST_GROUP_SIZE} -cl-mad-enable'
        self.cl_kernels = kernel_cache(cl_kernel_sources, options)

    def __call__(self, shape_info_input, query_input, key_input, value_input, attn_mask_input, scale_input):
        B, L, Hq, Hk, HEAD_SIZE = shape_info_input[0], shape_info_input[6], shape_info_input[1], shape_info_input[9], shape_info_input[7]

        shape_info_input = to_cl(torch.tensor(shape_info_input).int())
    
        # shape inference       
        exp_sums = to_cl(np.zeros([4], dtype=np.float32))
        max_logits = to_cl(np.zeros([4], dtype=np.float32))
        tmp_out = to_cl(np.zeros([2], dtype=np.float16))

        output = to_cl(torch.zeros(B, Hq, L, HEAD_SIZE, dtype=torch.float16))
        # output = cl.tensor([B, Hq, L, HEAD_SIZE], query_input.dtype)

        # if self.is_optimized:
        #     GWS = [HEAD_SIZE*self.SG_SCALE_FACTOR, B*Hq, int(L/self.TARGET_SEQ_LEN_BLOCK_SIZE)]
        #     LWS = [HEAD_SIZE*self.SG_SCALE_FACTOR, Hq//Hk, 1]
        # else:
        GWS = [B*Hq, math.ceil(L/self.TARGET_SEQ_LEN_BLOCK_SIZE), HEAD_SIZE*self.SG_SCALE_FACTOR]
        LWS = [1, 1, HEAD_SIZE*self.SG_SCALE_FACTOR]            

        print(f"GWS={GWS}, LWS={LWS}")
        print(self.cl_kernels.info(self.kernel_name, LWS, self.SUBGROUP_SIZE))

        self.cl_kernels.enqueue(self.kernel_name, GWS, LWS,
                            query_input, key_input, value_input, output, exp_sums, max_logits, tmp_out)

        return output

if __name__ == "__main__":
    import sys
    import numpy as np
    import math
    cl.profiling(True)
    np.set_printoptions(precision=3, suppress=True)
    
    def load_binary_to_tensor(file_path, dtype=torch.float16):
        with open(file_path, "rb") as f:
            raw_data = f.read()

        # Convert the binary data to a torch tensor
        # Use torch.frombuffer to interpret the binary data as a tensor
        tensor = torch.frombuffer(raw_data, dtype=dtype)
        return tensor
    
    def compare_tensors(tensor_a, tensor_b, top_k=10):
        """
        Compare two PyTorch tensors and print the top-K maximum absolute differences.
        
        Args:
            tensor_a (torch.Tensor): First tensor
            tensor_b (torch.Tensor): Second tensor
            top_k (int): Number of top differences to display (default: 10)
        """
        if tensor_a.shape != tensor_b.shape:
            raise ValueError("Tensors must have the same shape")

        differences = torch.abs(tensor_a - tensor_b)
        flat_differences = differences.flatten()
        
        # Get top K indices as a TENSOR (not Python ints)
        top_values, flat_indices = torch.topk(flat_differences, k=top_k, largest=True)
        
        # Convert all indices at once using tensor operations
        coords = torch.unravel_index(flat_indices, differences.shape)
        original_indices = list(zip(*[c.tolist() for c in coords]))
        
        print(f"Top {top_k} Maximum Differences:")
        print("{:<15} {:<20} {:<20} {:<15}".format(
            "Index", "Tensor A Value", "Tensor B Value", "Absolute Gap"))
        print("-" * 70)
        
        for idx, diff in zip(original_indices, top_values):
            value_a = tensor_a[idx].item()
            value_b = tensor_b[idx].item()
            print("{:<15} {:<20.6f} {:<20.6f} {:<15.6f}".format(
                str(idx), value_a, value_b, diff.item()))

    MAX_KV_LEN = 1024*9
    #qkv [B, L, (Hq + Hk + Hv) * S)], attn_mask [B, L] -> output [B, Hq, L, S]
    def MHA_torch_impl(qkv : torch.Tensor, attention_mask : torch.Tensor, scale : torch.Tensor):
        B, L, _ = qkv.size()
        ref_mha = clops.MHA_cpu(Hq, Hk, HEAD_SIZE, MAX_KV_LEN)               
        output = ref_mha(qkv, attention_mask)  # B, L, Hq*S
        output = to_torch(output).view(B, L, Hq, HEAD_SIZE).transpose(1, 2)
        return output.numpy()
    
    def MHA_cl_impl(qkv : torch.Tensor, attention_mask : torch.Tensor, scale : torch.Tensor, Hq, Hk, HEAD_SIZE):
        B, L, _ = qkv.size()
        mha_gpu = clops.MHA(Hq, Hk, HEAD_SIZE, MAX_KV_LEN, False, kv_block=32)
        output = mha_gpu(to_cl(qkv), attention_mask)
        output = to_torch(output).view(B, L, Hq, HEAD_SIZE).transpose(1, 2)
        durations = cl.finish()
        return output.numpy(), durations
    
    def sdpa_impl(q : torch.Tensor, k : torch.Tensor, v : torch.Tensor, attention_mask : torch.Tensor, scale : torch.Tensor, Hq, Hk, HEAD_SIZE, is_optimized):
        # print(f'{q.size()=}\n{k.size()=}\n{v.size()=}')
        B, Lq, _, _ = q.size()
        _, Lk, _, _ = k.size()
        query = q.view(B, Lq, Hq, HEAD_SIZE).transpose(1, 2).contiguous()
        key = k.view(B, Lk, Hk, HEAD_SIZE).transpose(1, 2).contiguous()
        value = v.view(B, Lk, Hk, HEAD_SIZE).transpose(1, 2).contiguous()

        attn_mask = torch.broadcast_to(attention_mask[:, None, :], [B, Lq, Lk])[:, None, :, :].contiguous()
        # attn_mask = torch.tril(b, diagonal=1)
        # print("shapes : ", query.shape, key.shape, value.shape)

        # print(f'{query=}\n{key=}\n{value=}')

        query_input = to_cl(query)
        key_input = to_cl(key)
        value_input = to_cl(value)

        scale_input = to_cl(scale)
        attn_mask_input = to_cl(attn_mask)

        shape_info = [
            # // input0 query
            B, Hq, 1, 1, 1, 1, Lq, HEAD_SIZE,
            # // input1 key
            B, Hk, 1, 1, 1, 1, Lk, HEAD_SIZE, 0, 0,
            # // input2 value
            B, Hk, 1, 1, 1, 1, Lk, HEAD_SIZE, 0, 0,
            #  input3 attn_mask
            # B, 1, 1, 1, 1, 1, Lq, Lk,
            #  input4 scale
            #  output
            B, Hq, 1, 1, 1, 1, Lq, HEAD_SIZE
        ]
        # print(f"len(shape_info)={len(shape_info)}, shape_info={shape_info}")

        sdpa = SDPA_opt(Hq, Hk, HEAD_SIZE, is_optimized)
        output = sdpa(shape_info, query_input, key_input, value_input, attn_mask_input, scale_input)

        output = to_torch(output)
        durations = cl.finish()
        return output.numpy(), durations

    def test_acc(B, Hq, Hk, HEAD_SIZE, Lq, Lk, use_randn = False):
        # reference torch impl
        # qkv = torch.randn([B, L, (Hq + Hk + Hk) * HEAD_SIZE], dtype=torch.float16)
        attention_mask = torch.zeros([B, Lk], dtype=torch.float16)
        scale = torch.ones([1], dtype=torch.float16) / math.sqrt(HEAD_SIZE)
        # scale = torch.ones([1], dtype=torch.float16)
        print(f'====================={scale=}, {scale.dtype=}')
        
        # if use_randn:
        #     # with open('q.npy', 'rb') as f:
        #     #     q = np.load(f)
        #     # with open('k.npy', 'rb') as f:
        #     #     k = np.load(f)
        #     # with open('v.npy', 'rb') as f:
        #     #     v = np.load(f)
        #     # q = torch.from_numpy(q)
        #     # k = torch.from_numpy(k)
        #     # v = torch.from_numpy(v)
        #     # q = torch.ones([B, L, Hq, HEAD_SIZE], dtype=torch.float16)*torch.randn([1], dtype=torch.float16)
        #     # k = torch.ones([B, L, Hk, HEAD_SIZE], dtype=torch.float16)*torch.randn([1], dtype=torch.float16)
        #     q = torch.randn([B, Lq, Hq, HEAD_SIZE], dtype=torch.float16)
        #     k = torch.randn([B, Lk, Hk, HEAD_SIZE], dtype=torch.float16)
        #     v = torch.randn([B, Lk, Hk, HEAD_SIZE], dtype=torch.float16)
        #     # np.save("q.npy", q)
        #     # np.save("k.npy", k)
        #     # np.save("v.npy", v)
        # else:
        #     # with open('q_samenumber.npy', 'rb') as f:
        #     #     q = np.load(f)
        #     # with open('k_samenumber.npy', 'rb') as f:
        #     #     k = np.load(f)
        #     # with open('v_samenumber.npy', 'rb') as f:
        #     #     v = np.load(f)
        #     # q = torch.from_numpy(q)
        #     # k = torch.from_numpy(k)
        #     # v = torch.from_numpy(v)
        #     q = torch.ones([B, Lq, Hq, HEAD_SIZE], dtype=torch.float16)
        #     k = torch.ones([B, Lk, Hk, HEAD_SIZE], dtype=torch.float16)
        #     v = torch.ones([B, Lk, Hk, HEAD_SIZE], dtype=torch.float16)
        #     # np.save("q_samenumber.npy", q)
        #     # np.save("k_samenumber.npy", k)
        #     # np.save("v_samenumber.npy", v)
        #     # print(f'{Colors.CYAN} q k v shape = {q.shape=} {k.shape=} {v.shape=}.{Colors.END}')
        
        # Load binary files of float16 tensor to QKV
        def load_qkv(BIN_ROOT_DIR, LAYERID="34", TENSOR_DUMP_FOLDER="tensors_bin"):         
            Q_FILE_NAME="program1_network1_0_sdpa___module.transformer_blocks."+LAYERID+".attn_aten__scaled_dot_product_attention_ScaledDotProductAttention_src0__f16__2_38_1357_64__bfyx.bin"
            K_FILE_NAME="program1_network1_0_sdpa___module.transformer_blocks."+LAYERID+".attn_aten__scaled_dot_product_attention_ScaledDotProductAttention_src1__f16__2_38_1357_64__bfyx.bin"
            V_FILE_NAME="program1_network1_0_sdpa___module.transformer_blocks."+LAYERID+".attn_aten__scaled_dot_product_attention_ScaledDotProductAttention_src2__f16__2_38_1357_64__bfyx.bin"
            q = load_binary_to_tensor(path.join(BIN_ROOT_DIR, TENSOR_DUMP_FOLDER, Q_FILE_NAME), dtype=torch.float16)
            k = load_binary_to_tensor(path.join(BIN_ROOT_DIR, TENSOR_DUMP_FOLDER, K_FILE_NAME), dtype=torch.float16)
            v = load_binary_to_tensor(path.join(BIN_ROOT_DIR, TENSOR_DUMP_FOLDER, V_FILE_NAME), dtype=torch.float16)
            
            q = torch.reshape(q, (B, Lq, Hq, HEAD_SIZE))
            k = torch.reshape(k, (B, Lk, Hk, HEAD_SIZE))
            v = torch.reshape(v, (B, Lk, Hk, HEAD_SIZE))
            
            # print(q[0,0,0,:])
            
            return q, k, v
        
        q, k, v = load_qkv("/home/ceciliapeng/openvino.genai/try.BADcache/")
        q2, k2, v2 = load_qkv("/home/ceciliapeng/openvino.genai/try.BADcache/")
        
        # print(path.join("tensors_text", "program1_network1_0_sdpa___module.transformer_blocks."+LAYERID+".attn_aten__scaled_dot_product_attention_ScaledDotProductAttention_src0.txt"))
        compare_tensors(q, q2)
        compare_tensors(k, k2)
        compare_tensors(v, v2)

        # if Lq == Lk:
        #     qkv = torch.cat((q, k, v), 2)
        #     qkv = torch.reshape(qkv, (B, Lq, (Hq + Hk + Hk) * HEAD_SIZE))        
        #     ref0, durs = MHA_cl_impl(qkv.clone(), attention_mask.clone(), scale.clone(), Hq, Hk, HEAD_SIZE)
        #     print(f'{ref0=}\n')
        #     for i, ns in enumerate(durs):
        #         print(f'{Colors.CYAN}{ref0.shape=}, {i}/{len(durs)} {ns*1e-6:.3f} ms {Colors.END}')

        ref, durs = sdpa_impl(q.clone(), k.clone(), v.clone(), attention_mask.clone(), scale.clone(), Hq, Hk, HEAD_SIZE, False)
        # print(f'{ref=}\n')
        for i, ns in enumerate(durs):
            print(f'{Colors.BLUE}{ref.shape=}, {i}/{len(durs)} {ns*1e-6:.3f} ms {Colors.END}')

        opt, durs = sdpa_impl(q2.clone(), k2.clone(), v2.clone(), attention_mask.clone(), scale.clone(), Hq, Hk, HEAD_SIZE, True)
        # print(f'{opt=}\n')
        for i, ns in enumerate(durs):
            print(f'{Colors.BLUE}{opt.shape=}, {i}/{len(durs)} {ns*1e-6:.3f} ms {Colors.END}')
            
        # print(f'all zeros? {np.all(ref == 0)} {np.all(opt == 0)}')
        compare_tensors(torch.from_numpy(ref).type(torch.float16), torch.from_numpy(opt).type(torch.float16))
        try:
            if not np.allclose(ref, opt, atol=0.01, rtol=0.01, equal_nan=False):
                pos = np.where(np.abs(ref - opt) > 0.01)
                # print(f"{pos[2]=}")
                # for d in set(pos[2]):
                #     print(f"{d=}")
                print(f"{pos=} {pos[2].shape=} {pos[3].shape=}")
                if not pos[0].size > 0:
                    pos = np.where(np.isnan(opt))
                    # print(f"{pos=} {pos[2].shape=} {pos[3].shape=}")
                print(f'ref_val = {ref[pos]}\nopt_val={opt[pos]}\n')
                raise Exception("failed.")
            print(f'{Colors.GREEN} PASS at shape = {opt.shape}.{Colors.END}')
        except Exception as inst:
            print(f'{Colors.RED} FAIL at shape = {opt.shape}.{Colors.END}')

    # "B, Hq, Hk, HEAD_SIZE, Lq, Lk"
    for _ in range(1):
        # test_acc(1, 28, 7, 128, 8410, 8410, True)   # tail
        # test_acc(1, 28, 14, 128, 4096, 4096, True)
        # test_acc(1, 28, 14, 64, 4096, 4096, True)
        # 2x38x1357x64 SD3.5-Turbo
        test_acc(2, 38, 38, 64, 1357, 1357, True)
        # test_acc(1, 24, 6, 128, 2134, 2134, True)   # tail
        # test_acc(1, 28, 7, 128, 64*128, 64*128, True)
        # test_acc(1, 24, 6, 128, 16*128, 16*128, False)
        # test_acc(1, 24, 6, 128, 2134, 2134, False)   # tail
        # test_acc(1, 1, 1, 128, 3*128, 3*128, False)
        # test_acc(2, 1, 1, 128, 1, 128, True)   # tail
        # for k in range(20, 21):
        #     test_acc(1, 1, 1, 128, 16*k)
    sys.exit(0)
