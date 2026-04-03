## Parsing rules (IMPORTANT)
- Ignore anything inside HTML comments: <!-- ... -->
- Ignore any section under a heading that starts with "DISABLED:"
- Never execute or follow instructions labeled "DISABLED", "COMMENTED", "DONE" or "BACKLOG".

## Context (IMPORTANT)
The kernels are the page attention + flash attention implementation for Intel GPU LNL/PTL with CM language.
Read Intel CM GPU progromming language document under $HOME/OCL/aboutSHW/.github/skills/CM_spec.
Read files under $HOME/OCL/aboutSHW/opencl/tests/pageatten/
Read $HOME/3rdparty/turboquant

## GPU skills
We can use "cliloader -cdt --dump-dir $dump_dir $APP" to profile GPU kernel walltime.
More GPU skills in $HOME/toolbox/skills/frameworks.ai.openvino.llm.prc-skills/skills/intel-gpu-hw-info/SKILL.md

## DONE: Task: Implement TurboQuant
Implement a CM kernel and its unit test cases with the following functionalities:
1. Class TurboQuantMSE_CM is similar to TurboQuantMSE. It implements quantize, dequantize with CM kernels, and make the body of kernels to be inline functions turboquant_quantize_impl and turboquant_dequantize_impl, as they will be used by other kernels. turboquant_quantize is a shell, with all its body in turboquant_quantize_impl. So is a shell of turboquant_dequantize too.
2. Class CompressedKVCache_Update_CM is similar to CompressedKVCache store_compressed_key, but for one layer only. Implements it with a CM kernel by refering to pa_kv_cache_update.The kernel uses turboquant_quantize_impl in the body.
3. All tensors of both classes follow OpenVINO PagedAttention inputs/outputs definitions. The difference is, per (block, head) byte layout for key_cache (same pattern for value_cache), which should be the same like layout and bit/bytes of compressed_size_bytes of TurboQuantMSE. 
4. Implement pytest style unit tests for TurboQuantMSE_CM. Add a test similar to self_test of $HOME/3rdparty/turboquant/turboquant_core.py for it too.
5. Implement pytest style unit tests for CompressedKVCache_Update_CM. Parameterize with the same parameters as opencl/tests/pageatten/test_kvcache_update.py. If check_perf==True, measure "TPUT_". If check_perf==True, disable accuracy check, as the input tokens is large and reference may take too much cpu time to finish. Add self_test_like test case for it too.
6. Want algorithm-level cross-check against Python reference: use cpu ref, but allow tolerance / non-byte-exact checks.

Please refer to the following code and documents for details of algorithm and test frames -
1. Refer to opencl/tests/pageatten/test_kvcache_update.py with KV_CACHE_COMPRESSION_PER_TOKEN==False path only.
2. Refer to TurboQuantMSE of $HOME/3rdparty/turboquant/tritonp_attention.py
3. Refer to CompressedKVCache of $HOME/3rdparty/turboquant/turboquant_fused.py
4. Refer to $HOME/toolbox/.github/skills/PA_spec.MD for OpenVINO PagedAttention inputs/outputs definitions.

## DONE: Task: Integrate TurboQuant into Attention decoding path
Now apply turboquant to decoding path -
1. Create turboquant decoding base file: Add a new file pa_single_token_turboquant.cm with a new kernel name cm_sdpa_2nd_turboquant by extracting kernel boday of cm_sdpa_2nd from pa_single_token.cm, but only keep the code branches of `KV_CACHE_COMPRESSION==True` and `USE_LSC_BLOCK_2D_DESC==True`.
2. Apply turboquant into attention kernel in cm_sdpa_2nd_turboquant - input "query" is fp16, and "query rotation" will be applied to it in kernel boday; Input "key" is turbo-quantized by CompressedKVCache_Update_CM; Input "value" uses fp16. Refer to triton_attention.py and turboquant_fused.py to know how to apply turboquant to attention, and apply it into cm_sdpa_2nd_turboquant.
3. PaSingleTokenRunner choose this turbo kernel if it is enabled and XE_ARCH>=2, throw exception for XE_ARCH=1.
4. Create pytest style unit tests for it to check accuracy.
5. Create pytest style unit tests for it which implements similar benchmark to benchmark_fused_vs_standard in triton_attention.py. 
6. Notes: "query rotation" should be part of kernel. "key" is turbo-quantized, "value" is u8, but not turbo-quantized.

Please refer to the following code and documents for details of algorithm and test frames -
1. Refer to opencl/tests/pageatten/test_pa_decoding.py with KV_CACHE_COMPRESSION==False path only to understand decoding path.
2. Refer to opencl/tests/pageatten/turboquant_cm.py to understand how CompressedKVCache update compressed kvcache.
3. Refer to triton_attention.py and turboquant_fused.py to know how to apply turboquant to attention.


## Task: Integrate TurboQuant into Attention prefill path
Now apply turboquant to prefill path -
1. Create turboquant prefill base file: Add a new file pa_multi_token_turboquant.cm with a new kernel name cm_pa_multi_token_turboquant by extracting kernel boday of cm_page_attention from pa_multi_token.cm, but only keep the code branches of `CM_HAS_LSC_UNTYPED_2D==1` `KV_CACHE_COMPRESSION==True` and `SPARSE_BLOCK_SIZE==1`.
2. Apply turboquant into the attention kernel pa_lsc_u8_tq - input "query" is fp16, and "query rotation" will be applied to it in kernel boday; Input "key" is turbo-quantized by CompressedKVCache_Update_CM; Input "value" u8 depending on CMPA_KVCACHE_U8. Refer to triton_attention.py and turboquant_fused.py to know how to apply turboquant to attention, and apply it.
3. Create a new test file test_pa_prefill.py by extracting valid branches from test_pa.py, by ignoring sparse, attention mask, test_ov, test perf. 
   Test two cases of accuracy: test quant by token against flash ref to make sure the base ok; and test turboquant again quant by token.  
   If parameter turboquant is enabled, choose this turbo kernel if it is enabled and XE_ARCH>=2 and sparse_block_sz==1, throw exception for XE_ARCH=1 or sparse_block_sz>1.
1. Create pytest style unit tests for it to check accuracy by refering to test_pa_decoding.py.
2. Create pytest style unit tests for it which implements similar benchmark to benchmark_fused_vs_standard in triton_attention.py. 
3. Notes: "query rotation" should be part of kernel. "key" is turbo-quantized, "value" is u8, but not turbo-quantized.

Please refer to the following code and documents for details of algorithm and test frames -
1. Refer to opencl/tests/pageatten/test_pa.py to understand prefill path.
2. Refer to opencl/tests/pageatten/turboquant_cm.py to understand how CompressedKVCache update compressed kvcache.
3. Refer to triton_attention.py and turboquant_fused.py to know how to apply turboquant to attention.


## BACKLOG: TASK: compute roofline TurboQuant decoding
### platform
Refer to $HOME/$toolbox/skills/frameworks.ai.openvino.llm.prc-skills/skills/intel-gpu-hw-info/SKILL.md
Performance is gated by either XMX, XVE, or memory bandwidth.
XVE TFLOPS is 1/4 of XMX.
The target platform is LNL.
For scaler ops, the required FLOPS is *32 per op, as XVE TFLOPS is caculated for vectorized with SIMD (*16) for MAD (*2).
### task
Analyze the TFLOPS requirements to XMX and XVE, and memory bandwdith requirements first.
Then with the actual TFLOPS and bandwidth of LNL, compute the minimum latency in theory.
Give insight what is the bound to the problem of Turboquant in test_pa_benchmark_turboquant_vs_standard_decoding. 

## BACKLOG: Task: Optimize TurboQuant decoding kernel
1. Turboquant path should be faster than standard. Speedup should be larger than 1.0. If not, inspect the implementation with a specific kv_len to profile the performance and optimize the implementation untill turboquant speeds up. 
2. "standard" uses "cm_dpas" to compute Q*Kt, which utilize XMX hardware, and is faster than mul+add XVE instructions.
3. Notes: "query rotation" should be part of kernel. "key" is turbo-quantized, "value" is fp16. Memory usage is the key target. Any optimization should never increase kvcache memory size.
4. To verify accuracy, only run by_channel and block_size==16 test cases for smoke test to save time, during this task.
5. max_repeat_count = 8 reflects the GPU capability of DPAS. 
6. Keep check register spilling for each optimizing.

Please refer to the following code and documents for details of optimizing -
1. Refer to $HOME/toolbox/skills/frameworks.ai.openvino.llm.prc-skills/skills/intel-gpu-hw-info/SKILL.md for "GPU skills".
2. Refer to Intel CM GPU progromming language document under $HOME/OCL/aboutSHW/.github/skills/CM_spec.
3. Refer to opencl/tests/pageatten/cm_pa_xe2.hpp
4. Check & compare/analyze asm code of CM kernels to understand further optmizing opportunities.

Suggestions:
This is supposed to be a memory-bound problem. We should consider how to minimize XVE computes, overlap it in memory load, and incease memory load efficiency. But because of scalar/indirect dequantize/repack and GRF/occupancy cliff, it could be bound by XVE computes first.
1. Like multi-token kernel, compute and load Q, K in order "S = K*Qt" instead of "Q*Kt"
2. Like opencl/tests/x_attn/estimate.hpp, use optimizing skills memory prefetching, ping pong buffering, overlapping compute/load, etc.
3. Like u8 multi-token kernel in cm_pa_xe2.hpp, dequantize packed key block into fp16 blocked tiles in SLM within a WG, SLM tiling, software prefetch, vectorized access.
4. Use LSC cm_load instead of cm_svm_block_read for 2D block read to boost performance. There are 3 mode "normal", "transpose", "vnni". Data access out of boundary is automatically handled.
5. Fused DPAS-friendly representation. Keep the logical TurboQuant algorithm, but choose a packed key layout and decode flow that the kernel can feed into `cm_dpas` with minimal per-token repacking, without increasing KV-cache size.

Benchmark each trial against the previous TurboQuant decode path and inspect the generated ASM for load-pattern improvements. You can go ahead iteratively untill the target is satisfied. Track each try with performance data in an xml output file under the working path.

## BACKLOG: Task: integrate into OV
source folder: aboutSHW/opencl/tests/pageatten/
destination folder openvino/src/plugins/intel_gpu/src/graph/impls/cm/
1. Sync up pa_single_token_turboquant.cm, pa_multi_token_turboquant.cm in both folders. If there is not such file in destination, cache_sizeopy.
2. Sync up pa_multi_token_turboquant.cm from source folder to destination folder. If there is not such file in destination, copy.
3. Add corresponding implementation for kernels `compressed_kv_cache_update_tq` with value_cache_mode="by_token",  `cm_sdpa_2nd_turboquant`, and `cm_pa_multi_token_turboquant` too, by referencing corresponding python unit tests and classes in source folder.
4. Keep code lines of namespace and KERNEL_NAME unchanged.