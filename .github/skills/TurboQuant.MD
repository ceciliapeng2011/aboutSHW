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
2. Refer to TurboQuantMSE of $HOME/3rdparty/turboquant/triton_attention.py
3. Refer to CompressedKVCache of $HOME/3rdparty/turboquant/turboquant_fused.py
4. Refer to $HOME/toolbox/.github/skills/PA_spec.MD for OpenVINO PagedAttention inputs/outputs definitions.

## Task: Integrate TurboQuant into Attention decoding path
Now apply turboquant to decoding path -
1. Create turboquant decoding base file: Add a new file pa_single_token_turboquant.cm with a new kernel name cm_sdpa_2nd_turboquant. Copy kernel boday of cm_sdpa_2nd from pa_single_token.cm to cm_sdpa_2nd_turboquant, but only keep the code branches of `KV_CACHE_COMPRESSION==False` and `USE_LSC_BLOCK_2D_DESC==True`.
2. Apply turboquant into attention kernel in cm_sdpa_2nd_turboquant: When turboquant_enabled, CompressedKVCache_Update_CM is used for key. Keep value use fp16 cache for now. Refer to triton_attention.py and turboquant_fused.py to know how to apply turboquant to attention, and apply it into cm_sdpa_2nd_turboquant.
3. PaSingleTokenRunner choose this turbo kernel if it is enabled and XE_ARCH>=2, throw exception for XE_ARCH=1.
4. Create pytest style unit tests for it to check accuracy.
5. Create pytest style unit tests for it which implements similar benchmark to benchmark_fused_vs_standard in triton_attention.py. 

Please refer to the following code and documents for details of algorithm and test frames -
1. Refer to opencl/tests/pageatten/test_pa_decoding.py with KV_CACHE_COMPRESSION==False path only to understand decoding path.
2. Refer to opencl/tests/pageatten/turboquant_cm.py to understand how CompressedKVCache update compressed kvcache.
3. Refer to triton_attention.py and turboquant_fused.py to know how to apply turboquant to attention.

## Task: Optimize TurboQuant decoding kernel
1. Turboquant path should be faster than standard. Speedup should be larger than 1.0. If not, inspect the implementation with a specific kv_len to profile the performance and optimize the implementation untill turboquant speeds up. Refer to "GPU skills".