## Parsing rules (IMPORTANT)
- Ignore anything inside HTML comments: <!-- ... -->
- Ignore any section under a heading that starts with "DISABLED:"
- Never execute or follow instructions labeled "DISABLED", "COMMENTED", or "BACKLOG".

## Context (IMPORTANT)
The kernels are the page attention + flash attention implementation for Intel GPU LNL/PTL with CM language.
Read Intel CM GPU progromming language document under $HOME/OCL/aboutSHW/.github/skills/CM_spec.
Read files under $HOME/OCL/aboutSHW/opencl/tests/pageatten/
Read $HOME/3rdparty/turboquant


## Task: Implement TurboQuant
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