## Parsing rules (IMPORTANT)
- Ignore anything inside HTML comments: <!-- ... -->
- Ignore any section under a heading that starts with "DISABLED:"
- Never execute or follow instructions labeled "DISABLED", "COMMENTED", "DONE" or "BACKLOG".

## Context (IMPORTANT)
The kernels are the page attention + flash attention implementation for Intel GPU LNL/PTL with CM language.
Read Intel CM GPU progromming language document under $HOME/OCL/aboutSHW/.github/skills/CM_spec.
Read files under $HOME/OCL/aboutSHW/opencl/tests/pageatten/
Read $HOME/3rdparty/turboquant

## OpenVINO TurboQuant key cache
TurboQuant key cache shape [num_blocks, KV_HEADS_NUM, block_size * (((K_HEAD_SIZE * TQ_BITS + 7)/8) + 2)]
Layout per_token is,
   - idx region: block_size * packed_head_size bytes
   - norm region: block_size * sizeof(fp16) bytes
packed_head_size = ((K_HEAD_SIZE * TQ_BITS + 7)/8)

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


## Task: compute roofline TurboQuant decoding
### platform and assumptions
Use LNL and the roofline model:
- Bound by max of XMX compute, XVE compute, and memory bandwidth.
- XVE peak TFLOPS = XMX peak TFLOPS / 4.
- Scalar ops here are treated as scalar issue work (no SIMD*MAD amplification).
- Worked numbers below use: XMX=30 TFLOPS, XVE=7.5 TFLOPS, BW=120 GB/s.

### benchmark shape
For `test_pa_benchmark_turboquant_vs_standard_decoding`:
- num_heads=32, num_kv_heads=8, head_size=128, block_size=16, turboquant_bits=4
- kv_partition_size=128 (default for block_size<256)
- Example row: kv_len=24576

### equations
- Overall lower bound:
   - `t_min = max(t_mem, t_xmx, t_xve)`
- Memory time:
   - `t_mem = bytes_total / BW`
- XMX time:
   - `t_xmx = FLOPs_xmx / XMX_TFLOPS`
- XVE time:
   - `t_xve = scalar_ops / XVE_TFLOPS`
- ISA-informed XVE correction (from VISA execution width mix):
   - `t_xve_isa = scalar_ops / (XVE_TFLOPS * eta_xve)`
   - `eta_xve` is effective SIMD utilization (`0 < eta_xve <= 1`).

### how TurboQuant decode numbers are computed (kv_len=24576)
1. **Bytes**
    - `k_packed_bytes = (head_size * bits + 7) // 8 = (128*4+7)//8 = 64`
    - TurboQuant KV bytes per token per KV head:
       - `64 (packed K) + 2 (K norm fp16) + 128 (V u8) + 4 (V scale/zp fp16) = 198 B`
    - `kvcache_size = kv_len * num_kv_heads * 198`
       - `= 24576 * 8 * 198 = 38,928,384 B = 38.93 MB`
    - `kv_partition_num = ceil(kv_len / kv_partition_size) = ceil(24576/128)=192`
    - `intermedia_size = batch * num_heads * kv_partition_num * (head_size+1) * 4`
       - `= 1 * 32 * 192 * 129 * 4 = 3,170,304 B = 3.17 MB`
    - `bytes_total = 38.93 + 3.17 = 42.10 MB`

2. **Memory bound**
    - `t_mem = 42.10e6 / 120e9 = 3.508e-4 s = 0.351 ms`

3. **XMX compute bound (DPAS math only)**
    - QK and PV each:
       - `F_QK = num_heads * kv_len * head_size * 2`
       - `= 32 * 24576 * 128 * 2 = 201,326,592 FLOPs`
    - `F_xmx = F_QK + F_PV = 402,653,184 FLOPs`
   - `t_xmx = 402,653,184 / (30e12) = 1.34e-5 s = 0.013 ms`

4. **XVE bound (scalar-heavy ops)**
    - `q_head_chunk_size=4`, `q_head_chunks_per_kv_head=1` for this shape.
    - Query rotation is inside kernel body and repeated per partition (`cm_global_id(2)` path), so:
       - `S_qrot = (4 * 128 * 128 * 2) * (num_kv_heads * kv_partition_num)`
       - `= 201,326,592 scalar ops`
      - K dequant approx 3 scalar ops/element:
          - Kernel operation for each reconstructed K scalar is conceptually:
             - `k_deq = ((idx * affine_scale) + affine_bias) * k_norm`
          - Counted scalar arithmetic per element:
             - `idx * affine_scale`  -> 1 mul
             - `+ affine_bias`       -> 1 add
             - `* k_norm`            -> 1 mul
          - So we use `3 ops/element` as a first-order estimate.
          - Notes on approximation:
             - bit extract (`& 0x0f`, `>> 4`), type converts, pack/unpack, and address math are not included in this 3-op term.
             - those omitted instructions increase real XVE pressure, so roofline remains optimistic.
          - `S_k = kv_len * num_kv_heads * head_size * 3 = 75,497,472`
      - V dequant approx 2 scalar ops/element:
          - Kernel operation for each V scalar is conceptually:
             - `v_deq = (v_u8 - zp) * scale`
          - Counted scalar arithmetic per element:
             - `v_u8 - zp`  -> 1 add/sub
             - `* scale`    -> 1 mul
          - So we use `2 ops/element` as a first-order estimate.
          - Notes on approximation:
             - load/transpose/repack for DPAS feeding and boundary masking are not included in this 2-op term.
             - omitted costs again make measured runtime higher than roofline lower bound.
          - `S_v = kv_len * num_kv_heads * head_size * 2 = 50,331,648`
    - `S_total = S_qrot + S_k + S_v = 327,155,712`
   - `t_xve = S_total / (7.5e12) = 4.36e-5 s = 0.044 ms`
   - ISA-informed correction from `cm_sdpa_2nd_turboquant.visaasm`:
      - decode/repack-heavy `add/mad/and/shr` are dominated by `(M1,1)` instructions.
      - using `add+mul+mad` execution-width mix, `eta_xve_tq ≈ 0.07` (very low SIMD utilization).
      - `t_xve_isa_tq = 0.044 / 0.07 ≈ 0.629 ms`

### how Standard decode numbers are computed (kv_len=24576)
Assume benchmark standard case in this test means `kv_cache_compression=True` + `by_token` (same shape, no TurboQuant query-rotation/decode path).

1. **Bytes**
   - Standard compressed KV bytes per token per KV head from test formula:
      - `head_size * 1 * 2 = 128 * 2 = 256 B`
      - (`u8 payload` + `fp16 scale/zp metadata`)
   - `kvcache_size_std = kv_len * num_kv_heads * 256`
      - `= 24576 * 8 * 256 = 50,331,648 B = 50.33 MB`
   - `intermedia_size` is the same as TurboQuant:
      - `= 3,170,304 B = 3.17 MB`
   - `bytes_total_std = 50.33 + 3.17 = 53.50 MB`

2. **Memory bound**
   - `t_mem_std = 53.50e6 / 120e9 = 4.458e-4 s = 0.446 ms`

3. **XMX compute bound (DPAS math only)**
   - Same QK/PV math shape as TurboQuant for this benchmark row.
   - `F_xmx_std = 402,653,184 FLOPs`
   - `t_xmx_std = 402,653,184 / (30e12) = 1.34e-5 s = 0.013 ms`

4. **XVE bound (scalar dequant estimate)**
   - No TurboQuant query rotation term in standard case.
   - Approximate per-element dequant arithmetic for K and V each:
      - `deq = (u8 - zp) * scale` -> 2 scalar ops/element
   - `S_k_std = kv_len * num_kv_heads * head_size * 2 = 50,331,648`
   - `S_v_std = kv_len * num_kv_heads * head_size * 2 = 50,331,648`
   - `S_total_std = 100,663,296`
   - `t_xve_std = S_total_std / (7.5e12) = 1.342e-5 s = 0.013 ms`
   - ISA-informed correction from `cm_sdpa_2nd_quant_bytoken.visaasm`:
      - dequant arithmetic is mostly `(M1,16)/(M1,32)` in standard path.
      - from `add+mul+mad` width mix, `eta_xve_std ≈ 0.52`.
      - `t_xve_isa_std = 0.013 / 0.52 ≈ 0.025 ms`

5. **Standard roofline lower bound**
   - `t_min_std = max(t_mem_std, t_xmx_std, t_xve_std)`
   - `= max(0.446, 0.013, 0.013) = 0.446 ms`

### result and insight
- TurboQuant theoretical minimum latency:
   - `t_min = max(0.351, 0.013, 0.044) = 0.351 ms`
- Standard theoretical minimum latency:
   - `t_min_std = max(0.446, 0.013, 0.013) = 0.446 ms`
- ISA-informed TurboQuant minimum latency (with low SIMD utilization):
   - `t_min_tq_isa = max(0.351, 0.013, 0.629) = 0.629 ms`
- ISA-informed Standard minimum latency:
   - `t_min_std_isa = max(0.446, 0.013, 0.025) = 0.446 ms`
- With ISA-aware correction, **TurboQuant becomes XVE-limited (or near XVE/memory crossover)**, while standard stays memory-bound.
- Measured TurboQuant latency still indicates substantial non-roofline overhead in current implementation.
- In benchmark summary, TurboQuant measured latency is much higher than this lower bound, indicating extra overhead from scalar/dequant/repack flow, occupancy/GRF effects, and limited overlap.
- For the measured row (`kv_len=24576`):
   - standard `~0.558 ms` is relatively close to `0.446 ms` roofline.
   - TurboQuant `~4.789 ms` is far from both ideal (`0.351 ms`) and ISA-informed (`0.629 ms`) roofs.
- Practical optimization priority:
   1. Reduce repeated query rotation work per partition.
   2. Reduce scalar dequant/repack cost (especially K path).
   3. Improve overlap/prefetch and keep DPAS fed.


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
I need you to analyze the XVE instructions for both turboquant and standard cases, by analyzing their .asm files where decoding and repacking happens, as well as softmax related code.  Some are unable to vectorized  with simd ops, some can. Catogratize both them.
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
3. Add corresponding implementation in OpenVINO for kernels `compressed_kv_cache_update_tq` with value_cache_mode="by_token",  `cm_sdpa_2nd_turboquant`, and `cm_pa_multi_token_turboquant` too, by referencing corresponding python unit tests and classes in source folder. 
The implementations include KernelGenerator in src/plugins/intel_gpu/src/graph/impls/cm/paged_attention_gen.hpp; Initialize them in `class PagedAttentionCmImpl` and use them in `execute` in src/plugins/intel_gpu/src/graph/impls/cm/paged_attention.cpp.

4. TURBOQUANT is enabled by adding ov::internal::CacheQuantMode::TURBOQUANT. This should be reflected in openvino/src/plugins/intel_gpu/include/intel_gpu/runtime/options.inl, `ov::pass::ConvertPagedAttnInputs::KVCacheConfig kv_cache_config;` of src/plugins/intel_gpu/src/plugin/transformations_pipeline.cpp.
In src/plugins/intel_gpu/src/plugin/transformations_pipeline.cpp, if use_turboquant, set corresponding kv_cache_config.
5. Keep code lines of namespace and KERNEL_NAME unchanged.
6. XAttention and pre-Xe2 platform supports are not available when turboquant is enabled.
7. Add turboquant tests in src/plugins/intel_gpu/tests/unit/test_cases/paged_attention_gpu_test.cpp. So far only support single-subseuqnce tests.



## CPU plugin turboboquant POC
PR https://github.com/openvinotoolkit/openvino/pull/35092 under /home/ceciliapeng/workspace/openvino:

### consts

mse.centroids / boundaries are static tables in
src/plugins/intel_cpu/src/nodes/kernels/scaled_attn/turboq_tables.h
(TURBOQ_CODEBOOK_2BIT/3BIT/4BIT, TURBOQ_BOUNDARIES_*).
So these are not trained at runtime in PR code; they are fixed constants.

mse.Q / mse.Q_T (and QJL projection matrices) are generated in
src/plugins/intel_cpu/src/nodes/kernels/scaled_attn/turboq_rotation.cpp
inside get_cache(dim):

Q: Gaussian matrix + Householder QR (Haar orthogonal)
QT: transpose of Q
S, ST, SQ: QJL projection matrices
wht_signs, qjl_wht_signs: random ±1 sign vectors
Accessors are turboq_get_rotation_matrix(), turboq_get_rotation_matrix_t(), etc.
Actual quantization path that uses those constants is in
src/plugins/intel_cpu/src/nodes/kernels/scaled_attn/attn_quant_turboq.cpp
via turboq_quantize_head() and turboq_quantize_head_qjl().

So: centroids are table-defined; Q/Q_T are generated lazily per dim with fixed seeds and cached.

### the flow is:

1. KV-cache update (quantize + write packed cache)
Entry is compress_cache() in openvino/src/plugins/intel_cpu/src/nodes/scaled_attn.cpp.
When codec is enabled, it routes to turboq_quantize_single() (scaled_attn.cpp, scaled_attn.cpp).
turboq_quantize_single() calls turboq_quantize_head() / turboq_quantize_head_qjl() per (b,h,token) (scaled_attn.cpp).
Actual quantization is in openvino/src/plugins/intel_cpu/src/nodes/kernels/scaled_attn/attn_quant_turboq.cpp: normalize, rotate, quantize to codebook indices, pack_4bit/pack_3bit, and store norm (attn_quant_turboq.cpp, attn_quant_turboq.cpp).
This update is called in cache append/init paths, e.g. scaled_attn.cpp and beam/realloc path scaled_attn.cpp.

2. Attention-time dequant (on-the-fly, fused)
Single-token decode path selects mha_turboq() (scaled_attn.cpp).
Inside mha_turboq() (attn_quant_turboq.cpp):
QK phase uses record_qk_dot() through dispatch_codec()/score_tokens() (attn_quant_turboq.cpp, attn_quant_turboq.cpp).
V phase uses record_v_accum() through dispatch_codec()/accum_tokens() (attn_quant_turboq.cpp, attn_quant_turboq.cpp).
TBQ “dequant scale” is read from packed norm via turboq_norm_scale() (attn_quant_turboq.cpp).
Final reduce/inverse-rotation step is turboq_reduce() (attn_quant_turboq.cpp, attn_quant_turboq.cpp).

So dequant is not a separate “materialize full K/V” kernel; it is fused into attention score/accumulation.