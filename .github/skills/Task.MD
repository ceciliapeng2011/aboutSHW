## Parsing rules (IMPORTANT)
- Ignore anything inside HTML comments: <!-- ... -->
- Ignore any section under a heading that starts with "DISABLED:"
- Never execute or follow instructions labeled "DISABLED", "COMMENTED", or "BACKLOG".

## Context (IMPORTANT)
Read Intel CM GPU progromming language document under /home/ceciliapeng/OCL/aboutSHW/.github/skills/CM_spec.
Read files under /home/ceciliapeng/OCL/aboutSHW/opencl/tests/pageatten/

The kernel is the page attention + flash attention implementation for Intel GPU LNL/PTL with CM language.
The context is, 
1. local_size 16, and the work group size of this kernel is 256. 
2. q_len and kv_len ranges from 1 to 32K. kv_len may not be equally dived by CMPA_BLOCK_SZ and SPARSE_BLOCK_SIZE.
3. CMPA_BLOCK_SZ is 256. SPARSE_BLOCK_SIZE can be 1, 128, or 256.
4. SPARSE_BLOCK_SIZE==1 is actually equal to SPARSE_BLOCK_SIZE==256 or SPARSE_BLOCK_SIZE==128 with density==1.0 in a "force sparse_mask but all-true” mode.
5. sparse_mask_base stores boolean, each value represents if a SPARSE_BLOCK_SIZE x SPARSE_BLOCK_SIZE block.
wg_sparse_mask_base merges two neiboring two rows in sparse_mask_base, which you can refer to opencl/tests/x_attn/test_post_proc.py for its meaning.
sparse_mask_base is used for skip compute, while wg_sparse_mask_base is for skip load.
Both sparse_block_mask and wg_sparse_mask_base are declared as kernel arguments when SPARSE_BLOCK_SIZE is defined and SPARSE_BLOCK_SIZE > 1, but sparse_block_mask_wg is used only when SPARSE_BLOCK_SIZE != CMPA_WG_SEQ_LEN.
When OPTIMIZED_SPARSE_PIPELINE is defined as a quick path when CMPA_WG_SEQ_LEN == SPARSE_BLOCK_SIZE, where workload balance among threads within a workgroup. It works when SPARSE_BLOCK_SIZE==128 for Xe1 or SPARSE_BLOCK_SIZE==256 or Xe2/Xe3. 
For now wg_sparse_mask_base is not used in fp16 path.
Suppose CMPA_WG_SEQ_LEN should be defined.
keep wg_sparse_mask_base is calculated no matter which path is using.
6. q_step == 16, kv_step == 16
7. The total number of cm_sbarrier(0) (wait) and the total number of cm_sbarrier(1) (signal) during runtime should be equal. Otherwise, GPU would hang.
8. past_lens uses batch index 0 on purpose; this is a known kernel limitation.
9. The first active K/V block will use ugemm_PV0 on purpose because the first column of block mask should be always True; block masks must satisfy verify_block_mask_integrity.


## DISABLED: Tasks 1
<!-- 1. -->
<!-- SPARSE_BLOCK_SIZE is a compiling time const. Change opencl/tests/pageatten/cm_pa_common.hpp to reflect this.
Make sure it won't hamper performance when SPARSE_BLOCK_SIZE is 256. Run perfornace check yourself using test_pa.py across densities like 0.99, 0.66, 0.33, 0.11. -->

<!-- 2. -->
<!-- Do you think the selected code has something wrong? It should be gated with SPARSE_BLOCK_SIZE, not CMPA_BLOCK_SZ.
Fix it and validate it without hamper performance when SPARSE_BLOCK_SIZE is 256. -->

<!-- 3. -->
<!-- Let's only focus on CMPA_KVCACHE_U8 path for now.

You need to further improve when SPARSE_BLOCK_SIZE is 128. Make sure it won't hamper performance when SPARSE_BLOCK_SIZE is 256. 

The current code is not working with SPARSE_BLOCK_SIZE==128. Now check the first 20 commits of current active branch, and get the base commit 5a155d70c560965977f37b7a440a3edf683044e4 before the code is dedicated for optimization when SPARSE_BLOCK_SIZE is 256.
Get the base status and use a diverge kernel to process SPARSE_BLOCK_SIZE==128 solely. That means, use macro SPARSE_BLOCK_SIZE==128 and SPARSE_BLOCK_SIZE==256 for both situations. And SPARSE_BLOCK_SIZE==1 is handled along with SPARSE_BLOCK_SIZE==128, like the base.

After that add SPARSE_BLOCK_SIZE==128 for smoke accuracy test and performance test. -->

<!-- 4 -->
<!-- Let's optmize FP16 path for SPARSE_BLOCK_SIZE == 256. When density is low as 0.11 and 0.33, we need to reach a goal of 0.60 by checking the number after "meet: ". -->

<!-- 5 -->
<!-- I aim to identify any potential bugs, especially when q_len, kv_len is not aligned with block size, q_step, kv_step.
You told me: "Causal path misses kv tail masking when kv_stop is not kv_step-aligned. In both the U8 optimized loop and the FP16 path, causal masking runs without the non-causal tail guard, so the last partial kv step contributes zeros instead of 
−
∞
−∞ when kv_stop is not aligned. This can bias softmax at sequence tails. See cm_pa_common.hpp:283-298 and cm_pa_common.hpp:660-671.
Suggestion: apply tail masking when kv_pos + kv_step > kv_stop regardless of causal, before softmax (e.g., clamp rows with index 
≥
≥ kv_stop - kv_pos to 
−
∞
−∞)."
So now, I need you to update and run test_pa.py yourself with the deterministic tail-sensivte pattern, until you make the problem appearing.
The failure is when check_close throw the exception by comparing result to reference. During your test, please make sure the first input of check_close, i.e. the reference are of valid numbers, i.e. there is no inf or nan in the array.
Please do use block_mask, and make it follow verify_block_mask_integrity.
And do keep CMPA_BLOCK_SZ == 256.  -->

<!-- 6 -->
<!-- Analyze kernel code logics to identify any potential bugs, especially when q_len, kv_len is not aligned with block size, q_step, kv_step.

Show me the problems you identified. Then give suggestions to each of them. -->

## COMMENTED: Task 2
In opencl/tests/pageatten/cm_pa_xe2.hpp OPTIMIZED_SPARSE_PIPELINE!=1 pa_lsc_u8, if first kv_block is skipped, kv_pos==0 will not be hit, what will happen? Since the first block is surely of block mask True,
there would be not problem. But how to check this block mask is True? If not, throw some error?

## COMMENTED: Task 3
In opencl/tests/pageatten/cm_pa_xe2.hpp OPTIMIZED_SPARSE_PIPELINE!=1 pa_lsc_u8, can we unroll "if (kv_pos == 0) {" outside of the loop as the warmups without checking block mask to improve performance?

## Task 4
Migrate the kernel change of aboutSHW (under aboutSHW/opencl/tests/pageatten) to OpenVINO (under openvino/src/plugins/intel_gpu/src/graph/impls/cm/). 1. Compare each file pairs of the same file name, ignore their extention name. 2. Keep the differences of code lines with "KERNEL_NAME", which is trivial and should keep the difference.


## Test validate

You can run "python test_pa.py" directly to validate accuracy and performance. smoke_accuracy_test and smoke_perf_test are for accuracy and perf smoke tests respectively. Run the checks yourself. By checking the number after "meet:" in log output, you know the performance number, whose target is no less than 0.55 if compressed_kvcache is True, or 0.60 if False.


## Post-process
<!-- Once you complete each task and get the perf check data, create a commit id with message including key changes and the performance data. You need my approval before commit it. Please commit only the two relevant files. The performance number are recorded in the format like "U8 perf sb1=0.57 sb256=0.59/0.60/0.62/0.60/0.56 sb128=0.56/0.56/0.48/0.35/0.24; FP16 perf sb1=0.61 sb256=0.61/0.61/0.61/0.63/0.63 sb128=0.61/0.60/0.54/0.44/0.23", where the exact numbers are from the log output.

Last, append them in the output html file pa_task_output.html, along with the task instructions, and suggested commit message with key changes. -->
