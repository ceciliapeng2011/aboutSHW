#!/bin/bash
# Run every test touched by syncing cm kernels from
# openvino.pipeline.mx/thirdparty/openvino/src/plugins/intel_gpu/src/graph/impls/cm into this
# tree (see git log for the individual syncs: license/style-only, pa_kv_cache_update_ref rename,
# HAS_QQ_BIAS speculative-decoding mask, cm_sdpa_opt decode NaN-guards, draft_block_mask rename).
# Continues past failures in one suite so you get a full picture across all of them.
set -uo pipefail

cd "$(dirname "$0")"

export HW_PEAK_FLOPS="${HW_PEAK_FLOPS:-59}"
export HW_PEAK_BW_GBS="${HW_PEAK_BW_GBS:-100}"

echo "##### [1/4] x_attn -- xattn_find_block.cm / xattn_gemm_qk.cm / xattn_post_proc.cm (license-only sync) #####"
pytest -v x_attn/test_find_block.py x_attn/test_gemm_qk.py x_attn/test_post_proc.py \
          x_attn/test_sort.py x_attn/test_xattn.py

echo "##### [2/4] pageatten -- pa_multi_token.cm (HAS_QQ_BIAS), pa_single_token*.cm, #####"
echo "#####             cm_attention_common.hpp / cm_pa_xe1.hpp / cm_pa_xe2.hpp, pa_kv_cache_update_ref.cm #####"
pytest -v pageatten/test_kvcache_update.py pageatten/test_pa.py pageatten/test_pa_decoding.py \
          pageatten/test_pa_multiseq.py pageatten/test_cb.py
# Optional, long-running perf variants (opt-in via env):
# PA_PERF=1 RUN_PA_PERF=1 pytest -v pageatten/test_pa_multiseq_perf.py pageatten/test_pa_decoding.py

echo "##### [3/4] sdpa_cm -- cm_sdpa_common.hpp, cm_sdpa_vlen.cm, cm_sdpa_opt_decode*.cm (NaN-guard fixes) #####"
pytest -v sdpa_cm/test_sdpa_prefill.py sdpa_cm/test_sdpa_decode.py sdpa_cm/test_sdpa_prefill_block_mask.py
(cd sdpa_cm && python3 bench_sdpa.py)

echo "##### [4/4] sdpa_cm/flashvsr -- draft_block_mask_{pool,score_softmax,topk_mask}.cm, flashvsr_mask_adapter.py #####"
pytest -v sdpa_cm/flashvsr/test_flashvsr_mask.py sdpa_cm/flashvsr/test_flashvsr_configs.py \
          sdpa_cm/flashvsr/test_flashvsr_block_sparse_attention.py
(cd sdpa_cm/flashvsr && python3 bench_flashvsr_mask.py)
(cd sdpa_cm/flashvsr && python3 bench_flashvsr_e2e.py)

echo "##### done #####"
