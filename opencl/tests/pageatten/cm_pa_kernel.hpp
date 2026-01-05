
#include "cm_pa_common.hpp"


extern "C" _GENX_MAIN_ void cm_page_attention(
    //query [q_len, num_heads, S]
#ifdef CM_HAS_LSC_UNTYPED_2D
    half* query [[type("svmptr_t")]],
#else
    SurfaceIndex q_gather [[type("buffer_t")]],
#endif
#ifdef CM_HAS_LSC_UNTYPED_2D
#if CMPA_KVCACHE_U8
    int8_t* k_cache [[type("svmptr_t")]],
    int8_t* v_cache [[type("svmptr_t")]],
#else
    half* k_cache [[type("svmptr_t")]],
    half* v_cache [[type("svmptr_t")]],
#endif
#else
#if CMPA_KVCACHE_U8
    int8_t* k_cache [[type("svmptr_t")]],
    int8_t* v_cache [[type("svmptr_t")]],
#else
    half* k_cache [[type("svmptr_t")]],
    SurfaceIndex v_cache_stateful [[type("buffer_t")]],
#endif
#endif
    int32_t* past_lens [[type("svmptr_t")]],
    int32_t* block_indices [[type("svmptr_t")]],
    int32_t* block_indices_begins [[type("svmptr_t")]],
    int32_t* subsequence_begins [[type("svmptr_t")]],
    half* output [[type("svmptr_t")]],
#if CMPA_KVCACHE_U8
#if SPARSE_BLOCK_SIZE > 1
    SurfaceIndex sparse_block_mask [[type("buffer_t")]],
    SurfaceIndex sparse_block_mask_wg [[type("buffer_t")]],
    int q_len,
    int num_q_blocks,
    int num_k_blocks,
    uint8_t validate_u8
) {
#else
    int q_len
) {
#endif
#else
#if SPARSE_BLOCK_SIZE > 1
    bool* sparse_block_mask [[type("svmptr_t")]],
    bool* sparse_block_mask_wg [[type("svmptr_t")]],
    int q_len,
    int num_q_blocks,
    int num_k_blocks,
    // validate sparse atten process
    bool validate) {
#else
    int q_len) {
#endif
#endif
    constexpr int is_causal = CMFLA_IS_CAUSAL;
    constexpr int num_heads = CMFLA_NUM_HEADS;
    constexpr int head_size = CMFLA_HEAD_SIZE;
    constexpr int num_kv_heads = CMFLA_NUM_KV_HEADS;
    constexpr int pa_block_sz = CMPA_BLOCK_SZ;
    //# query [q_len, num_heads, S]
    //# k_cache [kv_len, num_heads, S]
    //# v_cache [kv_len, num_heads, S]
#if CMPA_KVCACHE_U8
    constexpr uint K_SLM_SIZE = (4*kv_step * head_size * sizeof(half));
    constexpr uint V_SLM_SIZE = (4*kv_step * head_size * sizeof(half));
    constexpr uint Q_SLM_SIZE = 0;

    cm_slm_init(K_SLM_SIZE + V_SLM_SIZE + Q_SLM_SIZE);

    auto slm_K = cm_slm_alloc(K_SLM_SIZE);
    auto slm_V = cm_slm_alloc(V_SLM_SIZE);

#endif
    auto batch = cm_group_id(0);
    auto h = cm_group_id(1);
    auto hkv = h / (num_heads/num_kv_heads);
    auto wg_id = cm_group_id(2); // each work-group handles a sequence
    auto wg_local_id = cm_local_id(2);
    int  local_size  = cm_local_size(2);

    // Split query across WGs / SGs
    int wg_seq_len  = local_size * q_step;
    int past_q_lens = past_lens[0];
    int kv_seq_len  = q_len + past_q_lens;

    int q_start_sg = (wg_id * local_size + wg_local_id) * q_step;
    int q_len_sg   = (q_start_sg + q_step > q_len) ? (q_len - q_start_sg) : q_step;
    if (q_len_sg < 0) q_len_sg = 0;

    // causal => wg-level kv_stop
    int kv_stop = kv_seq_len;
    if constexpr (is_causal) {
        kv_stop = (wg_id + 1) * wg_seq_len + past_q_lens;
        if (kv_stop > kv_seq_len) kv_stop = kv_seq_len;
    }

    // Q/O offset: [B, L, H, S] flattened
    // Use 64-bit to avoid overflow for long seq
    uint64_t q_offset_elems = ((uint64_t)q_start_sg * (uint64_t)num_heads + (uint64_t)h) * (uint64_t)head_size;
    uint32_t q_offset_bytes = (uint32_t)(q_offset_elems * (uint64_t)sizeof(half));

    
#if CMPA_KVCACHE_U8
#if SPARSE_BLOCK_SIZE > 1
    const bool validate = (validate_u8 != 0);
    uint32_t block_mask_base_idx = 0;
    uint32_t wg_block_mask_base_idx = 0;

    if (validate) {
        const uint32_t q_start_block = (uint32_t)(q_start_sg / SPARSE_BLOCK_SIZE);
        block_mask_base_idx = (uint32_t)((h * (uint32_t)num_q_blocks + q_start_block) * (uint32_t)num_k_blocks);
        wg_block_mask_base_idx = (uint32_t)((h * (uint32_t)cm_group_count(2) + (uint32_t)wg_id) * (uint32_t)num_k_blocks);
    }

    uint32_t block_mask_base_byte_off    = block_mask_base_idx * (uint32_t)sizeof(uint8_t);
    uint32_t wg_block_mask_base_byte_off = wg_block_mask_base_idx * (uint32_t)sizeof(uint8_t);
#endif
    uint64_t kv_offset_elems = (uint64_t)hkv * (uint64_t)(head_size + 4) * (uint64_t)pa_block_sz;

    pa_lsc_u8<is_causal, num_heads, num_kv_heads, head_size, 0>(
        slm_K,
        slm_V,
        wg_local_id,
        local_size,
        q_start_sg,     // q_start for SG
        kv_stop,
        q_len_sg,       // q_step
        kv_seq_len,     // kv_len
#if USE_LSC
        reinterpret_cast<svmptr_t>(query + q_offset_elems),
#else
        q_gather,
        q_offset_bytes,
#endif
        reinterpret_cast<svmptr_t>(k_cache + kv_offset_elems),
        reinterpret_cast<svmptr_t>(v_cache + kv_offset_elems),
#if SPARSE_BLOCK_SIZE > 1
        sparse_block_mask,
        sparse_block_mask_wg,
        block_mask_base_byte_off,
        wg_block_mask_base_byte_off,
        validate_u8,
#endif
        reinterpret_cast<svmptr_t>(output + q_offset_elems),
        past_q_lens,
        block_indices
    );
#else
#if SPARSE_BLOCK_SIZE > 1
    bool *block_mask_base, *wg_block_mask_base;
    if (validate) {
        //# sparse_block_mask [num_heads, num_q_blocks, num_k_blocks]
        //# sparse_block_mask_wg [num_heads, wg_count_along_query, num_k_blocks]
        auto q_start_block = q_start_sg/ SPARSE_BLOCK_SIZE;
        block_mask_base = sparse_block_mask + (h * num_q_blocks + q_start_block) * num_k_blocks;
        wg_block_mask_base = sparse_block_mask_wg + (h * cm_group_count(2) + wg_id) * num_k_blocks;
    }
 #endif
    uint kv_offset = hkv*head_size*pa_block_sz;
    uint kv_offset_bytes = kv_offset * sizeof(half);
    pa_kernel_lsc_prefetch_f16<is_causal, num_heads, num_kv_heads, head_size, 0, 16>(
                            wg_local_id,
                            q_start_sg, //q_start for SG,
                            kv_stop,
                            q_len_sg, //q_step,
                            kv_seq_len, //kv_len,
#ifdef CM_HAS_LSC_UNTYPED_2D
                            reinterpret_cast<svmptr_t>(query + q_offset_elems),
#else
                            q_gather,
                            q_offset_bytes,
#endif
                            reinterpret_cast<svmptr_t>(k_cache + kv_offset),
#ifdef CM_HAS_LSC_UNTYPED_2D
                            reinterpret_cast<svmptr_t>(v_cache + kv_offset),
#else
                            v_cache_stateful,
                            kv_offset_bytes,
#endif
#if SPARSE_BLOCK_SIZE > 1
                            reinterpret_cast<svmptr_t>(block_mask_base),
                            reinterpret_cast<svmptr_t>(wg_block_mask_base),
                            validate,
#endif
                            reinterpret_cast<svmptr_t>(output + q_offset_elems),
                            past_q_lens,
                            block_indices);
#endif
}