# PA kernel spec for CM kernels

This document is a kernel-level companion to the primitive-level PA specification.
It does not change the primitive contract. Instead, it defines how CM kernels consume
primitive inputs, plus an optional subgroup-selection extension for zero-copy mixed split.

## Goal

Support mixed-only split routing without repacking:
- no new query buffer
- no new key_cache/value_cache buffers
- no new block_indices buffer
- no output scatter after kernel completion

The primitive-level PA inputs remain unchanged. CM kernels add optional sequence-selection
metadata and interpret original primitive buffers through that indirection.

## Base mapping from PA primitive inputs to CM kernel inputs

Primitive-level canonical inputs:
- query: [batch_size_in_tokens, num_heads * head_size]
- key_cache: [num_blocks, num_kv_heads, block_size, head_size] or kernel-specific packed variant
- value_cache: [num_blocks, num_kv_heads, block_size, head_size] or kernel-specific packed variant
- past_lens: i32[batch_size_in_sequences]
- subsequence_begins: i32[batch_size_in_sequences + 1]
- block_indices: i32[num_blocks]
- block_indices_begins: i32[batch_size_in_sequences + 1]
- max_context_len: i32 scalar

CM kernels keep consuming these original tensors directly.

## Solution B: optional subgroup selection

### New optional kernel inputs

These inputs are kernel-level only. They do not change the primitive-level PA spec.

1. selected_sequence_ids
- Type: i32[selected_sequence_count]
- Semantics: local subgroup sequence index -> original primitive sequence index
- Example: [0, 2] means the subgroup executes original sequences 0 and 2 only

2. selected_sequence_count
- Type: i32 scalar
- Semantics: number of active entries in selected_sequence_ids

### Optional enable rule

- If selected_sequence_ids is not provided, kernels use the legacy dense behavior:
  local sequence i maps to original sequence i.
- If selected_sequence_ids is provided, kernels must use indirect sequence lookup.

## Exact kernel-side interpretation

For each kernel-local subgroup sequence index local_seq_idx:

- orig_seq_idx = selected_sequence_ids[local_seq_idx]
- q_start = subsequence_begins[orig_seq_idx]
- q_end = subsequence_begins[orig_seq_idx + 1]
- q_len = q_end - q_start
- block_start = block_indices_begins[orig_seq_idx]
- block_end = block_indices_begins[orig_seq_idx + 1]
- block_count = block_end - block_start
- past_len = past_lens[orig_seq_idx]

Kernels must then use:
- query tokens from original query[q_start:q_end]
- block table entries from original block_indices[block_start:block_end]
- cache blocks via physical block IDs in original key_cache/value_cache
- output writes to original output token positions derived from q_start/q_end

No repacked subgroup query/cache/block buffers are created.

## Multi-token kernel contract

The multi-token CM kernel must support:
- iterating over selected_sequence_count logical subgroup sequences
- mapping each local subgroup sequence to orig_seq_idx through selected_sequence_ids
- deriving q_start/q_end and block table base from original primitive metadata
- writing output directly into the original output tensor using absolute token offsets

Legacy behavior:
- if subgroup selection is absent, behavior is identical to current dense packed mode

## Single-token kernel contract

The single-token CM kernel must support:
- iterating over selected_sequence_count logical subgroup sequences
- mapping each local subgroup sequence to orig_seq_idx through selected_sequence_ids
- deriving q_token_idx = subsequence_begins[orig_seq_idx + 1] - 1
- reading KV metadata from original block_indices/block_indices_begins/past_lens
- writing final output directly to the original output token position q_token_idx

Reduction kernel requirement:
- reduction must preserve absolute output token placement
- no host-side scatter should be required after kernel completion

## Host-side invariants

These invariants must hold whenever selected_sequence_ids is used.

1. Range validity
- selected_sequence_count >= 0
- len(selected_sequence_ids) == selected_sequence_count
- every selected_sequence_ids[i] is in [0, batch_size_in_sequences)

2. Split-routing partition rule
- For mixed-only split, decode subgroup contains exactly the original sequences with q_len == 1
- For mixed-only split, prefill subgroup contains exactly the original sequences with q_len > 1
- Decode and prefill selected_sequence_ids must be disjoint
- Their union must equal the full original sequence set participating in the call

3. Primitive metadata remains canonical
- past_lens, subsequence_begins, block_indices, and block_indices_begins always describe the full original PA call
- subgroup selection never mutates or rewrites these primitive tensors

4. Query/output alias rule
- subgroup kernels read from the original query tensor
- subgroup kernels write directly into the original output tensor at absolute token offsets
- host must not assume dense subgroup-local output layout

5. Block-table contiguity rule
- For each original sequence orig_seq_idx, block_indices[block_start:block_end] remains contiguous as defined by the primitive spec
- Kernel subgroup selection may skip whole sequences, but must not reinterpret partial block spans of a selected sequence

6. Sequence identity rule
- Each subgroup entry corresponds to one original primitive sequence
- Solution B supports subset selection and reordering of original sequences
- Solution B does not define synthetic sequence layouts such as splitting one original sequence into two subgroup entries or merging multiple original sequences into one subgroup entry

7. Duplicate-ID policy
- Default: selected_sequence_ids must not contain duplicates
- If duplicates are ever allowed for a special use case, output write conflicts must be explicitly defined by the kernel contract before enabling them

8. Empty subgroup rule
- selected_sequence_count may be zero only if the corresponding kernel dispatch is skipped by the host
- Kernels are not required to process an empty selected_sequence_ids table

## Backward compatibility

Legacy packed mode remains valid:
- omit selected_sequence_ids and selected_sequence_count
- kernels interpret the full primitive call in dense order

Zero-copy split mode:
- provide selected_sequence_ids and selected_sequence_count
- reuse all primitive tensors unchanged

## Host-side construction for mixed-only split

Given the full primitive metadata:
- token_counts[i] = subsequence_begins[i + 1] - subsequence_begins[i]
- decode_selected_sequence_ids = [i for i where token_counts[i] == 1]
- prefill_selected_sequence_ids = [i for i where token_counts[i] > 1]

Dispatches:
- single-token kernel uses decode_selected_sequence_ids
- multi-token kernel uses prefill_selected_sequence_ids

No subgroup-local copies of query, cache, or block table are required.
