# vLLM patches carried by this branch

Applied to an installed vLLM with `deploy/scicom/apply-vllm-patches.sh <venv python>`.

## `vllm-0.31.0-nixl-per-region-kv-heads.patch`

Lets a gemma-4-31B prefill and decode engine run with different TP sizes, for example prefill
TP4 and decode TP8.

**Why stock vLLM fails.** gemma-4-31B has 50 sliding-window layers with 16 KV heads (head dim 256)
and 10 global layers with 4 KV heads (head dim 512). The NIXL connector uses one model-wide head
count (16) to size and slice every KV region. At TP8 the 4 global heads cannot be split further,
so each GPU keeps one whole head and the global region does not halve between TP4 and TP8. The
decode rejects the prefill's KV:

    AssertionError: SPLIT region 5: remote P KV block_len 131072 must equal
    local 131072 * remote_heads 4 // local_heads 2 // block_size_ratio 1.

Region 5 is layer 5, the first global layer.

**What the patch does.** Each side advertises the KV heads per rank of every region
(`NixlAgentMetadata.region_kv_heads`). The handshake check and the read plan use that per-region
count instead of the model-wide one: a sliding region still reads its half of the prefill's
block, a global region with one head on both sides reads the whole block. A peer that does not
advertise the field keeps the old behaviour.

**Required flag on both engines:** `--disable-hybrid-kv-cache-manager`. With the hybrid manager
one tensor is shared by a sliding and a global layer, which cannot be re-sharded; the patch
rejects that with a clear error. The cost is KV capacity: the sliding layers then keep the whole
context instead of the last 1,024 tokens. Measured on H20 at 0.85 memory: decode TP4 holds 477,824
tokens (prod decode TP4 with the hybrid manager: 4,355,284).

**Limits.** Decode TP must be at least prefill TP for layers with fewer heads; prefill TP above
decode TP with differing head counts raises `NotImplementedError`.

**Origin.** Written against vLLM main `fa75d34`; ported to 0.31.0 here (one hunk by hand, the
`AttentionSpec` import that 0.31.0's `base_worker.py` lacks, and the two tests adapted to 0.31.0's
per-block descriptors). 64 of 64 tests pass in `tests/v1/kv_connector/unit/test_tp_mapping.py` and
the handshake, per-region and gemma4 tests of `test_nixl_connector.py`.

**Measured on tm-h20 (2026-10-17, vLLM 0.31.0 cu129, `ai-dynamo` 1.6.0.dev20261016):**

| Prefill -> decode | vLLM | Recall check (9 prompts, 1.6k-12.6k tokens) | KV transfer |
|---|---|---|---|
| TP2 -> TP4, one node | patched | 9/9 | `tp_ratio=2`, 0 errors |
| TP4 (1021) -> TP8 (1019), RoCE | patched | 9/9 | 0 failed, ~5.7 GB/s |
| TP4 (1021) -> TP8 (1019), RoCE | stock | HTTP 500 | `SPLIT region 5` assert, `handshake_failed` |
