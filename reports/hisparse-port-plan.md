# HiSparse + sketch-selected sparse decode for HyperQwen (Qwen3.8-27B, RTX 3090)

Date: 2026-10-31. Sources: vLLM PRs #53781, #51323, #56629, #59209, #60602 and issue #60072 (all read through the GitHub REST API). Our fork was read only: the HyperQwen checkout at `ce9615c`, with the installed `venv/.../site-packages/vllm`.

## 0. Facts that shape the plan

- **Installed vLLM:** 0.31.0, `db9527a46`, committed 2026-10-02. HyperQwen applies 51 patches (`patches/series`) on top, and pthfa is installed as a plugin, outside those patches. Production already runs **Model Runner V2**: the logs come from `v1/worker/gpu/model_runner.py`, and HiSparse requires V2.
- **Production serve line:** `--attention-backend CUSTOM` (pthfa), `--kv-cache-dtype int8_per_token_head`, MTP k=7 (drafter also CUSTOM), `FULL_AND_PIECEWISE`, `--async-scheduling`, `--enable-prefix-caching --mamba-cache-mode align`, `--max-num-batched-tokens 2048`, `max-num-seqs 2`, `max-model-len 153600`. The boot log prints **attention block size 864 tokens** and a GPU KV pool of about 210K tokens (1.37 requests at 150K).
- **KV cost per token per layer:** 4 heads × 2 (K and V) × (256 + 4-byte inline scale) = 2,080 B. Over 16 layers that is 33.3 KB/token, so 150K tokens is 5.0 GB.
- **Expected gain (estimate, not measured):**
  - A dense verify step at 150K reads about 5.0 GB of KV, roughly 6 ms on a 3090. W4A16 weights add about 13–14 GB, roughly 16 ms.
  - Sparse decode should cut each step by about 4.5 ms at 150K, around 20%. The gain is nil below 32K.
  - The main payoff is **capacity**. A 16 GB pinned pool per server holds about 480K tokens. The VM has 78 GB RAM with about 44 GB available, shared by two servers.
- **The 0.31.0 tree already contains merged HiSparse** (#51323/#53781/#56629 era): `vllm/v1/hisparse/{binding,coordinator,layout,runtime,block_pool,types}.py`, `distributed/kv_transfer/kv_connector/v1/hisparse/`, `SparseCacheRole`, `HiSparseResidentSpec`/`HiSparseHotSpec`, and the CUDA ops `hisparse_gather_plan`/`hisparse_resolve_residency` in `_C_stable_libtorch.abi3.so`.
- **What 0.31.0 is missing:** everything after Oct 2. That includes #59209, #60602 (no `_maybe_dequant` in our `qwen4_exp/nvidia/ops/qsa.py`), and the October HiSparse fixes listed in §a.

## (a) What upstream implements, and its state

**HiSparse base (#51323 → #53781, merged 2026-08-25 and 2026-09-12; #56629, merged 2026-09-12)**

- It adds a pinned-host tier *below* the normal GPU KV cache. Pages stay resident on GPU and spill to host only under pressure.
- Each request gets a fixed **GPU hot buffer** of `device_buffer_size` token rows, with an int16 LRU, so the cap is 32,768 rows per request per layer.
- At decode, a fused CUDA resolver (`hisparse_resolve_residency`) maps each selected token to a resident page, a hot row (LRU hit) or host (miss, gathered by `hisparse_gather_plan` from pinned memory). All on GPU, CUDA-graph replayable.
- Scheduler side: `HiSparseCoordinator` owns host blocks and spill policy. `HiSparseConnector` and `HiSparseConnectorWorker` carry the copies. `HiSparseRuntime` (`v1/hisparse/runtime.py`) holds the hot/host views.
- The selector (the "indexer") stays a GPU-resident, prefix-cacheable cache group with `SparseCacheRole.INDEXER`.
- #56629 lets TP ranks share one host pool (MLA only).

**#59209 "HiSparse for QSA" (OPEN; head `caf4f7ae`, 2 commits, +4361/−115, 21 files, base `b8141bd` of 2026-10-30, mergeable_state "unstable")**

- Adds `SparseFullAttentionSpec(FullAttentionSpec)` with `top_k` and `total_num_kv_heads` (`v1/kv_cache_interface.py`). Each layer is its own index-group leader.
- Forces the `BLNHC` layout (`v1/attention/backends/utils.py::resolve_kv_cache_layout`), so one host/hot row holds a token's K+V for every local head.
- Changes in `v1/hisparse/layout.py`, `binding.py` and `runtime.py`:
  - generalises MLA to "sparse source + indexer";
  - adds logical host accounting per unique KV head;
  - adds `HiSparseRuntime.gather_selected_cache()`, which stages the rows of independent query rows without touching LRU state.
- `single_type_kv_cache_manager.py::HiSparseResidentManager` and `coordinator.py::plan_prefix_tail_restore` add private writable tails on partial prefix hits.
- Model side (`models/qwen4_exp/nvidia/qsa.py::_forward_hisparse`): `write_target` → `do_kv_cache_update` → decode `cache.swap_in(...)` (logical top-k → hot-buffer rows) → kernel with `physical_indices=True`. `Qwen4ExpQSAHiSparseMetadata.req_id_per_token` masks FULL padding rows to −1. MTP verify rows under capture use `gather_selected_cache`; nonresident prefill uses `build_hisparse_prefill_staging_plan` + `gather_prefill_cache`.
- Kernel (`ops/qsa.py::_qsa_sparse_paged_gqa_splitk_kernel`): grid (row, kv_head, split); packed indices `[rows, TOPK+1]` with the **valid count in the trailing column** as the graph-safe loop bound; new `PHYSICAL_INDICES` constexpr bypasses `block_table`; fused sigmoid output gate; **one selection per row, shared by all KV heads.**
- Validated: TP2, BF16/FP8, MTP3 + `FULL_DECODE_ONLY` + async; prefix lifecycle at TP1/MTP off. Final increment got compatibility review only; human review pending. GSM8K 197/200 on and off.
- **Performance:** AIPerf shows **no throughput gain**. Low concurrency is **17–20% slower with zero misses** (resolver, gather launches, index copies, eager host mirroring). This matters for us because we run 1–2 users.

**#60072 items that matter to us**

- **Merged after our base (must cherry-pick):** #60682 swap-capacity off-by-one, #59688 reject `cudagraph_mode=FULL`, #60083/#60103 residency caching and fused slot invalidation, #60805 hot-buffer sizing, #61263 hash probes, #60647 prefix hits adopting GPU copies, #62739 resident-prefill read race, #62979 async-load admission deadlock. (#59235 MTP residency race and #59309 MTP collapse under FULL are already in 0.31.0.)
- **Still open:** **HiSparse prefill is not graph-capturable** (FULL rejected; `FULL_DECODE_ONLY`/`FULL_AND_PIECEWISE` fine, we use the latter); #64559/#64562 will change `hisparse_gather_plan`/`hisparse_resolve_residency` signatures; the low-concurrency overhead above. (FP8-on-Ampere #60602 is irrelevant if we stay int8.)

## (b) Our fork versus what the PRs assume

| Area | 0.31.0 + HyperQwen | #59209 base (`b8141bd`) | Rebase difficulty |
|---|---|---|---|
| Upstream distance | release branch | 2,113 commits ahead, 27 behind | Full rebase is not realistic. **Port, don't rebase.** |
| `v1/hisparse/*.py` | runtime 1,335 lines, coordinator 859 | 1,369 / 922 (348 / 173 changed lines) | Medium. Cherry-pick the 9 fixes, then the #59209 hunks. |
| KV planning | `v1/core/kv_cache_utils.py` (3,212 lines) | split out into `v1/core/kv_cache_planning.py` (layout.py imports it) | **High.** Our `hybrid-kv-groups-v2-cudagraph` and `kv-cache-seat-budget` patch `kv_cache_utils.py`. Keep our file and re-point the imports in layout.py (lines 73 and 325). |
| `single_type_kv_cache_manager.py` | ours +107 lines (`mamba-align-checkpoint-order`) | 373 changed lines | Medium. Hand-merge the `HiSparseResidentManager` hunks. |
| `kv_cache_interface.py` | has `int8_per_token_head` / `KVQuantMode` | adds `SparseFullAttentionSpec` | Low. |
| QSA model files | `qwen4_exp/nvidia/qsa.py` (534 lines), no `common/qsa_indexer.py` | 638 lines + common indexer | Do not port. Copy only the kernel. |
| Attention backend | pthfa CUSTOM; cache `[NB,Hkv,BS,2·(256+4)]` int8 with inline fp32 scale, Triton writer | BF16/FP8, per-tensor scales, BLNHC | **High.** The binding asserts `shape[1:] == (H,BS,hs+hs_v)` and `row_width = Hkv·2·hd`. Both must use the padded int8 row (2,080 B/token/layer). pthfa checks only the inner stride; confirm `csrc/pth_attn/launch.cu` honours the token/head strides under BLNHC. |
| Spec decode | MTP k=7, `spec-decode-attn`/`spec-decode-int8-kv` patches | MTP3 validated | Medium. See the shared-memory limit in §c. |
| Block size | 864 (mamba align) | 64/128 in tests | Low–medium. The hot buffer is rounded to 864-row blocks. Spill granularity is 864 tokens. |

Recommended method: new files in `patches/` appended to `patches/series`, in three parts. (1) `hisparse-oct-fixes.patch`: the 9 upstream fixes. (2) `hisparse-sparse-full-spec.patch`: the generic #59209 hunks only (spec, layout, binding, runtime, single_type, coordinator, `resolve_kv_cache_layout`). (3) `sketch-sparse-attn.patch`: our code. Leave pthfa untouched; it stays the prefill, drafter and dense-fallback kernel.

## (c) Where a training-free sketch selector replaces the QSA indexer

**Hook point.** `model_executor/models/qwen3_next.py::Qwen3NextAttention.__init__` (line ~344, `self.attn = Attention(...)`; Qwen3_5 reuses it). When `hisparse_config.selector == "sketch"`, build a new `SketchSparseAttention` owner layer in `vllm/model_executor/layers/attention/sketch_sparse_attention.py`. It mirrors `Qwen4ExpQSAAttention`:

- `get_kv_cache_spec()` returns `SparseFullAttentionSpec(top_k=U_max, total_num_kv_heads=4, dtype=int8, kv_quant_mode=INT8_PER_TOKEN_HEAD)`.
- It owns `self.hisparse_cache = create_hisparse_cache_handle(vllm_config, U_max, is_index_group_leader=True, row_width=2080, kv_dtype=torch.int8)`.
- It owns a child module `SketchKeyCache` with layer name `{prefix}.sketch`. `get_kv_cache_spec` returns `MLAAttentionSpec(block_size, num_kv_heads=1, head_size=4·r, dtype=fp16 or int8, cache_role=SparseCacheRole.INDEXER)`, the same pattern as `QSACompressedKeyCache`.
- `layout.py::_partition_hisparse_specs` then puts the sketch into the INDEXER group with no further change. It is GPU-resident, never spilled, and prefix-cached alongside the main KV blocks.
- Cost at r=32: 256 B/token/layer in fp16, or 132 B in int8 + scale. That is 4 KB/token across 16 layers, or 600 MB at 150K. Use int8 sketches.

**Data structures** (match the QSA/HiSparse contracts, so `swap_in` and the kernel stay unchanged):

- `topk_indices_buffer: int32 [max_num_batched_tokens, U_max + 1]`. Logical token positions, padded with −1; the last column is `valid_count`. `logical[:, :-1]` goes to `swap_in(..., return_valid_counts=True)`.
- `physical_topk_indices_buffer`: the same shape, filled from `swap_in` (hot-buffer rows); the count column is copied across.
- `sel_row_of_token: int32 [max_tokens]`. Maps each verify row to its request's selection row (MTP sharing, §d).
- `basis: fp16 [max_num_reqs(state slot), 16 layers, 4, 256, r]` (1 MB/request). Plus `gram: fp32 [slot, 16, 4, 256, 256]`, used only during prefill.

**Selection per decode step, per layer** (Triton, new file `vllm/v1/attention/ops/sketch_select.py`):

1. Project the queries: `q̂[row, h] = q[row, h] · V_r[kv(h)]`, giving 8 rows × 24 heads × r. This is tiny.
2. `score[req, kvh, n] = max over (verify rows, the 6 q-heads in the group) of z[n, kvh] · q̂`. This is a group-pooled score matching the lab method. It reads the sketch blocks through the INDEXER group's block table.
3. Top-k per KV head with the existing CUDA `ops.top_k_per_row_decode` (`_custom_ops.py:3090`, used by DSA). It is graph-safe; don't use `torch.topk`.
4. Merge: union across the 4 heads + sinks (first 64 tokens) + a recent window W (≥ 1,024, and always covering this step's k+1 new tokens). Dedupe, write `valid_count`. Below `U_max` tokens, select everything, so short contexts stay exact.

**Hard sm_86 limit.** `runtime.py::_check_residency_shared_memory` needs `8·(max_union_rows+1) + 12·⌈H/32⌉ + 2·H ≤ 101,376 B`.

- Example fits: `U_max=6144` with hot buffer H=20,480 rows (13.6% of 150K) uses 97.8 KB; `U_max=4096` with H=24,576 uses 91 KB.
- The default `max_union_rows = max_query_len·top_k` (8·U) cannot fit. Patch `ResolvedHiSparseConfig.from_vllm_config` with a `shared_selection` flag so that `max_union_rows = top_k` and `device_buffer_size ≥ top_k`.

**When the SVD basis is fit (decision gate in Stage 0)**

- **Option G, preferred if the lab confirms it:** a *global* per-(layer, KV head) basis calibrated offline on post-RoPE keys and stored as a model buffer.
  - A sketch is then a pure function of the key. Prefix caching just works, and nothing is fit at runtime.
  - Note `partial_rotary_factor=0.25`: only 64 of 256 dims rotate, which favours a global basis.
- **Option D, the lab-proven method:** a per-document basis.
  - During prefill chunks, `do_kv_cache_update` also accumulates `gram += kᵀk` from dequantised post-RoPE keys.
  - On the step where `num_computed_tokens` reaches the prompt length, run `torch.linalg.eigh` over 64 matrices of 256×256 (about 5 ms) to get `V_r`. One kernel then projects the prompt's resident and host rows into the sketch cache.
  - Prefix-cache consistency: keep a worker-side `basis_id` per sketch block. A request inherits the basis of its hit blocks, because a chain is always written with its root writer's basis.
  - If the hit covers less than 50% of the new prompt (for example a shared system prompt), fit a fresh basis and write *private* sketch blocks. That needs a `cache_role` flag that turns off prefix caching for that request's sketch group.
  - This is about 3 extra days, and it is the main argument for Option G.

**Appended tokens.** The KV write path computes `z = k_postrope · V_r` and stores it at the same `slot_mapping` in the sketch cache. That works because both groups share block size 864. Implement it as a small Triton kernel called right after `do_kv_cache_update`. Re-fitting is never needed inside a request. On later turns, the inherited basis projects the new turn's tokens.

## (d) Kernel plan

**Primary: port the QSA kernel.** Copy `_qsa_sparse_paged_gqa_splitk_kernel`, `_qsa_merge_splitk_kernel` and `_select_config` into `vllm/v1/attention/ops/sketch_sparse_gqa.py`. Changes:

1. **int8 per-token-head dequant.** Load the K and V tiles as int8 and cast them to bf16/fp16 (native, no inline asm on sm_86). Load the per-(token, head) fp32 scales from the inline 4-byte pad (`k_ptr + HEAD_DIM` bytes into each head row). Apply them as `scores *= k_scale[None, :]`, and scale `probabilities * v_scale[None, :]` before the PV dot.
2. **Causal mask for shared selections.** Add `positions_ptr` and set `valid &= token <= row_position` (row_position = seq_len − q_len + i). QSA does not need this because its indexer is per row; we share one list across 8 verify rows.
3. **Row indirection.** Load indices from `sel_row_of_token[row]`, so 8 verify rows read one selection row.
4. **Drop the fused output gate** (`Qwen3NextAttention.forward` applies the gate itself), or keep it behind a constexpr later.
5. Keep `PHYSICAL_INDICES` and the device-side `valid_count` loop bound. The grid is fixed (rows, 4, splits), so it captures in `FULL_AND_PIECEWISE` decode graphs.

**Alternative: compaction + stock pthfa.**

- Gather the selected rows (hot or resident → a per-request staging region, reusing `gather_selected_cache`). Place the window and new tokens last, in order.
- Then call the unmodified `pth_flash_attn_fwd` with a synthetic `block_table` over the staging pages, `seq_lens = valid_count`, and causal on. Causality among the verify tokens stays correct because they are contiguous at the end.
- Zero CUDA changes, and it keeps the tuned FULL-graph verify kernel.
- Cost: an extra copy of about 200 MB per step at U=6K, roughly 0.5 ms.
- Teaching pthfa a token-row gather mode means editing vendored, SHA-pinned, SASS-proven CUDA (`pthfa/csrc`, frozen op schema). That is the last resort.

**Decision:** build the Triton port in Stage 1 and keep compaction as the fallback if the Triton kernel loses to pthfa by more than 0.3 ms per layer-step.

**MTP sharing.**

- One selection per request per step, computed from all k+1 = 8 verify queries (max-pooled). All 8 rows use it through `sel_row_of_token`, so `swap_in` resolves **one row per request** rather than 8 (pass the first-row token mapping as `req_id_per_token`).
- In the builder, copy #59209's `_hisparse_token_to_req_buffer` masking of FULL padding rows.
- The MTP drafter layer (one full-attention layer, about 312 MB at 150K) stays dense on pthfa in a normal GPU group. QSA-style drafter selection reuse is optional (Stage 5).

## (e) Staged build order

| # | Stage | Days | Test plan |
|---|---|---|---|
| 0 | **Lab gate on Qwen3.8-27B** (on the lab rig, not production): global vs per-document basis, `U_max` union cap (4K/6K) vs per-head 2%, 8-row max-pooled shared selection, int8 sketches, sink/window sizes, at 32K/64K/128K | 2 | Later-turn recall and needle tests equal to dense. Selection recall ≥ 0.95 vs exact top-k. Measure union size distribution and the working set per answer. |
| 1 | **Sparse decode over the resident cache, no host tier:** `SketchKeyCache` (plain GPU group), selector, Triton int8 kernel with `block_table`, MTP shared selection, `FULL_AND_PIECEWISE` | 5 | Kernel vs fp64 reference on random int8 caches, including block id > 2,383 (#86). Graph replay equals eager. Greedy-output diff vs dense on 50 long prompts. MTP acceptance within 2% of dense. Decode ms/step at 32/64/150K. |
| 2 | **Port HiSparse to 0.31.0:** `hisparse-oct-fixes.patch` + generic #59209 hunks; re-point `kv_cache_planning` → `kv_cache_utils`; int8 padded row geometry in `binding.py` (`expected_shape`, `row_width`) and `layout.py` (`logical_block_bytes`); BLNHC with pthfa and the Triton writer | 4 | `patches/apply.sh` + `check_vllm_series.sh` clean. Ported upstream tests: `tests/v1/core/test_prefix_caching.py`, `test_kv_cache_utils.py`, `tests/v1/worker/test_attn_utils.py`, `test_hisparse_connector.py`. pthfa `bench/test_pthfa_attn.py` under BLNHC. Production config boots on a **spare** box. |
| 3 | **HiSparse integration:** `SparseFullAttentionSpec` in `SketchSparseAttention`, `_forward_hisparse`-style write_target/swap_in/`PHYSICAL_INDICES`; shared-selection patch to `ResolvedHiSparseConfig`; nonresident prefill through the staging plan + pthfa on the staged view | 5 | Forced spill (small GPU pool, `host_pool_gib=16`): outputs match Stage 1. Hot hit/miss and H2D counters (#58949 metrics). Preemption/recompute, cancel, and partial prefix hits (#59209 tail tests). Shared-memory check passes at k=7. |
| 4 | **Production feature matrix:** async scheduling, prefix caching + mamba align, chunked prefill 2048, two requests at 150K, `eager_host_mirror=False` when not spilling; basis Option D registry if Stage 0 picked it (+3) | 3 (+3) | 24 h soak on spare hardware. Multi-turn chat with prefix hits. Watch the #62979/#62217 admission deadlock and #59494 livelock patterns. |
| 5 | **Tuning and A/B:** U, H, W, splits; drafter selection reuse; GPU1 (x4) policy | 3 | Paired off/on at 1 and 2 users, 32K–150K, plus maximum context. Ship only if on ≥ off at 1 user. |

**Total:** about 22 days, or about 25 with Option D. Stage 1 delivers the decode speedup by itself. Stages 2–4 deliver the capacity.

## Risks

1. **Low-concurrency overhead.** #59209 is 17–20% slower at moderate concurrency even with zero misses. With 1–2 users we may never recover it. Mitigation: Stage 1 has no HiSparse at all. Enable the host tier only when context exceeds the GPU pool.
2. **sm_86 shared-memory cap.** It forces `U_max` around 4–6K and H around 20–24K rows. That is below the lab's 2%-per-head at 150K once unioned across heads, and H falls short of 15% beyond about 150K context. The int16 LRU caps H at 32K anyway. Stage 0 must confirm recall under the cap.
3. **PCIe x4 on GPU1 (6.2 GB/s).** 1,000 missed rows per layer-step cost 33 MB, about 5.4 ms, versus 1.45 ms on GPU0 (x16). Selection reuse is only 51% step to step, so this relies on the hot buffer absorbing the working set (about 11% of the document). Keep GPU1 on Stage 1 only unless measured otherwise.
4. **Prefill over spilled pages.** It stages the whole context per chunk. A new turn on a fully spilled 150K context moves about 5 GB per chunk group. TTFT risk. HiSparse prefill also stays out of CUDA graphs (#60072 open item).
5. **Upstream churn.** #59209 is unmerged and unreviewed. #64559/#64562 will change op signatures. Our ported patches become a maintenance burden at the next pin (`PATCHES.md` row per patch).
6. **Basis validity.** A global post-RoPE basis is unproven; Option D's prefix-cache consistency is subtle.
7. **pthfa under BLNHC.** If `launch.cu` assumes the HND stride, prefill and drafter break. The compaction fallback inherits the same requirement.
8. **Production safety.** All work happens on a separate venv or box. The production host serves two servers and must not be touched.
