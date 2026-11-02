# Pruning without forgetting: query-aware sparse KV for the 27B on 2×3090

*Report from round 3. Written 2026-10-31 by Claude.*

## Summary

TokenGate breaks multi-turn agents. It decides once, before the question exists, which tokens to throw away forever, and later turns lose the content (code recall 0.48 → 0.28).

The fix is to never throw anything away. Every token's KV stays in host RAM. The GPU holds three things:

- a compact **key sketch** for every token: rank-64 projection of the keys, int8, 4 KB/token;
- a **hot buffer** holding about 20% of each context in full;
- the recent and sink tokens.

At every decode step, each attention layer scores all tokens through the sketch, picks the **top 2%**, and runs exact attention on just those.

**Results, Qwen3.5 hybrids (same architecture family as our 27B):**

| Later-turn recall (question asked after the context) | Full context | **Sketch @2%** | TokenGate-style choose-once @25% |
|---|---|---|---|
| 2B: needle / sentence recall / code next-line | 1.00 / 0.67 / 0.48 | **1.00 / 0.72 / 0.47** | 0.53 / 0.26 / 0.22 |
| 9B: needle / sentence recall / code next-line | 1.00 / 0.87 / 0.93 | **1.00 / 0.89 / 0.93** | 0.65 / 0.28 / 0.45 |
| **27B (production W4A16)**: needle / sentence recall / code next-line | 1.00 / 0.99 / 1.00 | **1.00 / 0.97 / 1.00** | 0.47 / 0.16 / 0.27 |
| 9B SCBench, 5-turn shared context: KV lookup / RepoQA / prefix-suffix | 1.00 / 0.61 / 0.33 | **1.00 / 0.65 / 0.44** | 0.03 / 0.20 / 0.00 |

**Projected for the 27B on each 3090** (a model built from the measurements below, not yet built and benchmarked). Design point: shared-head rank-64 int8 sketch, 1% budget, 10% hot buffer.

| | Today (dense server, int8 KV) | With sparse KV |
|---|---|---|
| GPU memory per context token | 32 KB | **≈7.2 KB** (4 KB sketch + 10% hot buffer) → **4.4×** |
| Agents at 128K per card | ~1.9 | **~7** (cache budget 8.1 GB; ~0.15 GB recurrent state per agent) |
| Agents at 256K per card | ~1.0 | **~4** |
| Whole box at 256K (2 cards) | ~2 agents | **~6** with int8 KV in host RAM (~48 GB pinned of 78 GB), **~8** if the host tier stores int4 |
| Fetch per decode step, 256K agent | 0 | ~8 MB → 0.8 ms (x16) / 1.2 ms (x4), overlapped with linear layers |
| Decode step, 2 agents at 128K / 256K | 1.0× | **~1.45× / ~1.8× faster** (7× fewer attention bytes) |
| Information lost | none | **none** (every token is retrievable) |

## What we proved, in order

1. **TokenGate fails for agents, because the decision is final.** Its gate keeps the same tokens whether the question comes before or after the document: it doesn't read the question. At 25% keep, ordinary sentences keep 25% of their tokens and 0% survive intact.
2. **Sparsity was never the problem; deciding early was.** An oracle that picks the top 2% *per query* (exact attention scores) matches full context on all tasks. Picking once at the context boundary (SnapKV-style) or keeping only a recent window collapses to 0.10–0.37. Research agents found the same pattern in the literature (SCBench, ICLR'25).
3. **A cheap selector matches the oracle.**
   - A low-rank sketch of the keys picks as well as the oracle.
   - Rank 64 is needed at 2B; at 9B even rank 32 is lossless.
   - Quest-style page min/max bounds fail (sentence recall 0.26–0.43).
   - *Correction:* round 2 reported "rank 32 = oracle". That was a caching bug: the rank-64 sketch was being reused.
4. **One fixed basis works for every document.** A sketch basis fitted on a *different* document, even an essay used for code, is as good as one fitted on the document itself. So the basis can be fitted offline per layer and frozen. New tokens are projected when their KV is written, and prefix caching keeps working unchanged.
5. **The int8 sketch is free.** Results on all SCBench tasks are identical to the fp16 sketch, which halves the sketch to 4 KB/token.
6. **Shared selection across KV heads works** (0.97–1.00). One pick per layer for all 4 KV heads cuts the per-step selection the 3090 kernel must hold by 4×. That matters because the 3090's ~101 KB of shared memory caps the selection at about 4–6K tokens per step.
7. **Cross-layer reuse does not work.** Reusing one layer's pick for the next 1–2 attention layers drops sentence recall to 0.29–0.31. Each attention layer must pick for itself.
8. **Long answers need a bigger buffer than short ones.**
   - A 512-token answer touches about 40% of the document over its lifetime, against 11% for a 40-token answer.
   - Step-to-step overlap of the pick is only about 61%.
   - An LRU hot buffer of 20% of the context, tracked per token, misses about 6% of picks. That is about 10 MB per step for a 256K agent.
   - 16-token pages need 30% or more buffer, so fetch per token.
9. **Fine-grained fetch costs nothing extra.** The GPU reading pinned host RAM directly (zero-copy) runs at about 10 GB/s on the x16 card and 6.2 GB/s on the x4 card (link-saturated), *even with 512-byte rows*. So 10 MB/step costs about 1.0 ms (x16) or 1.6 ms (x4), and the fetch overlaps with the 3 linear-attention layers between attention layers.
   - An earlier CPU-side gather looked slow; the CPU gather was the bottleneck, not PCIe.
10. **Decode attention gets much cheaper.**
    - At 27B shapes (24 q heads, 4 KV heads, d256, 8 MTP verify queries), sketch scan + top-2% attention is **5–9× faster than full fp16 attention** at 128K–256K in a PyTorch microbenchmark.
    - It reads 7× fewer bytes against our int8 KV.
    - Our production kernel (pth-flash-attn int8) is faster than PyTorch's, so the real gain is projected from bytes moved: weights (~14 GB/step) plus attention bytes. That gives about 1.45× (128K) to 1.8× (256K) faster decode steps at batch 2.

## Squeeze test on the 27B: how lean can it go?

| 27B (24K context) | Full | per-head r64 @1% | shared-head r64 @2% | **shared-head r64 @1%** | per-head r32 @2% |
|---|---|---|---|---|---|
| Needle | 1.00 | 1.00 | 1.00 | **1.00** | 1.00 |
| Sentence recall | 0.99 | 0.97 | 0.97 | **1.00** | 0.97 |
| Code next-line | 1.00 | 1.00 | 1.00 | **1.00** | 0.98 |

**Design point:** rank-64 int8 sketch, **one pick per layer shared by all 4 KV heads**, and a **1% budget**.
- At 256K this is about 2.6K tokens per step. That fits under the 3090 kernel's ~4–6K selection cap with room to spare.
- It halves the exact-attention and fetch volume compared with 2%.

## Design to build into HyperQwen

- **Tiers:** full int8 KV lives in pinned host RAM, viewed by the GPU via UVA. The GPU keeps:
  - the int8 rank-64 key sketch for all tokens;
  - a token-granular LRU hot buffer of about 10% of each context;
  - the sink and recent windows;
  - the GatedDeltaNet state, unchanged.
- **Selection:** per attention layer and decode step:
  1. Project the 8 MTP verify queries through the frozen per-layer basis.
  2. Score them against the sketch.
  3. Pool across heads and queries, pick the top 1% **shared by all KV heads**, with a floor of about 1K and a cap of about 4K tokens.
  4. Missing rows come in over zero-copy, then exact attention runs on the picked set plus recent tokens.
- **Prefill:** dense, as today. Sketch rows are written alongside KV rows.
- **Upstream base:** vLLM HiSparse is already merged in our fork's 0.31 base, and open PR #59209 adds GQA, head_dim 256, the GDN hybrid cache manager and MTP selection reuse. Our new piece is the training-free sketch selector in place of the trained QSA indexer.
  - Plan: `reports/hisparse-port-plan.md` — about 22 working days in 6 stages.
- **Quickest standalone win (stage 2 of the plan):** sparse decode over the normal GPU cache, no host tier. That gives the decode-speed gain alone at long context (estimated −20% step time at 150K) with no PCIe risk.

## Risks and open questions

- **Validated on the real 27B (24K context).** A custom packed-W4A16 loader (`lab/q27loader.py`, 13.5 GB) runs the production weights in the lab harness. Rank-64 foreign-basis sketch at 2% matched full context (above table). Longer contexts (128K+) on the 27B remain to be measured; the trend from 2B to 9B to 27B only got better.
- **Small samples on two SCBench subsets** (9 and 12 questions). KV lookup (30) is solid. Contexts were 32K, not 128K+.
- **#59209 reports 17–20% slower decode at low concurrency** even with zero misses, from upstream overheads. It needs profiling in our single- and dual-user regime.
- **Hot-buffer miss rate on real agent traces** (tool-heavy, many turns) is unmeasured. The 10% buffer sizing comes from 9 cases of 512-token answers over 32K documents on the 9B: mean 9.6% misses, worst 16% (verbatim copying).
- **Fallback if the build stalls:** plain int4 KV gives 2× capacity with no PCIe traffic.

## Artefacts

- **Lab scripts:** `lab/{killtest,oracle,selector,variants,longans,scale,scb,overlap,kbench,uva,bw}.py`, plus logs `lab/*.log`.
- **Dashboard:** `dashboard/` (sample state from this round in `dashboard/www/state.json`).
