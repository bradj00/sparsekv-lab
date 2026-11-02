# sparsekv-lab

This repo holds the research that led to `sparsekv-hyperqwen`. We first tested learned prefill pruning (TokenGate) for multi-turn coding agents and rejected it. What replaced it is query-aware top-k over a frozen low-rank key sketch, with the full KV kept in host RAM. On the production Qwen3.8-27B it matched full context at a 1% budget (128K: needle / sentence / code 1.00 / 1.00 / 0.97 = full).

## What we found (the steps that made the serving build possible)

- **Pruning before the question kills later turns.** TokenGate at 25% keep: 0% of ordinary sentences survive intact, and code next-line recall falls 0.48 → 0.28. The gate keeps the same tokens whether the question comes before or after the document (`killtest.py`).
- **Pick per query, not once.** An oracle top-2% per decode query matches full context. Choose-once (SnapKV-style) and a sliding window both collapse to 0.10-0.37 (`oracle.py`).
- **A rank-64 key sketch picks as well as the oracle.** Quest page bounds do not (essay 0.26). The round-2 "rank 32 is enough" claim was a caching bug; rank 32 only holds at 9B (`selector.py`, `scale.py`).
- **One frozen basis works for every document.** A basis fitted on a different document, even an essay used for code, scores the same, so prefix caching survives. An int8 sketch equals fp16, and one pick shared across the 4 KV heads works. Cross-layer pick reuse fails (`variants.py`, `scb.py`: SCBench KV lookup 1.00 = full, RepoQA 0.65 vs 0.61).
- **Long answers drift.** A 512-token answer touches ~40% of the document, and step-to-step overlap is ~61%. So size the hot buffer by context (not by top-k), with token-granular fetch (`longans*.py`, `overlap.py`).
- **The PCIe link is fast enough for row-sized fetches.** Zero-copy UVA reads from pinned RAM reach ~10 GB/s on x16 and 6.2 GB/s on x4, even with 512 B rows. A CPU-side gather was the bottleneck, not the link (`uva.py`, `bw.py`, `kbench.py`).
- **27B in HF on a 24 GB card.** A packed W4A16 loader fits in 13.5 GB without decompressing (stock decompression needs 54 GB) (`q27loader.py`, `overlay27.py`).
- **27B validated, rank-64 shared-head sketch at 1%** (`lc27.py` → `lc27.json`): 32K 1.00/1.00/0.90 vs full 1.00/1.00/1.00; 64K 1.00/1.00/0.82 vs 1.00/1.00/0.90; 128K 1.00/1.00/0.97 = full. At 0.5%, sentence recall drops to 0.87, so stay at ≥1%.
- **HiSparse port path rejected for serving.** `reports/hisparse-port-plan.md` was estimated at ~22 days. We built a pthfa decode hook plus our own host tier instead (see `sparsekv-hyperqwen`).

## Results

| test (question after the context) | full context | sketch | TokenGate-style choose-once |
|---|---|---|---|
| 2B @2%: needle / sentence / code | 1.00 / 0.67 / 0.48 | 1.00 / 0.72 / 0.47 | 0.53 / 0.26 / 0.22 |
| 9B @2%: needle / sentence / code | 1.00 / 0.87 / 0.93 | 1.00 / 0.89 / 0.93 | 0.65 / 0.28 / 0.45 |
| 27B W4A16 @2%, 24K | 1.00 / 0.99 / 1.00 | 1.00 / 0.97 / 1.00 | 0.47 / 0.16 / 0.27 |
| 27B shared-head r64 @1%, 128K | 1.00 / 1.00 / 0.97 | 1.00 / 1.00 / 0.97 | n/a |
| 9B SCBench 32K: KV lookup / RepoQA / prefix-suffix | 1.00 / 0.61 / 0.33 | 1.00 / 0.65 / 0.44 | 0.03 / 0.20 / 0.00 |

The "choose-once" columns are untrained emulations of TokenGate-style selection, not the released checkpoints. The report's capacity projections were later beaten by the measured build: 8 × 250K on two cards (see `sparsekv-hyperqwen`).

## The rig

Same box as `sparsekv-hyperqwen`: a VM with 78 GB RAM on an i9-12900K / 96 GB DDR5 host, with an RTX 3090 Ti (PCIe Gen4 x16) and an RTX 3090 (Gen4 at x4), driver 595.99.02 / CUDA 13.2. Both cards are capped at 250 W with fans at 100%. The lab ran on the 3090 (GPU1) in its own venv: torch 2.14 cu130, transformers 5.22, TokenGate repo `token-gate-llms/TokenGate` @ a07581f.

## Layout

- `lab/`: experiment scripts with their logs (`*.log`, `lc27.json`). `install.sh`, `build-kernel.sh` and `smoke.sh` set up TokenGate. The FA2 bias kernel needed C++20, `FA2BIAS_ARCHS=8.6` and the head-dim-256 build; transformers 5.22 renamed `layer_type` to `block_type`.
- `reports/`: `sparse-kv-breakthrough-report.md` (round 3 findings) and `hisparse-port-plan.md`.
- `dashboard/`: the follow-along lab dashboard. `bin/tglab` is the CLI that writes `www/state.json`; `gpu-collector.sh`, `eta-recorder.py`, `thermwatch-lab.sh` and `bridge-lc27.sh` feed it, and `notify.py` prints (optionally posts) a status line. `www/index.html` and `report.html` are static pages; the shipped `state.json` / `eta.json` / `gpu.json` are a sample from round 3. `plan-r*.json` are round plans for `tglab init`.

## Configuration

Paths come from environment variables (see `lab/paths.py`): `LAB_DIR` (default `lab/`, holds `repo/` = TokenGate checkout, `venv/`, `hf/`), `TG_REPO`, `MODEL_DIR` / `MODEL_27B` (the Qwen3.8-27B W4A16 checkpoint), `Q27_HF`, and `CODE_CORPUS` (code context; default is the installed `vllm` package). The dashboard feeders use `WWW_DIR`, `SSH_TARGET` (user@gpu-host; local if unset for `gpu-collector.sh`), `TGLAB_SSH_KEY`, `REMOTE_LAB_DIR`, and optionally `NOTIFY_WEBHOOK_URL` / `DASHBOARD_URL`.

## License & credits

Our code is Apache-2.0 (`LICENSE`); third-party attributions are in `NOTICE`.

- [TokenGate](https://github.com/token-gate-llms/TokenGate) (Apache-2.0) is used as an installed
  dependency, not included.
- Models: Qwen3.5-2B / 9B and [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (Apache-2.0).
- Benchmarks: [SCBench](https://huggingface.co/datasets/microsoft/SCBench) (MIT), loaded at run time.
  Needle-in-a-haystack method after Greg Kamradt's
  [NIAH](https://github.com/gkamradt/LLMTest_NeedleInAHaystack) (MIT). The essay haystacks are not
  included (the essays are copyrighted by their author, and quoted essay text in recorded results is
  redacted); `lab/paths.py` reads TokenGate's `examples/data/niah_32k_samples.jsonl`.
- Uses PyTorch (BSD-3-Clause), transformers and compressed-tensors (Apache-2.0), vLLM and
  pth-flash-attn (Apache-2.0).
- Shared for non-commercial, open use in the spirit of "power to the people";
  Apache-2.0 allows any use.
