#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 sparsekv-lab contributors
"""KILL TEST: does TokenGate keep what an agent needs LATER?

Arm E (essays, hybrid Qwen3.5-2B needle gate = our model family):
  A  question BEFORE the document (the trained format: the gate can see the question)
  B  question AFTER the document (agent case: the gate prunes before the question exists;
     with causal attention this is identical to asking it turns later)
  metrics per keep-budget K: answer accuracy, needle-token survival, and survival of
  ordinary (non-needle) sentences — what a later, unforeseen question would need.
Arm C (code, Qwen3-1.7B RepoBench gate): 32K tokens of real vLLM source as context; for
  random lines deep in it, the in-file prefix (after the context) ends at that line and the
  model must produce the next line = retrieving arbitrary earlier code. Edit similarity.
Writes progress lines "PROGRESS <pct> <note>" and a final JSON line "RESULT {...}".
"""
import glob, json, os, random, re, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import LAB_DIR, TG_REPO, NIAH, MODEL_27B, Q27_HF, code_corpus  # noqa: E402
sys.path.insert(0, TG_REPO)
import torch
from transformers import AutoTokenizer
from tokengate import apply_chat_template, load_model, split_state_span
from tokengate.datasets.tasks import get_task
from tokengate.datasets.tasks.longbench import _normalize, _edit_sim

R = {"essays": {}, "code": {}}
def prog(p, note): print(f"PROGRESS {p} {note}", flush=True)

def survival(kept_set, st, spans):
    """spans: list of (tok_start, tok_end) within the state -> (mean frac kept, frac of spans >=90% kept)"""
    fr = []
    for a, b in spans:
        n = b - a
        if n <= 0: continue
        fr.append(sum((st + i) in kept_set for i in range(a, b)) / n)
    return (sum(fr) / len(fr) if fr else 0.0, sum(f >= 0.9 for f in fr) / len(fr) if fr else 0.0)

def char_spans_to_tok(offsets, spans):
    out = []
    for cs, ce in spans:
        ids = [i for i, (a, b) in enumerate(offsets) if a < ce and b > cs]
        if ids: out.append((ids[0], ids[-1] + 1))
    return out

# ---------------- Arm E: essays, hybrid gate ----------------
prog(2, "loading hybrid Qwen3.5-2B needle gate")
m = load_model("liana313/tokengate-niah-qwen3.5-2b", gate_attn=os.environ.get("GATE_ATTN", "fa2"))
tok = AutoTokenizer.from_pretrained(m.base_model)
task = get_task("niah_kamradt")
S = [json.loads(l) for l in open(NIAH)]
intro = "Answer the question using only the document below. Keep your answer short and direct."
rng = random.Random(7)
total_e = len(S) * 2 * 4
done = 0
for s in S:
    body = s["body"]
    enc = tok(body, return_offsets_mapping=True, add_special_tokens=False)
    offs = enc["offset_mapping"]
    n_state = len(enc["input_ids"])
    # needle span (by answer phrase) and 40 ordinary sentences
    ni = body.find(s["answer"][:30])
    needle_cs = body.rfind(".", 0, ni) + 1 if ni >= 0 else None
    needle_ce = body.find(".", ni) + 1 if ni >= 0 else None
    sents = [(mm.start(), mm.end()) for mm in re.finditer(r"[^.!?]{60,240}[.!?]", body)]
    sents = [x for x in sents if needle_cs is None or x[1] < needle_cs or x[0] > needle_ce]
    sents = rng.sample(sents, min(40, len(sents)))
    tok_needle = char_spans_to_tok(offs, [(needle_cs, needle_ce)]) if ni >= 0 else []
    tok_sents = char_spans_to_tok(offs, sents)
    prompts = {
        "A_question_first": task.format_prompt("{state}\n\nQuestion: " + s["question"]),
        "B_question_after": f"{intro}\n\nDocument:\n{{state}}\n\nQuestion: {s['question']}\n\nAnswer:",
    }
    for mode, pr in prompts.items():
        ids = apply_chat_template(tok, pr, state=body)
        cids, st, en = split_state_span(list(ids))
        assert en - st == n_state or True
        for k in (None, n_state // 4, 2000, 500):
            out = m.generate([ids], tok, max_kept_prefill_tokens=k, max_new_tokens=24)
            kept = set(out.kept_pos[0])
            acc = task.evaluate(out.answers[0], s["answer"])
            nd, _ = survival(kept, st, tok_needle)
            om, of = survival(kept, st, tok_sents)
            key = f"{mode}|K={'all' if k is None else ('25%' if k == n_state // 4 else k)}"
            d = R["essays"].setdefault(key, {"acc": [], "needle_kept": [], "ordinary_kept": [], "ordinary_full": [], "ttft": []})
            d["acc"].append(acc); d["needle_kept"].append(nd); d["ordinary_kept"].append(om); d["ordinary_full"].append(of); d["ttft"].append(out.prefill_s)
            done += 1
            prog(5 + int(55 * done / total_e), f"essays {mode} {key.split('|')[1]}: acc {acc:.0f} needle {nd:.0%} ordinary {om:.0%}")
for d in R["essays"].values():
    for kk in list(d):
        d[kk] = round(sum(d[kk]) / len(d[kk]), 3)
del m; torch.cuda.empty_cache()

# ---------------- Arm C: code, RepoBench gate ----------------
prog(62, "loading Qwen3-1.7B RepoBench (code) gate")
m = load_model("liana313/tokengate-repobench-qwen3-1.7b", gate_attn=os.environ.get("GATE_ATTN", "fa2"))
tok = AutoTokenizer.from_pretrained(m.base_model)
ctask = get_task("longbench_repobench_p")
files = sorted(glob.glob(f"{code_corpus()}/**/*.py", recursive=True))
random.Random(3).shuffle(files)
ctx, n = [], 0
for f in files:
    t = open(f, errors="ignore").read()
    if 2000 < len(t) < 30000:
        ctx.append(f"# {f.split('site-packages/')[-1]}\n{t}\n"); n += len(tok(t, add_special_tokens=False)["input_ids"])
    if n > 30000: break
code = "".join(ctx)
lines = code.split("\n")
cands = [i for i in range(10, int(len(lines) * 0.8)) if len(lines[i].strip()) > 25 and len(lines[i + 1].strip()) > 25
         and not lines[i + 1].strip().startswith(("#", '"""')) and code.count(lines[i + 1]) == 1]
picks = random.Random(5).sample(cands, 24)
n_state = len(tok(code, add_special_tokens=False)["input_ids"])
total_c = len(picks) * 3; done = 0
for i in picks:
    prefix = "\n".join(lines[i - 5:i + 1])
    pr = ctask.format_prompt("{state}\n\nQuestion: " + prefix)
    ids = apply_chat_template(tok, pr, state=code)
    for k in (None, n_state // 4, 2000):
        out = m.generate([ids], tok, max_kept_prefill_tokens=k, max_new_tokens=48)
        es = _edit_sim(out.answers[0].strip().split("\n")[0].strip(), lines[i + 1].strip())
        key = f"K={'all' if k is None else ('25%' if k == n_state // 4 else k)}"
        R["code"].setdefault(key, []).append(es)
        done += 1
        prog(62 + int(36 * done / total_c), f"code {key}: next-line edit-sim {es:.2f}")
R["code"] = {k: round(sum(v) / len(v), 3) for k, v in R["code"].items()}
R["code_state_tokens"] = n_state
print("RESULT " + json.dumps(R), flush=True)
