#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 sparsekv-lab contributors
"""PRACTICAL SELECTORS vs ORACLE (same harness as oracle.py):
  quest    16-token pages, per-page elementwise key min/max; page score = sum_d max(q_d*min_d, q_d*max_d); top pages
  lr64/lr32  rank-64/32 SVD sketch of each KV head's document keys; approximate scores pick the top-k, exact attention on them
ORACLE BOUND: does query-aware selection keep later-turn recall where query-agnostic pruning fails?

Plain Qwen3.5-2B (hybrid, same family as our 27B; no TokenGate). The document is prefilled with FULL
attention (every token's KV exists). The question arrives AFTER the document (agent case). For the
question prefill and every decode step, the full-attention layers may only attend to a budget of
document tokens, chosen by one of:
  full    all document tokens (reference)
  oracle  per query position + KV head, the exact top-k document keys by attention (query-aware)
  snap    top-k chosen ONCE at the document boundary from the last 32 document queries (query-agnostic,
          SnapKV-style; this is what irreversible eviction amounts to)
  window  4 sink tokens + the most recent document tokens (StreamingLLM-style)
New tokens (question + generated) are always visible. Linear-attention (GatedDeltaNet) layers run unchanged.
Tasks (questions about ORDINARY content, unforeseeable at prefill time):
  essay-complete  5 ~32K haystacks x 5 random ordinary sentences: "complete this sentence from the document"
  needle          the planted needle question per haystack
  code-next       30K tokens of real vLLM source, 16 random lines: "output the line that follows"
Budgets: 25% / 6% / 2% of the document. Prints PROGRESS lines and a final RESULT json.
"""
import copy, glob, json, math, os, random, re, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import LAB_DIR, TG_REPO, NIAH, MODEL_27B, Q27_HF, code_corpus  # noqa: E402
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AttentionInterface

MODEL = os.environ.get("MODEL", "Qwen/Qwen3.5-2B")
ST = {"mode": "full", "budget": None, "doc_len": 0, "record": False, "snap": {}, "sink": 4, "pages": {}, "lr": {}}

def topk_attn(module, q, k, v, attention_mask, scaling=None, dropout=0.0, **kw):
    B, Hq, Lq, D = q.shape; Hkv, Lk = k.shape[1], k.shape[2]; g = Hq // Hkv
    scaling = scaling or 1 / math.sqrt(D)
    li = getattr(module, "layer_idx", 0)
    if Lq > 512:                                   # document prefill: full causal attention
        if ST["record"]:                           # query-agnostic scores from the last 32 doc queries
            obs = q[:, :, -32:].float()
            s = torch.einsum("bhqd,bhkd->bhqk", obs, k.float().repeat_interleave(g, 1)) * scaling
            pos_q = torch.arange(Lk - 32, Lk, device=q.device)[:, None]
            s = s.masked_fill(torch.arange(Lk, device=q.device)[None] > pos_q, float("-inf")).softmax(-1)
            ST["snap"][li] = s.sum(2).view(B, Hkv, g, Lk).sum(2)[0]          # [Hkv, Lk]
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scaling, enable_gqa=True)
        return out.transpose(1, 2).contiguous(), None
    # question prefill / decode against the cache: restrict document keys to the budget
    doc = ST["doc_len"]; kk = k.repeat_interleave(g, 1); vv = v.repeat_interleave(g, 1)
    s = torch.einsum("bhqd,bhkd->bhqk", q.float(), kk.float()) * scaling      # [B,Hq,Lq,Lk]
    qpos = torch.arange(Lk - Lq, Lk, device=q.device)[:, None]
    allow = torch.arange(Lk, device=q.device)[None] <= qpos                     # causal [Lq,Lk]
    allow = allow[None, None].expand(B, Hkv, Lq, Lk).clone()
    bud, mode = ST["budget"], ST["mode"]
    if mode != "full" and bud is not None and bud < doc:
        keep = torch.zeros(B, Hkv, Lq, doc, dtype=torch.bool, device=q.device)
        sink = ST["sink"]
        if mode == "window":
            keep[..., :sink] = True; keep[..., doc - (bud - sink):] = True
        elif mode == "snap":
            sc = ST["snap"][li][:, :doc].clone(); sc[:, :sink] = float("inf")
            idx = sc.topk(bud, -1).indices                                      # [Hkv, bud]
            keep[0].scatter_(-1, idx[:, None, :].expand(Hkv, Lq, bud), True)
        elif mode == "quest":
            P = 16; npg = doc // P
            if li not in ST["pages"]:
                kd = k[0, :, :npg * P].float().view(Hkv, npg, P, D)
                ST["pages"][li] = (kd.amin(2), kd.amax(2))                          # [Hkv,npg,D] each
            mn, mx = ST["pages"][li]
            qg = q[0].float().view(Hkv, g, Lq, D)
            ub = torch.maximum(torch.einsum("hgqd,hpd->hgqpd", qg, mn), torch.einsum("hgqd,hpd->hgqpd", qg, mx)).sum(-1).amax(1)  # [Hkv,Lq,npg]
            ub[..., 0] = float("inf")
            npick = max(1, min(npg, bud // P))
            pid = ub.topk(npick, -1).indices                                         # [Hkv,Lq,npick]
            tok_idx = (pid[..., None] * P + torch.arange(P, device=q.device)).flatten(-2)
            keep[0].scatter_(-1, tok_idx, True)
            if doc > npg * P: keep[..., npg * P:doc] = True                           # ragged tail page
        elif mode.startswith("lr"):
            r = int(mode[2:])
            if li not in ST["lr"]:
                kd = k[0, :, :doc].float()                                            # [Hkv,doc,D]
                U, Sv, Vh = torch.linalg.svd(kd, full_matrices=False)
                ST["lr"][li] = (U[:, :, :r] * Sv[:, None, :r], Vh[:, :r])             # sketch [Hkv,doc,r], basis [Hkv,r,D]
            Kr, Vb = ST["lr"][li]
            qr = torch.einsum("hgqd,hrd->hgqr", q[0].float().view(Hkv, g, Lq, D), Vb)
            ap = torch.einsum("hgqr,hkr->hgqk", qr, Kr).mul(scaling).softmax(-1).sum(1)   # [Hkv,Lq,doc]
            ap[..., :sink] = float("inf")
            keep[0].scatter_(-1, ap.topk(bud, -1).indices, True)
        elif mode == "oracle":
            p = s[..., :doc].view(B, Hkv, g, Lq, doc).softmax(-1).sum(2)       # group-pooled attention mass
            p[..., :sink] = float("inf")
            idx = p.topk(bud, -1).indices
            keep.scatter_(-1, idx, True)
        allow[..., :doc] &= keep
    s = s.view(B, Hkv, g, Lq, Lk).masked_fill(~allow[:, :, None], float("-inf")).view(B, Hq, Lq, Lk)
    out = torch.einsum("bhqk,bhkd->bhqd", s.softmax(-1).to(vv.dtype), vv)
    return out.transpose(1, 2).contiguous(), None

AttentionInterface.register("topk", topk_attn)

def prog(p, note): print(f"PROGRESS {p} {note}", flush=True)
prog(2, f"loading {MODEL}")
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="topk").cuda().eval()
EOS = {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")}

def norm(t): return re.sub(r"[^a-z0-9 ]", " ", t.lower()).split()
def word_recall(pred, gold):
    g = norm(gold); p = set(norm(pred))
    return sum(w in p for w in g) / max(1, len(g))
def edit_sim(a, b):
    a, b = a.strip(), b.strip()
    if not a and not b: return 1.0
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, dp[0] = dp[0], i
        for j, cb in enumerate(b, 1):
            prev, dp[j] = dp[j], min(dp[j] + 1, dp[j - 1] + 1, prev + (ca != cb))
    return 1 - dp[-1] / max(len(a), len(b))

@torch.no_grad()
def prefill_doc(text):
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
    ST.update(mode="full", budget=None, record=True, snap={}, pages={}, lr={})
    out = model(ids, use_cache=True)
    ST["record"] = False; ST["doc_len"] = ids.shape[1]
    return out.past_key_values, ids.shape[1]

@torch.no_grad()
def ask(cache, n_doc, qtext, mode, budget, max_new=40):
    c = copy.deepcopy(cache)
    ST.update(mode=mode, budget=budget, doc_len=n_doc)
    ids = tok(qtext, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
    out = model(ids, past_key_values=c, use_cache=True)
    gen = []
    for _ in range(max_new):
        nt = out.logits[0, -1].argmax().item()
        if nt in EOS: break
        gen.append(nt)
        out = model(torch.tensor([[nt]], device="cuda"), past_key_values=out.past_key_values, use_cache=True)
    del c
    return tok.decode(gen, skip_special_tokens=True)

HEAD = "<|im_start|>user\nAnswer using only the document below. Be exact.\n\nDocument:\n"
TAIL = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
ARMS = [("full", None)] + [(m, f) for m in ("oracle", "quest", "lr64", "lr32") for f in (0.06, 0.02)]
R = {}
def rec(task, arm, frac, score):
    R.setdefault(task, {}).setdefault(f"{arm}@{'all' if frac is None else f'{int(frac*100)}%'}", []).append(score)

# ---------------- essays ----------------
S = [json.loads(l) for l in open(NIAH)]
rng = random.Random(11)
total = len(S) * 6 * len(ARMS) + 16 * len(ARMS); done = 0
for si, s in enumerate(S):
    body = s["body"]
    cache, n_doc = prefill_doc(HEAD + body)
    ni = body.find(s["answer"][:30])
    sents = [m.group(0).strip() for m in re.finditer(r"[^.!?]{80,260}[.!?]", body)
             if ni < 0 or abs(m.start() - ni) > 400]
    picks = rng.sample(sents, 5)
    qs = [("needle", f"\n\nQuestion: {s['question']} Answer briefly." + TAIL, s["answer"], word_recall)]
    for sent in picks:
        w = sent.split(); h = max(4, len(w) // 2)
        qs.append(("essay-complete", f"\n\nQuote the rest of this sentence from the document exactly: \"{' '.join(w[:h])}\"" + TAIL,
                   " ".join(w[h:]), word_recall))
    for task, qt, gold, metric in qs:
        for arm, frac in ARMS:
            b = None if frac is None else max(16, int(n_doc * frac))
            pred = ask(cache, n_doc, qt, arm, b)
            rec(task, arm, frac, metric(pred, gold)); done += 1
            prog(3 + int(94 * done / total), f"essay {si+1}/5 {task} {arm}@{frac or 'all'}: {metric(pred, gold):.2f}")
    del cache; torch.cuda.empty_cache()

# ---------------- code ----------------
files = sorted(glob.glob(f"{code_corpus()}/**/*.py", recursive=True))
random.Random(3).shuffle(files)
ctx, n = [], 0
for f in files:
    t = open(f, errors="ignore").read()
    if 2000 < len(t) < 30000:
        ctx.append(f"# file: {f.split('site-packages/')[-1]}\n{t}\n"); n += len(tok(t, add_special_tokens=False).input_ids)
    if n > 30000: break
code = "".join(ctx); lines = code.split("\n")
cands = [i for i in range(10, int(len(lines) * 0.85)) if len(lines[i].strip()) > 25 and len(lines[i + 1].strip()) > 25
         and code.count(lines[i]) == 1 and code.count(lines[i + 1]) == 1]
cache, n_doc = prefill_doc(HEAD + code)
for i in random.Random(5).sample(cands, 16):
    qt = f"\n\nIn the code above, which line comes right after this one? Output only that line.\n{lines[i].strip()}" + TAIL
    for arm, frac in ARMS:
        b = None if frac is None else max(16, int(n_doc * frac))
        pred = ask(cache, n_doc, qt, arm, b).strip().split("\n")[0]
        rec("code-next", arm, frac, edit_sim(pred, lines[i + 1].strip())); done += 1
        prog(3 + int(94 * done / total), f"code {arm}@{frac or 'all'}: {edit_sim(pred, lines[i+1].strip()):.2f}")
R = {t: {k: round(sum(v) / len(v), 3) for k, v in d.items()} for t, d in R.items()}
print("RESULT " + json.dumps(R), flush=True)
