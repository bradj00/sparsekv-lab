#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 sparsekv-lab contributors
"""STAGE 0 GATE: does the shared-head rank-64 key sketch still match full attention on the production 27B
at long context (32K / 64K / 128K)? One 24 GB card holds 13.5 GB of packed W4A16 weights plus up to 8 GB of
bf16 KV, so this harness avoids every extra copy:
  * full-attention KV goes into a preallocated buffer (no torch.cat growth),
  * each question restores a snapshot (attention lengths + GatedDeltaNet state) instead of deep-copying the cache,
  * chunked prefill uses flash attention twice (past keys, then the causal chunk) merged by log-sum-exp,
  * question/decode attention runs in row blocks so nothing scales with Lq x Lk.
Arms: full | oracle@1% (exact top-k per KV head) | sh64x@1% and @0.5% (one pick shared by all KV heads,
rank-64 sketch from a frozen basis fitted on a different essay). Every arm also records the share of the exact
attention mass its pick captured, and the working set (tokens touched over the whole answer).
Env: LENS=32000,64000,128000  OUT=$LAB_DIR/lc27.json  SELFTEST=1 (checks only)."""
import copy, glob, json, math, os, random, re, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import LAB_DIR, TG_REPO, NIAH, MODEL_27B, Q27_HF, code_corpus  # noqa: E402
import torch, torch.nn.functional as F
from transformers import AutoTokenizer, AttentionInterface
from transformers import cache_utils

LENS = [int(x) for x in os.environ.get("LENS", "32000,64000,128000").split(",")]
OUT = os.environ.get("OUT", f"{LAB_DIR}/lc27.json")
CAP = max(LENS) + 1024
RB = 8                                                     # query rows per block in the small path
ST = {"mode": "full", "budget": None, "doc_len": 0, "foreign": {}, "fit_foreign": False, "lr": {},
      "sink": 4, "mass": [], "ws": {}}
fa = torch.ops.aten._scaled_dot_product_flash_attention

# ---------- preallocated KV for full-attention layers ----------
def _update(self, key_states, value_states, *args, **kwargs):
    if not self.is_initialized:
        self.lazy_initialization(key_states, value_states)
        B, H, _, D = key_states.shape
        self.buf_k = torch.empty(B, H, CAP, D, dtype=key_states.dtype, device=key_states.device)
        self.buf_v = torch.empty_like(self.buf_k); self.n = 0
    m = key_states.shape[-2]
    self.buf_k[:, :, self.n:self.n + m] = key_states; self.buf_v[:, :, self.n:self.n + m] = value_states
    self.n += m
    self.keys, self.values = self.buf_k[:, :, :self.n], self.buf_v[:, :, :self.n]
    return self.keys, self.values
cache_utils.DynamicLayer.update = _update

def _clone(x):
    if torch.is_tensor(x): return x.clone()
    if isinstance(x, dict): return {k: _clone(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return type(x)(_clone(v) for v in x)
    return copy.copy(x)
SKIP = ("buf_k", "buf_v", "keys", "values")
def snapshot(cache):
    return [({k: _clone(v) for k, v in vars(L).items() if k not in SKIP}, getattr(L, "n", None)) for L in cache.layers]
def restore(cache, snap):
    for L, (d, n) in zip(cache.layers, snap):
        for k, v in d.items(): setattr(L, k, _clone(v))
        if n is not None:
            L.n = n; L.keys, L.values = L.buf_k[:, :, :n], L.buf_v[:, :, :n]

# ---------- attention ----------
def prefill_attn(q, k, v, scaling):
    """q [1,Hq,Lq,D]; k,v [1,Hkv,Lk,D] = past keys + this chunk, causal within the chunk."""
    B, Hq, Lq, D = q.shape; Hkv, Lk = k.shape[1], k.shape[2]; g = Hq // Hkv; off = Lk - Lq
    kc = k[:, :, off:].repeat_interleave(g, 1); vc = v[:, :, off:].repeat_interleave(g, 1)
    ob, lb = fa(q.contiguous(), kc, vc, 0.0, True, False, scale=scaling)[:2]
    if off == 0:
        return ob
    qg = q.reshape(B, Hkv, g * Lq, D).contiguous()           # GQA: the g query heads of a KV head become rows
    oa, la = fa(qg, k[:, :, :off].contiguous(), v[:, :, :off].contiguous(), 0.0, False, False, scale=scaling)[:2]
    oa = oa.reshape(B, Hq, Lq, D); la = la.reshape(B, Hq, Lq)
    m = torch.maximum(la, lb); wa = (la - m).exp()[..., None]; wb = (lb - m).exp()[..., None]
    return ((oa.float() * wa + ob.float() * wb) / (wa + wb)).to(q.dtype)

def sketch_of(li, K, doc):
    key = li
    if key not in ST["lr"]:
        Vb = ST["foreign"][li][:, :64]                        # [Hkv, r, D], frozen basis from another document
        q8 = torch.empty(K.shape[0], doc, Vb.shape[1], dtype=torch.int8, device=K.device)
        sc = torch.empty(K.shape[0], doc, 1, dtype=torch.float32, device=K.device)
        for a in range(0, doc, 16384):                         # int8 sketch with per-token scale, built in chunks
            x = torch.einsum("hkd,hrd->hkr", K[:, a:min(doc, a + 16384)].float(), Vb)
            s = x.abs().amax(-1, keepdim=True).clamp_min(1e-6) / 127
            q8[:, a:a + 16384] = (x / s).round().to(torch.int8); sc[:, a:a + 16384] = s
        ST["lr"][key] = (q8, sc, Vb)
    return ST["lr"][key]

def small_attn(module, q, k, v, scaling):
    B, Hq, Lq, D = q.shape; Hkv, Lk = k.shape[1], k.shape[2]; g = Hq // Hkv
    li = module.layer_idx; doc = ST["doc_len"]; mode, bud = ST["mode"], ST["budget"]
    qg = q[0].reshape(Hkv, g, Lq, D); K, V = k[0], v[0]
    sel_on = mode != "full" and bud is not None and bud < doc
    if sel_on and mode.startswith("sh"):
        Kq, Ks, Vb = sketch_of(li, K, doc)
    out = torch.empty(Hkv, g, Lq, D, dtype=q.dtype, device=q.device)
    kpos = torch.arange(Lk, device=q.device)
    for a in range(0, Lq, RB):
        b = min(Lq, a + RB); rb = b - a
        S = torch.bmm(qg[:, :, a:b].reshape(Hkv, g * rb, D), K.transpose(-1, -2)).view(Hkv, g, rb, Lk).float() * scaling
        qpos = torch.arange(Lk - Lq + a, Lk - Lq + b, device=q.device)
        S.masked_fill_(kpos[None] > qpos[:, None], float("-inf"))
        if sel_on:
            exact = S[..., :doc].softmax(-1).mean(1)                                   # [Hkv,rb,doc] exact mass
            if mode == "oracle":
                score = exact.clone()
            else:
                qr = torch.einsum("hgqd,hrd->hgqr", qg[:, :, a:b].float(), Vb)
                score = (torch.bmm(qr.reshape(Hkv, g * rb, -1), (Kq.float() * Ks).transpose(-1, -2)).view(Hkv, g, rb, doc) * scaling).softmax(-1).sum(1)
                score = score.sum(0, keepdim=True).expand(Hkv, rb, doc).contiguous()     # one pick for all KV heads
            score[..., :ST["sink"]] = float("inf")
            idx = score.topk(bud, -1).indices                                           # [Hkv,rb,bud]
            keep = torch.zeros(Hkv, rb, doc, dtype=torch.bool, device=q.device).scatter_(-1, idx, True)
            ST["mass"].append(exact.gather(-1, idx).sum(-1).mean().item())
            ws = ST["ws"].get(li)
            ST["ws"][li] = keep.any(0).any(0) if ws is None else ws | keep.any(0).any(0)
            S[..., :doc].masked_fill_(~keep[:, None], float("-inf"))
        P = S.softmax(-1).to(V.dtype); del S
        out[:, :, a:b] = torch.bmm(P.reshape(Hkv, g * rb, Lk), V).view(Hkv, g, rb, D)
    return out.reshape(1, Hq, Lq, D)

def topk_attn(module, q, k, v, attention_mask, scaling=None, dropout=0.0, **kw):
    scaling = scaling or 1 / math.sqrt(q.shape[-1])
    if q.shape[2] > 512:
        if ST["fit_foreign"]:
            ST["foreign"][module.layer_idx] = torch.linalg.svd(k[0].float(), full_matrices=False)[2][:, :64]
        o = prefill_attn(q, k, v, scaling)
    else:
        o = small_attn(module, q, k, v, scaling)
    return o.transpose(1, 2).contiguous(), None
AttentionInterface.register("topk", topk_attn)

def prog(p, note): print(f"PROGRESS {p} {note}", flush=True)

# ---------- self-test of the merged flash prefill against plain SDPA ----------
def selftest_attn():
    torch.manual_seed(0)
    for Lq, off in ((1024, 0), (1024, 3000)):
        q = torch.randn(1, 24, Lq, 256, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(1, 4, off + Lq, 256, device="cuda", dtype=torch.bfloat16); v = torch.randn_like(k)
        am = torch.arange(off + Lq, device="cuda")[None] <= (torch.arange(Lq, device="cuda")[:, None] + off)
        ref = F.scaled_dot_product_attention(q.float(), k.float().repeat_interleave(6, 1), v.float().repeat_interleave(6, 1),
                                             attn_mask=am, scale=1 / 16)
        got = prefill_attn(q, k, v, 1 / 16).float()
        print(f"SELFTEST prefill Lq={Lq} off={off} max|err|={(ref - got).abs().max().item():.4f}", flush=True)
selftest_attn()

prog(1, "loading Qwen3.8-27B W4A16 (packed loader)")
from q27loader import load as _load27, M as _M27
model, _cfg = _load27(); model.eval()
for _m in model.modules():
    if hasattr(_m, "config") and hasattr(_m.config, "_attn_implementation"): _m.config._attn_implementation = "topk"
tok = AutoTokenizer.from_pretrained(_M27)
EOS = {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")}
print(f"MEM after load {torch.cuda.memory_allocated() / 2**30:.2f} GiB", flush=True)

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
def prefill(ids):
    ST.update(mode="full", budget=None, lr={}); cache = None; CH = 4096; t = time.time()
    for a in range(0, ids.shape[1], CH):
        out = model(ids[:, a:a + CH], past_key_values=cache, use_cache=True, logits_to_keep=1)
        cache = out.past_key_values
    ST["doc_len"] = ids.shape[1]
    print(f"PREFILL {ids.shape[1]} tok in {time.time() - t:.0f}s, mem {torch.cuda.memory_allocated() / 2**30:.2f} GiB, "
          f"peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)
    return cache

@torch.no_grad()
def ask(cache, snap, qtext, mode, budget, max_new):
    restore(cache, snap)
    ST.update(mode=mode, budget=budget, mass=[], ws={})
    ids = tok(qtext, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
    out = model(ids, past_key_values=cache, use_cache=True, logits_to_keep=1)
    gen = []
    for _ in range(max_new):
        nt = out.logits[0, -1].argmax().item()
        if nt in EOS: break
        gen.append(nt)
        out = model(torch.tensor([[nt]], device="cuda"), past_key_values=cache, use_cache=True, logits_to_keep=1)
    mass = sum(ST["mass"]) / len(ST["mass"]) if ST["mass"] else None
    ws = (sum(w.float().mean().item() for w in ST["ws"].values()) / len(ST["ws"])) if ST["ws"] else None
    return tok.decode(gen, skip_special_tokens=True), mass, ws

@torch.no_grad()
def fit_foreign(text):
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids[:, :4096].cuda()
    ST.update(fit_foreign=True, foreign={}); model(ids, use_cache=False, logits_to_keep=1); ST["fit_foreign"] = False

HEAD = "<|im_start|>user\nAnswer using only the document below. Be exact.\n\nDocument:\n"
TAIL = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
ARMS = [("full", None), ("oracle", 0.01), ("sh64x", 0.01), ("sh64x", 0.005)]
S = [json.loads(l) for l in open(NIAH)]
fit_foreign(HEAD + S[4]["body"])                    # ONE frozen basis for every document below (an essay never asked about)

if os.environ.get("SELFTEST"):                       # snapshot/restore determinism + memory at the largest length
    L = max(LENS); body = "\n\n".join(s["body"] for s in S[:4])
    ids = tok(HEAD + body, return_tensors="pt", add_special_tokens=False).input_ids[:, :L].cuda()
    cache = prefill(ids); snap = snapshot(cache)
    qt = f"\n\nQuestion: {S[0]['question']} Answer briefly." + TAIL
    for arm in (("full", None), ("full", None), ("sh64x", 0.01)):
        b = None if arm[1] is None else int(L * arm[1])
        t = time.time(); r = ask(cache, snap, qt, arm[0], b, 16)
        print(f"SELFTEST {arm} -> {r[0]!r} mass={r[1]} ws={r[2]} ({time.time() - t:.0f}s) peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)
    print("SELFTEST gold:", S[0]["answer"], flush=True); sys.exit(0)

def code_text(target):
    files = sorted(glob.glob(f"{code_corpus()}/**/*.py", recursive=True))
    random.Random(3).shuffle(files); ctx, n = [], 0
    for f in files:
        t = open(f, errors="ignore").read()
        if 2000 < len(t) < 30000:
            ctx.append(f"# file: {f.split('site-packages/')[-1]}\n{t}\n"); n += len(tok(t, add_special_tokens=False).input_ids)
        if n > target * 1.05: break
    return "".join(ctx)

def stratified(cands, k, rng):
    if len(cands) <= k: return cands
    bins = [cands[i * len(cands) // k:(i + 1) * len(cands) // k] for i in range(k)]
    return [rng.choice(b) for b in bins if b]

R = json.load(open(OUT)) if os.path.exists(OUT) else {}
for L in LENS:
    tag = f"{L // 1000}K"
    if tag in R and R[tag].get("complete"): continue
    print(f"PHASE {tag}", flush=True)
    rec = {"rows": []}; rng = random.Random(L)
    # ---- essays: needles + ordinary sentences, spread over depth ----
    body = "\n\n".join(s["body"] for s in S[:4])
    ids = tok(HEAD + body, return_tensors="pt", add_special_tokens=False).input_ids[:, :L].cuda()
    text = tok.decode(ids[0])
    qs = []
    for s in S[:4]:
        if s["answer"][:30] in text and len([q for q in qs if q[0] == "needle"]) < 2:
            qs.append(("needle", f"\n\nQuestion: {s['question']} Answer briefly." + TAIL, s["answer"], 24))
    npos = [text.find(s["answer"][:30]) for s in S if s["answer"][:30] in text]
    sents = [m for m in re.finditer(r"[^.!?\n]{80,260}[.!?]", text)
             if all(abs(m.start() - p) > 400 for p in npos) and m.start() < len(text) * 0.92]
    sents = [m.group(0).strip() for m in sents if text.count(m.group(0).strip()) == 1]
    for sent in stratified(sents, 6, rng):
        w = sent.split(); h = max(4, len(w) // 2)
        qs.append(("essay-complete", f"\n\nQuote the rest of this sentence from the document exactly: \"{' '.join(w[:h])}\"" + TAIL,
                   " ".join(w[h:]), 32))
    code = code_text(L)
    cids = tok(HEAD + code, return_tensors="pt", add_special_tokens=False).input_ids[:, :L].cuda()
    ctext = tok.decode(cids[0]); lines = ctext.split("\n")
    cands = [i for i in range(10, int(len(lines) * 0.92)) if len(lines[i].strip()) > 25 and len(lines[i + 1].strip()) > 25
             and ctext.count(lines[i]) == 1 and ctext.count(lines[i + 1]) == 1]
    cq = [("code-next", f"\n\nIn the code above, which line comes right after this one? Output only that line.\n{lines[i].strip()}" + TAIL,
           lines[i + 1].strip(), 32) for i in stratified(cands, 8, rng)]
    total = (len(qs) + len(cq)) * len(ARMS); done = 0
    for doc_ids, qlist in ((ids, qs), (cids, cq)):
        prog(1, f"{tag}: prefilling {doc_ids.shape[1]} tokens")
        cache = prefill(doc_ids); snap = snapshot(cache); n = doc_ids.shape[1]
        for task, qt, gold, mx in qlist:
            for arm, frac in ARMS:
                b = None if frac is None else max(16, int(n * frac))
                pred, mass, ws = ask(cache, snap, qt, arm, b, mx)
                if task == "code-next":
                    pred = pred.strip().split("\n")[0]; sc = edit_sim(pred, gold)
                else:
                    sc = word_recall(pred, gold)
                rec["rows"].append({"task": task, "arm": f"{arm}@{frac or 'all'}", "score": sc, "mass": mass, "ws": ws,
                                    "pred": pred[:200], "gold": gold[:200]})
                done += 1
                prog(2 + int(97 * done / total), f"{tag} {task} {arm}@{frac or 'all'}: {sc:.2f}"
                     + (f" mass {mass:.2f}" if mass is not None else ""))
                R[tag] = rec; json.dump(R, open(OUT, "w"), indent=1)
        del cache, snap; ST["lr"] = {}; torch.cuda.empty_cache()
    summ = {}
    for r in rec["rows"]:
        d = summ.setdefault(r["task"], {}).setdefault(r["arm"], {"score": [], "mass": [], "ws": []})
        for k in ("score", "mass", "ws"):
            if r[k] is not None: d[k].append(r[k])
    rec["summary"] = {t: {a: {k: (round(sum(v) / len(v), 3) if v else None) for k, v in d.items()} for a, d in ad.items()}
                      for t, ad in summ.items()}
    rec["complete"] = True; R[tag] = rec; json.dump(R, open(OUT, "w"), indent=1)
    print(f"RESULT {tag} " + json.dumps(rec["summary"]), flush=True)
print("ALLDONE", flush=True)
