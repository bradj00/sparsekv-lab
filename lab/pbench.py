#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 sparsekv-lab contributors
"""STAGE 1 PROTOTYPE BENCH with the PRODUCTION kernel (pth-flash-attn, int8 per-token-head KV, block 864).
One full-attention layer of Qwen3.8-27B, MTP-7 verify (8 query rows per request), batch 1 and 2:
  dense : pthfa over the whole context (the dense baseline)
  sparse: sketch scores (int8 rank-64, shared across KV heads) -> top-1% -> gather the picked rows plus
          sinks (64) and a recent window (1024, holds the 8 new tokens, last and in order so causal masking
          stays exact) into a small staging cache -> the same pthfa kernel over ~C = 1% + 1088 tokens.
Reports ms per layer for each piece and x16 for a decode step. Run on the lab card in the serving venv (vLLM + pth-flash-attn)."""
import json, math, sys, time, torch
import vllm  # noqa: F401  (registers torch.ops._C)
from vllm import _custom_ops as vops
import pth_flash_attn.ops as ops
from pth_flash_attn.planner import plan_launch, workspace_token_splits
from pth_flash_attn.vllm_backend import PthFlashAttnImpl
DEV = "cuda"; HQ, HKV, D, BS, QL, R = 24, 4, 256, 864, 8, 64; G = HQ // HKV
SINK, WIN, KSEL = 64, 1024, 2048          # DSA's fast decode top-k is built for 2048
NSM = torch.cuda.get_device_properties(0).multi_processor_count
impl = PthFlashAttnImpl(num_heads=HQ, head_size=D, scale=D ** -0.5, num_kv_heads=HKV, alibi_slopes=None,
                        sliding_window=None, kv_cache_dtype="int8_per_token_head")
WS_TS = workspace_token_splits(max_num_seqs=2, q_dec_max=QL, Hkv=HKV, D=D, num_sms=NSM)

def bench(fn, n=30):
    for _ in range(5): fn()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / n * 1000

def make_kv(nb):                                     # production layout: [NB, Hkv, BS, 2*(D+4)] int8, inline scales
    kv = torch.empty(nb, HKV, BS, 2 * (D + 4), dtype=torch.int8, device=DEV)
    k, v, ks, vs = impl._caches(kv)
    k.copy_(torch.randint_like(k, -127, 128)); v.copy_(torch.randint_like(v, -127, 128))
    ks.fill_(0.5 / 127); vs.fill_(0.5 / 127)
    return kv, (k, v, ks, vs)

def launcher(caches, bt, seq_lens, B):
    q = torch.randn(B * QL, HQ, D, device=DEV, dtype=torch.bfloat16); out = torch.empty_like(q)
    qsl = torch.arange(0, B * QL + 1, QL, device=DEV, dtype=torch.int32)
    plan = plan_launch(num_tokens=B * QL, num_reqs=B, max_query_len=QL, G=G, Hkv=HKV, D=D, num_sms=NSM,
                       is_prefill=False, ws_token_splits=WS_TS)
    ws_o = torch.empty(plan.ws_o_shape, dtype=torch.float32, device=DEV)
    ws_lse = torch.empty(plan.ws_lse_shape, dtype=torch.float32, device=DEV)
    return q, lambda: ops.pth_flash_attn_fwd(out, q, *caches, bt, seq_lens, qsl, QL, plan.wm, plan.num_splits,
                                             impl.scale, ws_o, ws_lse, 0, True)

R_ = {"gpu": torch.cuda.get_device_name(0)}
for L in (32768, 65536, 131072, 153600, 262144):
    for B in (1, 2):
        nbr = math.ceil(L / BS)
        try:
            kv, caches = make_kv(nbr * B)
            bt = torch.arange(nbr * B, device=DEV, dtype=torch.int32).view(B, nbr)
            sl = torch.full((B,), L, device=DEV, dtype=torch.int32)
            q, dense = launcher(caches, bt, sl, B)
            t_dense = bench(dense)
            # ---- sparse pieces ----
            K = KSEL; C = K + SINK + WIN; nbs = math.ceil(C / BS)
            sk = torch.randint(-127, 128, (B, HKV, L, R), device=DEV, dtype=torch.int8)        # sketch (int8)
            sks = torch.rand(B, HKV, L, 1, device=DEV, dtype=torch.float16) / 127
            Vb = torch.randn(HKV, D, R, device=DEV, dtype=torch.bfloat16) / 16
            stage_nhd = torch.empty(B * nbs, BS, HKV, 2 * (D + 4), dtype=torch.int8, device=DEV)
            stage = stage_nhd.transpose(1, 2)                                                 # same view type as cache
            st_caches = impl._caches(stage)
            st_bt = torch.arange(B * nbs, device=DEV, dtype=torch.int32).view(B, nbs)
            st_sl = torch.full((B,), C, device=DEV, dtype=torch.int32)
            _, sparse_attn = launcher(st_caches, st_bt, st_sl, B)
            tok = torch.arange(L, device=DEV)
            win_idx = torch.arange(L - WIN, L, device=DEV).expand(B, WIN)
            sink_idx = torch.arange(SINK, device=DEV).expand(B, SINK)
            mask = torch.zeros(L, device=DEV, dtype=torch.bool); mask[:SINK] = True; mask[L - WIN:] = True
            state = {}
            def select():
                qg = q.view(B, QL, HKV, G, D).permute(0, 2, 1, 3, 4).reshape(B, HKV, QL * G, D)
                qr = torch.matmul(qg, Vb)                                                      # [B,Hkv,48,R]
                s = torch.matmul(qr.half(), (sk.half() * sks).transpose(-1, -2))              # [B,Hkv,48,L]
                p = (s.float() * impl.scale).softmax(-1).sum((1, 2))                           # [B,L] shared by heads
                p.masked_fill_(mask, float("-inf"))
                state["idx"] = torch.cat([p.topk(K, -1, sorted=False).indices, sink_idx, win_idx], 1)  # window last
            def gather():
                idx = state["idx"]                                                             # [B,C] token ids
                blk = torch.gather(bt.long(), 1, idx // BS); off = idx % BS
                rows = kv[blk.reshape(-1), :, off.reshape(-1)]                                 # [B*C,Hkv,2080]
                for b in range(B):
                    stage_nhd.view(-1, HKV, 2 * (D + 4))[b * nbs * BS: b * nbs * BS + C] = rows[b * C:(b + 1) * C]
            pbuf = torch.randn(B, L, device=DEV); tk = torch.empty(B, K, dtype=torch.int32, device=DEV)
            t_topk_torch = bench(lambda: pbuf.topk(K, -1, sorted=False))
            try:
                t_topk_dsa = bench(lambda: vops.top_k_per_row_decode(pbuf, 1, sl, tk, B, pbuf.stride(0), pbuf.stride(1), K))
            except Exception as e:
                print("dsa topk failed:", repr(e)[:200], flush=True); t_topk_dsa = None
            t_sel = bench(select); select(); t_gat = bench(gather); t_att = bench(sparse_attn)
            t_sparse = bench(lambda: (select(), gather(), sparse_attn()))
            key = f"{L // 1024}K,B={B}"
            R_[key] = {"dense_ms": round(t_dense, 3), "select_ms": round(t_sel, 3), "gather_ms": round(t_gat, 3),
                       "sparse_attn_ms": round(t_att, 3), "topk_torch_ms": round(t_topk_torch, 3),
                       "topk_dsa_ms": t_topk_dsa and round(t_topk_dsa, 3), "sparse_total_ms": round(t_sparse, 3),
                       "dense_step16_ms": round(16 * t_dense, 2), "sparse_step16_ms": round(16 * t_sparse, 2),
                       "saved_per_step_ms": round(16 * (t_dense - t_sparse), 2), "C": C}
            print(key, json.dumps(R_[key]), flush=True)
            del kv, caches, sk, sks, stage_nhd, stage, st_caches; torch.cuda.empty_cache()
        except torch.OutOfMemoryError:
            print(f"{L} B={B} OOM", flush=True); torch.cuda.empty_cache()
print("RESULT " + json.dumps(R_), flush=True)
