#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 sparsekv-lab contributors
"""Decode-attention micro-benchmark at Qwen3.8-27B shapes on one 3090.
Per full-attention layer, per decode step: batch B agents, Q=8 query tokens (MTP-7 verify), 24 q heads, 4 KV heads,
head_dim 256, context N. Compares
  full    : exact attention over all N keys (SDPA / flash, fp16)
  sketch  : rank-32 key sketch scores (fp16) -> per-KV-head top-k (2% of N, group+query pooled) -> gather K,V -> exact attention
  sketch_thr: same but selection via a fixed-size candidate pass (topk on a 1/8-subsampled score then refine) [cheaper topk]
Reports ms per layer and x16 layers per step, plus memory bytes touched."""
import json, math, time, torch, torch.nn.functional as F
torch.backends.cuda.matmul.allow_tf32 = True
dev = "cuda"; dt = torch.float16
import os
HQ, HKV, D, Q, R = 24, 4, 256, 8, int(os.environ.get("R", 32))
G = HQ // HKV
def bench(fn, n=20):
    for _ in range(3): fn()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / n * 1000
out = {"gpu": torch.cuda.get_device_name(0)}
for N in tuple(int(x) for x in os.environ.get("NS", "32768,131072,262144").split(",")):
    for B in (1, 2):
        try:
            k = torch.randn(B, HKV, N, D, device=dev, dtype=dt); v = torch.randn_like(k)
            q = torch.randn(B, HQ, Q, D, device=dev, dtype=dt)
            kr = torch.randn(B, HKV, N, R, device=dev, dtype=dt)          # sketch (resident)
            basis = torch.randn(HKV, R, D, device=dev, dtype=dt)
            kb = max(64, int(0.02 * N))
            sc = 1 / math.sqrt(D)
            def full():
                return F.scaled_dot_product_attention(q, k, v, scale=sc, enable_gqa=True)
            def sketch():
                qg = q.view(B, HKV, G * Q, D)
                qr = torch.einsum("bhqd,hrd->bhqr", qg, basis)              # project queries
                s = torch.einsum("bhqr,bhnr->bhqn", qr, kr)                 # [B,HKV,G*Q,N]
                pooled = s.amax(2)                                         # [B,HKV,N] max over group x queries
                idx = pooled.topk(kb, -1, sorted=False).indices             # [B,HKV,kb]
                ks = torch.gather(k, 2, idx[..., None].expand(B, HKV, kb, D))
                vs = torch.gather(v, 2, idx[..., None].expand(B, HKV, kb, D))
                return F.scaled_dot_product_attention(q, ks, vs, scale=sc, enable_gqa=True)
            def select_only():
                qg = q.view(B, HKV, G * Q, D)
                qr = torch.einsum("bhqd,hrd->bhqr", qg, basis)
                s = torch.einsum("bhqr,bhnr->bhqn", qr, kr)
                return s.amax(2).topk(kb, -1, sorted=False).indices
            tf, ts, tsel = bench(full), bench(sketch), bench(select_only)
            key = f"N={N//1024}K,B={B}"
            out[key] = {"full_ms_layer": round(tf, 3), "sketch_ms_layer": round(ts, 3), "select_ms_layer": round(tsel, 3),
                        "full_ms_step16": round(16 * tf, 2), "sketch_ms_step16": round(16 * ts, 2),
                        "speedup": round(tf / ts, 2),
                        "full_MB_read_step": round(16 * B * HKV * N * D * 2 * 1 / 2**20),        # int8 K+V
                        "sketch_MB_read_step": round(16 * B * HKV * (N * R * 2 + kb * D * 2) / 2**20)}
            print(key, json.dumps(out[key]), flush=True)
            del k, v, kr; torch.cuda.empty_cache()
        except torch.OutOfMemoryError:
            print(f"N={N} B={B} OOM", flush=True); torch.cuda.empty_cache()
print("RESULT " + json.dumps(out), flush=True)
