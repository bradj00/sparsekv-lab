#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 sparsekv-lab contributors
"""Host<->GPU fetch cost for KV offload on this box. Simulates one decode step of a 27B agent:
16 full-attention layers x 4 KV heads x 256 dim, int8 K+V (2 KB per token per layer for all heads).
Measures (a) bulk pinned H2D bandwidth, (b) gathering top-k scattered tokens from a pinned host pool
(token-granular vs 16-token pages), reporting ms per decode step and the tok/s ceiling it implies."""
import json, sys, time, torch
dev = torch.device("cuda")
R = {"gpu": torch.cuda.get_device_name(0)}
def timeit(fn, n=20):
    fn(); torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / n
# bulk
for mb in (64, 256):
    h = torch.empty(mb << 20, dtype=torch.uint8).pin_memory(); d = torch.empty_like(h, device=dev)
    s = timeit(lambda: d.copy_(h, non_blocking=True))
    R[f"bulk_{mb}MB_GBps"] = round(mb / 1024 / s, 2)
# scattered gather: pool of N tokens x 16 layers x (4 heads*256*2 bytes K+V = 2048 B)
N = 256 * 1024; L = 16; TOKB = 2048
pool = torch.empty(N, L * TOKB, dtype=torch.uint8).pin_memory()          # 8 GB host pool = one 256K agent
for frac in (0.02, 0.06):
    k = int(N * frac)
    for page in (1, 16):
        npg = k // page
        idx = torch.randperm(N // page)[:npg]
        stage = torch.empty(npg, page * L * TOKB, dtype=torch.uint8).pin_memory()
        dst = torch.empty_like(stage, device=dev)
        pv = pool.view(N // page, page * L * TOKB)
        def step():
            torch.index_select(pv, 0, idx, out=stage)       # CPU gather (host memcpy)
            dst.copy_(stage, non_blocking=True)            # H2D
        s = timeit(step, 5)
        R[f"top{int(frac*100)}%_page{page}_ms_per_step"] = round(s * 1000, 1)
        R[f"top{int(frac*100)}%_page{page}_MB"] = round(stage.numel() / 2**20)
print("RESULT " + json.dumps(R), flush=True)
