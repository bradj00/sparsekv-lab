#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 sparsekv-lab contributors
"""Zero-copy (UVA) scattered fetch from pinned host RAM, as a GPU kernel would do it (no CPU gather).
Host pool = one 256K-token agent's full KV for ONE layer-group slice: rows of R bytes. A GPU index_select on the
CUDA view of the pinned tensor reads only the selected rows over PCIe. Row size = fetch granularity:
512 B (1 token, 1 KV head, int8 K+V), 2 KB (1 token, all 4 heads), 8 KB (4 tokens), 32 KB (16-token page).
Fetch volume per step = miss_rate x 2% x 256K tokens x 32 KB (16 layers); we time a 10 MB and a 20 MB fetch."""
import json, time, torch
import vllm  # noqa: F401  (registers torch.ops._C)
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor as view
dev = "cuda"; R = {"gpu": torch.cuda.get_device_name(0)}
POOL_BYTES = 8 << 30                                                  # 8 GB = 256K tokens x 32 KB
for row in (512, 2048, 8192, 32768):
    n_rows = POOL_BYTES // row
    host = torch.empty(n_rows, row // 4, dtype=torch.int32, pin_memory=True)
    hv = view(host)
    for mb in (10, 20):
        k = (mb << 20) // row
        idx = torch.randperm(n_rows, device=dev)[:k]
        out = torch.empty(k, row // 4, dtype=torch.int32, device=dev)
        for _ in range(3): torch.index_select(hv, 0, idx, out=out)
        torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(10): torch.index_select(hv, 0, idx, out=out)
        torch.cuda.synchronize(); ms = (time.perf_counter() - t) / 10 * 1000
        R[f"row{row}B_{mb}MB_ms"] = round(ms, 2); R[f"row{row}B_GBps"] = round(mb / 1024 / (ms / 1000), 2)
        print(f"row {row} B, {mb} MB: {ms:.2f} ms ({mb/1024/(ms/1000):.1f} GB/s)", flush=True)
    del host, hv
print("RESULT " + json.dumps(R), flush=True)
