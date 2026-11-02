# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 sparsekv-lab contributors
import os, sys, time, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from q27loader import load, M
from transformers import AutoTokenizer
t0 = time.time(); model, cfg = load(); print("loaded", round(time.time() - t0), "s", round(torch.cuda.memory_allocated() / 2**30, 2), "GiB", flush=True)
tok = AutoTokenizer.from_pretrained(M)
ids = tok("<|im_start|>user\nWhat is 17*23? Answer with the number only.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", return_tensors="pt").input_ids.cuda()
with torch.no_grad():
    out = model(ids, use_cache=True, logits_to_keep=1); gen = []
    for _ in range(6):
        nt = out.logits[0, -1].argmax().item(); gen.append(nt)
        t1 = time.time(); out = model(torch.tensor([[nt]], device="cuda"), past_key_values=out.past_key_values, use_cache=True)
print("answer:", repr(tok.decode(gen)), "decode step", round((time.time() - t1) * 1000), "ms", flush=True)
