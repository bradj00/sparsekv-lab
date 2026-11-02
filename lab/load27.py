# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 sparsekv-lab contributors
import time, torch, sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import LAB_DIR, TG_REPO, NIAH, MODEL_27B, Q27_HF, code_corpus  # noqa: E402
from transformers import AutoConfig, AutoTokenizer
import transformers
M = Q27_HF
t0 = time.time()
cfg = AutoConfig.from_pretrained(M)
cls = getattr(transformers, cfg.architectures[0])
from transformers import CompressedTensorsConfig
model = cls.from_pretrained(M, dtype=torch.bfloat16, device_map="cuda:0", quantization_config=CompressedTensorsConfig(run_compressed=True))
print("loaded", round(time.time() - t0), "s", torch.cuda.memory_allocated() / 2**30, "GiB", flush=True)
tok = AutoTokenizer.from_pretrained(M)
ids = tok("<|im_start|>user\nWhat is 17*23? Answer with the number only.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", return_tensors="pt").input_ids.cuda()
out = model.generate(ids, max_new_tokens=8, do_sample=False)
print("answer:", tok.decode(out[0, ids.shape[1]:]), flush=True)
