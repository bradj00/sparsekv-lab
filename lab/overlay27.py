# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 sparsekv-lab contributors
"""Overlay of the 27B W4A16 checkpoint for HF transformers: dequantize the int8-packed embedding to bf16
(transformers compressed-tensors cannot load a quantized Embedding); everything else symlinked, still compressed."""
import json, os, torch
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import LAB_DIR, TG_REPO, NIAH, MODEL_27B, Q27_HF, code_corpus  # noqa: E402
from safetensors import safe_open
from safetensors.torch import save_file
from compressed_tensors.compressors.pack_quantized.helpers import unpack_from_int32
SRC = MODEL_27B; DST = Q27_HF
os.makedirs(DST, exist_ok=True)
for f in os.listdir(SRC):
    if f in ("config.json", "model.safetensors.index.json"): continue
    if not os.path.exists(f"{DST}/{f}"): os.symlink(f"{SRC}/{f}", f"{DST}/{f}")
idx = json.load(open(f"{SRC}/model.safetensors.index.json")); wm = idx["weight_map"]
P = "model.language_model.embed_tokens."
with safe_open(f"{SRC}/{wm[P+'weight_packed']}", "pt") as f:
    packed, scale, shape = f.get_tensor(P + "weight_packed"), f.get_tensor(P + "weight_scale"), f.get_tensor(P + "weight_shape")
q = unpack_from_int32(packed, 8, torch.Size(shape.tolist()))
w = (q.float() * scale.float().repeat_interleave(q.shape[1] // scale.shape[1], 1)).to(torch.bfloat16)
print("embed", tuple(w.shape), float(w.abs().mean()))
save_file({P + "weight": w.contiguous()}, f"{DST}/embed_bf16.safetensors")
for k in ("weight_packed", "weight_scale", "weight_shape"): wm.pop(P + k)
wm[P + "weight"] = "embed_bf16.safetensors"
json.dump(idx, open(f"{DST}/model.safetensors.index.json", "w"))
cfg = json.load(open(f"{SRC}/config.json")); qc = cfg.get("quantization_config") or cfg["text_config"]["quantization_config"]
qc["config_groups"].pop("group_2", None); qc["ignore"].append("re:.*embed_tokens$")
json.dump(cfg, open(f"{DST}/config.json", "w"), indent=1)
print("overlay ready", DST)
