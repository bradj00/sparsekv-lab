# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 sparsekv-lab contributors
"""Load Qwen3.8-27B W4A16 (compressed-tensors pack-quantized, group 128) into HF transformers WITHOUT decompressing:
every quantized Linear becomes PackedLinear (packed int32 + bf16 scales on GPU, dequantized per call)."""
import json, os, torch, torch.nn as nn, torch.nn.functional as F
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import LAB_DIR, TG_REPO, NIAH, MODEL_27B, Q27_HF, code_corpus  # noqa: E402
from safetensors import safe_open
from transformers import AutoConfig
import transformers
from compressed_tensors.compressors.pack_quantized.helpers import unpack_from_int32
M = MODEL_27B   # original checkpoint; embedding stays packed

class PackedLinear(nn.Module):
    def __init__(self, packed, scale, shape, bits):
        super().__init__()
        self.register_buffer("packed", packed, persistent=False); self.register_buffer("scale", scale, persistent=False)
        self.shape = torch.Size(shape); self.bits = bits; self.g = shape[1] // scale.shape[1]
    def weight_bf16(self):
        q = unpack_from_int32(self.packed, self.bits, self.shape)
        return (q.to(torch.bfloat16) * self.scale.repeat_interleave(self.g, 1)).to(torch.bfloat16)
    def forward(self, x):
        n = self.shape[0]
        if n <= 65536:
            return F.linear(x, self.weight_bf16())
        outs = []                                            # huge (lm_head): unpack 16K output rows at a time
        for a in range(0, n, 16384):
            b = min(n, a + 16384)
            q = unpack_from_int32(self.packed[a:b], self.bits, torch.Size((b - a, self.shape[1])))
            w = q.to(torch.bfloat16) * self.scale[a:b].repeat_interleave(self.g, 1)
            outs.append(F.linear(x, w))
        return torch.cat(outs, -1)

class PackedEmbedding(nn.Module):
    """int8 group-quantized embedding: unpack only the looked-up rows."""
    def __init__(self, packed, scale, shape, bits):
        super().__init__()
        self.register_buffer("packed", packed, persistent=False); self.register_buffer("scale", scale, persistent=False)
        self.shape = torch.Size(shape); self.bits = bits; self.g = shape[1] // scale.shape[1]
    def forward(self, ids):
        flat = ids.reshape(-1)
        q = unpack_from_int32(self.packed[flat], self.bits, torch.Size((flat.numel(), self.shape[1])))
        w = q.to(torch.bfloat16) * self.scale[flat].repeat_interleave(self.g, 1)
        return w.view(*ids.shape, self.shape[1])

def _set(model, path, tensor):
    mod_path, _, name = path.rpartition(".")
    mod = model.get_submodule(mod_path)
    old = getattr(mod, name)
    t = tensor.to("cuda", dtype=old.dtype if old.is_floating_point() else tensor.dtype)
    if isinstance(old, nn.Parameter): setattr(mod, name, nn.Parameter(t, requires_grad=False))
    else: mod.register_buffer(name, t)

def load():
    cfg = AutoConfig.from_pretrained(M)
    qc = getattr(cfg, "quantization_config", None) or cfg.text_config.quantization_config
    bits_of = {}
    for gname, g in qc["config_groups"].items():
        for tgt in g["targets"]: bits_of[tgt] = g["weights"]["num_bits"]
    for obj in (cfg, getattr(cfg, "text_config", None)):
        if obj is not None and hasattr(obj, "quantization_config"): delattr(obj, "quantization_config")
    cls = getattr(transformers, cfg.architectures[0])
    with torch.device("meta"):
        model = cls._from_config(cfg, dtype=torch.bfloat16)
    model.eval()
    idx = json.load(open(f"{M}/model.safetensors.index.json"))["weight_map"]
    files = sorted(set(idx.values()))
    packed = {}
    for fn in files:
        with safe_open(f"{M}/{fn}", "pt") as f:
            for k in f.keys():
                if k.startswith(("mtp.", "model.visual.")) or k == "model.language_model.embed_tokens.weight": continue
                if k.endswith((".weight_packed", ".weight_scale", ".weight_shape")):
                    base, _, suf = k.rpartition(".")
                    packed.setdefault(base, {})[suf] = f.get_tensor(k)
                    if len(packed[base]) == 3:
                        p = packed.pop(base)
                        bits = 8 if base.endswith("embed_tokens") else (bits_of.get("re:.*lm_head$", 4) if base == "lm_head" else 4)
                        mod_path, _, name = base.rpartition(".")
                        parent = model.get_submodule(mod_path) if mod_path else model
                        Cls = PackedEmbedding if base.endswith("embed_tokens") else PackedLinear
                        setattr(parent, name, Cls(p["weight_packed"].cuda(), p["weight_scale"].cuda().to(torch.bfloat16), p["weight_shape"].tolist(), bits))
                else:
                    try: _set(model, k, f.get_tensor(k))
                    except AttributeError: pass
    model.model.visual = None
    # re-materialize anything still on meta (rotary inv_freq etc.)
    for name, mod in model.named_modules():
        if "rotary_emb" in name.split(".")[-1]:
            with torch.device("cuda"):
                new = type(mod)(config=mod.config)
            parent_path, _, child = name.rpartition(".")
            setattr(model.get_submodule(parent_path), child, new)
    left = [n for n, t in list(model.named_parameters()) + list(model.named_buffers()) if t.is_meta]
    if left: print("STILL META:", left[:10], len(left), flush=True)
    return model, cfg
