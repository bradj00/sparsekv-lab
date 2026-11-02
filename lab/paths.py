# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 sparsekv-lab contributors
"""Paths shared by the lab scripts. Override any of them with environment variables."""
import importlib.util, os

LAB_DIR = os.environ.get("LAB_DIR", os.path.dirname(os.path.abspath(__file__)))
TG_REPO = os.environ.get("TG_REPO", os.path.join(LAB_DIR, "repo"))            # TokenGate checkout
NIAH = os.path.join(TG_REPO, "examples", "data", "niah_32k_samples.jsonl")    # 32K needle haystacks
MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(LAB_DIR, "models"))
MODEL_27B = os.environ.get("MODEL_27B", os.path.join(MODEL_DIR, "Qwen3.8-27B-W4A16-AutoRound-fast"))
Q27_HF = os.environ.get("Q27_HF", os.path.join(LAB_DIR, "q27-hf"))             # overlay written by overlay27.py


def code_corpus():
    """Directory of Python source used as the code context (default: the installed vllm package)."""
    if os.environ.get("CODE_CORPUS"):
        return os.environ["CODE_CORPUS"]
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit("set CODE_CORPUS to a directory of .py files (or install vllm)")
    return list(spec.submodule_search_locations)[0]
