#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 sparsekv-lab contributors
LAB_DIR="${LAB_DIR:-$(cd "$(dirname "$0")" && pwd)}"
cd "$LAB_DIR/repo"
export HF_HOME="${HF_HOME:-$LAB_DIR/hf}" TMPDIR="$LAB_DIR/tmp" CUDA_VISIBLE_DEVICES=1 CUDA_DEVICE_ORDER=PCI_BUS_ID PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
PY="$LAB_DIR/venv/bin/python"
echo "=== HYBRID fa2"
taskset -c 8-23 $PY examples/niah_demo.py --checkpoint liana313/tokengate-niah-qwen3.5-2b --k 2000 500 100 --gate-attn fa2 || { echo "=== HYBRID flex fallback"; taskset -c 8-23 $PY examples/niah_demo.py --checkpoint liana313/tokengate-niah-qwen3.5-2b --k 2000 500 100 --gate-attn flex; }
echo SMOKE-EXIT $?
