#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 sparsekv-lab contributors
# Build TokenGate's FA2 key-bias kernel for sm_86 with head dim 256. LAB_DIR holds repo/ (TokenGate) and venv/.
set -x
LAB_DIR="${LAB_DIR:-$(cd "$(dirname "$0")" && pwd)}"
cd "$LAB_DIR/repo"
export TMPDIR="$LAB_DIR/tmp" CUDA_HOME=/usr/local/cuda PATH="/usr/local/cuda/bin:$LAB_DIR/venv/bin:$PATH"
export FA2BIAS_ARCHS=8.6 MAX_JOBS=8 FA2BIAS_HDIM256=1 CUDA_VISIBLE_DEVICES=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
time taskset -c 8-23 "$LAB_DIR/venv/bin/python" kernels/fa2bias/build.py
echo BUILD-EXIT $?
