#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 sparsekv-lab contributors
# Create the lab venv in $LAB_DIR and install TokenGate from $LAB_DIR/repo (clone token-gate-llms/TokenGate there first).
set -ex
LAB_DIR="${LAB_DIR:-$(cd "$(dirname "$0")" && pwd)}"
cd "$LAB_DIR"
export TMPDIR="$LAB_DIR/tmp"; mkdir -p "$TMPDIR"
UV="${UV:-uv}"
[ -x venv/bin/python ] || $UV venv -p 3.12 --seed venv
$UV pip install -p venv/bin/python "torch>=2.9" --index-url https://download.pytorch.org/whl/cu130
$UV pip install -p venv/bin/python -e "repo[data]" --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match
$UV pip install -p venv/bin/python flash-linear-attention ninja packaging
venv/bin/python -c "import torch, transformers, tokengate; print(\"torch\", torch.__version__, \"cuda\", torch.version.cuda, \"tf\", transformers.__version__, torch.cuda.get_device_name(0))"
echo INSTALL-DONE
