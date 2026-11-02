#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 sparsekv-lab contributors
# Polls the GPUs every 3 s into www/gpu.json for the dashboard. Runs nvidia-smi locally, or on SSH_TARGET if set.
# Env: WWW_DIR (default ../www), SSH_TARGET (optional user@gpu-host), TGLAB_SSH_KEY (optional ssh key).
WWW_DIR="${WWW_DIR:-$(cd "$(dirname "$0")/../www" && pwd)}"; OUT="$WWW_DIR/gpu.json"
Q='nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,power.draw,power.limit,memory.used,memory.total,fan.speed --format=csv,noheader,nounits'
SSH=(ssh -o ConnectTimeout=5 -o BatchMode=yes); [ -n "$TGLAB_SSH_KEY" ] && SSH+=(-i "$TGLAB_SSH_KEY")
while true; do
  if [ -n "$SSH_TARGET" ]; then raw=$("${SSH[@]}" "$SSH_TARGET" "$Q" 2>/dev/null); else raw=$($Q 2>/dev/null); fi
  j=$(printf '%s\n' "$raw" | python3 -c '
import sys, json, time
g = []
for l in sys.stdin:
    f = [x.strip() for x in l.split(",")]
    if len(f) < 9: continue
    n = lambda x: float(x) if x.replace(".", "", 1).isdigit() else None
    g.append({"i": int(f[0]), "name": f[1].replace("NVIDIA GeForce ", ""), "temp": n(f[2]), "util": n(f[3]),
              "power": n(f[4]), "plimit": n(f[5]), "mem": n(f[6]), "memtot": n(f[7]), "fan": n(f[8])})
if g: print(json.dumps({"t": time.time(), "gpus": g}))')
  if [ -n "$j" ]; then echo "$j" > "$OUT.tmp" && mv "$OUT.tmp" "$OUT"; fi
  sleep 3
done
