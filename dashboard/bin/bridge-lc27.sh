#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 sparsekv-lab contributors
# Stream $REMOTE_LAB_DIR/lc27.log from the GPU host into the dashboard: PHASE 32K -> phase s0_32, PROGRESS -> pct + note.
# Env: SSH_TARGET (user@gpu-host), TGLAB_SSH_KEY (optional ssh key), REMOTE_LAB_DIR (lab dir on the GPU host).
: "${SSH_TARGET:?set SSH_TARGET to user@gpu-host}"; : "${REMOTE_LAB_DIR:?set REMOTE_LAB_DIR to the lab dir on the GPU host}"
SSH=(ssh -o ServerAliveInterval=30); [ -n "$TGLAB_SSH_KEY" ] && SSH+=(-i "$TGLAB_SSH_KEY")
T="$(cd "$(dirname "$0")" && pwd)/tglab"; cur=""
declare -A MAP=([32K]=s0_32 [64K]=s0_64 [128K]=s0_128)
"${SSH[@]}" "$SSH_TARGET" "tail -n +1 -F '$REMOTE_LAB_DIR/lc27.log'" | while IFS= read -r line; do
  case "$line" in
    PHASE*) tag=${line#PHASE }; [ -n "$cur" ] && $T done "$cur"; cur=${MAP[$tag]}; $T start "$cur";;
    PROGRESS*) [ -z "$cur" ] && continue; rest=${line#PROGRESS }; pct=${rest%% *}; note=${rest#* }; $T progress "$cur" "$pct" "$note";;
    RESULT*) [ -n "$cur" ] && $T done "$cur" "results in"; $T log "${line:0:400}";;
    *Traceback*|*Error*|*OutOfMemory*) $T log "lab error: ${line:0:200}";;
    ALLDONE*) exit 0;;
  esac
done
