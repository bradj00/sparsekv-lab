#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 sparsekv-lab contributors
# Thermal watchdog: every 15 s read both GPUs (core temp, power, fan RPM) and, optionally, hotspot/VRAM temps
# from a host-side sqlite thermlog. On 2 consecutive bad reads (core>=85, hotspot>=95, VRAM>=100, or 0 RPM at
# >150 W) it kills lab jobs ($REMOTE_LAB_DIR/venv python) on the GPU host and logs THERMAL STOP.
# Never touches power caps or any serving process.
# Env: SSH_TARGET (user@gpu-host), TGLAB_SSH_KEY (optional), REMOTE_LAB_DIR (lab dir on the GPU host),
#      REMOTE_PY (python with pynvml on the GPU host, default python3),
#      optional THERMLOG_SSH (user@host holding the thermlog), THERMLOG_DB (sqlite path), THERMLOG_PREFIX (sensor prefix).
: "${SSH_TARGET:?set SSH_TARGET to user@gpu-host}"; : "${REMOTE_LAB_DIR:?set REMOTE_LAB_DIR}"
SSH=(ssh -o ConnectTimeout=6); [ -n "$TGLAB_SSH_KEY" ] && SSH+=(-i "$TGLAB_SSH_KEY")
REMOTE_PY="${REMOTE_PY:-python3}"; T="$(cd "$(dirname "$0")" && pwd)/tglab"
bad=0; n=0
while true; do
  g=$("${SSH[@]}" "$SSH_TARGET" "$REMOTE_PY - <<PY
import pynvml as n
from ctypes import byref
n.nvmlInit(); fn=n._nvmlGetFunctionPointer('nvmlDeviceGetFanSpeedRPM'); out=[]
for i in range(2):
    h=n.nvmlDeviceGetHandleByIndex(i); r=[]
    for f in range(2):
        s=n.c_nvmlFanSpeedInfo_t(); s.version=n.nvmlFanSpeedInfo_v1; s.fan=f; fn(h,byref(s)); r.append(s.speed)
    nm='Ti' if 'Ti' in n.nvmlDeviceGetName(h) else '3090'
    out.append('%s %d %d %d'%(nm,n.nvmlDeviceGetTemperature(h,0),n.nvmlDeviceGetPowerUsage(h)//1000,min(r)))
print(';'.join(out))
PY" 2>/dev/null)
  hv=""
  if [ -n "$THERMLOG_SSH" ] && [ -n "$THERMLOG_DB" ]; then
    hv=$(ssh -o ConnectTimeout=6 "$THERMLOG_SSH" "D='$THERMLOG_DB'; for g in 0 1; do for s in hotspot_temp vram_temp; do sqlite3 \$D \"select cast(value as int) from samples where sensor='${THERMLOG_PREFIX:-gpu}\$g/\$s' and ts>strftime('%s','now')-90 order by ts desc limit 1\"; done; done" 2>/dev/null | tr "\n" " ")
  fi
  if [ -z "$g" ]; then sleep 20; continue; fi
  read h0 v0 h1 v1 <<<"$hv"; why=""
  IFS=';' read -ra G <<<"$g"
  for x in "${G[@]}"; do read nm t w rmin <<<"$x"
    [ "$t" -ge 85 ] && why+="$nm core ${t}C; "
    [ "$w" -ge 150 ] && [ "$rmin" -eq 0 ] && why+="$nm fan 0rpm at ${w}W; "
  done
  for x in "$h0:hot0" "$h1:hot1"; do v=${x%%:*}; [ -n "$v" ] && [ "$v" -ge 95 ] && why+="${x#*:} ${v}C; "; done
  for x in "$v0:vram0" "$v1:vram1"; do v=${x%%:*}; [ -n "$v" ] && [ "$v" -ge 100 ] && why+="${x#*:} ${v}C; "; done
  if [ -n "$why" ]; then bad=$((bad+1)); echo "$(date +%H:%M:%S) ALERT ($bad/2): $why [$g | $hv]"
    if [ $bad -ge 2 ]; then
      "${SSH[@]}" "$SSH_TARGET" "for p in \$(pgrep -f '^$REMOTE_LAB_DIR/venv/bin/python|^venv/bin/python'); do kill \$p; done"
      echo "$(date +%H:%M:%S) THERMAL STOP: lab jobs killed ($why)"; $T log "THERMAL STOP: lab jobs killed ($why)"; bad=0; sleep 300
    fi
  else bad=0; fi
  n=$((n+1)); [ $((n % 40)) -eq 1 ] && echo "$(date +%H:%M) ok: $g | hot/vram gpu0 $h0/$v0 gpu1 $h1/$v1"
  sleep 15
done
