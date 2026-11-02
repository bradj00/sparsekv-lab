#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 sparsekv-lab contributors
"""Every 60 s, record the dashboard's overall ETA (same formula as index.html phaseEta) into www/eta.json,
so the page can draw how the ETA has moved. History resets when a new round is initialised (state.created)."""
import json, os, time, tempfile
WWW = os.environ.get("WWW_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "www"))
OUT = f"{WWW}/eta.json"

def phase_eta(p, now):
    est = p["est_min"] * 60
    if p["status"] in ("done", "failed", "skipped"): return 0
    if p["status"] == "pending" or not p.get("started"): return est
    el = now - p["started"]; f = min(p["pct"], 99.5) / 100
    by_est = max(est - el, est * 0.08 * (1 - f))
    if f < 0.05: return by_est
    by_rate = el * (1 - f) / f; w = min(1, f * 1.6)
    return w * by_rate + (1 - w) * by_est

while True:
    try:
        s = json.load(open(f"{WWW}/state.json")); now = time.time(); ph = s["phases"]
        try: h = json.load(open(OUT))
        except Exception: h = {}
        if h.get("created") != s["created"]: h = {"created": s["created"], "points": []}
        stopped = any(p["status"] == "failed" for p in ph)
        tot = sum(p["est_min"] for p in ph)
        done_w = sum(p["est_min"] * (1 if p["status"] in ("done", "skipped", "failed") else p["pct"] / 100) for p in ph)
        eta = 0 if stopped else sum(phase_eta(p, now) for p in ph)
        h["points"].append([round(now), round(eta), round(100 * done_w / tot, 1)])
        h["points"] = h["points"][-20000:]
        fd, tmp = tempfile.mkstemp(dir=WWW)
        with os.fdopen(fd, "w") as f: json.dump(h, f, separators=(",", ":"))
        os.chmod(tmp, 0o644); os.replace(tmp, OUT)
    except Exception as e:
        print("eta-recorder:", e, flush=True)
    time.sleep(60)
