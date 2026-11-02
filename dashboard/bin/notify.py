#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 sparsekv-lab contributors
"""Optional status notifier: one plain-language line (% done, time left, latest `tglab exec` status).
Prints the line; if NOTIFY_WEBHOOK_URL is set, also POSTs {"content": line} as JSON to it.
Optional DASHBOARD_URL is appended as a link. Usage: notify.py [--test]"""
import json, os, sys, time, urllib.request
ROOT = os.environ.get("TGLAB_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
s = json.load(open(os.path.join(ROOT, "www", "state.json")))
ph = s["phases"]; tot = sum(p["est_min"] for p in ph)
done_w = sum(p["est_min"] * (1 if p["status"] in ("done", "skipped", "failed") else p["pct"] / 100) for p in ph)
pct = round(100 * done_w / tot); left = max(0, tot - done_w)
eta = f"~{left/60:.1f} h left" if left >= 60 else f"~{int(left)} min left"
status = s.get("exec") or "Work in progress."
msg = f"**{pct}% · {eta}** — {status}"
_last = os.path.join(ROOT, ".notify-last.json")
try: _prev = json.load(open(_last))
except Exception: _prev = {}
if _prev.get("status") == status and "--test" not in sys.argv:
    msg += f" _(no new status since {time.strftime('%-I:%M %p', time.localtime(_prev['since']))})_"
elif "--test" not in sys.argv:
    json.dump({"status": status, "since": time.time()}, open(_last, "w"))
if "--test" in sys.argv:
    msg = "*Test notification.*\n" + msg
if os.environ.get("DASHBOARD_URL"):
    msg += f"\n[Live dashboard]({os.environ['DASHBOARD_URL']})"
print(msg)
hook = os.environ.get("NOTIFY_WEBHOOK_URL")
if hook:
    req = urllib.request.Request(hook, json.dumps({"content": msg[:1900]}).encode(),
                                 {"Content-Type": "application/json", "User-Agent": "sparsekv-lab/1.0"})
    with urllib.request.urlopen(req, timeout=15) as r:
        print("sent", r.status)
