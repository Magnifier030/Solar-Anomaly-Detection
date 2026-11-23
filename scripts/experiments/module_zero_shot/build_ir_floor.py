"""Submission: EO + crack reasons from BASE; IR rebuilt from the test cache with
hotspot = z(mutual, c) + |z(std, c)| >= THR (fill < 0.85 -> 0), modules det >= 0.5.
Usage: python build_ir_floor.py BASE OUT CACHE THR C
"""
import json, subprocess, sys
from pathlib import Path
import numpy as np
args = sys.argv[1:]; sys.argv = sys.argv[:1] + ["ir"]
import raw_eval as R

PROJECT = Path("/home/M11509204/researchs/Solar-Anomaly-Detection")
KIT = PROJECT / "data/raw/evaluation_kit/author-kit_1012"
base, out, cache = Path(args[0]), Path(args[1]), Path(args[2]); thr, c = float(args[3]), float(args[4])
REASON = "推測模組板破損或內部隱裂、電性異常，導電路徑受損使局部電阻升高，進而產生異常發熱。"

sub = json.loads((base / "submission.json").read_text()); ir = json.loads(cache.read_text())
counts = {"solar_panel": 0, "hotspot": 0}
for key, modules in ir.items():
    mods = [m for m in modules if m["det_score"] >= 0.5]
    if len(mods) >= 3 and all("raw" in m for m in mods):
        s = R.z(np.array([m["raw"] for m in mods]), c) + np.abs(R.z(np.array([m["std"] for m in mods]), c))
        s = np.where(np.array([m["fill"] for m in mods]) < R.FILL_MIN, 0, s)
    else:
        s = np.zeros(len(mods))
    objects = []
    for m, v in zip(mods, s):
        if v >= thr:
            objects.append({"label": "hotspot", "points": m["polygon"], "reason": REASON}); counts["hotspot"] += 1
        else:
            objects.append({"label": "solar_panel", "points": m["polygon"]}); counts["solar_panel"] += 1
    sub[key] = objects
out.mkdir(parents=True, exist_ok=False)
path = out / "submission.json"
path.write_text(json.dumps(sub, ensure_ascii=False, allow_nan=False) + "\n")
chk = subprocess.run([sys.executable, str(KIT / "make_submission.py"), "--check", str(path)], capture_output=True, text=True)
(out / "format_check.txt").write_text(chk.stdout + chk.stderr)
(out / "manifest.json").write_text(json.dumps({"base": str(base), "ir_rule": "z(mutual,c)+|z(std,c)| >= thr",
    "thr": thr, "floor_c": c, "ir_counts": counts, "format_check_exit_code": chk.returncode, "uploaded": False},
    ensure_ascii=False, indent=1))
print(counts); print(chk.stdout, chk.stderr)
