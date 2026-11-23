"""Rebuild a submission: keep IR (and reasons) from a base submission, replace EO with
modules from the test cache where crack = Mask R-CNN label OR (per-frame top-1 zero-shot z >= THR).
Usage: python build_eo_crack.py BASE_DIR OUT_DIR CACHE_JSON THR FLOOR_C [DET_FLOOR]
"""
import json, subprocess, sys
from pathlib import Path
import numpy as np
sys.argv, args = sys.argv[:1] + ["eo"], sys.argv[1:]
import raw_eval as R

PROJECT = Path("/home/M11509204/researchs/Solar-Anomaly-Detection")
KIT = PROJECT / "data/raw/evaluation_kit/author-kit_1012"
base, out, cache = Path(args[0]), Path(args[1]), Path(args[2])
thr, c = float(args[3]), float(args[4])
det_floor = float(args[5]) if len(args) > 5 else 0.7
REASON = "模組板破裂，破損處的導電路徑受損。"

sub = json.loads((base / "submission.json").read_text())
eo = json.loads(cache.read_text())
counts = {"solar_panel": 0, "crack": 0, "crack_from_zs": 0, "crack_from_det": 0}
for key, modules in eo.items():
    mods = [m for m in modules if m["det_score"] >= det_floor]
    if len(mods) >= 3 and all("raw" in m for m in mods):
        s = R.z(np.array([m["raw"] for m in mods]), c)
        s = np.where(np.array([m["fill"] for m in mods]) < R.FILL_MIN, 0, s)
    else:
        s = np.zeros(len(mods))
    top = int(np.argmax(s)) if len(s) else -1
    objects = []
    for j, (m, v) in enumerate(zip(mods, s)):
        by_zs, by_det = j == top and v >= thr, m["det_label"] == 2
        if by_zs or by_det:
            objects.append({"label": "crack", "points": m["polygon"], "reason": REASON})
            counts["crack"] += 1
            counts["crack_from_zs"] += int(by_zs)
            counts["crack_from_det"] += int(by_det)
        else:
            objects.append({"label": "solar_panel", "points": m["polygon"]})
            counts["solar_panel"] += 1
    sub[key] = objects
out.mkdir(parents=True, exist_ok=False)
path = out / "submission.json"
path.write_text(json.dumps(sub, ensure_ascii=False, allow_nan=False) + "\n")
chk = subprocess.run([sys.executable, str(KIT / "make_submission.py"), "--check", str(path)], capture_output=True, text=True)
(out / "format_check.txt").write_text(chk.stdout + chk.stderr)
(out / "manifest.json").write_text(json.dumps({
    "base": str(base), "eo_cache": str(cache), "eo_rule": "maskrcnn label OR per-frame top-1 zero-shot z>=thr",
    "thr": thr, "floor_c": c, "det_floor": det_floor, "eo_counts": counts,
    "format_check_exit_code": chk.returncode, "uploaded": False}, ensure_ascii=False, indent=1))
print(counts)
print(chk.stdout, chk.stderr)
