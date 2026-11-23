"""IR rebuilt from test cache: s = z(mutual,c)+|z(std,c)| (fill<0.85 -> 0), modules within 5px of border excluded,
hotspot = top-K per frame with s>=thr and s>=rel*max. EO kept from BASE. Usage: BASE OUT CACHE C THR K REL"""
import json, subprocess, sys
from pathlib import Path
import numpy as np, cv2
args = sys.argv[1:]; sys.argv = sys.argv[:1] + ["ir"]
import raw_eval as R
PROJECT = Path("/home/M11509204/researchs/Solar-Anomaly-Detection"); KIT = PROJECT / "data/raw/evaluation_kit/author-kit_1012"
TEST = PROJECT / "data/raw/datasets/9_30_test_set"
base, out, cache = Path(args[0]), Path(args[1]), Path(args[2]); c, thr, K, rel = float(args[3]), float(args[4]), int(args[5]), float(args[6])
PKW = float(args[7]); HI = float(args[8]); NDENSE = int(args[9])
REASON = "推測模組板破損或內部隱裂、電性異常，導電路徑受損使局部電阻升高，進而產生異常發熱。"
sub = json.loads((base / "submission.json").read_text()); ir = json.loads(cache.read_text())
counts = {"solar_panel": 0, "hotspot": 0}
for key, modules in ir.items():
    h, w = cv2.imread(str(TEST / key), cv2.IMREAD_UNCHANGED).shape[:2]
    mods = [m for m in modules if m["det_score"] >= 0.5]
    keep = np.zeros(len(mods), bool)
    if len(mods) >= 3 and all("raw" in m for m in mods):
        s = R.z(np.array([m["raw"] for m in mods]), c) + np.abs(R.z(np.array([m["std"] for m in mods]), c)) + PKW * R.z(np.array([m["pk"] for m in mods]), c)
        s = np.where(np.array([m["fill"] for m in mods]) < R.FILL_MIN, 0, s)
        def edge(p):
            p = np.asarray(p); return p[:, 0].min() <= 5 or p[:, 1].min() <= 5 or p[:, 0].max() >= w - 6 or p[:, 1].max() >= h - 6
        s = np.where(np.array([edge(m["polygon"]) for m in mods]), -1e9, s)
        keep[np.argsort(-s)[:K]] = True; t = HI if len(mods) >= NDENSE else thr; keep &= (s >= t) & (s >= rel * s.max())
    objects = []
    for m, k in zip(mods, keep):
        if k: objects.append({"label": "hotspot", "points": m["polygon"], "reason": REASON}); counts["hotspot"] += 1
        else: objects.append({"label": "solar_panel", "points": m["polygon"]}); counts["solar_panel"] += 1
    sub[key] = objects
out.mkdir(parents=True, exist_ok=False); path = out / "submission.json"
path.write_text(json.dumps(sub, ensure_ascii=False, allow_nan=False) + "\n")
chk = subprocess.run([sys.executable, str(KIT / "make_submission.py"), "--check", str(path)], capture_output=True, text=True)
(out / "format_check.txt").write_text(chk.stdout + chk.stderr)
(out / "manifest.json").write_text(json.dumps({"base": str(base), "ir_rule": "z(raw)+|z(std)|+PKW*z(bright peak); border5 excl, topK, s>=thr, s>=rel*max", "pk_weight": PKW, "dense_thr": HI, "dense_min_modules": NDENSE, "c": c, "thr": thr, "K": K, "rel": rel,
    "ir_counts": counts, "format_check_exit_code": chk.returncode, "uploaded": False}, ensure_ascii=False, indent=1))
print(counts, chk.returncode)
