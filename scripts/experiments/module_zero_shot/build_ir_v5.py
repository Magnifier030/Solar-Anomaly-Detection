"""IR v5: v3 score; separate (K,thr,rel) for dense (>=NDENSE modules) and sparse frames. Reasons for hotspots already in BASE are copied
(keeps diode routing); new hotspots get the default reason. EO kept from BASE.
Usage: BASE OUT CACHE NDENSE Kd THRd RELd Ks THRs RELs"""
import json, subprocess, sys
from pathlib import Path
import numpy as np, cv2
args = sys.argv[1:]; sys.argv = sys.argv[:1] + ["ir"]
import raw_eval as R
PROJECT = Path("/home/M11509204/researchs/Solar-Anomaly-Detection"); KIT = PROJECT / "data/raw/evaluation_kit/author-kit_1012"
TEST = PROJECT / "data/raw/datasets/9_30_test_set"
base, out, cache = Path(args[0]), Path(args[1]), Path(args[2]); ND = int(args[3])
Kd, Td, Rd, Ks, Ts, Rs = int(args[4]), float(args[5]), float(args[6]), int(args[7]), float(args[8]), float(args[9])
C = 0.5; PKW = 1.0
REASON = "推測模組板破損或內部隱裂、電性異常，導電路徑受損使局部電阻升高，進而產生異常發熱。"
sub = json.loads((base / "submission.json").read_text()); ir = json.loads(cache.read_text())
old = {}
for key, objs in sub.items():
    for o in objs:
        if o["label"] == "hotspot": old[(key, json.dumps(o["points"]))] = o["reason"]
counts = {"solar_panel": 0, "hotspot": 0, "hot_dense": 0, "hot_sparse": 0, "kept_from_base": 0}
for key, modules in ir.items():
    h, w = cv2.imread(str(TEST / key), cv2.IMREAD_UNCHANGED).shape[:2]
    mods = [m for m in modules if m["det_score"] >= 0.5]
    keep = np.zeros(len(mods), bool); dense = len(mods) >= ND
    if len(mods) >= 3 and all("raw" in m for m in mods):
        s = R.z(np.array([m["raw"] for m in mods]), C) + np.abs(R.z(np.array([m["std"] for m in mods]), C)) + PKW * R.z(np.array([m["pk"] for m in mods]), C)
        s = np.where(np.array([m["fill"] for m in mods]) < R.FILL_MIN, 0, s)
        def edge(p):
            p = np.asarray(p); return p[:, 0].min() <= 5 or p[:, 1].min() <= 5 or p[:, 0].max() >= w - 6 or p[:, 1].max() >= h - 6
        s = np.where(np.array([edge(m["polygon"]) for m in mods]), -1e9, s)
        K, t, rel = (Kd, Td, Rd) if dense else (Ks, Ts, Rs)
        keep[np.argsort(-s)[:K]] = True; keep &= (s >= t) & (s >= rel * s.max())
    objects = []
    for m, k in zip(mods, keep):
        if k:
            r = old.get((key, json.dumps(m["polygon"])))
            counts["kept_from_base"] += r is not None
            objects.append({"label": "hotspot", "points": m["polygon"], "reason": r or REASON}); counts["hotspot"] += 1
            counts["hot_dense" if dense else "hot_sparse"] += 1
        else: objects.append({"label": "solar_panel", "points": m["polygon"]}); counts["solar_panel"] += 1
    sub[key] = objects
out.mkdir(parents=True, exist_ok=False); path = out / "submission.json"
path.write_text(json.dumps(sub, ensure_ascii=False, allow_nan=False) + "\n")
chk = subprocess.run([sys.executable, str(KIT / "make_submission.py"), "--check", str(path)], capture_output=True, text=True)
(out / "format_check.txt").write_text(chk.stdout + chk.stderr)
(out / "manifest.json").write_text(json.dumps({"base": str(base), "ir_rule": "v3 score; dense/sparse split", "ndense": ND, "dense": [Kd, Td, Rd], "sparse": [Ks, Ts, Rs],
    "ir_counts": counts, "format_check_exit_code": chk.returncode, "uploaded": False}, ensure_ascii=False, indent=1))
print(counts, chk.returncode)
