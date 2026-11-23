"""Add-only cracks: every test EO module (det>=0.7) with CLIP B/16 crack prob >= P and crop mean grey <= BRIGHT -> crack. IR/other untouched.
Usage: BASE OUT P BRIGHT"""
import json, subprocess, sys
from pathlib import Path
import numpy as np, cv2
from rectify import rectify
P = Path("/home/M11509204/researchs/Solar-Anomaly-Detection"); TEST = P / "data/raw/datasets/9_30_test_set"; KIT = P / "data/raw/evaluation_kit/author-kit_1012"
base, out = Path(sys.argv[1]), Path(sys.argv[2]); pthr, bright = float(sys.argv[3]), float(sys.argv[4])
REASON = "模組板破裂，破損處的導電路徑受損。"
sub = json.loads((base / "submission.json").read_text())
eo = json.load(open("test_cache_eo_module-maskrcnn-eo-all.json")); clip = json.load(open("test_eo_clip_all.json"))
cnt = {"added": 0, "already": 0, "bright_skip": 0, "nomatch": 0}
for k, mods in eo.items():
    mods = [m for m in mods if m["det_score"] >= 0.7]
    cand = [m for m, p in zip(mods, clip[k]) if p >= pthr]
    if not cand: continue
    g = cv2.imread(str(TEST / k), cv2.IMREAD_GRAYSCALE)
    idx = {json.dumps(o["points"]): o for o in sub[k]}
    for m in cand:
        o = idx.get(json.dumps(m["polygon"]))
        if o is None: cnt["nomatch"] += 1; continue
        if o["label"] == "crack": cnt["already"] += 1; continue
        if rectify(g, np.asarray(m["polygon"], float), 64, inset=0.05).mean() > bright: cnt["bright_skip"] += 1; continue
        o["label"] = "crack"; o["reason"] = REASON; cnt["added"] += 1
cnt["cracks_total"] = sum(o["label"] == "crack" for v in sub.values() for o in v)
cnt["hotspots_total"] = sum(o["label"] == "hotspot" for v in sub.values() for o in v)
out.mkdir(parents=True, exist_ok=False); path = out / "submission.json"
path.write_text(json.dumps(sub, ensure_ascii=False, allow_nan=False) + "\n")
chk = subprocess.run([sys.executable, str(KIT / "make_submission.py"), "--check", str(path)], capture_output=True, text=True)
(out / "format_check.txt").write_text(chk.stdout + chk.stderr)
(out / "manifest.json").write_text(json.dumps({"base": str(base), "eo_rule": "add crack where CLIP B/16 prob>=P and crop mean grey<=BRIGHT", "P": pthr, "bright": bright, "counts": cnt, "format_check_exit_code": chk.returncode, "uploaded": False}, ensure_ascii=False, indent=1))
print(cnt, chk.returncode)
