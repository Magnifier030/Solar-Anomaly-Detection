"""Relabel EO cracks whose rectified module crop has > SAT fraction of pixels >= 245 grey (sun glare)
back to solar_panel. Train cracks above 0.1: 2/352. Usage: python glare_filter.py SRC OUT SAT"""
import json, subprocess, sys, cv2, numpy as np, importlib.util
from pathlib import Path
from rectify import rectify
P = Path("/home/M11509204/researchs/Solar-Anomaly-Detection")
spec = importlib.util.spec_from_file_location("mm", P / "scripts/module_maskrcnn.py"); mm = importlib.util.module_from_spec(spec); spec.loader.exec_module(mm)
src, out, thr = Path(sys.argv[1]), Path(sys.argv[2]), float(sys.argv[3])
TEST = P / "data/raw/datasets/9_30_test_set"; KIT = P / "data/raw/evaluation_kit/author-kit_1012"
sub = json.loads((src / "submission.json").read_text()); n = kept = 0
for k, objs in sub.items():
    if "/EO/" not in k or not any(o["label"] == "crack" for o in objs):
        continue
    img = mm.load_image(TEST / k, "eo")
    for o in objs:
        if o["label"] != "crack":
            continue
        g = cv2.cvtColor(rectify(img, np.asarray(o["points"], float), 128, inset=0.03), cv2.COLOR_RGB2GRAY)
        if (g >= 245).mean() > thr:
            pts = o["points"]; o.clear(); o.update({"label": "solar_panel", "points": pts}); n += 1
        else:
            kept += 1
out.mkdir(parents=True, exist_ok=False)
path = out / "submission.json"
path.write_text(json.dumps(sub, ensure_ascii=False, allow_nan=False) + "\n")
chk = subprocess.run([sys.executable, str(KIT / "make_submission.py"), "--check", str(path)], capture_output=True, text=True)
(out / "format_check.txt").write_text(chk.stdout + chk.stderr)
m = json.loads((src / "manifest.json").read_text()); m.update(glare_filter={"sat_thr": thr, "relabelled": n, "cracks_kept": kept}, derived_from=str(src), format_check_exit_code=chk.returncode, uploaded=False)
(out / "manifest.json").write_text(json.dumps(m, ensure_ascii=False, indent=1))
print("relabelled", n, "kept", kept); print(chk.stdout, chk.stderr)
