"""Relabel IR hotspots whose polygon lies within M px of the image border to solar_panel. Usage: SRC OUT M"""
import json, sys, subprocess
from pathlib import Path
import numpy as np, cv2
SRC, OUT, M = Path(sys.argv[1]), Path(sys.argv[2]), float(sys.argv[3])
P = Path("/home/M11509204/researchs/Solar-Anomaly-Detection"); TEST = P / "data/raw/datasets/9_30_test_set"
KIT = P / "data/raw/evaluation_kit/author-kit_1012"
sub = json.loads((SRC / "submission.json").read_text()); n = kept = 0; sizes = {}
for key, objs in sub.items():
    if not any(o["label"] == "hotspot" for o in objs): continue
    h, w = cv2.imread(str(TEST / key), cv2.IMREAD_UNCHANGED).shape[:2]; sizes[(w, h)] = sizes.get((w, h), 0) + 1
    for o in objs:
        if o["label"] != "hotspot": continue
        p = np.asarray(o["points"], float)
        if p[:, 0].min() <= M or p[:, 1].min() <= M or p[:, 0].max() >= w - 1 - M or p[:, 1].max() >= h - 1 - M:
            o["label"] = "solar_panel"; o.pop("reason", None); n += 1
        else: kept += 1
OUT.mkdir(parents=True, exist_ok=False)
(OUT / "submission.json").write_text(json.dumps(sub, ensure_ascii=False, allow_nan=False) + "\n")
chk = subprocess.run([sys.executable, str(KIT / "make_submission.py"), "--check", str(OUT / "submission.json")], capture_output=True, text=True)
(OUT / "format_check.txt").write_text(chk.stdout + chk.stderr)
(OUT / "manifest.json").write_text(json.dumps({"base": str(SRC), "rule": f"IR hotspot within {M}px of border -> solar_panel",
    "relabelled": n, "kept": kept, "format_check_exit_code": chk.returncode, "uploaded": False}, ensure_ascii=False, indent=1))
print("sizes", sizes, "relabelled", n, "kept", kept); print(chk.stdout[-500:], chk.stderr[-300:])
