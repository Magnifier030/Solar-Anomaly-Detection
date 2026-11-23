"""Relabel cracks that sit on a Mask R-CNN class-2 module (all 28 scored 0 on the platform) to solar_panel. Usage: SRC OUT"""
import json, sys, subprocess
from pathlib import Path
SRC, OUT = Path(sys.argv[1]), Path(sys.argv[2])
P = Path("/home/M11509204/researchs/Solar-Anomaly-Detection"); KIT = P / "data/raw/evaluation_kit/author-kit_1012"
cache = json.load(open(Path(__file__).parent / "test_cache_eo_module-maskrcnn-eo-all.json"))
det = {k: {json.dumps(m["polygon"]) for m in v if m["det_label"] == 2} for k, v in cache.items()}
sub = json.loads((SRC / "submission.json").read_text()); n = kept = 0
for k, objs in sub.items():
    for o in objs:
        if o["label"] == "crack":
            if json.dumps(o["points"]) in det.get(k, ()): o["label"] = "solar_panel"; o.pop("reason", None); n += 1
            else: kept += 1
print("det cracks in cache", sum(map(len, det.values())), "relabelled", n, "kept", kept)
OUT.mkdir(parents=True, exist_ok=False); (OUT / "submission.json").write_text(json.dumps(sub, ensure_ascii=False, allow_nan=False) + "\n")
chk = subprocess.run([sys.executable, str(KIT / "make_submission.py"), "--check", str(OUT / "submission.json")], capture_output=True, text=True)
(OUT / "format_check.txt").write_text(chk.stdout + chk.stderr)
(OUT / "manifest.json").write_text(json.dumps({"base": str(SRC), "rule": "drop cracks on Mask R-CNN class-2 modules (0/28 on platform 2026-11-05)", "relabelled": n, "kept": kept, "format_check_exit_code": chk.returncode, "uploaded": False}, indent=1))
print(chk.stdout[-400:])
