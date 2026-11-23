"""Route reasons: IR hotspots with a bypass-diode band shape (vertical 1/3 strip, uniform, spanning the module)
get the diode sentence; others keep their reason. Usage: SRC_DIR OUT_DIR"""
import json, sys, subprocess, numpy as np, cv2
SRC, OUT = sys.argv[1], sys.argv[2]
from band_feat import band_features, mm
P = "/home/M11509204/researchs/Solar-Anomaly-Detection/"
TEST = mm.Path(P + "data/raw/datasets/9_30_test_set"); KIT = mm.Path(P + "data/raw/evaluation_kit/author-kit_1012")
DIODE = "推測旁路二極體導通或異常，導致該電池串工作異常而發熱。"
sub = json.load(open(f"{SRC}/submission.json")); n = 0
for key, objs in sub.items():
    hs = [o for o in objs if o["label"] == "hotspot"]
    if not hs: continue
    grey = cv2.cvtColor(mm.load_image(TEST / key, "ir"), cv2.COLOR_RGB2GRAY)
    for o in hs:
        f = band_features(grey, o["points"])
        if f[0] >= 2.5 and f[1] >= 0.7 and f[4] >= 0.8 and f[5] <= 0.1:
            o["reason"] = DIODE; n += 1
out = mm.Path(OUT); out.mkdir(parents=True, exist_ok=False)
(out / "submission.json").write_text(json.dumps(sub, ensure_ascii=False, allow_nan=False) + "\n")
chk = subprocess.run([sys.executable, str(KIT / "make_submission.py"), "--check", str(out / "submission.json")], capture_output=True, text=True)
(out / "format_check.txt").write_text(chk.stdout + chk.stderr)
(out / "manifest.json").write_text(json.dumps({"base": SRC, "diode_routed": n, "diode_reason": DIODE,
    "rule": "band contrast>=2.5, cover>=0.7, in_frac>=0.8, out_frac<=0.1", "format_check_exit_code": chk.returncode, "uploaded": False}, ensure_ascii=False, indent=1))
print("routed", n); print(chk.stdout[-600:], chk.stderr[-300:])
