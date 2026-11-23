"""Submission: IR + reasons from BASE; EO crack = Mask R-CNN label OR
(per-frame DINOv2 top-1 with floored z >= THR AND CLIP 'cracked' prob >= P).
Usage: python build_eo_clipgate.py BASE OUT CACHE THR C P
"""
import json, subprocess, sys
from pathlib import Path
import numpy as np, torch
args = sys.argv[1:]; sys.argv = sys.argv[:1] + ["eo"]
import raw_eval as R
import clip_crack as C
from rectify import rectify
from transformers import CLIPModel, CLIPProcessor
import importlib.util
PROJECT = Path("/home/M11509204/researchs/Solar-Anomaly-Detection")
spec = importlib.util.spec_from_file_location("mm", PROJECT / "scripts/module_maskrcnn.py")
mm = importlib.util.module_from_spec(spec); spec.loader.exec_module(mm)
KIT = PROJECT / "data/raw/evaluation_kit/author-kit_1012"; TEST = PROJECT / "data/raw/datasets/9_30_test_set"
base, out, cache = Path(args[0]), Path(args[1]), Path(args[2]); thr, c, pgate = map(float, args[3:6])
REASON = "模組板破裂，破損處的導電路徑受損。"

with torch.no_grad():
    hub = PROJECT / ".hf_cache/hub"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16", cache_dir=hub).cuda().eval()
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16", cache_dir=hub)
    t = proc(text=C.NORMAL + C.CRACK, return_tensors="pt", padding=True).to("cuda")
    txt = torch.nn.functional.normalize(C.feat(model.get_text_features(**t)), dim=-1)
    txt = torch.stack([torch.nn.functional.normalize(txt[:4].mean(0), dim=0), torch.nn.functional.normalize(txt[4:].mean(0), dim=0)])
    sub = json.loads((base / "submission.json").read_text()); eo = json.loads(cache.read_text())
    counts = {"solar_panel": 0, "crack": 0, "from_zs_gated": 0, "from_det": 0, "top1_candidates": 0}
    for key, modules in eo.items():
        mods = [m for m in modules if m["det_score"] >= 0.7]
        gated = -1
        if len(mods) >= 3 and all("raw" in m for m in mods):
            s = R.z(np.array([m["raw"] for m in mods]), c)
            s = np.where(np.array([m["fill"] for m in mods]) < R.FILL_MIN, 0, s)
            top = int(np.argmax(s))
            if s[top] >= thr:
                counts["top1_candidates"] += 1
                image = mm.load_image(TEST / key, "eo")
                crop = rectify(image, np.asarray(mods[top]["polygon"], float), 224, inset=0.03)
                x = proc(images=[crop], return_tensors="pt")["pixel_values"].cuda()
                img = torch.nn.functional.normalize(C.feat(model.get_image_features(pixel_values=x)), dim=-1)
                if float((100 * img @ txt.T).softmax(-1)[0, 1]) >= pgate:
                    gated = top
        objects = []
        for j, m in enumerate(mods):
            by_zs, by_det = j == gated, m["det_label"] == 2
            if by_zs or by_det:
                objects.append({"label": "crack", "points": m["polygon"], "reason": REASON})
                counts["crack"] += 1; counts["from_zs_gated"] += int(by_zs); counts["from_det"] += int(by_det)
            else:
                objects.append({"label": "solar_panel", "points": m["polygon"]}); counts["solar_panel"] += 1
        sub[key] = objects
out.mkdir(parents=True, exist_ok=False)
path = out / "submission.json"
path.write_text(json.dumps(sub, ensure_ascii=False, allow_nan=False) + "\n")
chk = subprocess.run([sys.executable, str(KIT / "make_submission.py"), "--check", str(path)], capture_output=True, text=True)
(out / "format_check.txt").write_text(chk.stdout + chk.stderr)
(out / "manifest.json").write_text(json.dumps({"base": str(base), "eo_rule": "maskrcnn OR (dino top1 z>=thr AND clip p>=P)",
    "thr": thr, "floor_c": c, "clip_p": pgate, "eo_counts": counts, "format_check_exit_code": chk.returncode, "uploaded": False}, ensure_ascii=False, indent=1))
print(counts); print(chk.stdout, chk.stderr)
