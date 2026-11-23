"""CLIP B/16 crack prob for every test EO module (det>=0.7). Output test_eo_clip_all.json {key: [prob per kept module]} + crops cache."""
import json, sys
from pathlib import Path
import numpy as np, torch
sys.argv = sys.argv[:1] + ["eo"]
import raw_eval as R
import clip_crack as C
from rectify import rectify
from transformers import CLIPModel, CLIPProcessor
import importlib.util
PROJECT = Path("/home/M11509204/researchs/Solar-Anomaly-Detection"); S = Path(__file__).parent
spec = importlib.util.spec_from_file_location("mm", PROJECT / "scripts/module_maskrcnn.py"); mm = importlib.util.module_from_spec(spec); spec.loader.exec_module(mm)
TEST = PROJECT / "data/raw/datasets/9_30_test_set"
with torch.no_grad():
    hub = PROJECT / ".hf_cache/hub"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16", cache_dir=hub).cuda().eval()
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16", cache_dir=hub)
    t = proc(text=C.NORMAL + C.CRACK, return_tensors="pt", padding=True).to("cuda")
    txt = torch.nn.functional.normalize(C.feat(model.get_text_features(**t)), dim=-1)
    txt = torch.stack([torch.nn.functional.normalize(txt[:4].mean(0), dim=0), torch.nn.functional.normalize(txt[4:].mean(0), dim=0)])
    eo = json.loads((S / "test_cache_eo_module-maskrcnn-eo-all.json").read_text()); out = {}
    for i, (key, modules) in enumerate(eo.items(), 1):
        mods = [m for m in modules if m["det_score"] >= 0.7]
        if not mods: out[key] = []; continue
        image = mm.load_image(TEST / key, "eo")
        crops = [rectify(image, np.asarray(m["polygon"], float), 224, inset=0.03) for m in mods]
        x = proc(images=crops, return_tensors="pt")["pixel_values"].cuda()
        img = torch.nn.functional.normalize(C.feat(model.get_image_features(pixel_values=x)), dim=-1)
        out[key] = (100 * img @ txt.T).softmax(-1)[:, 1].cpu().numpy().round(4).tolist()
        if i % 100 == 0: print(i, flush=True)
(S / "test_eo_clip_all.json").write_text(json.dumps(out)); print("done")
