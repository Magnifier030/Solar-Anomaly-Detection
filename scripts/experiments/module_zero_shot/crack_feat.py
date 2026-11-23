"""High-res module features for supervised crack learning.
Train: GT polygons (solar_panel=0, crack=1) of every train EO frame. Test: Mask R-CNN cache modules (det>=0.7).
Per module: rectify 448px (inset .03) -> DINOv2-reg-base patch tokens (32x32) -> [mean, max] (1536) ;
texture: grey Laplacian energy, high-pass std at 3 scales, fraction of high-pass outliers (6).
Usage: crack_feat.py train|test"""
import json, sys
from pathlib import Path
import numpy as np, cv2, torch
MODE = sys.argv[1]; sys.argv = ["raw_eval.py", "eo"]
import raw_eval as R
from rectify import rectify
from zs_score import _model
mm = R.mm; S = Path(__file__).parent; TEST = R.PROJECT / "data/raw/datasets/9_30_test_set"
SZ = 448
def tex(c):
    g = cv2.cvtColor(c, cv2.COLOR_RGB2GRAY).astype(np.float32)
    out = [float(np.abs(cv2.Laplacian(g, cv2.CV_32F)).mean())]
    for s in (1.0, 2.0, 4.0):
        hp = g - cv2.GaussianBlur(g, (0, 0), s); out.append(float(hp.std()))
    hp = g - cv2.GaussianBlur(g, (0, 0), 2.0); sd = hp.std() + 1e-6
    out += [float((hp > 3 * sd).mean()), float((hp < -3 * sd).mean())]
    return out
@torch.no_grad()
def emb(crops):
    model = _model()
    mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1); std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)
    res = []
    for i in range(0, len(crops), 16):
        x = torch.from_numpy(np.stack(crops[i:i + 16])).cuda().permute(0, 3, 1, 2).float() / 255.0
        with torch.autocast("cuda", dtype=torch.float16):
            h = model(pixel_values=(x - mean) / std).last_hidden_state[:, 1 + model.config.num_register_tokens:].float()
        h = torch.nn.functional.normalize(h, dim=-1)
        res.append(torch.cat([h.mean(1), h.max(1).values], 1).cpu().numpy().astype(np.float16))
    return np.concatenate(res)
F, T, META = [], [], []
if MODE == "train":
    G = mm.frames_by_group()
    for i, n in enumerate(sorted(G), 1):
        try: shapes = mm.load_shapes(n, "eo")
        except Exception: continue
        shapes = [(l, p) for l, p in shapes if l in ("solar_panel", "crack") and len(p) >= 4]
        if not shapes: continue
        image = mm.load_image(mm.image_path(mm.TRAIN / "EO" / "image", n), "eo")
        crops = [np.ascontiguousarray(rectify(image, p.astype(float), SZ, inset=0.03), np.uint8) for _, p in shapes]
        F.append(emb(crops)); T += [tex(c) for c in crops]
        META += [(n, G[n], int(l == "crack"), j) for j, (l, _) in enumerate(shapes)]
        if i % 100 == 0: print(i, len(G), flush=True)
else:
    eo = json.loads((S / "test_cache_eo_module-maskrcnn-eo-all.json").read_text())
    for i, (k, mods) in enumerate(eo.items(), 1):
        mods = [m for m in mods if m["det_score"] >= 0.7]
        if not mods: continue
        image = mm.load_image(TEST / k, "eo")
        crops = [np.ascontiguousarray(rectify(image, np.asarray(m["polygon"], float), SZ, inset=0.03), np.uint8) for m in mods]
        F.append(emb(crops)); T += [tex(c) for c in crops]; META += [(k, "test", -1, j) for j in range(len(mods))]
        if i % 100 == 0: print(i, len(eo), flush=True)
np.save(S / f"crackfeat_{MODE}_dino.npy", np.concatenate(F)); np.save(S / f"crackfeat_{MODE}_tex.npy", np.array(T, np.float32))
json.dump(META, open(S / f"crackfeat_{MODE}_meta.json", "w")); print("done", len(META))
