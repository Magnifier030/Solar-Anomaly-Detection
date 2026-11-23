"""Hotspot CL features: DINOv3-L small-defect descriptor (as crack_big3) on IR module crops, contrast-normalised per frame.
Crop normalisation: c' = 128 + (c - median(c)) * 40 / s_frame, s_frame = median over frame modules of (p95-p5) -> removes whole-module offset
and per-site contrast scale (pseudo-colour vs grey).
Sets (meta rows: key, group, label, module_idx, kind):
  train: predicted modules of all train IR frames in raw_ir/raw_ir_cal/raw_ir_dense; label 1 if IoU>=.5 with a GT hotspot. positives: clean + 6 aug.
  test : predicted modules (det>=.5) of test_cache_ir_module-maskrcnn-ir-all.json
  syn  : random modules (train frames of G01-G04 + test frames, any label) -> pair (clean kind 'c', synthetic hotspot 's:<type>', counterfactual 'n:<type>')
Usage: hot_feat.py train|test|syn"""
import json, sys, os
from pathlib import Path
import numpy as np, cv2, torch
MODE = sys.argv[1]; sys.argv = ["raw_eval.py", "ir"]
import raw_eval as R
from rectify import rectify
from shapely.geometry import Polygon
import timm, glob
torch.cuda.set_per_process_memory_fraction(0.35)
mm = R.mm; S = Path(__file__).parent; TEST = R.PROJECT / "data/raw/datasets/9_30_test_set"
SZ, K, BS = 448, 8, 16
rng = np.random.default_rng(11)
_p = glob.glob(str(R.PROJECT / ".hf_cache/hub/models--timm--vit_large_patch16_dinov3.lvd1689m/snapshots/*/model.safetensors"))[0]
model = timm.create_model("vit_large_patch16_dinov3.lvd1689m", pretrained=True, pretrained_cfg_overlay=dict(file=_p), dynamic_img_size=True).cuda().to(torch.bfloat16).eval()
NPRE = model.num_prefix_tokens
MEAN = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)

def frame_crops(grey, polys):
    raw = [rectify(grey, np.asarray(p, float), SZ, inset=0.03).astype(np.float32) for p in polys]
    sf = max(np.median([np.percentile(c, 95) - np.percentile(c, 5) for c in raw]), 4.0)
    return [128 + (c - np.median(c)) * 40 / sf for c in raw]          # float, un-clipped
def u8(c): return np.repeat(np.clip(c, 0, 255).astype(np.uint8)[..., None], 3, 2)

yy, xx = np.mgrid[:SZ, :SZ].astype(np.float32)
def smooth(g=5, sig=SZ / 8):
    return cv2.GaussianBlur(cv2.resize(rng.uniform(-1, 1, (g, g)).astype(np.float32), (SZ, SZ), interpolation=cv2.INTER_CUBIC), (0, 0), sig)
def blob(cy, cx, r): return np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * r * r))
# synthetic hotspots (class: positive)
def s_spot(c):
    o = c.copy()
    for _ in range(rng.integers(1, 3)):
        r = rng.uniform(.03, .08) * SZ; cy, cx = rng.uniform(.1, .9, 2) * SZ; a = rng.uniform(25, 110)
        if rng.random() < .5: o += a * blob(cy, cx, r)
        else:  # cell-shaped: soft rectangle ~ one cell
            h, w = rng.uniform(.08, .18, 2) * SZ; m = ((abs(yy - cy) < h / 2) & (abs(xx - cx) < w / 2)).astype(np.float32)
            o += a * .7 * cv2.GaussianBlur(m, (0, 0), 3)
    return o
def s_band(c):
    w = SZ / 3; i = rng.integers(0, 3); a = rng.uniform(15, 60); m = np.zeros((SZ, SZ), np.float32)
    if rng.random() < .8: m[:, int(i * w):int((i + 1) * w)] = 1
    else: m[int(i * w):int((i + 1) * w), :] = 1
    return c + a * cv2.GaussianBlur(m, (0, 0), 4) * (1 + .15 * smooth())
def s_patch(c):
    m = (smooth(6, SZ / 10) > rng.uniform(.05, .35)).astype(np.float32); m = cv2.GaussianBlur(m, (0, 0), rng.uniform(3, 10))
    if m.mean() < .05: m = blob(*(rng.uniform(.2, .8, 2) * SZ), .15 * SZ)
    return c + rng.uniform(20, 80) * m * (1 + .3 * smooth())
# counterfactual negatives
def n_grad(c):
    a = rng.uniform(0, 2 * np.pi); g = (np.cos(a) * (xx / SZ - .5) + np.sin(a) * (yy / SZ - .5)); return c + rng.uniform(20, 60) * g
def n_trunc(c):
    o = c.copy(); f = rng.uniform(.15, .45); side = rng.integers(4); n = int(f * SZ); bg = rng.uniform(0, 60)
    if side == 0: o[:n] = bg
    elif side == 1: o[-n:] = bg
    elif side == 2: o[:, :n] = bg
    else: o[:, -n:] = bg
    return o
def n_cold(c):
    o = c.copy()
    for _ in range(rng.integers(1, 3)): o -= rng.uniform(25, 90) * blob(*(rng.uniform(.1, .9, 2) * SZ), rng.uniform(.03, .1) * SZ)
    return o
def n_edge(c):  # bright frame / seam line or neighbour spot bleeding over the border
    o = c.copy(); w = rng.uniform(.01, .03) * SZ; side = rng.integers(4); a = rng.uniform(25, 90)
    if rng.random() < .5:
        d = {0: yy, 1: SZ - 1 - yy, 2: xx, 3: SZ - 1 - xx}[side]; o += a * np.exp(-d ** 2 / (2 * w * w))
    else:
        t = rng.uniform(.1, .9) * SZ; cy, cx = {0: (-.04 * SZ, t), 1: (1.04 * SZ, t), 2: (t, -.04 * SZ), 3: (t, 1.04 * SZ)}[side]
        o += a * blob(cy, cx, rng.uniform(.05, .09) * SZ)
    return o
SYN = {"s:spot": s_spot, "s:band": s_band, "s:patch": s_patch, "n:grad": n_grad, "n:trunc": n_trunc, "n:cold": n_cold, "n:edge": n_edge}
def aug(c):  # positives: flips / 90 rot / gain / noise
    c = np.rot90(c, rng.integers(4)) if rng.random() < .5 else c
    if rng.random() < .5: c = c[:, ::-1]
    c = cv2.resize(np.ascontiguousarray(c), (SZ, SZ)); return 128 + (c - 128) * rng.uniform(.7, 1.4) + rng.normal(0, 2, c.shape)

@torch.no_grad()
def desc(crops):
    out = []
    for i in range(0, len(crops), BS):
        x = torch.from_numpy(np.stack(crops[i:i + BS])).cuda().permute(0, 3, 1, 2).half() / 255.0
        h = model.forward_features(((x.float() - MEAN) / STD).to(torch.bfloat16))[:, NPRE:].float()
        h = torch.nn.functional.normalize(h, dim=-1)
        dev = h - h.median(1, keepdim=True).values; nrm = dev.norm(dim=-1)
        v, idx = nrm.topk(K, dim=1); g = idx[..., None].expand(-1, -1, h.shape[-1])
        out.append(torch.cat([h.mean(1), h.gather(1, g).mean(1), dev.gather(1, g).mean(1), v], 1).cpu().numpy().astype(np.float16))
    return np.concatenate(out)
F, META = [], []
def flush(crops, meta):
    if crops: F.append(desc([u8(c) for c in crops])); META.extend(meta)
def iou(a, b):
    a, b = Polygon(a).buffer(0), Polygon(b).buffer(0); i = a.intersection(b).area; return i / max(a.union(b).area, 1e-6)

def train_items():
    seen = {}
    for f in ("raw_ir_cal.json", "raw_ir.json", "raw_ir_dense.json"):
        for it in json.load(open(S / f)): seen[it["name"]] = it
    return [seen[k] for k in sorted(seen)]
IRF = mm.TRAIN / mm.FOLDER["ir"] / "image"
if MODE == "train":
    items = train_items(); print("frames", len(items), flush=True)
    for i, it in enumerate(items, 1):
        mods = [m for m in it["modules"] if m["det_score"] >= .5]
        if not mods: continue
        grey = mm.load_image(mm.image_path(IRF, it["name"]), "ir")[..., 0]
        crops = frame_crops(grey, [m["polygon"] for m in mods]); hs = it["truth"].get("hotspot", [])
        cs, meta = [], []
        for j, (m, c) in enumerate(zip(mods, crops)):
            y = int(any(iou(m["polygon"], h) >= .5 for h in hs)); cs.append(c); meta.append((it["name"], it["group"], y, j, "c"))
            if y:
                for v in range(6): cs.append(aug(c)); meta.append((it["name"], it["group"], 1, j, f"a{v}"))
        flush(cs, meta)
        if i % 100 == 0: print(i, len(items), flush=True)
elif MODE == "test":
    ir = json.load(open(S / "test_cache_ir_module-maskrcnn-ir-all.json"))
    for i, (k, mods) in enumerate(ir.items(), 1):
        mods = [m for m in mods if m["det_score"] >= .5]
        if not mods: continue
        grey = mm.load_image(TEST / k, "ir")[..., 0]
        flush(frame_crops(grey, [m["polygon"] for m in mods]), [(k, "test", -1, j, "c") for j in range(len(mods))])
        if i % 100 == 0: print(i, len(ir), flush=True)
else:  # syn: per base crop -> clean + 1 synthetic positive + 1 counterfactual negative
    items = [it for it in train_items() if it["group"] in ("G01", "G02", "G03", "G04")]
    ir = json.load(open(S / "test_cache_ir_module-maskrcnn-ir-all.json"))
    src = [("tr", it) for it in items] + [("te", k) for k in ir]
    NPER = {"tr": 3, "te": 6}
    for i, (kind, x) in enumerate(src, 1):
        if kind == "tr":
            mods = [m for m in x["modules"] if m["det_score"] >= .5]; key, grp = x["name"], x["group"]
            grey = mm.load_image(mm.image_path(IRF, key), "ir")[..., 0]
        else:
            mods = [m for m in ir[x] if m["det_score"] >= .5]; key, grp = x, "test"; grey = mm.load_image(TEST / x, "ir")[..., 0]
        if len(mods) < 3: continue
        crops = frame_crops(grey, [m["polygon"] for m in mods]); cs, meta = [], []
        for j in rng.choice(len(mods), min(NPER[kind], len(mods)), replace=False):
            c = crops[j]; sp = rng.choice(["s:spot", "s:spot", "s:band", "s:patch"]); ng = rng.choice(["n:grad", "n:trunc", "n:cold", "n:edge"])
            for t, cc in (("c", c), (sp, SYN[sp](c)), (ng, SYN[ng](c))): cs.append(cc); meta.append((key, grp, -1, int(j), t))
        flush(cs, meta)
        if i % 200 == 0: print(i, len(src), flush=True)
np.save(S / f"hotcl_{MODE}.npy", np.concatenate(F)); json.dump(META, open(S / f"hotcl_{MODE}_meta.json", "w")); print("done", len(META))
