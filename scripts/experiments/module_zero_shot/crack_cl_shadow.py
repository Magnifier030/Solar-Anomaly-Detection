"""[DINOv3 ViT-L/16 @768 variant of crack_big.py] Small-defect crack features with a large frozen backbone + geo3 / bi augmentation (from How-to-overcome-domain-shift).
Backbone: DINOv2-with-registers-LARGE, module rectified to 672 px (48x48 patches of 14 px).
Small-defect descriptor: patch tokens h (L2-normalised); dev = h - median_patch(h); top-K (K=8, ~0.35 % of the module)
patches by ||dev|| -> [mean h, mean h[topK], mean dev[topK], sorted ||dev|| topK] (3080 dims).
Augmentation (train views only):
  geo3 (domain-shift `nonegeo3`): zoom x2^U(-1,1), rotation U(-45,45) deg, shift U(-.35,.35) half-frame, keystone +-0.3, reflect border.
  bi (illumination part of IC-Light `dyncal15bi`, implemented directly): low-band luminance ratio field in [0.25, 4];
     plus glare blob / soft shadow / panel tint (background) perturbations.
Usage: crack_big.py train|test   (train: GT polygons; positives 1 clean + 6 aug views, negatives 1 view with aug p .5)"""
import json, sys, os
from pathlib import Path
import numpy as np, cv2, torch
sys.argv = ["raw_eval.py", "eo"]
import raw_eval as R
from rectify import rectify
import timm, glob
torch.cuda.set_per_process_memory_fraction(0.35)
mm = R.mm; S = Path(__file__).parent; TEST = R.PROJECT / "data/raw/datasets/9_30_test_set"
SZ, K, BS = 768, 8, 12
rng = np.random.default_rng(7)
_p = glob.glob(str(R.PROJECT / ".hf_cache/hub/models--timm--vit_large_patch16_dinov3.lvd1689m/snapshots/*/model.safetensors"))[0]
model = timm.create_model("vit_large_patch16_dinov3.lvd1689m", pretrained=True, pretrained_cfg_overlay=dict(file=_p), dynamic_img_size=True).cuda().to(torch.bfloat16).eval()
NPRE = model.num_prefix_tokens
MEAN = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)

def smooth_field(g=4):
    f = cv2.resize(rng.uniform(-1, 1, (g, g)).astype(np.float32), (SZ, SZ), interpolation=cv2.INTER_CUBIC)
    return cv2.GaussianBlur(f, (0, 0), SZ / 8)
def geo3(c):
    s = 2 ** rng.uniform(-1, 1); a = np.deg2rad(rng.uniform(-45, 45)); t = rng.uniform(-.35, .35, 2) * SZ / 2
    ctr = np.array([SZ / 2, SZ / 2])
    A = np.array([[s * np.cos(a), -s * np.sin(a)], [s * np.sin(a), s * np.cos(a)]])
    src = np.array([[0, 0], [SZ, 0], [SZ, SZ], [0, SZ]], np.float32)
    dst = (src - ctr) @ A.T + ctr + t + rng.uniform(-.3, .3, (4, 2)) * SZ / 4      # keystone: independent corner moves
    H = cv2.getPerspectiveTransform(src, dst.astype(np.float32))
    return cv2.warpPerspective(c, H, (SZ, SZ), borderMode=cv2.BORDER_REFLECT)
def bi(c):
    x = c.astype(np.float32)
    r = np.exp(smooth_field() * np.log(4.0) * rng.uniform(0.2, 0.6))[..., None]          # low-band illumination ratio (<= [0.25, 4])
    m0 = x.mean(); x = x * np.where(r > 1, 1 + (r - 1) * np.clip((235 - m0) / 120, 0.1, 1), r)   # brightening damped on bright panels
    if rng.random() < .3:                                                                  # glare blob
        yy, xx = np.mgrid[:SZ, :SZ]; cy, cx = rng.uniform(0, SZ, 2); r = rng.uniform(.08, .4) * SZ
        x += rng.uniform(60, 160) * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * r * r))[..., None]
    if rng.random() < .3:                                                                  # soft irregular shadow
        m = (smooth_field(6) > rng.uniform(-.2, .3)).astype(np.float32); m = cv2.GaussianBlur(m, (0, 0), rng.uniform(4, 20))
        x *= (1 - m * rng.uniform(.3, .7))[..., None]
    if rng.random() < .3:                                                                  # panel tint / background colour
        x = x * rng.uniform(.8, 1.2, 3) + rng.uniform(-30, 40)
    return np.clip(x, 0, 255).astype(np.uint8)
def augment(c):
    if rng.random() < .5: c = geo3(c)
    if rng.random() < .8: c = bi(c)
    return c

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

# ---- counterfactual tree-shadow negatives: clean non-crack train modules + large soft irregular dark blob (no crack) ----
def tree_shadow(c):
    x=c.astype(np.float32); f=smooth_field(rng.integers(3,7)); m=(f>rng.uniform(-.1,.4)).astype(np.float32)
    m=cv2.GaussianBlur(m,(0,0),rng.uniform(3,15)); x*= (1-m*rng.uniform(.4,.8))[...,None]
    if rng.random()<.3: yy,xx=np.mgrid[:SZ,:SZ]; cy,cx=rng.uniform(0,SZ,2); r=rng.uniform(.1,.35)*SZ; x+=rng.uniform(60,150)*np.exp(-((yy-cy)**2+(xx-cx)**2)/(2*r*r))[...,None]
    return np.clip(x,0,255).astype(np.uint8)
G = mm.frames_by_group(); names = sorted(G); rng.shuffle(names)
F, META = [], []; NMAX = int(os.environ.get("NSYN", 2400)); cnt = 0
for n in names:
    if cnt >= NMAX: break
    try: shapes = mm.load_shapes(n, "eo")
    except Exception: continue
    if any(l == "crack" for l, _ in shapes): continue
    shapes = [(j, p) for j, (l, p) in enumerate(shapes) if l == "solar_panel" and len(p) >= 4]
    if not shapes: continue
    image = mm.load_image(mm.image_path(mm.TRAIN / "EO" / "image", n), "eo"); crops, meta = [], []
    for j, p in [shapes[i] for i in rng.choice(len(shapes), min(3, len(shapes)), replace=False)]:
        crops.append(tree_shadow(np.ascontiguousarray(rectify(image, p.astype(float), SZ, inset=0.03), np.uint8))); meta.append((n, G[n], 3, j, 0))
    F.append(desc(crops)); META.extend(meta); cnt += len(crops)
    if len(META) <= 3: cv2.imwrite(str(S / "shadow_example.jpg"), crops[0])
np.save(S / "crackbig3_shadow.npy", np.concatenate(F)); json.dump(META, open(S / "crackbig3_shadow_meta.json", "w")); print("done", len(META))
