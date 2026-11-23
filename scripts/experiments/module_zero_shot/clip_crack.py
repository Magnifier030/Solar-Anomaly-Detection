"""WinCLIP-style text-prompt zero-shot crack score for EO modules (held-out + calibration caches).

Per rectified module crop: CLIP ViT-B/16 image embedding vs prompt ensembles (normal / cracked);
score = softmax(100*cos) prob of 'cracked'. Reports per-group AUC for: CLIP prob, its within-frame z,
the DINOv2 mutual z (cached raw), and their sum; plus crack TIoU for per-frame top-1 rules.
"""
import json, sys
from pathlib import Path
import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor

sys.argv = ["raw_eval.py", "eo"]
import raw_eval as R
from rectify import rectify

S = Path(__file__).parent
PROJECT = Path("/home/M11509204/researchs/Solar-Anomaly-Detection")
NORMAL = ["a photo of an intact solar panel", "a photo of a clean undamaged photovoltaic module",
          "an aerial photo of a normal solar panel", "a flawless solar module surface"]
CRACK = ["a photo of a cracked solar panel", "a photo of a broken photovoltaic module with shattered glass",
         "an aerial photo of a damaged solar panel with cracks", "a solar module with a broken glass surface"]


def feat(out):
    return out if torch.is_tensor(out) else out.pooler_output


def roc_auc_score(y, x):
    y = np.asarray(y, bool); x = np.asarray(x, float)
    order = np.argsort(x, kind='mergesort'); r = np.empty(len(x)); r[order] = np.arange(1, len(x) + 1)
    n1, n0 = y.sum(), (~y).sum()
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def cv_iou_truth(item, polys):
    import cv2
    truths = item["truth"].get("crack", [])
    y = np.zeros(len(polys), bool)
    for t in truths:
        tm = np.zeros((1080 // 4 + 1, 1920 // 4 + 1), np.uint8)
        cv2.fillPoly(tm, [np.int32(np.asarray(t) / 4)], 1)
        for j, p in enumerate(polys):
            pm = np.zeros_like(tm)
            cv2.fillPoly(pm, [np.int32(np.asarray(p) / 4)], 1)
            inter = (tm & pm).sum()
            if inter and inter / (tm | pm).sum() >= 0.5:
                y[j] = True
    return y


@torch.no_grad()
def main():
    cache = PROJECT / ".hf_cache"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16", cache_dir=cache / "hub").cuda().eval()
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16", cache_dir=cache / "hub")
    t = proc(text=NORMAL + CRACK, return_tensors="pt", padding=True).to("cuda")
    txt = torch.nn.functional.normalize(feat(model.get_text_features(**t)), dim=-1)
    txt = torch.stack([torch.nn.functional.normalize(txt[:4].mean(0), dim=0),
                       torch.nn.functional.normalize(txt[4:].mean(0), dim=0)])
    import importlib.util
    spec = importlib.util.spec_from_file_location("mm", PROJECT / "scripts/module_maskrcnn.py")
    mm = importlib.util.module_from_spec(spec); spec.loader.exec_module(mm)
    for tag, suffix in (("held", ""), ("cal", "_cal")):
        items = json.loads((S / f"raw_eo{suffix}.json").read_text())
        rows = []
        for n, it in enumerate(items, 1):
            mods = [m for m in it["modules"] if m["det_score"] >= 0.7 and "raw" in m]
            if len(mods) < 3:
                continue
            image = mm.load_image(mm.image_path(mm.TRAIN / "EO" / "image", it["name"]), "eo")
            crops = [rectify(image, np.asarray(m["polygon"], float), 224, inset=0.03) for m in mods]
            x = proc(images=crops, return_tensors="pt")["pixel_values"].cuda()
            img = torch.nn.functional.normalize(feat(model.get_image_features(pixel_values=x)), dim=-1)
            prob = (100 * img @ txt.T).softmax(-1)[:, 1].cpu().numpy()
            y = cv_iou_truth(it, [m["polygon"] for m in mods])
            zc = R.z(prob, 0.0)
            zd = R.z(np.array([m["raw"] for m in mods]), 0.1)
            for j in range(len(mods)):
                rows.append((it["group"], n, y[j], prob[j], zc[j], zd[j]))
            if n % 100 == 0:
                print(tag, n, len(items), flush=True)
        a = np.array([(r[2], r[3], r[4], r[5]) for r in rows], float)
        groups = np.array([r[0] for r in rows]); frames = np.array([r[1] for r in rows])
        print(f"== {tag}")
        for g in sorted(set(groups)):
            k = groups == g
            if a[k, 0].sum() == 0:
                continue
            aucs = [roc_auc_score(a[k, 0], a[k, i]) for i in (1, 2, 3)] + [roc_auc_score(a[k, 0], a[k, 2] + a[k, 3])]
            # per-frame top-1 hit rate among frames containing a crack
            hits = {}
            for name, col in (("clip", 2), ("dino", 3), ("sum", None)):
                h = tot = 0
                for f in set(frames[k]):
                    kk = k & (frames == f)
                    if a[kk, 0].sum() == 0:
                        continue
                    s = a[kk, 2] + a[kk, 3] if col is None else a[kk, col]
                    h += a[kk, 0][np.argmax(s)]; tot += 1
                hits[name] = f"{int(h)}/{tot}"
            print(f"{g}: n={k.sum()} pos={int(a[k,0].sum())} AUC clip-prob {aucs[0]:.3f} clip-z {aucs[1]:.3f} "
                  f"dino-z {aucs[2]:.3f} sum {aucs[3]:.3f} | top1 {hits}", flush=True)
        np.save(S / f"clip_crack_{tag}.npy", a)
        np.save(S / f"clip_crack_{tag}_meta.npy", np.array([(r[0], r[1]) for r in rows], dtype=object), allow_pickle=True)


if __name__ == "__main__":
    main()
