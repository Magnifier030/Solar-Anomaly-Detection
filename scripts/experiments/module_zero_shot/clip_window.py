"""WinCLIP-style windowed CLIP crack score vs whole-module CLIP score.

Per module: rectify to 448, CLIP 'cracked' prob for the whole crop, and for 2x2 and 3x3 windows
(each resized to 224 by the processor). Module score per variant = whole / max(2x2) / max(3x3) /
mean(whole, max 3x3). Reports per-group AUC and the gated top-1 rule (DINOv2 top-1 z>=1.5
AND score>=p) pooled over unseen held-out groups (G01/G03/G07) and over normal-only cal groups.
"""
import json, sys
from pathlib import Path
import numpy as np, torch
sys.argv = ["raw_eval.py", "eo"]
import raw_eval as R
import clip_crack as C
from rectify import rectify
from transformers import CLIPModel, CLIPProcessor
import importlib.util

S = Path(__file__).parent
PROJECT = Path("/home/M11509204/researchs/Solar-Anomaly-Detection")
spec = importlib.util.spec_from_file_location("mm", PROJECT / "scripts/module_maskrcnn.py")
mm = importlib.util.module_from_spec(spec); spec.loader.exec_module(mm)


def windows(crop, n):
    h = crop.shape[0] // n
    return [crop[i * h:(i + 1) * h, j * h:(j + 1) * h] for i in range(n) for j in range(n)]


@torch.no_grad()
def main():
    hub = PROJECT / ".hf_cache/hub"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16", cache_dir=hub).cuda().eval()
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16", cache_dir=hub)
    t = proc(text=C.NORMAL + C.CRACK, return_tensors="pt", padding=True).to("cuda")
    txt = torch.nn.functional.normalize(C.feat(model.get_text_features(**t)), dim=-1)
    txt = torch.stack([torch.nn.functional.normalize(txt[:4].mean(0), dim=0), torch.nn.functional.normalize(txt[4:].mean(0), dim=0)])

    def probs(imgs):
        out = []
        for i in range(0, len(imgs), 128):
            x = proc(images=imgs[i:i + 128], return_tensors="pt")["pixel_values"].cuda()
            f = torch.nn.functional.normalize(C.feat(model.get_image_features(pixel_values=x)), dim=-1)
            out.append((100 * f @ txt.T).softmax(-1)[:, 1].cpu())
        return torch.cat(out).numpy()

    rows = []
    for tag, suffix in (("held", ""), ("cal", "_cal")):
        for n, it in enumerate(json.loads((S / f"raw_eo{suffix}.json").read_text()), 1):
            mods = [m for m in it["modules"] if m["det_score"] >= 0.7 and "raw" in m]
            if len(mods) < 3:
                continue
            image = mm.load_image(mm.image_path(mm.TRAIN / "EO" / "image", it["name"]), "eo")
            crops = [rectify(image, np.asarray(m["polygon"], float), 448, inset=0.03) for m in mods]
            whole = probs(crops)
            w2 = probs([w for c in crops for w in windows(c, 2)]).reshape(len(mods), 4).max(1)
            w3 = probs([w for c in crops for w in windows(c, 3)]).reshape(len(mods), 9).max(1)
            y = C.cv_iou_truth(it, [m["polygon"] for m in mods])
            zd = R.z(np.array([m["raw"] for m in mods]), 0.1)
            zd = np.where(np.array([m["fill"] for m in mods]) < R.FILL_MIN, 0, zd)
            for j in range(len(mods)):
                rows.append((tag, it["group"], f"{tag}{n}", y[j], zd[j], whole[j], w2[j], w3[j], (whole[j] + w3[j]) / 2))
            if n % 100 == 0:
                print(tag, n, flush=True)
    names = ["whole", "max2x2", "max3x3", "mean(whole,3x3)"]
    tags = np.array([r[0] for r in rows]); g = np.array([r[1] for r in rows]); fr = np.array([r[2] for r in rows])
    a = np.array([r[3:] for r in rows], float); y = a[:, 0].astype(bool); zd = a[:, 1]
    np.save(S / "clip_window.npy", a); np.save(S / "clip_window_meta.npy", np.stack([tags, g, fr], 1))
    for grp in ("G01", "G07", "G04"):
        k = g == grp
        print(grp, " ".join(f"{nm} AUC {C.roc_auc_score(y[k], a[k, 2 + i]):.3f}" for i, nm in enumerate(names)))
    unseen = np.isin(g, ["G01", "G03", "G07"]); normal = ~np.isin(g, ["G01", "G03", "G07", "G04"])
    for i, nm in enumerate(names):
        col = a[:, 2 + i]
        for p in (0.4, 0.5, 0.6, 0.7, 0.8):
            stats = {}
            for name, mask in (("unseen", unseen), ("normal", normal), ("G04", g == "G04")):
                tp = fp = 0
                for f in np.unique(fr[mask]):
                    kk = np.where(fr == f)[0]
                    j = kk[np.argmax(zd[kk])]
                    if zd[j] >= 1.5 and col[j] >= p:
                        tp += y[j]; fp += not y[j]
                fn = y[mask].sum() - tp
                stats[name] = (int(tp), int(fp), int(fn), 2 * tp / max(2 * tp + fp + fn, 1))
            u = stats["unseen"]
            print(f"{nm:16s} p>={p}: unseen tp {u[0]} fp {u[1]} fn {u[2]} F1 {u[3]:.3f} | normal-group fp {stats['normal'][1]} "
                  f"| G04 F1 {stats['G04'][3]:.3f}", flush=True)


if __name__ == "__main__":
    main()
