"""Real-label check (no transplant, no per-frame cap): G01-G04 GT. Threshold = 1 % FPR on normal modules of the OTHER three groups, applied to the held group.
Scores: v3, ablations (drop |z(std)|, drop pk), tpl (frame-template residual peak), and combinations."""
import json, sys, numpy as np, cv2
from pathlib import Path
from multiprocessing import Pool
sys.argv = ["raw_eval.py", "ir"]; import raw_eval as R
from rectify import rectify
from shapely.geometry import Polygon
from clip_crack import roc_auc_score as auc
S = Path(__file__).parent; mm = R.mm; IRF = mm.TRAIN / mm.FOLDER["ir"] / "image"; SZ = 224; LAB = ["G01", "G02", "G03", "G04"]
def iou(a, b): a, b = Polygon(a).buffer(0), Polygon(b).buffer(0); return a.intersection(b).area / max(a.union(b).area, 1e-6)
def job(it):
    mods = [m for m in it["modules"] if m["det_score"] >= .5]
    if len(mods) < 3: return None
    g = mm.load_image(mm.image_path(IRF, it["name"]), "ir")[..., 0]; h, w = g.shape
    cr = [rectify(g, np.asarray(m["polygon"], float), SZ, inset=.03).astype(np.float32) for m in mods]; T = np.median([c - np.median(c) for c in cr], 0)
    tp = np.array([np.percentile(cv2.GaussianBlur(c - np.median(c) - T, (0, 0), 3.)[12:-12, 12:-12], 99.5) for c in cr])
    bad = np.array([(np.asarray(m["polygon"])[:, 0].min() <= 5) or (np.asarray(m["polygon"])[:, 1].min() <= 5) or (np.asarray(m["polygon"])[:, 0].max() >= w - 6) or (np.asarray(m["polygon"])[:, 1].max() >= h - 6) or m["fill"] < R.FILL_MIN for m in mods])
    y = np.array([any(iou(m["polygon"], q) >= .5 for q in it["truth"].get("hotspot", [])) for m in mods])
    Z = lambda k: R.z(np.array([m[k] for m in mods]), .5); zt = R.z(tp, .5); a, s, p = Z("raw"), np.abs(Z("std")), Z("pk")
    o = {"v3": a + s + p, "mutual": a, "mutual+pk": a + p, "mutual+std": a + s, "pk": p, "tpl": zt, "mutual+tpl": a + zt, "v3+tpl": a + s + p + zt}
    return it["group"], y, bad, len(it["truth"].get("hotspot", [])), o
if __name__ == "__main__":
    items = {}
    for f in ("raw_ir_cal.json", "raw_ir.json", "raw_ir_dense.json"):
        for it in json.load(open(S / f)): items[it["name"]] = it
    its = [it for it in items.values() if it["group"] in LAB]
    with Pool(16) as P: res = [r for r in P.map(job, its, chunksize=4) if r]
    G = np.concatenate([[r[0]] * len(r[1]) for r in res]); y = np.concatenate([r[1] for r in res]); bad = np.concatenate([r[2] for r in res]); ng = {g: sum(r[3] for r in res if r[0] == g) for g in LAB}
    print("modules", len(y), "GT", ng)
    print("method".ljust(12), "AUC " + "/".join(LAB), "| F1 (prec,rec) at 1% FPR from other groups, no per-frame cap")
    for k in res[0][4]:
        s = np.concatenate([r[4][k] for r in res]); A = [auc(y[G == g], s[G == g]) for g in LAB]; out = []; F = []
        for g in LAB:
            t = np.percentile(s[(G != g) & ~y & ~bad], 99); p = (s >= t) & ~bad & (G == g); tp = (p & y).sum(); f = 2 * tp / max(p.sum() + ng[g], 1); F.append(f); out.append(f"{f:.2f}({tp / max(p.sum(), 1):.2f},{tp / max(ng[g], 1):.2f})")
        print(k.ljust(12), "/".join(f"{a:.3f}" for a in A), "|", " ".join(out), f"| mean F1 {np.mean(F):.3f}", flush=True)
