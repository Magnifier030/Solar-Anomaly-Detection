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
    np.save(S / "hot_real.npy", {"G": G, "y": y, "bad": bad, "ng": ng, "s": {k: np.concatenate([r[4][k] for r in res]) for k in res[0][4]}}, allow_pickle=True)
    def gpd_thr(x, risk):  # peaks-over-threshold (SPOT-style): GPD fitted by moments to exceedances over q90
        u = np.percentile(x, 90); e = x[x > u] - u; m, v = e.mean(), e.var(); xi = np.clip(.5 * (1 - m * m / max(v, 1e-9)), -.5, .45); sg = max(.5 * m * (m * m / max(v, 1e-9) + 1), 1e-6); q = risk / .1
        return u + sg / xi * (q ** (-xi) - 1) if abs(xi) > 1e-4 else u - sg * np.log(q)
    def ev(k, norm, label):
        s0 = np.concatenate([r[4][k] for r in res]); s = s0.copy(); ok = ~bad
        for g in LAB:
            x = s0[(G == g) & ok]; med = np.median(x)
            if norm == "mad": s[G == g] = (s0[G == g] - med) / max(1.4826 * np.median(np.abs(x - med)), 1e-6)
            if norm == "tail": s[G == g] = (s0[G == g] - med) / max(np.percentile(x, 99) - med, 1e-6)
        out = []; F = []
        for g in LAB:
            if norm == "pot":
                t = gpd_thr(s0[(G == g) & ok], 3e-3); p = (s0 >= t) & ok & (G == g)
            else:
                grid = np.percentile(s[(G != g) & ok], np.linspace(90, 99.95, 60))
                def f(t):
                    v = []
                    for h in LAB:
                        if h == g: continue
                        q = (s >= t) & ok & (G == h); v.append(2 * (q & y).sum() / max(q.sum() + ng[h], 1))
                    return np.mean(v)
                t = max(grid, key=f); p = (s >= t) & ok & (G == g)
            tp = (p & y).sum(); fv = 2 * tp / max(p.sum() + ng[g], 1); F.append(fv); out.append(f"{fv:.2f}({tp / max(p.sum(), 1):.2f},{tp / max(ng[g], 1):.2f})")
        print(f"{k:11s} {label:9s}", " ".join(out), f"| mean F1 {np.mean(F):.3f}", flush=True)
    print("F1(prec,rec) G01 G02 G03 G04; thresholds never use the held group labels; no per-frame cap")
    for k in ("v3", "mutual+pk", "mutual+tpl", "v3+tpl"):
        for n, l in (("none", "raw"), ("mad", "site-MAD"), ("tail", "site-tail"), ("pot", "POT")): ev(k, n, l)
