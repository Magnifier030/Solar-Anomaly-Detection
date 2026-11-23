import json, sys, numpy as np, cv2, collections
from pathlib import Path
from multiprocessing import Pool
sys.argv = ["raw_eval.py", "ir"]; import raw_eval as R
from rectify import rectify
from clip_crack import roc_auc_score as auc
S = Path(__file__).parent; mm = R.mm; T = R.PROJECT / "data/raw/datasets/9_30_test_set"; SZ = 224; LAB = ["G01", "G02", "G03", "G04"]
def job(kv):
    k, mods = kv; mods = [m for m in mods if m["det_score"] >= .5]
    if len(mods) < 3 or not all("raw" in m for m in mods): return k, None
    g = mm.load_image(T / k, "ir")[..., 0]; h, w = g.shape
    cr = [rectify(g, np.asarray(m["polygon"], float), SZ, inset=.03).astype(np.float32) for m in mods]; Tp = np.median([c - np.median(c) for c in cr], 0)
    tp = np.array([np.percentile(cv2.GaussianBlur(c - np.median(c) - Tp, (0, 0), 3.)[12:-12, 12:-12], 99.5) for c in cr])
    bad = [bool((np.asarray(m["polygon"])[:, 0].min() <= 5) or (np.asarray(m["polygon"])[:, 1].min() <= 5) or (np.asarray(m["polygon"])[:, 0].max() >= w - 6) or (np.asarray(m["polygon"])[:, 1].max() >= h - 6) or m["fill"] < R.FILL_MIN) for m in mods]
    return k, ((R.z(np.array([m["raw"] for m in mods]), .5) + R.z(tp, .5)).tolist(), bad)
if __name__ == "__main__":
    d = np.load(S / "hot_real.npy", allow_pickle=True).item(); G, y, bad, ng = d["G"], d["y"], d["bad"], d["ng"]; s = d["s"]["mutual+tpl"]; ok = ~bad
    def F(t, gs=LAB): return [2 * (((s >= t) & ok & (G == g)) & y).sum() / max(((s >= t) & ok & (G == g)).sum() + ng[g], 1) for g in gs]
    grid = np.arange(2, 16.01, .5); M = [np.mean(F(t)) for t in grid]; t = float(grid[int(np.argmax(M))])
    print("threshold curve (t: mean group F1):", " ".join(f"{a:.0f}:{b:.2f}" for a, b in zip(grid[::2], M[::2]))); print("best t", t, "per group", np.round(F(t), 3), "| plateau (>=best-.03):", grid[np.array(M) >= max(M) - .03].min(), "-", grid[np.array(M) >= max(M) - .03].max())
    B = np.load(S / "hot_bench.npy", allow_pickle=True).item(); rec, lab = B["rec"], B["lab"]; print("transplant benchmark, mutual+tpl at 1% clean FPR:")
    cc = lambda n: np.concatenate(rec[n]["k1 only"]) + np.concatenate(rec[n]["tpl"]); th = np.percentile(cc(0), 99)
    for n in (1, 2, 4, 8, 16):
        q = cc(n); l = np.concatenate(lab[n]); p = q >= th; print(f"  n={n}: recall {(p & l).sum() / l.sum():.2f} precision {(p & l).sum() / max(p.sum(), 1):.2f} AUC {auc(l, q):.3f}")
    ir = json.load(open(S / "test_cache_ir_module-maskrcnn-ir-all.json")); ps = json.load(open(S / "test_ir_pseudo.json"))
    with Pool(16) as P: res = dict(P.map(job, list(ir.items()), chunksize=4))
    json.dump({k: v for k, v in res.items() if v}, open(S / "hot_tpl_test.json", "w"))
    for nm, f in (("pseudo", True), ("grey", False)):
        n = [sum(a >= t and not b for a, b in zip(*res[k])) for k in res if res[k] and ps[k] == f]; print(nm, "frames", len(n), "picked", sum(n), "frames>=1", sum(x > 0 for x in n), "per-frame p50/p90/max", np.percentile(n, 50), np.percentile(n, 90), max(n))
