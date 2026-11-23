"""Pooled LOGO comparison: one global rule for all 4 labelled groups (like test: one threshold for all sites).
Uses held-out logits hotcl_logo_{real,syn}.npy. Reports best pooled hotspot TIoU per method + per-group at that setting."""
import json, sys, numpy as np, cv2
from pathlib import Path
sys.argv = ["raw_eval.py", "ir"]; import raw_eval as R
S = Path(__file__).parent
M = json.load(open(S / "hotcl_train_meta.json")); grp = np.array([m[1] for m in M]); kind = np.array([m[4] for m in M]); y = np.array([m[2] for m in M])
LAB = ["G01", "G02", "G03", "G04"]; LG = {t: np.load(S / f"hotcl_logo_{t}.npy") for t in ("real", "syn")}; LG["avg"] = (LG["real"] + LG["syn"]) / 2
items = {}
for f in ("raw_ir_cal.json", "raw_ir.json", "raw_ir_dense.json"):
    for it in json.load(open(S / f)): items[it["name"]] = it
IRF = R.mm.TRAIN / R.mm.FOLDER["ir"] / "image"; rows = {}
for i in np.where((kind == "c") & np.isin(grp, LAB))[0]: rows.setdefault(M[i][0], []).append(i)
frames = []
for n, idx in rows.items():
    it = items[n]; mods = [m for m in it["modules"] if m["det_score"] >= .5]; idx = np.array(sorted(idx, key=lambda i: M[i][3]))
    hgt, wid = cv2.imread(str(R.mm.image_path(IRF, n)), cv2.IMREAD_GRAYSCALE).shape
    edge = lambda p: (np.asarray(p)[:, 0].min() <= 5) | (np.asarray(p)[:, 1].min() <= 5) | (np.asarray(p)[:, 0].max() >= wid - 6) | (np.asarray(p)[:, 1].max() >= hgt - 6)
    zz = (lambda v: R.z(v, .5)) if len(mods) >= 3 else (lambda v: np.zeros(len(v)))
    v3 = zz(np.array([m["raw"] for m in mods])) + np.abs(zz(np.array([m["std"] for m in mods]))) + zz(np.array([m["pk"] for m in mods]))
    frames.append(dict(y=y[idx].astype(bool), ng=len(it["truth"].get("hotspot", [])), group=it["group"], truth=it["truth"], mods=mods, v3=v3, bad=np.array([edge(m["polygon"]) or m["fill"] < R.FILL_MIN for m in mods]),
                       **{t: LG[t][idx] for t in LG}))
def tiou(fr, key, thr, K, rel):  # fast proxy: module F1, TP = picked module matched (IoU>=.5) to a GT hotspot
    tp = p = g = 0
    for f in fr:
        s = np.where(f["bad"], -1e9, key(f)); keep = np.zeros(len(s), bool)
        if len(s) >= 3: keep[np.argsort(-s)[:K]] = True; keep &= (s >= thr) & ((s >= rel * s.max()) if rel else True)
        tp += (keep & f["y"]).sum(); p += keep.sum(); g += f["ng"]
    return 2 * tp / max(p + g, 1)
zf = lambda v: R.z(v, .5) if len(v) >= 3 else np.zeros(len(v))
methods = {"v3": (lambda f: f["v3"], np.arange(1, 8.01, .25))}
for t in ("real", "syn", "avg"):
    methods[f"abs_{t}"] = (lambda f, t=t: f[t], np.arange(-6, 8.01, .5))
    methods[f"z_{t}"] = (lambda f, t=t: zf(f[t]), np.arange(.5, 8.01, .25))
    methods[f"v3+.3abs_{t}"] = (lambda f, t=t: f["v3"] + .3 * f[t], np.arange(-2, 10.01, .25))
for name, (key, grid) in methods.items():
    best = (-1,)
    for K in (2, 4):
        for rel in (0, .7):
            for t in grid:
                v = tiou(frames, key, t, K, rel)
                if v > best[0]: best = (v, t, K, rel)
    _, t, K, rel = best
    per = {g: round(tiou([f for f in frames if f["group"] == g], key, t, K, rel), 3) for g in LAB}
    print(f"{name:16s} pooled {best[0]:.3f} thr {t:.2f} K{K} rel{rel} | {per} | mean {np.mean(list(per.values())):.3f}", flush=True)
