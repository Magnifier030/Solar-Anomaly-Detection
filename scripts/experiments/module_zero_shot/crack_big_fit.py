"""Fit / validate the large-model small-defect crack classifier (features from crack_big.py).
LOGO over crack groups (G01, G04, G07); held-out metrics on clean views; robustness = AUC of clean positives vs
augmented (geo3/bi) held-out negatives. Test: H-added share in global top-N (platform-proven non-cracks) as a domain check.
Usage: crack_big_fit.py [test]"""
import json, sys, os, numpy as np, torch
PFX = os.environ.get("PFX", "crackbig")
from clip_crack import roc_auc_score as auc
from crack_sup_lib import fit
O = "/home/M11509204/researchs/Solar-Anomaly-Detection/outputs/submissions/"
X = np.load(PFX + "_train.npy").astype(np.float32); M = json.load(open(PFX + "_train_meta.json"))
grp = np.array([m[1] for m in M]); y = np.array([m[2] for m in M]); view = np.array([m[4] for m in M])
X[:, -8:] = np.log(X[:, -8:] + 1e-3)
mu, sd = X.mean(0), X.std(0) + 1e-6; X = (X - mu) / sd
print("rows", len(y), "pos rows", y.sum(), "pos modules", (y & (view == 0)).sum())
clean = (y == 0) & (view == 0) | (y == 1) & (view == 0)
def score(w, b, Z): return (torch.tensor(Z, device="cuda") @ w + b).cpu().numpy()
s = np.full(len(y), np.nan)
for g in ("G01", "G04", "G07"):
    tr = grp != g; w, b = fit(X[tr], y[tr]); s[~tr] = score(w, b, X[~tr])
    k = (grp == g)
    pc = k & (y == 1) & (view == 0); nc = k & (y == 0) & (view == 0); na = k & (y == 0) & (view == 1)
    print(f"{g}: AUC clean {auc(np.r_[np.ones(pc.sum()), np.zeros(nc.sum())], np.r_[s[pc], s[nc]]):.3f} | "
          f"pos vs aug-neg {auc(np.r_[np.ones(pc.sum()), np.zeros(na.sum())], np.r_[s[pc], s[na]]):.3f}", flush=True)
# other (crack-free) groups: held-out by fitting without them is not needed for AUC; report pooled over crack groups
k = np.isin(grp, ["G01", "G04", "G07"]) & (view == 0)
o = np.argsort(-s[k]); yy = y[k][o]
print("pooled crack-group AUC", round(auc(y[k], s[k]), 3), "top-N precision", {n: round(yy[:n].mean(), 3) for n in (50, 100, 200)})
if len(sys.argv) > 1:
    T = np.load(PFX + "_test.npy").astype(np.float32); TM = json.load(open(PFX + "_test_meta.json"))
    T[:, -8:] = np.log(T[:, -8:] + 1e-3); T = (T - mu) / sd
    eo = json.load(open("test_cache_eo_module-maskrcnn-eo-all.json"))
    pos = {}
    for kk, mods in eo.items():
        for j, m in enumerate([m for m in mods if m["det_score"] >= 0.7]): pos[(kk, json.dumps(m["polygon"]))] = j
    def cr(sub): return {(kk, json.dumps(o["points"])) for kk, v in sub.items() for o in v if o["label"] == "crack"}
    Gc = cr(json.load(open(O + "ir-v6union-sp7-diode-addcrack-zlow2-20261107/submission.json")))
    Hc = cr(json.load(open(O + "G-clipall70-crack-20261107/submission.json")))
    dom = json.load(open("test_eo_domain.json"))
    idx = {(m[0], m[3]): i for i, m in enumerate(TM)}
    Hneg = [idx[(kk, pos[(kk, p)])] for kk, p in Hc - Gc if (kk, p) in pos]
    D17 = [idx[(kk, pos[(kk, p)])] for kk, p in Gc if (kk, p) in pos and dom[kk][0] in ("G01", "G07")]
    G223 = [idx[(kk, pos[(kk, p)])] for kk, p in Gc if (kk, p) in pos and dom[kk][0] not in ("G01", "G07")]
    print("test rows", len(TM), "H-added", len(Hneg), "zero-hit G01/G07", len(D17), "G remaining", len(G223))
    for name, extra in (("train only", []), ("train + platform negatives", Hneg + D17)):
        Xa = np.vstack([X, T[extra]]); ya = np.r_[y, np.zeros(len(extra), int)]
        w, b = fit(Xa, ya); st = score(w, b, T); o = np.argsort(-st); rank = np.empty(len(st), int); rank[o] = np.arange(len(st))
        hs = set(Hneg); gs = set(G223)
        print(name, {n: (sum(i in hs for i in o[:n]), sum(i in gs for i in o[:n])) for n in (100, 200, 500)}, "(H-added, G-remaining) in top-N;",
              "median rank of G-remaining", int(np.median(rank[G223])), "of", len(st))
        np.save(PFX + "_test_scores_" + ("plain" if not extra else "pneg") + ".npy", st)
