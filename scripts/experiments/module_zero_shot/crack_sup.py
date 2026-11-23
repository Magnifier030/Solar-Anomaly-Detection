"""Leave-one-group-out supervised crack classifier on high-res module features. No per-frame cap.
Variants: feature set x (absolute | frame-centred). Logistic regression (torch, class-balanced, L2)."""
import json, numpy as np, torch
from clip_crack import roc_auc_score as auc
D = np.load("crackfeat_train_dino.npy").astype(np.float32); T = np.load("crackfeat_train_tex.npy"); M = json.load(open("crackfeat_train_meta.json"))
names = np.array([m[0] for m in M]); grp = np.array([m[1] for m in M]); y = np.array([m[2] for m in M])
print("modules", len(y), "cracks", y.sum(), {g: int(y[grp == g].sum()) for g in sorted(set(grp)) if y[grp == g].sum()})
def centre(X):
    X = X.copy()
    for n in np.unique(names):
        k = names == n; X[k] -= np.median(X[k], 0)
    return X
T = np.log1p(np.maximum(T, 0))
SETS = {"tex": T, "dino": D, "dino+tex": np.hstack([D, T])}
def fit(X, y, wd=1e-3, ep=300):
    X = torch.tensor(X, device="cuda"); yt = torch.tensor(y, dtype=torch.float32, device="cuda")
    w = torch.zeros(X.shape[1], device="cuda", requires_grad=True); b = torch.zeros(1, device="cuda", requires_grad=True)
    pw = (len(y) - y.sum()) / max(y.sum(), 1)
    opt = torch.optim.LBFGS([w, b], max_iter=ep, line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad(); l = torch.nn.functional.binary_cross_entropy_with_logits(X @ w + b, yt, pos_weight=torch.tensor(pw, device="cuda")) + wd * (w * w).sum(); l.backward(); return l
    opt.step(closure); return w.detach(), b.detach()
res = {}
for sname, X0 in SETS.items():
    for mode in ("abs", "centred"):
        X = centre(X0) if mode == "centred" else X0.copy()
        mu, sd = X.mean(0), X.std(0) + 1e-6; X = (X - mu) / sd
        s = np.zeros(len(y))
        for g in sorted(set(grp)):
            tr = grp != g; w, b = fit(X[tr], y[tr]); s[~tr] = (torch.tensor(X[~tr], device="cuda") @ w + b).cpu().numpy()
        res[(sname, mode)] = s
        per = {g: round(auc(y[grp == g], s[grp == g]), 3) for g in ("G01", "G04", "G07")}
        # pooled held-out precision at top-N globally (all groups, no cap)
        o = np.argsort(-s); prec = {n: round(y[o[:n]].mean(), 3) for n in (50, 100, 200, 400)}
        print(f"{sname:9s} {mode:8s} AUC {per} pooled AUC {auc(y, s):.3f} | top-N precision {prec}", flush=True)
np.save("crack_sup_scores.npy", np.array([res[k] for k in res])); json.dump([list(k) for k in res], open("crack_sup_keys.json", "w"))
