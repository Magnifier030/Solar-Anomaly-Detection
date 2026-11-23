"""LOGO precision-vs-threshold for dino-abs; final model on all train; score test modules."""
import json, numpy as np, torch
from crack_sup_lib import fit
from clip_crack import roc_auc_score as auc
D = np.load("crackfeat_train_dino.npy").astype(np.float32); M = json.load(open("crackfeat_train_meta.json"))
grp = np.array([m[1] for m in M]); y = np.array([m[2] for m in M])
Dt = np.load("crackfeat_test_dino.npy").astype(np.float32); Mt = json.load(open("crackfeat_test_meta.json"))
mu, sd = D.mean(0), D.std(0) + 1e-6; X = (D - mu) / sd; Xt = (Dt - mu) / sd
s = np.zeros(len(y))
for g in sorted(set(grp)):
    tr = grp != g; w, b = fit(X[tr], y[tr]); s[~tr] = (torch.tensor(X[~tr], device="cuda") @ w + b).cpu().numpy()
print("LOGO pooled AUC", round(auc(y, s), 3))
print("held-out score threshold -> n, precision, recall, per-group n>thr (non-crack groups = FP sites)")
for t in (0, 2, 4, 6, 8, 10, 12, 15):
    k = s >= t; fpg = {g: int(k[grp == g].sum()) for g in sorted(set(grp)) if k[grp == g].sum()}
    print(f"thr {t:3d}: n {k.sum():4d} prec {y[k].mean() if k.sum() else 0:.3f} rec {y[k].sum() / y.sum():.3f} {fpg}")
w, b = fit(X, y); st = (torch.tensor(Xt, device="cuda") @ w + b).cpu().numpy(); sa = (torch.tensor(X, device="cuda") @ w + b).cpu().numpy()
print("final model train-fit AUC", round(auc(y, sa), 3))
for t in (0, 2, 4, 6, 8, 10, 12, 15):
    k = st >= t; fr = {}
    for m, kk in zip(Mt, k):
        if kk: fr[m[0]] = fr.get(m[0], 0) + 1
    print(f"TEST thr {t:3d}: modules {k.sum():5d} frames {len(fr):4d} max/frame {max(fr.values()) if fr else 0}")
np.save("crack_test_scores.npy", st); np.save("crack_logo_scores.npy", s)
