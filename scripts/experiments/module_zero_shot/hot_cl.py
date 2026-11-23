"""Hotspot CL head (SupCon + BCE, linear) on hotcl_* features. Classes: 0 normal (real G01-G04 normals + syn clean),
1 hotspot (real, group-balanced + synthetic s:*), 2 counterfactual n:*.
LOGO over G01-G04: AUC and official hotspot TIoU (rule: border excl., top-K, s>=thr, s>=rel*max; thr chosen on the other 3 groups).
Scores compared: v3 = z(raw)+|z(std)|+z(pk); CL = z(logit); v3+w*CL; CL logit absolute.
Env: REAL=1 use real positives (0 = synthetic only); EP; SEED. Writes hotcl_test_logit[_tag].npy"""
import json, os, sys, numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn, cv2
from pathlib import Path
sys.argv = ["raw_eval.py", "ir"]; import raw_eval as R
torch.cuda.set_per_process_memory_fraction(0.35)
from clip_crack import roc_auc_score as auc
S = Path(__file__).parent; REAL = os.environ.get("REAL", "1") == "1"; EP = int(os.environ.get("EP", 1500)); TAG = os.environ.get("TAG", "")
def load(p): X = np.load(S / f"{p}.npy").astype(np.float32); X[:, -8:] = np.log(X[:, -8:] + 1e-3); return X, json.load(open(S / f"{p}_meta.json"))
X, M = load("hotcl_train"); Z, ZM = load("hotcl_syn"); T, TM = load("hotcl_test")
mu, sd = X.mean(0), X.std(0) + 1e-6; X, Z, T = [(a - mu) / sd for a in (X, Z, T)]
grp = np.array([m[1] for m in M]); y = np.array([m[2] for m in M]); kind = np.array([m[4] for m in M]); zk = np.array([m[4] for m in ZM]); zg = np.array([m[1] for m in ZM])
LAB = ["G01", "G02", "G03", "G04"]; lab_ok = np.isin(grp, LAB)
zc = np.where(zk == "c", 0, np.where(np.char.startswith(zk.astype(str), "s:"), 1, 2))

class Head(nn.Module):
    def __init__(s, d): super().__init__(); s.f = nn.Sequential(nn.Dropout(.5), nn.Linear(d, 128)); s.c = nn.Linear(128, 1)
    def forward(s, x): e = Fn.normalize(s.f(x), dim=-1); return e, s.c(e).squeeze(-1) * 10
def supcon(e, lab, tau=.1):
    sim = e @ e.T / tau; n = len(lab); eye = torch.eye(n, device=e.device, dtype=torch.bool); sim = sim.masked_fill(eye, -1e9)
    pm = (lab[:, None] == lab[None]) & ~eye; lp = sim - sim.logsumexp(1, keepdim=True); return -(lp * pm).sum(1).div(pm.sum(1).clamp(min=1)).mean()
def train(hold=None, seed=int(os.environ.get("SEED", 0)), bs=128):
    torch.manual_seed(seed); g = np.random.default_rng(seed)
    tr = lab_ok & (grp != hold)
    Xs = np.vstack([X[tr], Z]); ls = np.r_[y[tr], zc]; real = np.r_[np.ones(tr.sum(), bool), np.zeros(len(Z), bool)]; gg = np.r_[grp[tr], zg]
    n0 = np.where(ls == 0)[0]; rp = np.where((ls == 1) & real)[0]; sp = np.where((ls == 1) & ~real)[0]; cf = np.where(ls == 2)[0]
    w = np.ones(len(rp))
    for u in set(gg[rp]): w[gg[rp] == u] = 1 / (gg[rp] == u).sum()
    w /= w.sum(); h = Head(Xs.shape[1]).cuda(); opt = torch.optim.AdamW(h.parameters(), 3e-4, weight_decay=.1); Xt = torch.tensor(Xs, device="cuda")
    for it in range(EP):
        parts = [g.choice(n0, bs), g.choice(sp, bs // 2 if REAL else bs), g.choice(cf, bs // 2)]
        if REAL: parts.append(g.choice(rp, bs // 2, p=w))
        b = np.concatenate(parts); lab = torch.tensor(ls[b], device="cuda")
        e, lo = h(Xt[b] + .1 * torch.randn_like(Xt[b])); loss = supcon(e, lab) + Fn.binary_cross_entropy_with_logits(lo, (lab == 1).float())
        opt.zero_grad(); loss.backward(); opt.step()
    h.eval()
    with torch.no_grad(): return lambda A: torch.cat([h(torch.tensor(A[i:i + 4096], device="cuda"))[1].detach() for i in range(0, len(A), 4096)]).cpu().numpy()

# --- frame structures for pipeline eval (train) ---
items = {}
for f in ("raw_ir_cal.json", "raw_ir.json", "raw_ir_dense.json"):
    for it in json.load(open(S / f)): items[it["name"]] = it
IRF = R.mm.TRAIN / R.mm.FOLDER["ir"] / "image"
clean = kind == "c"; rows = {}
for i in np.where(clean & lab_ok)[0]: rows.setdefault(M[i][0], []).append(i)
frames = []
for n, idx in rows.items():
    it = items[n]; mods = [m for m in it["modules"] if m["det_score"] >= .5]; idx = sorted(idx, key=lambda i: M[i][3])
    hgt, wid = cv2.imread(str(R.mm.image_path(IRF, n)), cv2.IMREAD_GRAYSCALE).shape
    def edge(p): p = np.asarray(p); return p[:, 0].min() <= 5 or p[:, 1].min() <= 5 or p[:, 0].max() >= wid - 6 or p[:, 1].max() >= hgt - 6
    v3 = R.z(np.array([m["raw"] for m in mods]), .5) + np.abs(R.z(np.array([m["std"] for m in mods]), .5)) + R.z(np.array([m["pk"] for m in mods]), .5) if len(mods) >= 3 else np.zeros(len(mods))
    frames.append(dict(name=n, group=it["group"], truth=it["truth"], mods=mods, idx=np.array(idx), v3=v3,
                       bad=np.array([edge(m["polygon"]) or m["fill"] < R.FILL_MIN for m in mods])))
def tiou(fr, sc, thr, K=2, rel=.7):
    smp = []
    for f, s in zip(fr, sc):
        s = np.where(f["bad"], -1e9, s); keep = np.zeros(len(s), bool)
        if len(s) >= 3: keep[np.argsort(-s)[:K]] = True; keep &= (s >= thr) & (s >= rel * s.max())
        pred = {}
        for m, k in zip(f["mods"], keep): pred.setdefault("hotspot" if k else "solar_panel", []).append(m["polygon"])
        smp.append({"ground_truth": f["truth"], "prediction": pred, "group": f["group"]})
    return R.f1_tiou(smp, ("solar_panel", "hotspot")).get("TIoU_hotspot", 0)
def zf(v): return R.z(v, .5) if len(v) >= 3 else np.zeros(len(v))
def evaluate(scorefn, hold, grid):
    tr = [f for f in frames if f["group"] != hold]; te = [f for f in frames if f["group"] == hold]
    best = max(grid, key=lambda t: tiou(tr, [scorefn(f) for f in tr], t)); return tiou(te, [scorefn(f) for f in te], best), best

res = {}; LOGO = np.full(len(X), np.nan, np.float32)
for hold in LAB:
    fn = train(hold); L = fn(X); m = clean & (grp == hold); LOGO[grp == hold] = L[grp == hold]
    for f in frames: f["cl"] = L[f["idx"]]
    hf = [f for f in frames if f["group"] == hold]
    a_cl = auc(y[m], L[m]); a_v3 = auc(np.concatenate([y[f["idx"]] for f in hf]), np.concatenate([f["v3"] for f in hf]))
    print(hold, "auc_cl", round(a_cl, 3), "auc_v3", round(a_v3, 3), flush=True)
np.save(S / f"hotcl_logo{TAG}.npy", LOGO)
fn = train(None); np.save(S / f"hotcl_test_logit{TAG}.npy", fn(T)); np.save(S / f"hotcl_train_logit{TAG}.npy", fn(X))
print("saved", TAG, "REAL", REAL)
