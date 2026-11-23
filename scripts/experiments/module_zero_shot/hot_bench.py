"""Multi-hotspot benchmark (no test labels, no platform): transplant REAL GT hotspot residuals (G01-G04) into n random modules of real dense
train frames that have no hotspot labels (G05/G06/G08/G09/G12), n in {0,1,2,4,8,16}. Measures whether each score still finds them as n grows.
Scores (module-level, within-frame robust z, c=.5):
  k1    current DINOv2 mutual score (1st nearest other module)   -> masked when several modules share a defect?
  iv10/iv30/q30  MuSc-style interval average over the lowest 10/30 % of other modules / 30 % quantile
  std, pk  current intensity terms;  v3 = z(k1)+|z(std)|+z(pk)
  tpl   frame-template residual: pixelwise median of all rectified modules in the frame, peak of the positive residual
Threshold-free of per-frame counts: threshold fixed at 1 % module FPR on n=0 frames; reports recall and precision by n."""
import json, sys, numpy as np, cv2, torch
from pathlib import Path
sys.argv = ["raw_eval.py", "ir"]; import raw_eval as R
from rectify import rectify
from zs_score import _tokens, FILL_MIN, fill_ratio
from shapely.geometry import Polygon
torch.cuda.set_per_process_memory_fraction(0.35)
S = Path(__file__).parent; mm = R.mm; IRF = mm.TRAIN / mm.FOLDER["ir"] / "image"; SZ = 224; rng = np.random.default_rng(5)
NS = (0, 1, 2, 4, 8, 16); MAXF = 45
def grey_crops(g, polys): return [rectify(g, np.asarray(p, float), SZ, inset=.03).astype(np.float32) for p in polys]
def contrast(cr): return max(np.median([np.percentile(c, 95) - np.percentile(c, 5) for c in cr]), 3.)
# ---------- donors: residual of a real GT module against the pixelwise median of its frame-mates
items = {}
for f in ("raw_ir_cal.json", "raw_ir.json", "raw_ir_dense.json"):
    for it in json.load(open(S / f)): items[it["name"]] = it
def iou(a, b): a, b = Polygon(a).buffer(0), Polygon(b).buffer(0); return a.intersection(b).area / max(a.union(b).area, 1e-6)
don = {g: [] for g in ("G01", "G02", "G03", "G04")}
for g in don:
    names = [n for n, it in items.items() if it["group"] == g and it["truth"].get("hotspot")]
    for n in rng.permutation(names)[:40]:
        it = items[n]; mods = [m for m in it["modules"] if m["det_score"] >= .5]
        if len(mods) < 4: continue
        im = mm.load_image(mm.image_path(IRF, n), "ir")[..., 0]; cr = grey_crops(im, [m["polygon"] for m in mods]); sf = contrast(cr)
        hit = [j for j, m in enumerate(mods) if any(iou(m["polygon"], h) >= .5 for h in it["truth"]["hotspot"])]
        if not hit: continue
        rest = [c - np.median(c) for j, c in enumerate(cr) if j not in hit]
        if len(rest) < 3: continue
        tpl = np.median(rest, 0)
        for j in hit:
            r = cv2.GaussianBlur(cr[j] - np.median(cr[j]) - tpl, (0, 0), 1.); r = np.clip(r - np.percentile(np.abs(r), 60), 0, None) / sf
            if r.max() * sf > 8: don[g].append(r.astype(np.float32))
print("donors", {g: len(v) for g, v in don.items()}, flush=True)
DG = [g for g in don if don[g]]
# ---------- scores
@torch.no_grad()
def mutual_all(tok):
    m, p, d = tok.shape; k = max(1, int(round(p * .10))); out = {x: np.zeros(m) for x in ("k1", "iv10", "iv30", "q30")}
    for i in range(m):
        o = torch.cat([tok[:i], tok[i + 1:]]); D = (2 - 2 * tok[i] @ o.reshape(-1, d).T).view(p, m - 1, p).min(2).values.sort(1).values
        a, b = max(1, int(round(.1 * (m - 1)))), max(1, int(round(.3 * (m - 1))))
        for x, v in (("k1", D[:, 0]), ("iv10", D[:, :a].mean(1)), ("iv30", D[:, :b].mean(1)), ("q30", D[:, b - 1])): out[x][i] = float(v.topk(k).values.mean())
    return out
def intensity(cr):
    std, pk, tp = [], [], []; T = np.median([c - np.median(c) for c in cr], 0)
    for c in cr:
        s = cv2.resize(c, (64, 64), interpolation=cv2.INTER_AREA); b = cv2.GaussianBlur(s[6:58, 6:58], (0, 0), 1.5); std.append(s[4:60, 4:60].std()); pk.append(np.percentile(b, 99.5) - np.median(b))
        r = cv2.GaussianBlur(c - np.median(c) - T, (0, 0), 3.)[12:-12, 12:-12]; tp.append(np.percentile(r, 99.5))
    return np.array(std), np.array(pk), np.array(tp)
Z = lambda v: R.z(v, .5)
def combos(mu, std, pk, tp):
    I = np.abs(Z(std)) + Z(pk); o = {"v3(k1)": Z(mu["k1"]) + I}
    for x in ("iv10", "iv30", "q30"): o[f"v3({x})"] = Z(mu[x]) + I
    o["pk"] = Z(pk); o["tpl"] = Z(tp); o["v3(iv30)+tpl"] = o["v3(iv30)"] + Z(tp); o["k1 only"] = Z(mu["k1"]); o["iv30 only"] = Z(mu["iv30"])
    return o
# ---------- benchmark
frames = [it for it in items.values() if it["group"] in ("G05", "G06", "G08", "G09", "G12") and sum(m["det_score"] >= .5 for m in it["modules"]) >= 25]
frames = [frames[i] for i in rng.permutation(len(frames))[:MAXF]]; print("frames", len(frames), flush=True)
rec = {n: {} for n in NS}; lab = {n: [] for n in NS}
for fi, it in enumerate(frames, 1):
    mods = [m for m in it["modules"] if m["det_score"] >= .5 and fill_ratio(m["polygon"]) >= FILL_MIN][:100]
    im = mm.load_image(mm.image_path(IRF, it["name"]), "ir")[..., 0]; base = grey_crops(im, [m["polygon"] for m in mods]); sf = contrast(base)
    u8 = lambda c: np.repeat(np.clip(c, 0, 255).astype(np.uint8)[..., None], 3, 2); t0 = _tokens([u8(c) for c in base]); order = rng.permutation(len(mods))
    for n in NS:
        idx = order[:min(n, len(mods) // 3)]; cr = [c.copy() for c in base]; tok = t0.clone()
        if len(idx):
            for j in idx:
                r = don[DG[rng.integers(len(DG))]]; r = r[rng.integers(len(r))] if isinstance(r, list) else r
                r = np.rot90(r, 2) if rng.random() < .5 else r; r = r[:, ::-1] if rng.random() < .5 else r; cr[j] = cr[j] + np.ascontiguousarray(r) * sf
            tok[idx] = _tokens([u8(cr[j]) for j in idx])
        o = combos(mutual_all(tok), *intensity(cr)); y = np.zeros(len(mods), bool); y[idx] = True; lab[n].append(y)
        for k, v in o.items(): rec[n].setdefault(k, []).append(v)
    print(fi, len(frames), flush=True)
    del t0; torch.cuda.empty_cache()
np.save(S / "hot_bench.npy", {"rec": rec, "lab": lab}, allow_pickle=True)
from clip_crack import roc_auc_score as auc
names = list(rec[0]); print("\nmethod".ljust(16) + "thr@1%FPR | recall (precision) at n=" + " ".join(str(n) for n in NS[1:]) + " | AUC")
for k in names:
    t = np.percentile(np.concatenate(rec[0][k]), 99); parts = []; aucs = []
    for n in NS[1:]:
        s = np.concatenate(rec[n][k]); y = np.concatenate(lab[n]); p = s >= t; parts.append(f"{(p & y).sum() / y.sum():.2f}({(p & y).sum() / max(p.sum(), 1):.2f})"); aucs.append(auc(y, s))
    print(k.ljust(16) + f"{t:7.2f}  " + " ".join(parts) + " | " + " ".join(f"{a:.3f}" for a in aucs), flush=True)
