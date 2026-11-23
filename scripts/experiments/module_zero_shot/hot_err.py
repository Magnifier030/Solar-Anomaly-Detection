"""Validation error decomposition of the uploaded hotspot pipeline (v3 rule + CL swap with LOGO logits) on G01-G04.
FN causes: nomod (no predicted module IoU>=.5), border/fill excluded, below-thr, cut by K/rel, dropped by CL. FP: frame has no GT / GT frame wrong module (neighbour of GT?)."""
import json, sys, collections, numpy as np, cv2
from pathlib import Path
sys.argv = ["raw_eval.py", "ir"]; import raw_eval as R
from shapely.geometry import Polygon
S = Path(__file__).parent
M = json.load(open(S / "hotcl_train_meta.json")); grp = np.array([m[1] for m in M]); kind = np.array([m[4] for m in M])
L = (np.load(S / "hotcl_logo_real.npy") + np.load(S / "hotcl_logo_syn.npy")) / 2
items = {}
for f in ("raw_ir_cal.json", "raw_ir.json", "raw_ir_dense.json"):
    for it in json.load(open(S / f)): items[it["name"]] = it
rows = {}
for i in np.where((kind == "c") & np.isin(grp, ["G01", "G02", "G03", "G04"]))[0]: rows.setdefault(M[i][0], []).append(i)
IRF = R.mm.TRAIN / R.mm.FOLDER["ir"] / "image"
P = lambda p: Polygon(p).buffer(0)
def iou(a, b): i = a.intersection(b).area; return i / max(a.union(b).area, 1e-6)
st = collections.defaultdict(collections.Counter)
for n, idx in rows.items():
    it = items[n]; g = it["group"]; mods = [m for m in it["modules"] if m["det_score"] >= .5]; idx = np.array(sorted(idx, key=lambda i: M[i][3])); lg = L[idx]
    hgt, wid = cv2.imread(str(R.mm.image_path(IRF, n)), cv2.IMREAD_GRAYSCALE).shape
    bad = np.array([(np.asarray(m["polygon"])[:, 0].min() <= 5) | (np.asarray(m["polygon"])[:, 1].min() <= 5) | (np.asarray(m["polygon"])[:, 0].max() >= wid - 6) | (np.asarray(m["polygon"])[:, 1].max() >= hgt - 6) | (m["fill"] < R.FILL_MIN) for m in mods])
    dense = len(mods) >= 25; K, thr, rel = (4, 3.5, .6) if dense else (2, 3.25, .7)
    base = np.zeros(len(mods), bool); s = np.zeros(len(mods))
    if len(mods) >= 3:
        s = R.z(np.array([m["raw"] for m in mods]), .5) + np.abs(R.z(np.array([m["std"] for m in mods]), .5)) + R.z(np.array([m["pk"] for m in mods]), .5)
        s = np.where(bad, -1e9, s); base[np.argsort(-s)[:K]] = True; base &= (s >= thr) & (s >= rel * s.max())
    pred = base & (lg >= -4); add = np.where(~base & ~bad & (lg >= 2))[0]
    if len(add): pred[add[np.argmax(lg[add])]] = True
    mp = [P(m["polygon"]) for m in mods]; gts = [P(np.asarray(h, float)) for h in it["truth"].get("hotspot", [])]
    match = {}
    for gi, gp in enumerate(gts):
        ious = [iou(gp, q) for q in mp]; j = int(np.argmax(ious)) if ious else -1
        if j >= 0 and ious[j] >= .5: match[gi] = j
    c = st[g]; c["GT"] += len(gts); c["pred"] += int(pred.sum()); c["frames"] += 1; c["gt_frames"] += bool(gts)
    hit = set()
    for gi in range(len(gts)):
        j = match.get(gi)
        if j is None: c["FN_nomodule"] += 1; continue
        if pred[j]: c["TP"] += 1; hit.add(j); continue
        if bad[j]: c["FN_border/fill"] += 1
        elif base[j] and lg[j] < -4: c["FN_droppedByCL"] += 1
        elif len(mods) < 3: c["FN_<3mods"] += 1
        elif s[j] < thr: c["FN_belowThr"] += 1
        else: c["FN_cutByK/rel"] += 1
    for j in np.where(pred)[0]:
        if j in hit: continue
        if not gts: c["FP_noGTframe"] += 1
        else:
            near = any(mp[j].distance(mp[match[gi]]) < 3 for gi in match) if match else False
            c["FP_GTframe_neighbour" if near else "FP_GTframe_other"] += 1
    c["addedByCL"] += int(len(add) > 0); c["addedByCL_TP"] += int(len(add) > 0 and pred[add[np.argmax(lg[add])]] and add[np.argmax(lg[add])] in hit)
    c["droppedByCL"] += int((base & (lg < -4)).sum())
tot = collections.Counter()
for g in sorted(st): tot.update(st[g])
for g, c in list(sorted(st.items())) + [("ALL", tot)]:
    tp = c["TP"]; print(f"{g}: frames {c['frames']} (GT frames {c['gt_frames']}) GT {c['GT']} pred {c['pred']} TP {tp} prec {tp/max(c['pred'],1):.2f} rec {tp/max(c['GT'],1):.2f} F1 {2*tp/max(c['pred']+c['GT'],1):.3f}")
    print("    ", {k: v for k, v in c.items() if k.startswith(("FN", "FP", "added", "dropped"))})
