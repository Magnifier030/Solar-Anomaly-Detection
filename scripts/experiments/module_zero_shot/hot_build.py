"""Hotspot rebuild on top of BASE (EO/crack untouched).
swap  TD TA KA : keep BASE hotspots with avg-CL logit >= TD, add non-picked modules (not border/fill-bad) with logit >= TA, <= KA added per frame.
fuse  THR K REL KD THRD RELD : rescore every frame with s = v3 + .3*logit (v3 = z(raw)+|z(std)|+z(pk), c=.5); sparse (<25 mods) K/THR/REL, dense KD/THRD/RELD.
Usage: hot_build.py BASE OUT MODE args... [DRY]"""
import json, sys, subprocess, numpy as np, cv2
from pathlib import Path
a = sys.argv[1:]; sys.argv = ["raw_eval.py", "ir"]; import raw_eval as R
S = Path(__file__).parent; PJ = Path("/home/M11509204/researchs/Solar-Anomaly-Detection"); KIT = PJ / "data/raw/evaluation_kit/author-kit_1012"; T = PJ / "data/raw/datasets/9_30_test_set"
base, out, mode = Path(a[0]), Path(a[1]), a[2]; DRY = a[-1] == "DRY"; p = [float(x) for x in (a[3:-1] if DRY else a[3:])]
REASON = "推測模組板破損或內部隱裂、電性異常，導電路徑受損使局部電阻升高，進而產生異常發熱。"
L = np.load(S / "hotcl_test_logit_avg.npy"); TM = json.load(open(S / "hotcl_test_meta.json")); row = {(m[0], m[3]): i for i, m in enumerate(TM)}
ir = json.load(open(S / "test_cache_ir_module-maskrcnn-ir-all.json")); sub = json.loads((base / "submission.json").read_text())
c = dict(kept=0, dropped=0, added=0, hot=0, frames_changed=0)
for key, modules in ir.items():
    mods = [m for m in modules if m["det_score"] >= .5]
    if not mods: continue
    h, w = cv2.imread(str(T / key), cv2.IMREAD_GRAYSCALE).shape
    bad = np.array([(np.asarray(m["polygon"])[:, 0].min() <= 5) | (np.asarray(m["polygon"])[:, 1].min() <= 5) | (np.asarray(m["polygon"])[:, 0].max() >= w - 6)
                    | (np.asarray(m["polygon"])[:, 1].max() >= h - 6) | (m.get("fill", 1) < R.FILL_MIN) for m in mods])
    lg = np.array([L[row[(key, j)]] for j in range(len(mods))])
    objs = sub.get(key, []); old = {json.dumps(o["points"]): o for o in objs if o["label"] == "hotspot"}
    was = np.array([json.dumps(m["polygon"]) in old for m in mods]); other = [o for o in objs if json.dumps(o["points"]) not in {json.dumps(m["polygon"]) for m in mods}]
    if mode == "swap":
        TD, TA, KA = p; keep = was & (lg >= TD); cand = np.where(~was & ~bad & (lg >= TA))[0]; cand = cand[np.argsort(-lg[cand])][:int(KA)]
        new = keep.copy(); new[cand] = True
    else:
        THR, K, REL, KD, THRD, RELD = p; new = np.zeros(len(mods), bool)
        if len(mods) >= 3 and all("raw" in m for m in mods):
            s = R.z(np.array([m["raw"] for m in mods]), .5) + np.abs(R.z(np.array([m["std"] for m in mods]), .5)) + R.z(np.array([m["pk"] for m in mods]), .5) + .3 * lg
            s = np.where(bad, -1e9, s); k, t, r = (KD, THRD, RELD) if len(mods) >= 25 else (K, THR, REL)
            new[np.argsort(-s)[:int(k)]] = True; new &= (s >= t) & (s >= r * s.max())
    c["kept"] += int((new & was).sum()); c["dropped"] += int((was & ~new).sum()); c["added"] += int((new & ~was).sum()); c["hot"] += int(new.sum()); c["frames_changed"] += int((new != was).any())
    res = []
    for m, n_ in zip(mods, new):
        o = old.get(json.dumps(m["polygon"]))
        res.append({"label": "hotspot", "points": m["polygon"], "reason": o["reason"] if o else REASON} if n_ else {"label": "solar_panel", "points": m["polygon"]})
    sub[key] = res + other
print(c)
if DRY: sys.exit()
out.mkdir(parents=True, exist_ok=False); pa = out / "submission.json"; pa.write_text(json.dumps(sub, ensure_ascii=False, allow_nan=False) + "\n")
chk = subprocess.run([sys.executable, str(KIT / "make_submission.py"), "--check", str(pa)], capture_output=True, text=True); (out / "format_check.txt").write_text(chk.stdout + chk.stderr)
(out / "manifest.json").write_text(json.dumps({"base": str(base), "mode": mode, "params": p, "scores": "hotcl_test_logit_avg.npy (hot_cl.py real+syn avg)", "counts": c,
    "format_check_exit_code": chk.returncode, "uploaded": False}, ensure_ascii=False, indent=1)); print("check", chk.returncode)
