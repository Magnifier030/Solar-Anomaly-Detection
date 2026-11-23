"""Parallel sweep of EO crack rules on cached raw features (held-out + calibration).
rule: crack if zero-shot floored-z >= thr (plain) or additionally Mask R-CNN label (union);
also 'top1': per frame mark the top-scoring module if its z >= thr.
"""
import itertools, json, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
sys.argv = ["raw_eval.py", "eo"]
import raw_eval as R
S = Path(__file__).parent
SETS = {t: json.loads((S / f"raw_eo{s}.json").read_text()) for t, s in (("held", ""), ("cal", "_cal"))}

def scores(items, c):
    out = []
    for it in items:
        mods = [m for m in it["modules"] if m["det_score"] >= 0.7]
        if len(mods) >= 3 and all("raw" in m for m in mods):
            s = R.z(np.array([m["raw"] for m in mods]), c)
            s = np.where(np.array([m["fill"] for m in mods]) < R.FILL_MIN, 0, s)
        else:
            s = np.zeros(len(mods))
        out.append((mods, s))
    return out

def run(cfg):
    tag, mode, c, thr = cfg
    samples = []
    for it, (mods, s) in zip(SETS[tag], scores(SETS[tag], c)):
        pred = {}
        top = int(np.argmax(s)) if len(s) else -1
        for j, (m, v) in enumerate(zip(mods, s)):
            zs = (v >= thr) if mode != "top1" else (j == top and v >= thr)
            hit = zs or (mode in ("union", "top1") and m["det_label"] == 2)
            pred.setdefault("crack" if hit else "solar_panel", []).append(m["polygon"])
        samples.append({"ground_truth": it["truth"], "prediction": pred, "group": it["group"]})
    r, per = R.score(samples)
    n = sum(len(v) for s_ in samples for k, v in s_["prediction"].items() if k == "crack")
    return cfg, r, per, n

if __name__ == "__main__":
    cfgs = list(itertools.product(("held", "cal"), ("plain", "union", "top1"), (0.0, 0.1, 0.3, 0.5),
                                  np.round(np.arange(1.0, 8.01, 0.5), 2)))
    with Pool(60) as pool:
        res = pool.map(run, cfgs)
    table = {cfg: (r, per, n) for cfg, r, per, n in res}
    for mode in ("plain", "union", "top1"):
        for c in (0.0, 0.1, 0.3, 0.5):
            thrs = np.round(np.arange(1.0, 8.01, 0.5), 2)
            best = max(thrs, key=lambda t: table[("cal", mode, c, t)][0])
            h = table[("held", mode, c, best)]
            oracle = max(table[("held", mode, c, t)][0] for t in thrs)
            print(f"{mode:5s} c={c}: cal thr {best} cal {table[('cal', mode, c, best)][0]:.3f} -> held {h[0]:.3f} {h[1]} n={h[2]} | oracle {oracle:.3f}", flush=True)
    json.dump({"|".join(map(str, k)): v for k, v in table.items()}, open(S / "fast_sweep.json", "w"))
