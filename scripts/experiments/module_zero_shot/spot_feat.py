"""Small-scale spot contrast per module: 128px rectified grey (inset .05), tophat = c - medianBlur(c,k); sp = p99.9 of tophat, sp2 = 2nd-largest blob peak.
Writes aligned lists <cache>_sp.json (list per item/key of [sp7, sp15, sp31])."""
import json, sys
from pathlib import Path
from multiprocessing import Pool
import numpy as np, cv2
sys.argv = ["raw_eval.py", "ir"]
import raw_eval as R
from rectify import rectify
S = Path(__file__).parent
TEST = Path("/home/M11509204/researchs/Solar-Anomaly-Detection/data/raw/datasets/9_30_test_set")
PV = R.PROJECT / "data/raw/external/PV-HSD-2025/extracted/valid/images"
def feats(grey, poly):
    c = rectify(grey, np.asarray(poly, float), 128, inset=0.05)
    c = cv2.GaussianBlur(c, (0, 0), 1.0)
    out = []
    for k in (7, 15, 31):
        t = c.astype(np.float32) - cv2.medianBlur(c, k).astype(np.float32)
        out.append(float(np.percentile(t, 99.9)))
    return out
def grey_of(kind, name):
    if kind == "pv": return cv2.imread(str(PV / name), cv2.IMREAD_GRAYSCALE)
    if kind == "test": return cv2.cvtColor(R.mm.load_image(TEST / name, "ir"), cv2.COLOR_RGB2GRAY)
    return cv2.cvtColor(R.mm.load_image(R.mm.image_path(R.mm.TRAIN / "IR" / "image", name), "ir"), cv2.COLOR_RGB2GRAY)
def job(a):
    kind, name, polys = a
    if not polys: return []
    g = grey_of(kind, name); return [feats(g, p) for p in polys]
if __name__ == "__main__":
    with Pool(48) as P:
        for fn, kind in (("raw_ir_pvhsd.json", "pv"), ("raw_ir.json", "tr"), ("raw_ir_cal.json", "tr"), ("raw_ir_dense.json", "tr")):
            items = json.loads((S / fn).read_text())
            res = P.map(job, [(kind, it["name"], [m["polygon"] for m in it["modules"]]) for it in items], chunksize=2)
            (S / fn.replace(".json", "_sp.json")).write_text(json.dumps(res)); print("done", fn, flush=True)
        cache = json.loads((S / "test_cache_ir_module-maskrcnn-ir-all.json").read_text())
        res = P.map(job, [("test", k, [m["polygon"] for m in v]) for k, v in cache.items()], chunksize=2)
        (S / "test_cache_ir_sp.json").write_text(json.dumps(dict(zip(cache.keys(), res)))); print("done test", flush=True)
