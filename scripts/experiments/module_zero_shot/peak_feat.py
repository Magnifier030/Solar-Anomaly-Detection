"""Add per-module local peak contrast to IR caches: on a 64px rectified grey crop (inset 0.1, Gaussian blur),
pk = p99.5 - median (bright peak), pn = median - p0.5 (dark peak). Usage: python peak_feat.py"""
import json, sys
from pathlib import Path
from multiprocessing import Pool
import numpy as np, cv2
sys.argv = ["raw_eval.py", "ir"]
import raw_eval as R
from rectify import rectify
S = Path(__file__).parent
TEST = Path("/home/M11509204/researchs/Solar-Anomaly-Detection/data/raw/datasets/9_30_test_set")

def feats(grey, poly):
    c = cv2.GaussianBlur(rectify(grey, np.asarray(poly, float), 64, inset=0.1).astype(np.float32), (0, 0), 1.5)
    med = np.median(c)
    return float(np.percentile(c, 99.5) - med), float(med - np.percentile(c, 0.5)), float(med)

def train_job(item):
    if not item["modules"]: return []
    grey = cv2.cvtColor(R.mm.load_image(R.mm.image_path(R.mm.TRAIN / "IR" / "image", item["name"]), "ir"), cv2.COLOR_RGB2GRAY)
    return [feats(grey, m["polygon"]) for m in item["modules"]]

def test_job(kv):
    key, mods = kv
    if not mods: return []
    grey = cv2.cvtColor(R.mm.load_image(TEST / key, "ir"), cv2.COLOR_RGB2GRAY)
    return [feats(grey, m["polygon"]) for m in mods]

if __name__ == "__main__":
    with Pool(48) as P:
        for suf in ("", "_cal"):
            items = json.loads((S / f"raw_ir{suf}.json").read_text())
            for it, fs in zip(items, P.map(train_job, items, chunksize=4)):
                for m, f in zip(it["modules"], fs): m["pk"], m["pn"], m["med"] = f
            (S / f"raw_ir{suf}.json").write_text(json.dumps(items)); print("done", suf, flush=True)
        path = S / "test_cache_ir_module-maskrcnn-ir-all.json"; cache = json.loads(path.read_text())
        for (k, mods), fs in zip(cache.items(), P.map(test_job, list(cache.items()), chunksize=4)):
            for m, f in zip(mods, fs): m["pk"], m["pn"], m["med"] = f
        path.write_text(json.dumps(cache)); print("done test", flush=True)
