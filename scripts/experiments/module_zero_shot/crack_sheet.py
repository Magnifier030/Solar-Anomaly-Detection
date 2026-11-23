import json, random, sys
from pathlib import Path
import numpy as np, cv2
from rectify import rectify
P = Path("/home/M11509204/researchs/Solar-Anomaly-Detection"); TEST = P / "data/raw/datasets/9_30_test_set"
A = json.load(open(P / "outputs/submissions/ir-v3-peak-diode-clipgatecrack-20261106/submission.json"))
eo = json.load(open("test_cache_eo_module-maskrcnn-eo-all.json")); clip = json.load(open("test_eo_clip_all.json"))
acr = {(k, json.dumps(o["points"])) for k, v in A.items() for o in v if o["label"] == "crack"}
rows = []
for k, mods in eo.items():
    mods = [m for m in mods if m["det_score"] >= 0.7]
    for m, p in zip(mods, clip[k]): rows.append((k, m["polygon"], p, (k, json.dumps(m["polygon"])) in acr))
print("modules", len(rows), "A cracks found", sum(r[3] for r in rows))
ps = np.array([r[2] for r in rows])
for t in (0.5, 0.7, 0.9): 
    fr = {}
    for r in rows:
        if r[2] >= t: fr[r[0]] = fr.get(r[0], 0) + 1
    print(f"clip>={t}: modules {sum(ps>=t)} frames {len(fr)} max/frame {max(fr.values()) if fr else 0}")
def sheet(sel, fn):
    tiles = []
    for i, (k, poly, p, _) in enumerate(sel):
        im = cv2.imread(str(TEST / k)); c = rectify(im, np.asarray(poly, float), 192, inset=0.0)
        cv2.putText(c, f"{i} {p:.2f}", (3, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        tiles.append(c)
    while len(tiles) % 8: tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(fn, np.vstack([np.hstack(tiles[i:i + 8]) for i in range(0, len(tiles), 8)]))
random.seed(1)
a = [r for r in rows if r[3]]; sheet(random.sample(a, 32), "sheet_Acrack.jpg")
top = sorted([r for r in rows if not r[3]], key=lambda r: -r[2])[:32]; sheet(top, "sheet_topclip.jpg")
json.dump([(r[0], r[2]) for r in top], open("sheet_topclip.json", "w"))
