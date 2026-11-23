import json, numpy as np, cv2, importlib.util
from pathlib import Path
P=Path("/home/M11509204/researchs/Solar-Anomaly-Detection"); T=P/"data/raw/datasets/9_30_test_set"
spec=importlib.util.spec_from_file_location("mm",P/"scripts/module_maskrcnn.py"); mm=importlib.util.module_from_spec(spec); spec.loader.exec_module(mm)
s=json.load(open(P/"outputs/submissions/ir-v3-peak-top2-diode-clipwin-noglare-20261106/submission.json"))
tiles=[]
keys=[k for k in s if "/IR/" in k]
pseudo=[k for k in keys if np.abs(cv2.imread(str(T/k))[...,0].astype(int)-cv2.imread(str(T/k))[...,2]).mean()>5]
for k in pseudo[::25][:8]:
    orig=cv2.imread(str(T/k)); grey=cv2.cvtColor(mm.load_image(T/k,"ir"),cv2.COLOR_RGB2BGR)
    for o in s[k]:
        if o["label"]=="hotspot":
            cv2.polylines(orig,[np.int32(o["points"])],True,(0,255,0),2); cv2.polylines(grey,[np.int32(o["points"])],True,(0,255,0),2)
    tiles.append(np.hstack([cv2.resize(orig,(320,256)),cv2.resize(grey,(320,256))]))
cv2.imwrite("pseudo_sheet.jpg",np.vstack([np.hstack(tiles[i:i+2]) for i in range(0,len(tiles),2)]))
print(len(pseudo))
