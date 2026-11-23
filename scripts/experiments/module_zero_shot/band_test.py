import json, sys, importlib.util, numpy as np, cv2
ARG=sys.argv[1]
from band_feat import band_features, mm
P="/home/M11509204/researchs/Solar-Anomaly-Detection/"
TEST=mm.Path(P+"data/raw/datasets/9_30_test_set")
sub=json.load(open(ARG))
rows=[]
for key,objs in sub.items():
    hs=[o for o in objs if o["label"]=="hotspot"]
    if not hs: continue
    grey=cv2.cvtColor(mm.load_image(TEST/key,"ir"),cv2.COLOR_RGB2GRAY)
    for o in hs: rows.append((key,)+tuple(band_features(grey,o["points"])))
a=np.array([r[1:] for r in rows]); band=(a[:,0]>=2.5)&(a[:,1]>=0.7)
print(len(rows),"hotspots; band-like",band.sum(),"frames",len({rows[i][0] for i in np.where(band)[0]}))
json.dump([[rows[i][0]]+list(map(float,a[i])) for i in range(len(rows))],open("band_test.json","w"))
