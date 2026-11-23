import json, numpy as np, cv2
from band_feat import mm
from rectify import rectify
P="/home/M11509204/researchs/Solar-Anomaly-Detection/"
TEST=mm.Path(P+"data/raw/datasets/9_30_test_set")
sub=json.load(open(P+"outputs/submissions/ir-floor05-clipwin-noglare-20261105/submission.json"))
bt=json.load(open("band_test.json")); i=0; tiles=[]; non=[]
for key,objs in sub.items():
    hs=[o for o in objs if o["label"]=="hotspot"]
    if not hs: continue
    img=mm.load_image(TEST/key,"ir")
    for o in hs:
        f=bt[i]; i+=1
        t=cv2.resize(rectify(img,np.asarray(o["points"],float),96,inset=0.0),(80,120))
        (tiles if (f[1]>=2.5 and f[2]>=0.7 and f[5]>=0.8 and f[6]<=0.1) else non).append(t)
def grid(ts,n=12):
    ts=ts+[np.zeros_like(ts[0])]*((-len(ts))%n)
    return np.vstack([np.hstack(ts[r*n:(r+1)*n]) for r in range(len(ts)//n)])
cv2.imwrite("band_sheet.jpg",cv2.cvtColor(grid(tiles),cv2.COLOR_RGB2BGR))
cv2.imwrite("nonband_sheet.jpg",cv2.cvtColor(grid(non[::12][:60]),cv2.COLOR_RGB2BGR))
