import json, numpy as np, cv2
from pathlib import Path
from rectify import rectify
P=Path("/home/M11509204/researchs/Solar-Anomaly-Detection"); S=P/"outputs/submissions"; TEST=P/"data/raw/datasets/9_30_test_set"
import importlib.util
spec=importlib.util.spec_from_file_location("mm",P/"scripts/module_maskrcnn.py"); mm=importlib.util.module_from_spec(spec); spec.loader.exec_module(mm)
a=json.load(open(S/"ir-floor05-clipwin-noglare-diode-20261105/submission.json")); b=json.load(open(S/"ir-border-diode-clipwin-noglare-20261105/submission.json"))
tiles=[];ctx=[]
for k in a:
    for oa,ob in zip(a[k],b[k]):
        if oa["label"]=="hotspot" and ob["label"]!="hotspot":
            img=mm.load_image(TEST/k,"ir")
            tiles.append(cv2.resize(rectify(img,np.asarray(oa["points"],float),96),(80,120)))
            if len(ctx)<9 and len(tiles)%7==1:
                c=img.copy(); cv2.polylines(c,[np.int32(oa["points"])],True,(255,0,0),3); ctx.append(cv2.resize(c,(320,256)))
def grid(ts,n=12):
    ts=ts+[np.zeros_like(ts[0])]*((-len(ts))%n); return np.vstack([np.hstack(ts[r*n:(r+1)*n]) for r in range(len(ts)//n)])
cv2.imwrite("removed_tiles.jpg",cv2.cvtColor(grid(tiles[:72]),cv2.COLOR_RGB2BGR)); cv2.imwrite("removed_ctx.jpg",cv2.cvtColor(grid(ctx,3),cv2.COLOR_RGB2BGR))
