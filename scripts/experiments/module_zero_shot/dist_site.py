import json,sys,numpy as np,cv2
sys.argv=["x","ir"]
import raw_eval as R, border_eval_lib as B
from pathlib import Path
TEST=Path("/home/M11509204/researchs/Solar-Anomaly-Detection/data/raw/datasets/9_30_test_set")
sm=json.load(open("site_match.json"))["ir"]; t=json.load(open("test_cache_ir_module-maskrcnn-ir-all.json"))
agg={}
for k,mods in t.items():
    im=cv2.imread(str(TEST/k)); ps=np.abs(im[...,0].astype(int)-im[...,2]).mean()>5
    seg=("PSEUDO" if ps else sm[k][0]); mods=[m for m in mods if m["det_score"]>=0.5]
    a=agg.setdefault(seg,[0,0,0,0,0,[]]); a[0]+=1
    if len(mods)<3 or not all("pk" in m for m in mods): continue
    s=B.frame_score(mods,0.5,"raw+std+brightpk",1.0); o=(s>=3.25)&(s>=0.7*s.max())
    a[1]+=o.any(); a[2]+=min(o.sum(),2); a[3]+=o.sum(); a[4]+=len(mods); a[5].append(np.median([m["raw"] for m in mods]))
print("seg frames flagged K2hot nocaphot mods/frame")
for g,a in sorted(agg.items()): print(g,a[0],a[1],a[2],a[3],round(a[4]/max(a[0],1),1))
