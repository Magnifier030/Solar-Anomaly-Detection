"""zlow2 for the IR mutual term: per module store z-scored (c=0.5) mutual score under 6 transforms -> m['zr6']."""
import json, sys
from pathlib import Path
import numpy as np, cv2
sys.argv=["raw_eval.py","ir"]
import raw_eval as R
from rectify import rectify
from zs_score import SIZE, _mutual, _tokens
S=Path(__file__).parent; mm=R.mm
TEST=Path("/home/M11509204/researchs/Solar-Anomaly-Detection/data/raw/datasets/9_30_test_set")
def tf(c,k):
    h=c.shape[0]; ctr=(h/2,h/2)
    if k==0: return c
    if k==1: M=cv2.getRotationMatrix2D(ctr,0,0.75)
    if k==2: M=cv2.getRotationMatrix2D(ctr,0,1.33)
    if k==3: M=cv2.getRotationMatrix2D(ctr,15,1.0)
    if k==4: M=cv2.getRotationMatrix2D(ctr,-15,1.0)
    if k==5: return np.ascontiguousarray(c[:,::-1])
    return cv2.warpAffine(c,M,(h,h),borderMode=cv2.BORDER_REFLECT)

def run(mods,img):
    mods=[m for m in mods if m["det_score"]>=0.5]
    if len(mods)<3: return
    crops=[np.ascontiguousarray(rectify(img,np.asarray(m["polygon"],float),SIZE,inset=0.03),np.uint8) for m in mods]
    for k in range(6):
        raw=np.asarray(_mutual(_tokens([np.ascontiguousarray(tf(c,k)) for c in crops]),1),float)
        for m,v in zip(mods,raw): m.setdefault("raw6",[]).append(float(v))
for f in ("raw_ir.json","raw_ir_cal.json","raw_ir_dense.json"):
    items=json.load(open(S/f))
    for n,it in enumerate(items,1):
        run(it["modules"],mm.load_image(mm.image_path(mm.TRAIN/"IR"/"image",it["name"]),"ir"))
        if n%200==0: print(f,n,flush=True)
    json.dump(items,open(S/f.replace(".json","_z6.json"),"w"))
t=json.load(open(S/"test_cache_ir_module-maskrcnn-ir-all.json"))
for n,(k,mods) in enumerate(t.items(),1):
    run(mods,mm.load_image(TEST/k,"ir"))
    if n%200==0: print("test",n,flush=True)
json.dump(t,open(S/"test_cache_ir_z6.json","w")); print("done")
