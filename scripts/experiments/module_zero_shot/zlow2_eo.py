"""zlow2 (How-to-overcome-domain-shift Stage-1 pose search) on within-frame DINOv2 mutual crack scoring.
Per module crop: 6 generic transforms (identity, zoom-out .75, zoom-in 1.33, rot +-15, hflip); per transform the mutual
score is z-scored within the frame (c floor 0.1); zlow2 = mean of the 2 lowest z. Rows aligned with clip_window.npy."""
import json, sys
from pathlib import Path
import numpy as np, cv2
sys.argv=["raw_eval.py","eo"]
import raw_eval as R
from rectify import rectify
from zs_score import SIZE, _mutual, _tokens
S=Path(__file__).parent; mm=R.mm
def tf(c,k):
    h=c.shape[0]; ctr=(h/2,h/2)
    if k==0: return c
    if k==1: M=cv2.getRotationMatrix2D(ctr,0,0.75)
    if k==2: M=cv2.getRotationMatrix2D(ctr,0,1.33)
    if k==3: M=cv2.getRotationMatrix2D(ctr,15,1.0)
    if k==4: M=cv2.getRotationMatrix2D(ctr,-15,1.0)
    if k==5: return np.ascontiguousarray(c[:,::-1])
    return cv2.warpAffine(c,M,(h,h),borderMode=cv2.BORDER_REFLECT)
rows=[]
for tag,suf in (("held",""),("cal","_cal")):
    for n,it in enumerate(json.loads((S/f"raw_eo{suf}.json").read_text()),1):
        mods=[m for m in it["modules"] if m["det_score"]>=0.7 and "raw" in m]
        if len(mods)<3: continue
        img=mm.load_image(mm.image_path(mm.TRAIN/"EO"/"image",it["name"]),"eo")
        crops=[np.ascontiguousarray(rectify(img,np.asarray(m["polygon"],float),SIZE,inset=0.03),np.uint8) for m in mods]
        Z=[]
        for k in range(6):
            raw=np.asarray(_mutual(_tokens([np.ascontiguousarray(tf(c,k)) for c in crops]),2),float)
            Z.append(R.z(raw,0.1))
        Z=np.sort(np.array(Z),0)
        for j in range(len(mods)): rows.append((Z[:, j][0], Z[:2, j].mean(), np.median(Z[:, j]), Z[:, j].mean()))
        if n%100==0: print(tag,n,flush=True)
np.save(S/"zlow2_eo.npy",np.array(rows)); print("done",len(rows))
