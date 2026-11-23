import sys,json,numpy as np,cv2
sys.argv=["raw_eval.py","eo"]
import raw_eval as R
from rectify import rectify
mm=R.mm; G=mm.frames_by_group(); tiles=[]
for g in ("G01","G07","G04"):
    n_=0
    for n in sorted(G):
        if G[n]!=g: continue
        try: sh=mm.load_shapes(n,"eo")
        except Exception: continue
        cr=[p for l,p in sh if l=="crack" and len(p)>=4]
        if not cr: continue
        im=mm.load_image(mm.image_path(mm.TRAIN/"EO"/"image",n),"eo")
        for p in cr:
            c=np.ascontiguousarray(rectify(im,p.astype(float),160,inset=0.03),np.uint8).copy(); cv2.putText(c,g,(3,15),0,.5,(0,255,255),1); tiles.append(c); n_+=1
            if (g=="G04" and n_>=8) or n_>=36: break
        if (g=="G04" and n_>=8) or n_>=36: break
tiles=tiles[:50]; tiles+= [np.zeros_like(tiles[0])]*(-len(tiles)%10)
cv2.imwrite("train_crack_sheet.jpg",np.vstack([np.hstack(tiles[i:i+10]) for i in range(0,len(tiles),10)]))
