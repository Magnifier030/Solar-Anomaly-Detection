import sys,numpy as np,cv2
sys.argv=["raw_eval.py","eo"]
import raw_eval as R
from rectify import rectify
mm=R.mm; G=mm.frames_by_group(); rows=[]
for n in sorted(G):
    if G[n]!="G01": continue
    sh=[(l,p) for l,p in mm.load_shapes(n,"eo") if len(p)>=4]
    if not any(l=="crack" for l,_ in sh): continue
    im=mm.load_image(mm.image_path(mm.TRAIN/"EO"/"image",n),"eo")
    cr=[p for l,p in sh if l=="crack"][0]; c0=cr.mean(0)
    nb=sorted([p for l,p in sh if l=="solar_panel"],key=lambda p:np.linalg.norm(p.mean(0)-c0))[:7]
    t=[]
    for k,p in enumerate([cr]+nb):
        c=np.ascontiguousarray(rectify(im,p.astype(float),200,inset=0.03),np.uint8).copy()
        cv2.putText(c,"CRACK" if k==0 else "panel",(3,18),0,.6,(0,0,255) if k==0 else (0,255,0),2); t.append(c)
    rows.append(np.hstack(t))
    if len(rows)==4: break
cv2.imwrite("g01_neighbors.jpg",np.vstack(rows))
