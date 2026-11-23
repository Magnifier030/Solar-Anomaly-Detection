import sys,numpy as np,cv2
sys.argv=["raw_eval.py","eo"]
import raw_eval as R
mm=R.mm; G=mm.frames_by_group(); out=[]
for n in sorted(G):
    if G[n]!="G07": continue
    sh=mm.load_shapes(n,"eo"); cr=[p for l,p in sh if l=="crack"]
    if not cr: continue
    im=mm.load_image(mm.image_path(mm.TRAIN/"EO"/"image",n),"eo").copy(); im=im if im.ndim==3 else cv2.cvtColor(im,cv2.COLOR_GRAY2BGR)
    for p in cr: cv2.polylines(im,[np.int32(p)],True,(0,0,255),3)
    out.append(cv2.resize(im,(960,540)))
    if len(out)==2: break
cv2.imwrite("g07_frames.jpg",np.vstack(out))
