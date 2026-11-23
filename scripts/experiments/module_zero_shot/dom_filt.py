import json,numpy as np,cv2
from rectify import rectify
TEST="/home/M11509204/researchs/Solar-Anomaly-Detection/data/raw/datasets/9_30_test_set/"
ms=json.load(open("crackfeat_test_meta.json")); o=json.load(open("dom_rank.json")); eo=json.load(open("test_cache_eo_module-maskrcnn-eo-all.json"))
import sys; GL,RNG=float(sys.argv[1]),float(sys.argv[2])
def lf(c):
    g=cv2.cvtColor(c,cv2.COLOR_BGR2GRAY).astype(np.float32); b=cv2.GaussianBlur(g,(0,0),12)
    return float((g>235).mean()), float(np.percentile(b,97)-np.percentile(b,3))
keep=[];tiles=[];info=[]
for r,i in enumerate(o):
    k,_,_,j=ms[i]; m=[m for m in eo[k] if m["det_score"]>=0.7][j]
    c=rectify(cv2.imread(TEST+k),np.asarray(m["polygon"],float),160,inset=0.03).astype(np.uint8).copy()
    gl,rng=lf(c); info.append((r,gl,rng))
    if gl<GL and rng<RNG:
        keep.append(i); cv2.putText(c,str(r),(3,15),0,0.5,(0,255,255),1); tiles.append(c)
a=np.array(info); print(np.percentile(a[:,1],[25,50,75,90]),np.percentile(a[:,2],[10,25,50,75]))
print("kept",len(keep),"of",len(o))
json.dump(keep,open("dom_keep.json","w"))
tiles=tiles[:80]; tiles+= [np.zeros_like(tiles[0])]*(-len(tiles)%10)
cv2.imwrite("dom_filt80.jpg",np.vstack([np.hstack(tiles[i:i+10]) for i in range(0,len(tiles),10)]))
