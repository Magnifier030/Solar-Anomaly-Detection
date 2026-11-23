import json,numpy as np,cv2,collections
from rectify import rectify
O="/home/M11509204/researchs/Solar-Anomaly-Detection/outputs/submissions/"; TEST="/home/M11509204/researchs/Solar-Anomaly-Detection/data/raw/datasets/9_30_test_set/"
TM=json.load(open("crackbig_test_meta.json")); st=np.load("crackbig_test_scores_pneg.npy"); eo=json.load(open("test_cache_eo_module-maskrcnn-eo-all.json"))
dom=json.load(open("test_eo_domain.json"))
def cr(sub): return {(k,json.dumps(o["points"])) for k,v in sub.items() for o in v if o["label"]=="crack"}
Gc=cr(json.load(open(O+"ir-v6union-sp7-diode-addcrack-zlow2-20261107/submission.json"))); Hc=cr(json.load(open(O+"G-clipall70-crack-20261107/submission.json")))
mods={k:[m for m in v if m["det_score"]>=0.7] for k,v in eo.items()}
cand=[]
for i in np.argsort(-st):
    k,_,_,j,_=TM[i]; p=json.dumps(mods[k][j]["polygon"])
    if (k,p) in Gc or (k,p) in Hc: continue
    cand.append(int(i))
    if len(cand)==1200: break
json.dump(cand,open("big_cand.json","w"))
print("domains top100",collections.Counter(dom[TM[i][0]][0] for i in cand[:100]),"frames",len({TM[i][0] for i in cand[:100]}))
tiles=[]
for r,i in enumerate(cand[:80]):
    k,_,_,j,_=TM[i]; c=rectify(cv2.imread(TEST+k),np.asarray(mods[k][j]["polygon"],float),160,inset=0.03).astype(np.uint8).copy()
    cv2.putText(c,str(r),(3,15),0,0.5,(0,255,255),1); tiles.append(c)
cv2.imwrite("big_top80.jpg",np.vstack([np.hstack(tiles[i:i+10]) for i in range(0,80,10)]))
