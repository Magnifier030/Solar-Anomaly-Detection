import json,numpy as np,cv2
from rectify import rectify
TEST="/home/M11509204/researchs/Solar-Anomaly-Detection/data/raw/datasets/9_30_test_set/"
ms=json.load(open("crackfeat_test_meta.json")); o=json.load(open("dom_rank.json")); eo=json.load(open("test_cache_eo_module-maskrcnn-eo-all.json"))
tiles=[]
for r,i in enumerate(o[:80]):
    k,_,_,j=ms[i]; m=[m for m in eo[k] if m["det_score"]>=0.7][j]
    im=cv2.imread(TEST+k); c=rectify(im,np.asarray(m["polygon"],float),160,inset=0.03).astype(np.uint8).copy()
    cv2.putText(c,str(r),(3,15),0,0.5,(0,255,255),1); tiles.append(c)
rows=[np.hstack(tiles[i:i+10]) for i in range(0,80,10)]
cv2.imwrite("dom_top80.jpg",np.vstack(rows))
