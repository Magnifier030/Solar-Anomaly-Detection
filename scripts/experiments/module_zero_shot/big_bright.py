import json,numpy as np,cv2,collections
from rectify import rectify
O="/home/M11509204/researchs/Solar-Anomaly-Detection/outputs/submissions/"; TEST="/home/M11509204/researchs/Solar-Anomaly-Detection/data/raw/datasets/9_30_test_set/"
eo=json.load(open("test_cache_eo_module-maskrcnn-eo-all.json")); mods={k:[m for m in v if m["det_score"]>=0.7] for k,v in eo.items()}
dom=json.load(open("test_eo_domain.json"))
add=json.load(open(O+"thr35-bigcrack500-20261109/manifest.json"))["added"]
def br(k,j): return float(rectify(cv2.imread(TEST+k,0),np.asarray(mods[k][j]["polygon"],float),64,inset=.05).mean())
b=np.array([br(k,j) for k,j in add])
for a,z in ((0,100),(100,200),(200,300),(300,500)):
    seg=add[a:z]; print(f"rank {a}-{z}: white(>150) {int((b[a:z]>150).sum())}  frames {len({k for k,_ in seg})}  domains {dict(collections.Counter(dom[k][0] for k,_ in seg))}")
json.dump(b.tolist(),open("big500_bright.json","w"))
