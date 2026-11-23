import json,collections,numpy as np
from shapely.geometry import Polygon,Point
O="/home/M11509204/researchs/Solar-Anomaly-Detection/outputs/submissions/"
W=np.array([[1.342,0.043,-0.155],[0.148,1.084,-0.161]]); Ainv=np.linalg.inv(np.r_[W,[[0,0,1]]])
dom=json.load(open("test_eo_domain.json"))
sub=json.load(open(O+"thr35-dinov3L-CLwhitespot-crack100-20261116/submission.json"))
res=collections.Counter(); mapped={}
for k,v in sub.items():
    if "/IR/" not in k: continue
    ke=k.replace("/IR/","/EO/"); eo=sub.get(ke,[])
    for o in v:
        if o["label"]!="hotspot": continue
        c=np.mean(o["points"],0)/[640,512]; e=(Ainv@np.r_[c,1])[:2]
        if not (0<=e[0]<=1 and 0<=e[1]<=1): res["mapped outside EO"]+=1; continue
        pt=Point(e[0]*1920,e[1]*1080); hit=[o2 for o2 in eo if Polygon(o2["points"]).buffer(0).contains(pt)]
        if not hit: res["no EO module"]+=1; continue
        res["EO module found: "+hit[0]["label"]]+=1; mapped.setdefault(ke,[]).append(json.dumps(hit[0]["points"]))
        res["dom "+dom[ke][0]]+=1
print(dict(res))
cr={(k,json.dumps(o["points"])) for k,v in sub.items() for o in v if o["label"]=="crack"}
hm={(k,p) for k,L in mapped.items() for p in L}
print("CL cracks",len(cr),"that coincide with a predicted IR hotspot",len(cr&hm))
json.dump({k:L for k,L in mapped.items()},open("hot2eo.json","w"))
