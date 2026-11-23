import json,collections,numpy as np
from shapely.geometry import Polygon,Point
O="/home/M11509204/researchs/Solar-Anomaly-Detection/outputs/submissions/"
Wp=json.load(open("reg_warps.json")); dom=json.load(open("test_eo_domain.json"))
def run(fn,label):
    sub=json.load(open(O+fn+"/submission.json")); obs=0; exp=0.; nfr=0; hot_mod={}
    for ke,(w,i0,i1,g) in Wp.items():
        if i1<.6: continue
        w=np.array(w); k=ke.replace("/EO/","/IR/"); eo=sub[ke]; polys=[Polygon(o["points"]).buffer(0) for o in eo]
        hm=set()
        for o in sub[k]:
            if o["label"]!="hotspot": continue
            c=np.r_[np.mean(o["points"],0),1]; e=(w@c)*2; pt=Point(*e)
            for j,p in enumerate(polys):
                if p.contains(pt): hm.add(j); break
        hot_mod[ke]=hm
        cr=[j for j,o in enumerate(eo) if o["label"]=="crack"]
        if not cr: continue
        nfr+=1; obs+=sum(j in hm for j in cr); exp+=len(cr)*len(hm)/len(eo)
    print(f"{label}: frames w/ cracks (registered) {nfr} | cracks on a hotspot module: observed {obs}, random expectation {exp:.1f}")
    return hot_mod
hm=run("thr35-dinov3L-CLwhitespot-crack100-20261116","CL")
run("thr35-dinov3L-crack100-20261110","DINOv3-L")
run("thr35-bigcrack500-20261109","v2 big500 (+G223, ~4% prec)")
run("G-clipall70-crack-20261107","H (G+386 failed)")
tot=sum(len(v) for v in hm.values()); print("registered frames",len(hm),"hotspot-linked EO modules",tot, collections.Counter(dom[k][0] for k,v in hm.items() for _ in v).most_common())
json.dump({k:sorted(v) for k,v in hm.items()},open("hot_linked_eo.json","w"))
