import json,numpy as np,collections
O="/home/M11509204/researchs/Solar-Anomaly-Detection/outputs/submissions/"
tr=json.load(open("spot_train.json")); te=json.load(open("spot_test.json"))
def q(a): a=np.array(a); return f"n={len(a)} spot_px median {np.median(a[:,0]):.0f} | frac px>=20: {(a[:,0]>=20).mean():.2f} | p99.9 med {np.median(a[:,1]):.0f}"
d=collections.defaultdict(list)
for k,(g,y,px,mx) in tr.items(): d[(g,y)].append((px,mx))
print("TRAIN")
for k in sorted(d): print(" ",k,q(d[k]))
eo=json.load(open("test_cache_eo_module-maskrcnn-eo-all.json")); pos={}
for kk,mods in eo.items():
    for j,m in enumerate([m for m in mods if m["det_score"]>=0.7]): pos[(kk,json.dumps(m["polygon"]))]=j
def cr(f): s=json.load(open(O+f+"/submission.json")); return {(k,pos[(k,json.dumps(o["points"]))]) for k,v in s.items() for o in v if o["label"]=="crack" and (k,json.dumps(o["points"])) in pos}
Gc=cr("ir-v6union-sp7-diode-addcrack-zlow2-20261107"); Hc=cr("G-clipall70-crack-20261107")
v2=[tuple(x) for x in json.load(open(O+"thr30-bigcrack100-20261109/manifest.json"))["added"][:100]]
v3=[tuple(x) for x in json.load(open("tmp_crackbig3/manifest.json"))["added"][:100]]
v2b=[tuple(x) for x in json.load(open(O+"thr35-bigcrack500-20261109/manifest.json"))["added"][100:500]] if True else []
dom=json.load(open("test_eo_domain.json"))
sets={"v2 top100 (~40% hit)":v2,"v2 101-500 (~2%)":v2b,"H-added (0% hit)":sorted(Hc-Gc),"G old (~5%)":sorted(Gc),"DINOv3-L top100":v3,"all test":[tuple(k.rsplit("|",1)) for k in te]}
print("TEST")
for n,L in sets.items():
    a=[te[f"{k}|{j}"] for k,j in L]; print(" ",n,q(a))
json.dump({"v2":v2,"v2b":v2b},open("spot_sets.json","w"))
