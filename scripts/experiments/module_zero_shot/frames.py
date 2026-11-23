import json,collections,numpy as np
O="/home/M11509204/researchs/Solar-Anomaly-Detection/outputs/submissions/"
dom=json.load(open("test_eo_domain.json"))
def cracks(f):
    s=json.load(open(O+f+"/submission.json")); return [(k,o["points"]) for k,v in s.items() for o in v if o["label"]=="crack"]
eo=json.load(open("test_cache_eo_module-maskrcnn-eo-all.json")); nmod={k:len([m for m in v if m["det_score"]>=.7]) for k,v in eo.items()}
for f in ("thr35-dinov3L-crack100-20261110","thr35-dinov3L-CLwhitespot-crack100-20261116"):
    c=cracks(f); fr=collections.Counter(k for k,_ in c)
    print(f,"cracks",len(c),"frames",len(fr),"domains",collections.Counter(dom[k][0] for k,_ in c).most_common())
    print("  cracks per frame dist",sorted(collections.Counter(fr.values()).items()),"| frame ids",sorted(int(k.split("_")[-1][:3]) for k in fr)[:60])
L={k for k,_ in cracks("thr35-dinov3L-crack100-20261110")}; C={k for k,_ in cracks("thr35-dinov3L-CLwhitespot-crack100-20261116")}
a={(k,json.dumps(p)) for k,p in cracks("thr35-dinov3L-crack100-20261110")}; b={(k,json.dumps(p)) for k,p in cracks("thr35-dinov3L-CLwhitespot-crack100-20261116")}
print("only-L",len(a-b),collections.Counter(dom[k][0] for k,_ in a-b).most_common(),"| only-CL",len(b-a),collections.Counter(dom[k][0] for k,_ in b-a).most_common())
print("test frames per domain",collections.Counter(v[0] for v in dom.values()).most_common())
