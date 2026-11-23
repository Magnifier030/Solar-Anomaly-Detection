import json,glob,collections,os,numpy as np
P="/home/M11509204/researchs/Solar-Anomaly-Detection/data/raw/datasets/9_16_train_set/train/"
M=json.load(open("crackbig_train_meta.json")); g={m[0]:m[1] for m in M}
T=json.load(open("text_meta.json")); C,U=T["C"],T["U"]; F=np.load("text_F.npy"); ui={u:i for i,u in enumerate(U)}
R=collections.defaultdict(collections.Counter)
for mod in ("IR","EO"):
    for f in glob.glob(P+mod+"/label/*.json"):
        b=os.path.basename(f)[:-5]
        for s in json.load(open(f))["shapes"]:
            if s["label"]!="solar_panel" and s.get("reason"): R[(mod,g.get(b,"?"))][s["reason"].strip()]+=1
for k,c in sorted(R.items()): print(k,sum(c.values()),"unique",len(c))
# overlap of reason sets between clusters
ks=sorted(R); 
for a in ks:
    print(a,{b:len(set(R[a])&set(R[b])) for b in ks if b!=a and set(R[a])&set(R[b])})
json.dump({f"{k[0]}|{k[1]}":dict(v) for k,v in R.items()},open("text_clusters.json","w"),ensure_ascii=False,indent=0)
