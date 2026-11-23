import json,numpy as np,collections
O="/home/M11509204/researchs/Solar-Anomaly-Detection/outputs/submissions/"
mt=json.load(open("crackfeat_train_meta.json")); ms=json.load(open("crackfeat_test_meta.json"))
fr=collections.defaultdict(lambda:[0,0])
for n,g,l,i in mt: fr[n][0]=g; fr[n][1]+=l
for g in ["G01","G04","G07"]:
    v=[c for gg,c in fr.values() if gg==g]; print(g,"frames",len(v),"cracks",sum(v),"frames with crack",sum(c>0 for c in v))
dom=json.load(open("test_eo_domain.json")); sc=np.load("crack_test_scores.npy")
G=json.load(open(O+"ir-v6union-sp7-diode-addcrack-zlow2-20261107/submission.json")); H=json.load(open(O+"G-clipall70-crack-20261107/submission.json"))
def cr(sub,k): return {json.dumps(o["points"]) for o in sub[k] if o["label"]=="crack"}
cache=json.load(open("/home/M11509204/researchs/Solar-Anomaly-Detection/outputs/submissions/../../"+"x" if False else "test_eo_clip_all.json")) if False else None
Hframes=collections.Counter(); 
for k in G:
    if len(cr(H,k)-cr(G,k)): Hframes[dom[k][0]]+=1
print("H-added frames by domain",Hframes)
print("modules per domain", collections.Counter(dom[r[0]][0] for r in ms))
idx=[i for i,r in enumerate(ms) if dom[r[0]][0] in ("G01","G07")]
o=sorted(idx,key=lambda i:-sc[i])
for N in [50,100,200,300]:
    print(N,"frames",len({ms[i][0] for i in o[:N]}),"score>=",round(float(sc[o[N-1]]),2))
json.dump(o[:400],open("dom_rank.json","w"))
