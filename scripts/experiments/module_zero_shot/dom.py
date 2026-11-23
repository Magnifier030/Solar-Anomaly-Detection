import json,numpy as np,collections
mt=json.load(open("crackfeat_train_meta.json")); Xt=np.load("crackfeat_train_dino.npy")[:,:768]
ms=json.load(open("crackfeat_test_meta.json")); Xs=np.load("crackfeat_test_dino.npy")[:,:768]
def frames(meta,X):
    d=collections.defaultdict(list)
    for i,r in enumerate(meta): d[r[0]].append(i)
    names=sorted(d); F=np.stack([X[d[n]].mean(0) for n in names]); F/=np.linalg.norm(F,axis=1,keepdims=True); return names,F,d
tn,TF,td=frames(mt,Xt); sn,SF,sd=frames(ms,Xs)
tg={r[0]:r[1] for r in mt}; tg=np.array([tg[n] for n in tn])
# LOO sanity on train frames: nearest-neighbour group accuracy
Sim=TF@TF.T; np.fill_diagonal(Sim,-1); print("train frame 1-NN group acc",(tg[Sim.argmax(1)]==tg).mean())
S2=SF@TF.T; nn=S2.argsort(1)[:,::-1][:,:5]
assign=[collections.Counter(tg[r]).most_common(1)[0][0] for r in nn]; top=S2.max(1)
print("test frames per nearest train group",collections.Counter(assign))
G=json.load(open("/home/M11509204/researchs/Solar-Anomaly-Detection/outputs/submissions/ir-v6union-sp7-diode-addcrack-zlow2-20261107/submission.json"))
A=json.load(open("/home/M11509204/researchs/Solar-Anomaly-Detection/outputs/submissions/ir-v3-peak-diode-clipgatecrack-20261106/submission.json"))
for nm,sub in [("A",A),("G",G)]:
    c=collections.Counter()
    for n,a in zip(sn,assign):
        c[a]+=sum(o["label"]=="crack" for o in sub.get(n.replace("test/","",0),[]) or sub.get(n.split("/")[-1],[]))
    print(nm,"cracks by domain",c, "total",sum(o["label"]=="crack" for v in sub.values() for o in v))
print(list(G)[:3])
json.dump({n:[a,float(t)] for n,a,t in zip(sn,assign,top)},open("test_eo_domain.json","w"))
