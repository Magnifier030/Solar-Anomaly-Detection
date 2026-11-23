import json,numpy as np,cv2,collections
from rectify import rectify
TEST="/home/M11509204/researchs/Solar-Anomaly-Detection/data/raw/datasets/9_30_test_set/"
X=np.load("crackbig_train.npy")[:,:1024].astype(np.float32); M=json.load(open("crackbig_train_meta.json"))
clean=np.array([m[4]==0 for m in M]); X=X[clean]; X/=np.linalg.norm(X,axis=1,keepdims=True)
# panel-ness = max cosine to 2000 random clean train modules' mean tokens
rng=np.random.default_rng(0); B=X[rng.choice(len(X),3000,replace=False)]
tr=np.sort(X@B.T,axis=1)[:,-2]   # exclude self-match
thr=0.75; print("panel-ness train p1",thr)
T=np.load("crackbig_test.npy")[:,:1024].astype(np.float32); T/=np.linalg.norm(T,axis=1,keepdims=True)
pn=(T@B.T).max(1); np.save("crackbig_test_panelness.npy",pn)
TM=json.load(open("crackbig_test_meta.json")); eo=json.load(open("test_cache_eo_module-maskrcnn-eo-all.json")); mods={k:[m for m in v if m["det_score"]>=0.7] for k,v in eo.items()}
cand=json.load(open("big_cand.json"))
keep=[i for i in cand if pn[i]>=thr]
print("kept",len(keep),"of",len(cand)); print("rejected in top80:",[r for r,i in enumerate(cand[:80]) if pn[i]<thr])
# shadow: large dark low-frequency blob
def shadow(c):
    g=cv2.cvtColor(c,cv2.COLOR_BGR2GRAY).astype(np.float32); b=cv2.GaussianBlur(g,(0,0),12); return float(np.percentile(b,97)-np.percentile(b,3))
out=[];tiles=[]
for i in keep:
    k,_,_,j,_=TM[i]; c=rectify(cv2.imread(TEST+k),np.asarray(mods[k][j]["polygon"],float),160,inset=0.03).astype(np.uint8).copy()
    if shadow(c)>60: continue
    out.append(i)
    if len(tiles)<80: cv2.putText(c,str(len(out)-1),(3,15),0,0.5,(0,255,255),1); tiles.append(c)
print("after shadow",len(out)); json.dump(out,open("big_cand_f.json","w"))
cv2.imwrite("big_f80.jpg",np.vstack([np.hstack(tiles[i:i+10]) for i in range(0,80,10)]))
