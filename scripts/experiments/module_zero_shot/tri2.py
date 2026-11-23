import json,numpy as np,cv2,collections
from rectify import rectify
TEST="/home/M11509204/researchs/Solar-Anomaly-Detection/data/raw/datasets/9_30_test_set/"
X=np.load("crackbig3_train.npy").astype(np.float32); M=json.load(open("crackbig3_train_meta.json"))
T=np.load("crackbig3_test.npy").astype(np.float32); TM=json.load(open("crackbig3_test_meta.json")); dom=json.load(open("test_eo_domain.json"))
def nz(A): A=A[:,2048:3072]; return A/np.linalg.norm(A,axis=1,keepdims=True)   # mean deviation of top-K deviant patches
k=[i for i,m in enumerate(M) if m[1]=="G01" and m[2]==1 and m[4]==0]; proto=nz(X[k]).mean(0); proto/=np.linalg.norm(proto)
s=nz(T)@proto; o=np.argsort(-s)
add={tuple(x) for x in json.load(open("tmp_crackcl/manifest.json"))["added"]}
inCL=np.array([(TM[i][0],TM[i][3]) in add for i in range(len(TM))])
print("score of CL picks that are triangle-like: top quantiles",np.round(np.sort(s[inCL])[::-1][:35:5],3))
eo=json.load(open("test_cache_eo_module-maskrcnn-eo-all.json")); mods={kk:[m for m in v if m["det_score"]>=0.7] for kk,v in eo.items()}
tiles=[]
for r,i in enumerate(o[:80]):
    kk,_,_,j,_=TM[i]; c=rectify(cv2.imread(TEST+kk),np.asarray(mods[kk][j]["polygon"],float),160,inset=0.03).astype(np.uint8).copy()
    cv2.putText(c,f"{r}{'*' if inCL[i] else ''} {dom[kk][0]}",(3,15),0,.45,(0,255,255),1); tiles.append(c)
cv2.imwrite("tri_top80.jpg",np.vstack([np.hstack(tiles[i:i+10]) for i in range(0,80,10)]))
print("top80 by triangle-prototype: in CL",int(inCL[o[:80]].sum()),"| frames",len({TM[i][0] for i in o[:80]}),collections.Counter(dom[TM[i][0]][0] for i in o[:80]).most_common())
np.save("tri_scores.npy",s)
