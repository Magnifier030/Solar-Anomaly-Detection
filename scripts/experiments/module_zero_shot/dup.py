import json,collections,numpy as np,cv2
TEST="/home/M11509204/researchs/Solar-Anomaly-Detection/data/raw/datasets/9_30_test_set/"
O="/home/M11509204/researchs/Solar-Anomaly-Detection/outputs/submissions/"
dom=json.load(open("test_eo_domain.json")); keys=sorted(dom)
# global thumbnail descriptor: 64x36 grey, z-normalised -> near-duplicate frames
def th(k): g=cv2.resize(cv2.imread(TEST+k,0),(96,54)).astype(np.float32); return ((g-g.mean())/(g.std()+1e-6)).ravel()
A=np.stack([th(k) for k in keys]); A/=np.linalg.norm(A,axis=1,keepdims=True); Sm=A@A.T; np.fill_diagonal(Sm,-1)
nn=Sm.max(1); print("test frames",len(keys),"| nearest-neighbour corr quantiles",np.round(np.percentile(nn,[10,50,90,99]),3))
print("frames with a near-duplicate (corr>.9):",int((nn>.9).sum()),"(>.8):",int((nn>.8).sum()))
c=json.load(open(O+"thr35-dinov3L-CLwhitespot-crack100-20261116/submission.json")); cf={k for k,v in c.items() for o in v if o["label"]=="crack"}
ci=[keys.index(k) for k in cf]
print("crack frames",len(ci),"| their NN corr median",round(float(np.median(nn[ci])),3),"| crack frames whose NN is another crack frame",sum(keys[int(Sm[i].argmax())] in cf for i in ci))
# adjacency in id order
ids=sorted(int(k.split("_")[-1][:3]) for k in cf); print("adjacent-id crack frame pairs",sum(b-a==1 for a,b in zip(ids,ids[1:])))
idx={k:i for i,k in enumerate(keys)}
adj=[Sm[idx[f"test/EO/image/test_{i:03d}.jpg"],idx[f"test/EO/image/test_{i+1:03d}.jpg"]] for i in range(1,696) if f"test/EO/image/test_{i+1:03d}.jpg" in idx and f"test/EO/image/test_{i:03d}.jpg" in idx]
print("corr of consecutive ids median",round(float(np.median(adj)),3))
np.save("test_frame_corr.npy",Sm); json.dump(keys,open("test_frame_keys.json","w"))
