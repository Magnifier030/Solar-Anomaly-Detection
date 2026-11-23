import json,numpy as np,cv2
P="/home/M11509204/researchs/Solar-Anomaly-Detection/data/raw/datasets/9_30_test_set/"; O="/home/M11509204/researchs/Solar-Anomaly-Detection/outputs/submissions/"
W=np.array([[1.342,0.043,-0.155],[0.148,1.084,-0.161]]); Ainv=np.linalg.inv(np.r_[W,[[0,0,1]]])
sub=json.load(open(O+"thr35-dinov3L-CLwhitespot-crack100-20261116/submission.json")); dom=json.load(open("test_eo_domain.json"))
want={"G02":2,"G12":2,"G03":1,"G01":1}; tiles=[]
for k in sorted(sub):
    if "/IR/" not in k: continue
    ke=k.replace("/IR/","/EO/"); d=dom[ke][0]
    hs=[o for o in sub[k] if o["label"]=="hotspot"]
    if not hs or want.get(d,0)==0: continue
    want[d]-=1; e=cv2.imread(P+ke); ir=cv2.imread(P+k)
    for o in sub[ke]:
        cv2.polylines(e,[np.int32(o["points"])],True,(0,0,255) if o["label"]=="crack" else (0,200,0),2)
    for o in hs:
        p=np.asarray(o["points"])/[640,512]; q=(np.c_[p,np.ones(len(p))]@Ainv.T)[:,:2]*[1920,1080]
        cv2.polylines(e,[np.int32(q)],True,(255,0,255),5); cv2.polylines(ir,[np.int32(o["points"])],True,(255,0,255),2)
    ir=cv2.resize(ir,(1350,1080)); t=np.hstack([e,ir]); cv2.putText(t,d+" "+ke[-12:],(20,60),0,2,(0,255,255),4); tiles.append(cv2.resize(t,(1635,540)))
cv2.imwrite("ir2eo_vis.jpg",np.vstack(tiles[:6]))
