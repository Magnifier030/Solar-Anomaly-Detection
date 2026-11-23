"""usage: hot_sheet.py LOGIT_TAG -> hot_cmp{TAG}.jpg rows: picks lowest CL (3 rows), top CL not picked (3 rows), crops shown in ORIGINAL colours"""
import json,sys,numpy as np,cv2
P="/home/M11509204/researchs/Solar-Anomaly-Detection/"; T=P+"data/raw/datasets/9_30_test_set/"
TM=json.load(open("hotcl_test_meta.json")); ir=json.load(open("test_cache_ir_module-maskrcnn-ir-all.json"))
pos={};poly={}
for k,mods in ir.items():
    for j,m in enumerate([m for m in mods if m["det_score"]>=.5]): pos[(k,json.dumps(m["polygon"]))]=j; poly[(k,j)]=np.asarray(m["polygon"],float)
idx={(m[0],m[3]):i for i,m in enumerate(TM)}
sub=json.load(open(P+"outputs/submissions/thr35-CLv3shadow-fixedpipe-crack100-20261116/submission.json"))
H=[idx[(k,pos[(k,json.dumps(o["points"]))])] for k,v in sub.items() for o in v if o["label"]=="hotspot" and (k,json.dumps(o["points"])) in pos]
L=np.load(f"hotcl_test_logit{sys.argv[1]}.npy"); Hs=set(H)
low=sorted(H,key=lambda i:L[i])[:36]; top=[i for i in np.argsort(-L) if i not in Hs][:36]
def tile(i):
    k,_,_,j,_=TM[i]; im=cv2.imread(T+k); p=poly[(k,j)]; x0,y0=p.min(0); x1,y1=p.max(0); w,h=x1-x0,y1-y0
    X0,Y0=int(max(0,x0-w*.7)),int(max(0,y0-h*.7)); X1,Y1=int(min(im.shape[1],x1+w*.7)),int(min(im.shape[0],y1+h*.7))
    c=im.copy(); cv2.polylines(c,[p.astype(np.int32)],True,(0,255,0),1); c=cv2.resize(c[Y0:Y1,X0:X1],(130,130))
    cv2.putText(c,f"{k[-7:-4]} {L[i]:.1f}",(2,12),0,.4,(0,255,255),1); return c
rows=[np.hstack([tile(i) for i in S_[r*12:(r+1)*12]]) for S_ in (low,top) for r in range(3)]
sep=np.full((6,rows[0].shape[1],3),255,np.uint8)
cv2.imwrite(f"hot_cmp{sys.argv[1]}.jpg",np.vstack(rows[:3]+[sep]+rows[3:]))
