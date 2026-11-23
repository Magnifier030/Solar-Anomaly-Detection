"""Per-frame EO<->IR registration by ECC on module masks (init = train-G04 affine). Then: do CL crack modules coincide with predicted IR hotspots
more than random modules of the same frames? Saves per-frame warp (IR px -> EO px) in reg_warps.json."""
import json,collections,numpy as np,cv2
from shapely.geometry import Polygon,Point
O="/home/M11509204/researchs/Solar-Anomaly-Detection/outputs/submissions/"
sub=json.load(open(O+"thr35-dinov3L-CLwhitespot-crack100-20261116/submission.json")); dom=json.load(open("test_eo_domain.json"))
W=np.array([[1.342,0.043,-0.155],[0.148,1.084,-0.161]]); Ainv=np.linalg.inv(np.r_[W,[[0,0,1]]])
Se=np.diag([960,540,1.]); Si=np.diag([640,512,1.]); init=(Se@Ainv@np.linalg.inv(Si))[:2].astype(np.float32)   # IR px -> EO px (half-res EO)
def mask(objs,shape,scale):
    m=np.zeros(shape,np.uint8)
    for o in objs: cv2.fillPoly(m,[np.int32(np.asarray(o["points"])*scale)],255)
    return cv2.GaussianBlur(m,(0,0),3).astype(np.float32)/255
warps={}; stat=collections.Counter(); ious=[]
for k in sub:
    if "/IR/" not in k: continue
    ke=k.replace("/IR/","/EO/")
    if not sub[k] or not sub.get(ke): continue
    mi=mask(sub[k],(512,640),1); me=mask(sub[ke],(540,960),.5)
    def iou(w):
        we=cv2.warpAffine(me,w,(640,512),flags=cv2.INTER_LINEAR+cv2.WARP_INVERSE_MAP); a=mi>.5;b=we>.5; return (a&b).sum()/max((a|b).sum(),1)
    i0=iou(init); w=init.copy()
    try:
        _,w=cv2.findTransformECC(mi,me,w,cv2.MOTION_AFFINE,(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,200,1e-5),None,5)
        i1=iou(w)
    except cv2.error: i1=-1
    if i1<i0: w=init; i1=i0; stat["kept init"]+=1
    warps[ke]=[w.tolist(),float(i0),float(i1),dom[ke][0]]; ious.append((dom[ke][0],i0,i1))
json.dump(warps,open("reg_warps.json","w"))
d=collections.defaultdict(list)
for g,a,b in ious: d[g].append((a,b))
print("mask IoU before->after registration per EO domain:")
for g,v in sorted(d.items()): v=np.array(v); print(f"  {g} n={len(v)} {np.median(v[:,0]):.2f} -> {np.median(v[:,1]):.2f} | frac after>.6 {np.mean(v[:,1]>.6):.2f}")
print(dict(stat))
