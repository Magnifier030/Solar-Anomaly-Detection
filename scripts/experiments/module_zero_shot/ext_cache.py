"""PV-HSD-2025 valid split as unseen-site dense IR validation: modules (Mask R-CNN ir-all), v3 feats, hotspot = module containing a GT box centre."""
import json, sys, glob, random
from pathlib import Path
import numpy as np, cv2, torch
N=int(sys.argv[1]) if len(sys.argv)>1 else 500
sys.argv=["raw_eval.py","ir"]
import raw_eval as R
from rectify import rectify
from zs_score import SIZE, _mutual, _tokens, fill_ratio
from peak_feat import feats
mm=R.mm; S=Path(__file__).parent
ims=sorted(glob.glob(str(R.PROJECT/"data/raw/external/PV-HSD-2025/extracted/valid/images/*.JPG")))
random.seed(0); random.shuffle(ims); ims=ims[:N]
dev=torch.device("cuda"); model=mm.build_model("ir",pretrained=False)
model.load_state_dict(torch.load(R.PROJECT/"outputs/module-maskrcnn-ir-all/last.pt",map_location="cpu")["model"]); model.to(dev).eval()
out=[]
for i,p in enumerate(ims,1):
    image=cv2.cvtColor(cv2.imread(p,cv2.IMREAD_GRAYSCALE),cv2.COLOR_GRAY2RGB)
    if image.shape[:2]!=(512,640): continue
    h,w=image.shape[:2]; boxes=[]
    for l in open(p.replace("/images/","/labels/").rsplit(".",1)[0]+".txt"):
        c,x,y,bw,bh=map(float,l.split()); boxes.append([x*w,y*h,bw*w,bh*h])
    kept=mm.select(mm.predict(model,image,dev),0.5,0.5,0.5)
    mods=[{"polygon":d["polygon"],"det_score":d["score"]} for d in kept]
    if len(mods)>=3:
        grey=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY); polys=[np.asarray(m["polygon"],float) for m in mods]
        crops=[np.ascontiguousarray(rectify(image,q,SIZE,inset=0.03),np.uint8) for q in polys]
        raw=_mutual(_tokens(crops),1)
        for m,q,r in zip(mods,polys,raw):
            m["raw"]=float(r); m["std"]=float(rectify(grey,q,64,inset=0.06).astype(np.float32).std()); m["fill"]=float(fill_ratio(q))
            m["pk"],m["pn"],m["med"]=feats(grey,m["polygon"])
            c=q.astype(np.float32).reshape(-1,1,2)
            m["hot"]=int(sum(cv2.pointPolygonTest(c,(b[0],b[1]),False)>=0 for b in boxes))
    out.append({"name":Path(p).name,"modules":mods,"boxes":boxes})
    if i%50==0: print(i,len(ims),flush=True)
(S/"raw_ir_pvhsd.json").write_text(json.dumps(out)); print("done",len(out))
