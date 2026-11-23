"""Train IR frames of hotspot-free dense groups (+ all G02): modules from the all-data Mask R-CNN, zero-shot feats, GT."""
import json, sys
from pathlib import Path
import numpy as np, cv2, torch
sys.argv=["raw_eval.py","ir"]
import raw_eval as R
from rectify import rectify
from zs_score import SIZE, _mutual, _tokens, fill_ratio
from peak_feat import feats
mm=R.mm; S=Path(__file__).parent
G=mm.frames_by_group()
names=sorted(n for n,g in G.items() if g in {"G02","G05","G06","G08","G09","G10","G11","G12"})
dev=torch.device("cuda"); model=mm.build_model("ir",pretrained=False)
model.load_state_dict(torch.load(R.PROJECT/"outputs/module-maskrcnn-ir-all/last.pt",map_location="cpu")["model"]); model.to(dev).eval()
out=[]
for i,n in enumerate(names,1):
    try: shapes=mm.load_shapes(n,"ir")
    except Exception: continue
    image=mm.load_image(mm.image_path(mm.TRAIN/"IR"/"image",n),"ir")
    kept=mm.select(mm.predict(model,image,dev),0.5,0.5,0.5)
    mods=[{"polygon":d["polygon"],"det_score":d["score"],"det_label":d["label"]} for d in kept]
    if len(mods)>=3:
        grey=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY); polys=[np.asarray(m["polygon"],float) for m in mods]
        crops=[np.ascontiguousarray(rectify(image,p,SIZE,inset=0.03),np.uint8) for p in polys]
        raw=_mutual(_tokens(crops),1)
        for m,p,r in zip(mods,polys,raw):
            m["raw"]=float(r); m["std"]=float(rectify(grey,p,64,inset=0.06).astype(np.float32).std()); m["fill"]=float(fill_ratio(p))
            m["pk"],m["pn"],m["med"]=feats(grey,m["polygon"])
    truth={}
    for l,p in shapes: truth.setdefault(l,[]).append(p.tolist())
    out.append({"name":n,"group":G[n],"modules":mods,"truth":truth})
    if i%50==0: print(i,len(names),flush=True)
(S/"raw_ir_dense.json").write_text(json.dumps(out)); print("done",len(out))
