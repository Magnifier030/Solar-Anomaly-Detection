"""White-spot score per module: grey 256px rectified; residual = g - max(open_hline, open_vline) (keeps compact bright blobs, removes grid lines);
spot_px = #px residual>35, spot_max = p99.9 residual. Usage: spot.py train|test"""
import json,sys,numpy as np,cv2
from pathlib import Path
MODE=sys.argv[1]; sys.argv=["raw_eval.py","eo"]
import raw_eval as R
from rectify import rectify
mm=R.mm; S=Path(__file__).parent; TEST=R.PROJECT/"data/raw/datasets/9_30_test_set"
KH=cv2.getStructuringElement(cv2.MORPH_RECT,(25,1)); KV=cv2.getStructuringElement(cv2.MORPH_RECT,(1,25))
def score(c):
    g=cv2.cvtColor(c,cv2.COLOR_BGR2GRAY) if c.ndim==3 else c
    r=g.astype(np.int16)-np.maximum(cv2.morphologyEx(g,cv2.MORPH_OPEN,KH),cv2.morphologyEx(g,cv2.MORPH_OPEN,KV)).astype(np.int16)
    r=r[8:-8,8:-8]; return [int((r>35).sum()),float(np.percentile(r,99.9))]
out={}
if MODE=="train":
    G=mm.frames_by_group()
    for n in sorted(G):
        try: shapes=mm.load_shapes(n,"eo")
        except Exception: continue
        shapes=[(l,p) for l,p in shapes if l in ("solar_panel","crack") and len(p)>=4]
        if not shapes: continue
        im=mm.load_image(mm.image_path(mm.TRAIN/"EO"/"image",n),"eo")
        for j,(l,p) in enumerate(shapes): out[f"{n}|{j}"]=[G[n],int(l=="crack")]+score(np.ascontiguousarray(rectify(im,p.astype(float),256,inset=0.03),np.uint8))
else:
    eo=json.load(open(S/"test_cache_eo_module-maskrcnn-eo-all.json"))
    for k,mods in eo.items():
        mods=[m for m in mods if m["det_score"]>=0.7]
        if not mods: continue
        im=mm.load_image(TEST/k,"eo")
        for j,m in enumerate(mods): out[f"{k}|{j}"]=score(np.ascontiguousarray(rectify(im,np.asarray(m["polygon"],float),256,inset=0.03),np.uint8))
json.dump(out,open(S/f"spot_{MODE}.json","w")); print("done",len(out))
