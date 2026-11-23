import json,sys,numpy as np
sys.argv=["x","ir"]
import raw_eval as R, border_eval_lib as B
def stats(frames):
    nover=[];mx=[]
    for mods in frames:
        mods=[m for m in mods if m["det_score"]>=0.5]
        if len(mods)<3 or not all("pk" in m for m in mods): continue
        s=B.frame_score(mods,0.5,"raw+std+brightpk",1.0)
        mx.append(s.max()); nover.append(int(((s>=3.25)&(s>=0.7*s.max())).sum()))
    nover=np.array(nover); mx=np.array(mx)
    return len(nover), (nover>0).mean(), np.bincount(np.minimum(nover,6),minlength=7)/len(nover), np.percentile(mx,[25,50,75,90])
for tag,f in (("held","raw_ir.json"),("cal","raw_ir_cal.json")):
    items=json.load(open(f))
    by={}
    for it in items: by.setdefault(it["group"],[]).append(it)
    for g,its in sorted(by.items()):
        ngt=np.array([sum(1 for s in it["truth"].get("hotspot",[])) if isinstance(it["truth"],dict) else 0 for it in its])
        n,fl,bc,mx=stats([it["modules"] for it in its])
        print(tag,g,"frames",n,"GT-frame frac %.2f"%(ngt>0).mean(),"flag frac %.2f"%fl,"nover",np.round(bc,2),"max pct",np.round(mx,1))
t=json.load(open("test_cache_ir_module-maskrcnn-ir-all.json"))
import cv2
pseudo=json.load(open("site_match.json")) if False else None
n,fl,bc,mx=stats(list(t.values()))
print("test frames",n,"flag frac %.2f"%fl,"nover",np.round(bc,2),"max pct",np.round(mx,1))
