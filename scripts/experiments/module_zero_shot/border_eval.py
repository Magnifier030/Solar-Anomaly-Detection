import sys; sys.argv=["raw_eval.py","ir"]
import raw_eval as R, json, numpy as np
W,H=640,512
def touches(p,m):
    p=np.asarray(p); return p[:,0].min()<=m or p[:,1].min()<=m or p[:,0].max()>=W-1-m or p[:,1].max()>=H-1-m
sets={t:json.load(open(f"raw_ir{s}.json")) for t,s in (("held",""),("cal","_cal"))}
# GT hotspots touching border
for t,it in sets.items():
    n=b=0
    for i in it:
        for p in i["truth"].get("hotspot",[]): n+=1; b+=touches(p,3)
    print(t,"GT hotspots",n,"touch border",b)
def label(items,c,thr,m,extra):
    out=[]
    for it in items:
        mods=[x for x in it["modules"] if x["det_score"]>=0.5]; pred={}
        if len(mods)>=3:
            s=R.z(np.array([x["raw"] for x in mods]),c)+np.abs(R.z(np.array([x["std"] for x in mods]),c))
            s=np.where(np.array([x["fill"] for x in mods])<R.FILL_MIN,0,s)
            if m is not None:
                s=s-np.array([extra if touches(x["polygon"],m) else 0 for x in mods])
        else: s=np.zeros(len(mods))
        for x,v in zip(mods,s): pred.setdefault("hotspot" if v>=thr else "solar_panel",[]).append(x["polygon"])
        out.append({"ground_truth":it["truth"],"prediction":pred,"group":it["group"]})
    return out
grid=np.arange(1,8.01,0.25)
for c in (0.3,0.5,0.75):
  for m,extra in ((None,0),(2,1e9),(5,1e9),(2,2.0),(5,2.0)):
    best=max(grid,key=lambda t:R.score(label(sets["cal"],c,t,m,extra))[0])
    cal=R.score(label(sets["cal"],c,best,m,extra))[0]; held,per=R.score(label(sets["held"],c,best,m,extra))
    fixed,per25=R.score(label(sets["held"],c,2.5,m,extra))
    print(f"c={c} border m={m} pen={extra}: cal thr {best} cal {cal:.3f} held {held:.3f} {per} | thr2.5 held {fixed:.3f} {per25}",flush=True)
