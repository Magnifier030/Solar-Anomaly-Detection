import json,sys,numpy as np
from multiprocessing import Pool
sys.argv=["x","ir"]
import raw_eval as R, border_eval_lib as B
held=json.load(open("raw_ir_z6.json")); cal=json.load(open("raw_ir_cal_z6.json")); dense=json.load(open("raw_ir_dense_z6.json"))
SETS={"held":held,"cal":[i for i in cal if i["group"]=="G04"]+[i for i in dense if i["group"]=="G02"]}
def zr(mods,mode,c=0.5):
    if mode=="id": return R.z(np.array([x["raw"] for x in mods]),c)
    Z=np.sort(np.array([R.z(np.array([x["raw6"][k] for x in mods]),c) for k in range(6)]),0)
    return {"zlow2":Z[:2].mean(0),"zmed":np.median(Z,0),"noZoomLow2":None}[mode]
def score(mods,mode):
    z=lambda k:R.z(np.array([x[k] for x in mods]),0.5)
    s=zr(mods,mode)+np.abs(z("std"))+z("pk")
    s=np.where(np.array([x["fill"] for x in mods])<R.FILL_MIN,0,s)
    return np.where(np.array([B.touches(x["polygon"],5) for x in mods]),-1e9,s)
def lab(items,mode,thr,K=2,rel=0.7):
    out=[]
    for it in items:
        mods=[x for x in it["modules"] if x["det_score"]>=0.5]; keep=np.zeros(len(mods),bool)
        if len(mods)>=3 and all("raw6" in x for x in mods):
            s=score(mods,mode); keep[np.argsort(-s)[:K]]=True; keep&=(s>=thr)&(s>=rel*s.max())
        pred={}
        for x,k in zip(mods,keep): pred.setdefault("hotspot" if k else "solar_panel",[]).append(x["polygon"])
        out.append({"ground_truth":it["truth"],"prediction":pred,"group":it["group"]})
    return out
def job(a):
    mode,thr=a; r={}
    for n,it in SETS.items(): r[n]=R.score(lab(it,mode,thr))
    return a,r
if __name__=="__main__":
    with Pool(24) as P:
        for (mode,thr),r in P.map(job,[(m,t) for m in ("id","zlow2","zmed") for t in (2.5,3.0,3.25,3.5,4.0)]):
            print(f"{mode:6s} thr {thr}: held {r['held'][0]:.3f} {r['held'][1]} cal {r['cal'][0]:.3f} {r['cal'][1]} mean {(r['held'][0]+r['cal'][0])/2:.3f}",flush=True)
