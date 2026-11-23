import json,sys,numpy as np
from multiprocessing import Pool
sys.argv=["x","ir"]
import raw_eval as R, border_eval_lib as B
held=json.load(open("raw_ir.json")); cal=json.load(open("raw_ir_cal.json")); dense=json.load(open("raw_ir_dense.json"))
ALL=held+[i for i in cal if i["group"]=="G04"]+dense
def lab(thr_lo,thr_hi,N,K):
    out=[]
    for it in ALL:
        mods=[x for x in it["modules"] if x["det_score"]>=0.5]; keep=np.zeros(len(mods),bool)
        if len(mods)>=3:
            s=B.frame_score(mods,0.5,"raw+std+brightpk",1.0); t=thr_hi if len(mods)>=N else thr_lo
            keep[np.argsort(-s)[:K]]=True; keep&=(s>=t)&(s>=0.7*s.max())
        pred={}
        for x,k in zip(mods,keep): pred.setdefault("hotspot" if k else "solar_panel",[]).append(x["polygon"])
        out.append({"ground_truth":it["truth"],"prediction":pred,"group":it["group"]})
    return out
def job(a):
    lo,hi,N,K=a; s=lab(lo,hi,N,K); r,per=R.score(s)
    hot={g:v for g,v in per.items() if g in("G01","G02","G03","G04")}
    nfp=sum(len(x["prediction"].get("hotspot",[])) for x in s if x["group"] in("G05","G06","G08","G09","G10","G11","G12"))
    return a,r,np.mean(list(hot.values())),hot,nfp
if __name__=="__main__":
    grid=[(3.25,hi,N,2) for hi in (9,10,11,12,14,16,20) for N in (15,20,25,30)]
    with Pool(32) as P:
        for a,r,m,hot,nfp in sorted(P.map(job,grid),key=lambda x:-x[1]):
            print(f"lo {a[0]} hi {a[1]} N {a[2]} K {a[3]}: pooled {r:.3f} grpmean {m:.3f} {hot} normalFP {nfp}",flush=True)
