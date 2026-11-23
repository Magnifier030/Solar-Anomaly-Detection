import json,sys,numpy as np
from multiprocessing import Pool
sys.argv=["x","ir"]
import raw_eval as R, border_eval_lib as B
from clip_crack import roc_auc_score as auc
def attach(fn):
    it=json.load(open(fn)); sp=json.load(open(fn.replace(".json","_sp.json")))
    for i,s in zip(it,sp):
        for m,v in zip(i["modules"],s): m["sp7"],m["sp15"],m["sp31"]=v
    return it
held=attach("raw_ir.json"); cal=attach("raw_ir_cal.json"); dense=attach("raw_ir_dense.json"); pv=attach("raw_ir_pvhsd.json")
ALL=held+[i for i in cal if i["group"]=="G04"]+dense
def fs(mods,k,w):
    s=B.frame_score(mods,0.5,"raw+std+brightpk",1.0)
    if w: s=s+w*R.z(np.array([x[k] for x in mods]),0.5)
    return s
# PV-HSD AUC of raw spot feats and of combos
H=[];cols={}
for f in pv:
    m=[x for x in f["modules"] if x["det_score"]>=0.5 and "raw" in x]
    if len(m)<3: continue
    H+=[x["hot"]>0 for x in m]
    for k in ("sp7","sp15","sp31"): cols.setdefault(k,[]).extend(R.z(np.array([x[k] for x in m]),0.5))
H=np.array(H)
for k,v in cols.items(): print("PVHSD AUC",k,round(auc(H,np.array(v)),3))
def pv_f1(k,w,thr,K):
    tp=n=0;G=0
    for f in pv:
        m=[x for x in f["modules"] if x["det_score"]>=0.5 and "raw" in x]
        if len(m)<3: continue
        h=np.array([x["hot"]>0 for x in m]); G+=h.sum()
        s=fs(m,k,w); kp=np.zeros(len(m),bool); kp[np.argsort(-s)[:K]]=True; kp&=(s>=thr)&(s>=0.7*s.max()); tp+=(kp&h).sum(); n+=kp.sum()
    return tp/max(n,1), 2*tp/(n+G), n
def lab(k,w,thr,K):
    out=[]
    for it in ALL:
        mods=[x for x in it["modules"] if x["det_score"]>=0.5]; keep=np.zeros(len(mods),bool)
        if len(mods)>=3:
            s=fs(mods,k,w); keep[np.argsort(-s)[:K]]=True; keep&=(s>=thr)&(s>=0.7*s.max())
        pred={}
        for x,kk in zip(mods,keep): pred.setdefault("hotspot" if kk else "solar_panel",[]).append(x["polygon"])
        out.append({"ground_truth":it["truth"],"prediction":pred,"group":it["group"]})
    return out
def job(a):
    k,w,thr=a; s=lab(k,w,thr,2); r,per=R.score(s)
    hot={g:v for g,v in per.items() if g in("G01","G02","G03","G04")}
    return a,r,np.mean(list(hot.values())),hot,pv_f1(k,w,thr,2)
if __name__=="__main__":
    grid=[("sp7",0,3.25)]+[(k,w,t) for k in ("sp7","sp15","sp31") for w in (0.5,1.0,2.0) for t in (3.25,4.5)]
    with Pool(32) as P:
        for a,r,m,hot,pvr in P.map(job,grid):
            print(f"{a}: pooled {r:.3f} grpmean {m:.3f} {hot} | PVHSD prec {pvr[0]:.3f} F1 {pvr[1]:.3f} n {pvr[2]}",flush=True)
