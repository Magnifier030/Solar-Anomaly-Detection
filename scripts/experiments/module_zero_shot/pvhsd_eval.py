"""Module-level hotspot selection on PV-HSD (unseen dense sites). GT = modules containing a box centre."""
import json,sys,numpy as np
sys.argv=["x","ir"]
import border_eval_lib as B
D=json.load(open("raw_ir_pvhsd.json"))
F=[]
for f in D:
    mods=[m for m in f["modules"] if m["det_score"]>=0.5 and "raw" in m]
    if len(mods)<3: continue
    s=B.frame_score(mods,0.5,"raw+std+brightpk",1.0)
    F.append((s,np.array([m["hot"]>0 for m in mods]),len(mods)))
G=sum(h.sum() for _,h,_ in F); print("frames",len(F),"hot modules",G,"modules",sum(n for *_,n in F),"base rate",G/sum(n for *_,n in F))
# marginal precision by within-frame rank
for r in range(8):
    tp=sum(h[np.argsort(-s)[r]] for s,h,n in F if n>r); print(f"rank {r+1}: precision {tp/len(F):.3f}")
# precision by score bin
allS=np.concatenate([s for s,_,_ in F]); allH=np.concatenate([h for _,h,_ in F])
for lo,hi in [(2,2.5),(2.5,3.25),(3.25,4),(4,5),(5,7),(7,10),(10,99)]:
    m=(allS>=lo)&(allS<hi); print(f"score [{lo},{hi}): n {m.sum():5d} precision {allH[m].mean() if m.sum() else 0:.3f}")
def run(thr,K,rel):
    tp=npred=0
    for s,h,n in F:
        k=np.zeros(n,bool); k[np.argsort(-s)[:K]]=True; k&=(s>=thr)&(s>=rel*s.max()); tp+=(k&h).sum(); npred+=k.sum()
    return tp,npred,2*tp/(npred+G)
for K in (2,3,4,6,10,999):
  for thr in (2.0,2.5,3.25,4,5):
    for rel in (0.7,0.0):
      tp,n,f=run(thr,K,rel); print(f"K {K:3d} thr {thr} rel {rel}: pred {n:5d} tp {tp:4d} prec {tp/max(n,1):.3f} rec {tp/G:.3f} F1 {f:.3f}")
