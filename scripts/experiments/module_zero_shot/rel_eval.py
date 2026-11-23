import sys; sys.argv=["raw_eval.py","ir"]
import raw_eval as R, json, numpy as np
from multiprocessing import Pool
from border_eval_lib import label_v
SETS={t:json.load(open(f"raw_ir{s}.json")) for t,s in (("held",""),("cal","_cal"))}
def job(a):
    tag,K,rel,thr=a; r,per=R.score(label_v(SETS[tag],0.5,thr,"raw+std+brightpk",1.0,K,rel)); return a,r,per
if __name__=="__main__":
    cfg=[(2,0.7),(2,0.8),(2,0.85),(2,0.9),(1,0)]; grid=[3.0,3.25,3.5]
    jobs=[(t,K,r,th) for t in ("held","cal") for K,r in cfg for th in grid]
    with Pool(30) as P: D={a:(r,p) for a,r,p in P.map(job,jobs,chunksize=1)}
    for K,r in cfg:
        for th in grid:
            ca,h=D[("cal",K,r,th)],D[("held",K,r,th)]
            print(f"K={K} rel={r} thr {th}: cal {ca[0]:.3f} {ca[1]} held {h[0]:.3f} {h[1]} | mean {(ca[0]+h[0])/2:.3f}")
