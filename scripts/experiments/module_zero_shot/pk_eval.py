import sys; sys.argv=["raw_eval.py","ir"]
import raw_eval as R, json, numpy as np
from multiprocessing import Pool
from border_eval_lib import label_v
SETS={t:json.load(open(f"raw_ir{s}.json")) for t,s in (("held",""),("cal","_cal"))}
def job(a):
    tag,v,w,thr=a; r,per=R.score(label_v(SETS[tag],0.5,thr,v,w)); return a,r,per
if __name__=="__main__":
    cfg=[("raw+std+brightpk",0.75),("raw+std+brightpk",1.25),("raw+std+brightpk",1.5)]
    grid=[2.5,3.0,3.5,4.0,4.5]
    jobs=[(t,v,w,th) for t in ("held","cal") for v,w in cfg for th in grid]
    with Pool(60) as P: D={a:(r,p) for a,r,p in P.map(job,jobs,chunksize=1)}
    for v,w in cfg:
        for th in grid:
            ca,h=D[("cal",v,w,th)],D[("held",v,w,th)]
            print(f"{v} a={w} thr {th}: cal {ca[0]:.3f} {ca[1]} held {h[0]:.3f} {h[1]} | mean {(ca[0]+h[0])/2:.3f}")
