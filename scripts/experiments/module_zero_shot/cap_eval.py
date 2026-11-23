import sys; sys.argv=["raw_eval.py","ir"]
import raw_eval as R, json, numpy as np
from multiprocessing import Pool
from border_eval_lib import label_cap
SETS={t:json.load(open(f"raw_ir{s}.json")) for t,s in (("held",""),("cal","_cal"))}
for t,it in SETS.items(): print(t,"GT/frame hist",np.bincount([len(i["truth"].get("hotspot",[])) for i in it]))
def job(a):
    tag,c,K,rel,thr=a; r,per=R.score(label_cap(SETS[tag],c,thr,5,K,rel)); return a,r,per
if __name__=="__main__":
    cfg=[(2,0),(2,0.7)]
    grid=[1.25,1.5,1.75,2.0,2.25]
    jobs=[(t,c,K,rel,th) for t in ("held","cal") for c in (0.4,0.5,0.6) for K,rel in cfg for th in grid]
    with Pool(60) as P: D={a:(r,p) for a,r,p in P.map(job,jobs,chunksize=1)}
    for c in (0.4,0.5,0.6):
        for K,rel in cfg:
            for th in grid:
                ca,h=D[("cal",c,K,rel,th)],D[("held",c,K,rel,th)]
                print(f"c={c} K={K} rel={rel} thr {th}: cal {ca[0]:.3f} {ca[1]} held {h[0]:.3f} {h[1]} | mean {(ca[0]+h[0])/2:.3f}")
