import sys; sys.argv=["raw_eval.py","ir"]
import raw_eval as R, json, numpy as np
from multiprocessing import Pool
from border_eval_lib import label_area
SETS={t:json.load(open(f"raw_ir{s}.json")) for t,s in (("held",""),("cal","_cal"))}
def job(a):
    tag,c,m,amin,thr=a; r,per=R.score(label_area(SETS[tag],c,thr,m,amin)); return a,r,per
if __name__=="__main__":
    grid=[1.5,2.0,2.5,3.0,3.5]; cfg=[(5,0.7),(5,0.85),(5,0.95),(5,99)]
    jobs=[(t,c,m,a,th) for t in ("held","cal") for c in (0.3,0.5,0.75) for m,a in cfg for th in grid]
    with Pool(60) as P: D={a:(r,p) for a,r,p in P.map(job,jobs,chunksize=1)}
    for c in (0.3,0.5,0.75):
        for m,a in cfg:
            for th in grid:
                ca,h=D[("cal",c,m,a,th)],D[("held",c,m,a,th)]
                print(f"c={c} area<{a} thr {th}: cal {ca[0]:.3f} {ca[1]} held {h[0]:.3f} {h[1]} | mean {(ca[0]+h[0])/2:.3f}")
