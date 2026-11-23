import sys; sys.argv=["raw_eval.py","ir"]
import raw_eval as R, json, numpy as np, itertools
from multiprocessing import Pool
from border_eval_lib import label
SETS={t:json.load(open(f"raw_ir{s}.json")) for t,s in (("held",""),("cal","_cal"))}
def job(a):
    tag,c,m,pen,thr=a; r,per=R.score(label(SETS[tag],c,thr,m,pen)); return a,r,per
if __name__=="__main__":
    cfg=[(None,0),(2,1e9),(5,1e9),(2,1.0),(2,2.0),(5,2.0)]
    grid=list(np.arange(1.5,6.01,0.5))
    jobs=[(t,c,m,p,float(th)) for t in ("held","cal") for c in (0.3,0.5,0.75) for m,p in cfg for th in grid]
    with Pool(60) as P: res=P.map(job,jobs,chunksize=1)
    D={a:(r,per) for a,r,per in res}
    out=[]
    for c in (0.3,0.5,0.75):
        for m,p in cfg:
            best=max(grid,key=lambda th:D[("cal",c,m,p,float(th))][0])
            h=D[("held",c,m,p,float(best))]; o=max(D[("held",c,m,p,float(th))][0] for th in grid)
            print(f"c={c} m={m} pen={p}: cal thr {best} cal {D[('cal',c,m,p,float(best))][0]:.3f} {D[('cal',c,m,p,float(best))][1]} held {h[0]:.3f} {h[1]} | held-oracle {o:.3f}",flush=True)
    json.dump([[list(map(str,a)),r,per] for a,(r,per) in D.items()],open("border_fast.json","w"))
