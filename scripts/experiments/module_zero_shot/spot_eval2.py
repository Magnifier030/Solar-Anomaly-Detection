import spot_eval as E
from multiprocessing import Pool
if __name__=="__main__":
    grid=[("sp7",0,t) for t in (3.75,4.5,5.0)]+[("sp7",0.5,t) for t in (4.0,5.0,5.5)]+[("sp7",0.25,t) for t in (3.75,4.5)]
    with Pool(16) as P:
        for a,r,m,hot,pvr in P.map(E.job,grid): print(a,f"pooled {r:.3f} grpmean {m:.3f}",hot,f"PV prec {pvr[0]:.3f} n {pvr[2]}",flush=True)
