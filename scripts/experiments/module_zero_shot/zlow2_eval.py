import numpy as np, sys
sys.argv=["x","eo"]
import clip_crack as C
a=np.load("clip_window.npy"); m=np.load("clip_window_meta.npy",allow_pickle=True); zl=np.load("zlow2_eo.npy")
assert len(zl)==len(a), (len(zl),len(a))
g,fr=m[:,1],m[:,2]; y=a[:,0].astype(bool); clip=a[:,2]
S={"identity z":a[:,1],"zmin":zl[:,0],"zlow2":zl[:,1],"zmed":zl[:,2],"zmean":zl[:,3]}
for grp in ("G01","G07","G04"):
    k=g==grp; print(grp," ".join(f"{n} {C.roc_auc_score(y[k],v[k]):.3f}" for n,v in S.items()))
unseen=np.isin(g,["G01","G03","G07"]); normal=~np.isin(g,["G01","G03","G07","G04"])
def rule(mask,s,t,p):
    tp=fp=0
    for f in np.unique(fr[mask]):
        kk=np.where(fr==f)[0]; j=kk[np.argmax(s[kk])]
        if s[j]>=t and clip[j]>=p: tp+=y[j]; fp+=not y[j]
    return tp,fp,y[mask].sum()-tp
for n,s in S.items():
    for t in (0.5,1.0,1.5,2.0):
        u=rule(unseen,s,t,0.5); nn=rule(normal,s,t,0.5); g4=rule(g=="G04",s,t,0.5)
        print(f"{n:10s} t>={t}: unseen tp {u[0]} fp {u[1]} F1 {2*u[0]/max(2*u[0]+u[1]+u[2],1):.3f} | normal fp {nn[1]} | G04 tp {g4[0]} fp {g4[1]}")
