"""CL v2: add REAL white-spot negatives = v2 ranks 101-500 (platform ~2% precision), split by frame into halves A/B:
train with half A as white-spot class, measure how many half-B items reach the top-100 (held-out junk). Swap and average."""
import os,json,numpy as np
os.environ.setdefault("ARCH","lin"); os.environ.setdefault("EP","1500")
src=open("crack_cl.py").read().split('# labels: 0 normal')[0]; exec(src)
lab_tr=y.copy(); fr=sorted({TM[i][0] for i in v2b}); g=np.random.default_rng(0); g.shuffle(fr); half={f:k%2 for k,f in enumerate(fr)}
A=[i for i in v2b if half[TM[i][0]]==0]; B=[i for i in v2b if half[TM[i][0]]==1]
base=np.load("crackcl_test_scores_pneg.npy"); ob=np.argsort(-base)
def top(o,L,n=100): return sum(i in set(L) for i in o[:n])
print("held-out halves sizes",len(A),len(B),"| CL v1 (syn+pneg) top100: v2top100",top(ob,v2),"halfA",top(ob,A),"halfB",top(ob,B))
for tr_h,te_h,nm in ((A,B,"train A / test B"),(B,A,"train B / test A")):
    Xs=np.vstack([X,Z,T[Hneg+D17+tr_h]]); ls=np.r_[lab_tr,np.full(len(Z)+len(Hneg+D17+tr_h),2)]
    st=train_cl(Xs,ls)(T); o=np.argsort(-st)
    print(f"  {nm}: top100 v2top100 {top(o,v2)} | held-out junk half {top(o,te_h)} (v1: {top(ob,te_h)}) | top200 v2top100 {top(o,v2,200)} junk {top(o,te_h,200)} (v1 {top(ob,te_h,200)})",flush=True)
Xs=np.vstack([X,Z,T[Hneg+D17+v2b]]); ls=np.r_[lab_tr,np.full(len(Z)+len(Hneg+D17+v2b),2)]
st=train_cl(Xs,ls)(T); np.save("crackcl2_test_scores_pneg.npy",st); o=np.argsort(-st); print("full: top100 v2top100",top(o,v2))
