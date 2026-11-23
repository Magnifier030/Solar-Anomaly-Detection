import json,numpy as np,collections
X=np.load("crackbig3_train.npy").astype(np.float32); M=json.load(open("crackbig3_train_meta.json"))
def comps(F,thr):
    S=F@F.T; n=len(F); lab=-np.ones(n,int); c=0
    for i in range(n):
        if lab[i]>=0: continue
        st=[i]; lab[i]=c
        while st:
            u=st.pop()
            for v in np.where((S[u]>thr)&(lab<0))[0]: lab[v]=c; st.append(v)
        c+=1
    return lab
for part,sl in (("module mean",slice(0,1024)),("top-K deviant",slice(1024,2048))):
    print("feature:",part)
    for pos in (1,0):
        k=[i for i,m in enumerate(M) if m[2]==pos and m[4]==0]; F=X[k][:,sl]; F=F-F.mean(0) if False else F; F/=np.linalg.norm(F,axis=1,keepdims=True)
        g=np.array([M[i][1] for i in k])
        for grp in ("G01","G07","G04"):
            s=g==grp
            if pos==0: s=s&(np.random.default_rng(0).random(len(s))<(400/max(s.sum(),1)))
            r={t:len(set(comps(F[s],t))) for t in (.9,.95,.97)}
            print(f"  {'crack' if pos else 'normal(sample)'} {grp} n={s.sum()} components@cos>t {r}")
