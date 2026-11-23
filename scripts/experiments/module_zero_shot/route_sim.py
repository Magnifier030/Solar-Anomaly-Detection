import json, numpy as np
T=json.load(open("text_cat.json")); a=np.load("band_feat.npy"); meta=np.load("band_meta.npy"); g,k=meta[:,0],meta[:,1]
crack="模組板破裂，破損處的導電路徑受損。"
cur="推測模組板破損或內部隱裂、電性異常，導電路徑受損使局部電阻升高，進而產生異常發熱。"
hedge="推測模組板破裂或內部隱裂、電性異常，或旁路二極體導通，導致電池串局部電阻升高而異常發熱。"
hm=lambda v:len(v)/sum(1/max(x,1e-3) for x in v)
isband=(a[:,0]>=2.5)&(a[:,1]>=0.7)
print("band routed per (group,cat):",{(gg,kk):f"{int(isband[(g==gg)&(k==kk)].sum())}/{int(((g==gg)&(k==kk)).sum())}" for gg,kk in set(zip(g,k))})
def sim(other,diode,route):
    cats={}
    for j in range(len(k)):
        sent=diode if (route and isband[j]) else other
        cats.setdefault(k[j],[]).append(T[sent][k[j]])
    v=[np.mean(cats[c]) for c in ("board","internal","diode")]+[T[crack]["crack"]]
    return round(hm(v),3),[round(x,3) for x in v]
diodes=sorted(T,key=lambda c:-T[c]["diode"])[:5]
print("single current",sim(cur,cur,False)); print("single hedge",sim(hedge,hedge,False))
for d in diodes:
    print("route cur/",d,sim(cur,d,True)); print("route hedge/",d,sim(hedge,d,True))
# best non-diode sentence when routing
best=max(T,key=lambda c:sim(c,diodes[0],True)[0]); print("best other with routing",best,sim(best,diodes[0],True))
