import json,numpy as np
T=json.load(open("text_meta.json")); C,U=T["C"],T["U"]; F=np.load("text_F.npy"); ui={u:i for i,u in enumerate(U)}; ci={c:i for i,c in enumerate(C)}
K=json.load(open("text_clusters.json"))
cat={"IR-break":["IR|G04"],"IR-microcrack":["IR|G01","IR|G02"],"IR-diode":["IR|G03"],"EO-G04":["EO|G04"],"EO-G01":["EO|G01"],"EO-G07":["EO|G07"]}
refs={k:sorted({r for c in v for r in K[c]}) for k,v in cat.items()}
def s(c,k,loo=True):
    idx=[ui[r] for r in refs[k] if not (loo and r==c)]; return F[ci[c],idx].max()
cur={"IR":["推測模組板破損或內部隱裂、電性異常，導電路徑受損使局部電阻升高，進而產生異常發熱。","推測旁路二極體導通或異常，導致該電池串工作異常而發熱。"],"EO":["模組板破裂，破損處的導電路徑受損。"]}
print("current sentences, score vs each category (max over train refs):")
for m,L in cur.items():
    for c in L: print(" ",c[:20],{k:round(float(s(c,k,False)),3) for k in cat if k.startswith(m)})
print("\nbest LOO sentence per category (s if in hidden refs≈1; LOO = robust estimate) + cross-category scores")
best={}
for k in cat:
    m=k[:2]; cands=sorted({r for kk in cat if kk.startswith(m) for r in refs[kk]})
    sc=[(float(s(c,k)),c) for c in cands]; sc.sort(reverse=True); best[k]=sc[0][1]
    print(k,len(refs[k]),"refs | LOO",round(sc[0][0],3),sc[0][1],"| cross",{kk:round(float(s(sc[0][1],kk)),3) for kk in cat if kk.startswith(m) and kk!=k})
# a single compromise sentence per modality maximising harmonic mean of LOO across categories
for m,ks in (("IR",["IR-break","IR-microcrack"]),("EO",["EO-G04","EO-G01","EO-G07"]),("IR3",["IR-break","IR-microcrack","IR-diode"])):
    mm=m[:2]; cands=sorted({r for kk in cat if kk.startswith(mm) for r in refs[kk]})
    hm=[(len(ks)/sum(1/max(s(c,k),1e-3) for k in ks),c) for c in cands]; hm.sort(reverse=True)
    print("compromise",m,round(hm[0][0],3),hm[0][1],{k:round(float(s(hm[0][1],k)),3) for k in ks})
json.dump(best,open("text_best.json","w"),ensure_ascii=False,indent=1)
