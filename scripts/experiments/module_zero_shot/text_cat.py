"""Proxy per hidden category (board / internal / diode for hotspot; crack) using leave-self-out max BERTScore."""
import glob, json, re, csv, sys
import numpy as np
from bert_score import BERTScorer
P="/home/M11509204/researchs/Solar-Anomaly-Detection/"
R=P+"data/raw/datasets/9_16_train_set/train"
def cat(r):
    return "diode" if ("二極體" in r or "電池串" in r) else ("internal" if re.search("隱裂|微裂|電性|接觸", r) else "board")
refs={"board":set(),"internal":set(),"diode":set(),"crack":set()}
for mod,lab in (("IR","hotspot"),("EO","crack")):
    for f in glob.glob(f"{R}/{mod}/label/*.json"):
        for s in json.load(open(f))["shapes"]:
            if s["label"]==lab and s.get("reason"):
                refs["crack" if mod=="EO" else cat(s["reason"].strip())].add(s["reason"].strip())
refs={k:sorted(v) for k,v in refs.items()}
cands=sorted(set().union(*refs.values())|set(json.load(open("text_opt.json")))|{
 "推測模組板破損或內部隱裂、電性異常，導電路徑受損使局部電阻升高，進而產生異常發熱。",
 "模組板破裂，破損處的導電路徑受損。",
 "推測旁路二極體導通或異常，導致該電池串工作異常而發熱。",
 "該區呈條狀高溫，推測旁路二極體導通或電池串工作異常。",
 "推測模組板破損、內部隱裂或旁路二極體異常，使局部電阻升高而產生異常發熱。",
 "推測模組板破裂或內部隱裂、電性異常，或旁路二極體導通，導致電池串局部電阻升高而異常發熱。",
})
sc=BERTScorer(model_type="bert-base-chinese",num_layers=8,lang="zh",rescale_with_baseline=True)
T={}
for k,rs in refs.items():
    f=sc.score([c for c in cands for r in rs],[r for c in cands for r in rs],batch_size=512)[2].numpy().reshape(len(cands),len(rs))
    for i,c in enumerate(cands):
        row=f[i].copy(); row[[j for j,r in enumerate(rs) if r==c]]=-9
        T.setdefault(c,{})[k]=float(np.sort(row)[-2]) # 2nd best: pessimistic proxy for hidden refs
json.dump(T,open("text_cat.json","w"),ensure_ascii=False)
for k in refs: print(k,len(refs[k]))
for key in ("board","internal","diode","crack"):
    print("== best for",key)
    for c in sorted(T,key=lambda c:-T[c][key])[:4]: print(round(T[c][key],3),c)
hm=lambda v:len(v)/sum(1/max(x,1e-3) for x in v)
print("== best single hotspot sentence (hmean board/internal/diode)")
for c in sorted(T,key=lambda c:-hm([T[c][k] for k in ("board","internal","diode")]))[:6]:
    print(round(hm([T[c][k] for k in ("board","internal","diode")]),3),{k:round(T[c][k],3) for k in T[c]},c)
cur="推測模組板破損或內部隱裂、電性異常，導電路徑受損使局部電阻升高，進而產生異常發熱。"
print("current",{k:round(T[cur][k],3) for k in T[cur]})
