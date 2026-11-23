"""Reason optimisation: train reasons grouped by (modality, group) -> BERTScore (platform settings) matrix; LOO best sentence per cluster."""
import json,glob,collections,os,numpy as np,torch
os.environ.setdefault("HF_HUB_OFFLINE","1")
from bert_score import BERTScorer
P="/home/M11509204/researchs/Solar-Anomaly-Detection/data/raw/datasets/9_16_train_set/train/"
M=json.load(open("crackbig_train_meta.json")); g={os.path.basename(m[0]):m[1] for m in M}
R=collections.defaultdict(collections.Counter)
for mod in ("IR","EO"):
    for f in glob.glob(P+mod+"/label/*.json"):
        b=os.path.basename(f).replace(".json",".jpg")
        for s in json.load(open(f))["shapes"]:
            if s["label"]!="solar_panel" and s.get("reason"): R[(mod,g.get(b,"?"))][s["reason"].strip()]+=1
for k,c in sorted(R.items()): print(k,sum(c.values()),"unique",len(c))
U=sorted({r for c in R.values() for r in c})
EXTRA=["推測模組板破損或內部隱裂、電性異常，導電路徑受損使局部電阻升高，進而產生異常發熱。","模組板破裂，破損處的導電路徑受損。","推測旁路二極體導通或異常，導致該電池串工作異常而發熱。"]
C=U+[e for e in EXTRA if e not in U]
sc=BERTScorer(model_type="bert-base-chinese",num_layers=8,lang="zh",rescale_with_baseline=True,device="cuda")
cand=[a for a in C for b in U]; ref=[b for a in C for b in U]
_,_,F=sc.score(cand,ref,batch_size=256); F=F.numpy().reshape(len(C),len(U))
np.save("text_F.npy",F); json.dump({"C":C,"U":U,"R":{f"{k[0]}|{k[1]}":dict(v) for k,v in R.items()}},open("text_meta.json","w"),ensure_ascii=False)
print("done",F.shape)
