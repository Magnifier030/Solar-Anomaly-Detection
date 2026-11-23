import json,numpy as np
O="/home/M11509204/researchs/Solar-Anomaly-Detection/outputs/submissions/"
TM=json.load(open("crackbig3_test_meta.json")); idx={(m[0],m[3]):i for i,m in enumerate(TM)}
add=[idx[tuple(x)] for x in json.load(open(O+"thr35-bigcrack500-20261109/manifest.json"))["added"]]
top100=set(add[:100]); tail=set(add[100:500])
pn=np.load("crackbig_test_panelness.npy")
def cr(sub): return {(k,json.dumps(o["points"])) for k,v in sub.items() for o in v if o["label"]=="crack"}
for tag in ("crackbig3_test_scores_pneg.npy",):
    st=np.load(tag); st=np.where(pn>=0.75,st,-1e9)
    excl=set()
    o=[i for i in np.argsort(-st)]
    for N in (100,200):
        s=set(o[:N]); print(tag,"top",N,": v2 top100 (≈40% hit)",len(s&top100),"| v2 101-500 (≈2%)",len(s&tail),"| other",N-len(s&top100)-len(s&tail))
