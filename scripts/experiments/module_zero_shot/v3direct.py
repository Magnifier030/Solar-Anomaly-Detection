"""Direct DINOv3 crack test: same candidate pipeline as v2 (exclude G/H cracks, panel-ness>=.75, low-freq range<=60), PFX scores. -> cand file"""
import json,sys,numpy as np,cv2
from rectify import rectify
PFX=sys.argv[1]; O="/home/M11509204/researchs/Solar-Anomaly-Detection/outputs/submissions/"; TEST="/home/M11509204/researchs/Solar-Anomaly-Detection/data/raw/datasets/9_30_test_set/"
TM=json.load(open(PFX+"_test_meta.json")); TM2=json.load(open("crackbig_test_meta.json"))
assert [m[:4] for m in TM]==[m[:4] for m in TM2]
st=np.load(PFX+"_test_scores_pneg.npy"); pn=np.load("crackbig_test_panelness.npy")
eo=json.load(open("test_cache_eo_module-maskrcnn-eo-all.json")); mods={k:[m for m in v if m["det_score"]>=0.7] for k,v in eo.items()}
def cr(sub): return {(k,json.dumps(o["points"])) for k,v in sub.items() for o in v if o["label"]=="crack"}
Gc=cr(json.load(open(O+"ir-v6union-sp7-diode-addcrack-zlow2-20261107/submission.json"))); Hc=cr(json.load(open(O+"G-clipall70-crack-20261107/submission.json")))
v2=json.load(open(O+"thr35-big100only-20261110/manifest.json"))
v2top=json.load(open(O+"thr30-bigcrack100-20261109/manifest.json"))["added"][:100]; v2s={tuple(x) for x in v2top}
out=[];tiles=[]
for i in np.argsort(-st):
    k,_,_,j,_=TM[i]; p=json.dumps(mods[k][j]["polygon"])
    if (k,p) in Gc or (k,p) in Hc or pn[i]<0.75: continue
    c=rectify(cv2.imread(TEST+k),np.asarray(mods[k][j]["polygon"],float),160,inset=0.03).astype(np.uint8).copy()
    b=cv2.GaussianBlur(cv2.cvtColor(c,cv2.COLOR_BGR2GRAY).astype(np.float32),(0,0),12)
    if np.percentile(b,97)-np.percentile(b,3)>60: continue
    out.append(int(i))
    if len(tiles)<100: cv2.putText(c,str(len(out)-1)+("*" if (k,j) in v2s else ""),(3,15),0,0.5,(0,255,255),1); tiles.append(c)
    if len(out)==400: break
json.dump(out,open(PFX+"_cand_direct.json","w"))
cv2.imwrite(PFX+"_direct100.jpg",np.vstack([np.hstack(tiles[i:i+10]) for i in range(0,100,10)]))
print(PFX,"overlap with v2 top100 in first 100 cands:",sum((TM[i][0],TM[i][3]) in v2s for i in out[:100]))
