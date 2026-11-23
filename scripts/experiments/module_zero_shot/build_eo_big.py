"""Add-only cracks from the large-model small-defect classifier (crack_big + geo3/bi, platform negatives).
Candidates big_cand_f.json (not in G/H crack sets, panel-ness>=0.75, low-freq range<=60) minus glare (>2% px >=250)
and dark-blob shadows (>12% px of blurred grey < 0.45*median). Usage: BASE OUT N"""
import json,sys,subprocess,numpy as np,cv2
from pathlib import Path
from rectify import rectify
P=Path("/home/M11509204/researchs/Solar-Anomaly-Detection"); TEST=P/"data/raw/datasets/9_30_test_set"; KIT=P/"data/raw/evaluation_kit/author-kit_1012"
base,out,N=Path(sys.argv[1]),Path(sys.argv[2]),int(sys.argv[3])
REASON="模組板破裂，破損處的導電路徑受損。"
sub=json.loads((base/"submission.json").read_text()); TM=json.load(open("crackbig_test_meta.json")); cand=json.load(open(sys.argv[4] if len(sys.argv)>4 else "big_cand_f.json"))
eo=json.load(open("test_cache_eo_module-maskrcnn-eo-all.json")); mods={k:[m for m in v if m["det_score"]>=0.7] for k,v in eo.items()}
added=[];skip={"glare":0,"shadow":0,"nomatch":0,"already":0}
for i in cand:
    if len(added)==N: break
    k,_,_,j,_=TM[i]; m=mods[k][j]
    g=rectify(cv2.imread(str(TEST/k),cv2.IMREAD_GRAYSCALE),np.asarray(m["polygon"],float),160,inset=0.03).astype(np.float32)
    if (g>=250).mean()>0.02: skip["glare"]+=1; continue
    b=cv2.GaussianBlur(g,(0,0),8)
    if (b<0.45*np.median(b)).mean()>0.12: skip["shadow"]+=1; continue
    o={json.dumps(x["points"]):x for x in sub[k]}.get(json.dumps(m["polygon"]))
    if o is None: skip["nomatch"]+=1; continue
    if o["label"]=="crack": skip["already"]+=1; continue
    if o["label"]=="hotspot": continue
    o["label"]="crack"; o["reason"]=REASON; added.append([k,j])
n={l:sum(o["label"]==l for v in sub.values() for o in v) for l in ("solar_panel","hotspot","crack")}
out.mkdir(parents=True,exist_ok=False); p=out/"submission.json"; p.write_text(json.dumps(sub,ensure_ascii=False,allow_nan=False)+"\n")
chk=subprocess.run([sys.executable,str(KIT/"make_submission.py"),"--check",str(p)],capture_output=True,text=True)
(out/"format_check.txt").write_text(chk.stdout+chk.stderr)
(out/"manifest.json").write_text(json.dumps({"base":str(base),"rule":"add top-N large-model (DINOv2-L 672px small-defect top-K patches, geo3+bi aug, platform negatives) crack candidates","N":N,"skip":skip,"added":added,"counts":n,"format_check_exit_code":chk.returncode,"uploaded":False},ensure_ascii=False,indent=1))
print(len(added),skip,n,chk.returncode)
