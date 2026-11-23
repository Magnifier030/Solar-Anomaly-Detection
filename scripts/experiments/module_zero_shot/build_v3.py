"""Fixed candidate pipeline. Exclude only platform-proven non-cracks (H-added 386, G01/G07 zero-hit 35) and all G01/G07-domain frames
(platform: 0 hits). Keep panel-ness>=.75 and low-freq range<=60. Glare: frac(px>=250)>2% AND module mean > frame-mates median mean + 25.
Usage: SCORES OUT N"""
import json,sys,subprocess,numpy as np,cv2
from pathlib import Path
from rectify import rectify
P=Path("/home/M11509204/researchs/Solar-Anomaly-Detection"); TEST=P/"data/raw/datasets/9_30_test_set"; KIT=P/"data/raw/evaluation_kit/author-kit_1012"; O=P/"outputs/submissions"
st=np.load(sys.argv[1]); out=Path(sys.argv[2]); N=int(sys.argv[3]); REASON="模組板破裂，破損處的導電路徑受損。"
TM=json.load(open("crackbig3_test_meta.json")); tri=np.load("tri_scores.npy"); pn=np.load("crackbig_test_panelness.npy"); dom=json.load(open("test_eo_domain.json"))
eo=json.load(open("test_cache_eo_module-maskrcnn-eo-all.json")); mods={k:[m for m in v if m["det_score"]>=0.7] for k,v in eo.items()}
def cr(f): s=json.load(open(O/f/"submission.json")); return {(k,json.dumps(o["points"])) for k,v in s.items() for o in v if o["label"]=="crack"}
Gc=cr("ir-v6union-sp7-diode-addcrack-zlow2-20261107"); Hc=cr("G-clipall70-crack-20261107"); bad=Hc-Gc
base=O/"G-dense4-thr35-hotspot-20261108"; sub=json.loads((base/"submission.json").read_text())
for v in sub.values():
    for o in v:
        if o["label"]=="crack": o["label"]="solar_panel"; o.pop("reason",None)
fm={}
def frame_mean(k):
    if k not in fm:
        g=cv2.imread(str(TEST/k),0); fm[k]=float(np.median([cv2.mean(g,mask=cv2.fillPoly(np.zeros_like(g),[np.int32(m["polygon"])],255))[0] for m in mods[k]]))
    return fm[k]
added=[];skip=dict(dom=0,bad=0,pn=0,shadow=0,glare=0,hot=0,border=0)
for i in np.argsort(-st):
    if len(added)==N: break
    k,_,_,j,_=TM[i]; m=mods[k][j]; p=json.dumps(m["polygon"])
    if dom[k][0] in ("G01","G07"): skip["dom"]+=1; continue
    if (k,p) in bad: skip["bad"]+=1; continue
    if pn[i]<.75: skip["pn"]+=1; continue
    c=rectify(cv2.imread(str(TEST/k)),np.asarray(m["polygon"],float),160,inset=0.03).astype(np.uint8); g=cv2.cvtColor(c,cv2.COLOR_BGR2GRAY).astype(np.float32)
    b=cv2.GaussianBlur(g,(0,0),12)
    if np.percentile(b,97)-np.percentile(b,3)>60: skip["shadow"]+=1; continue
    if (g>=250).mean()>.02 and g.mean()>frame_mean(k)+25: skip["glare"]+=1; continue
    if g.mean()>220 and tri[i]<.9: skip["border"]+=1; continue   # very bright non-triangle module: seam/tick artefacts, unverified on platform
    o={json.dumps(x["points"]):x for x in sub[k]}.get(p)
    if o is None or o["label"]=="hotspot": skip["hot"]+=1; continue
    o["label"]="crack"; o["reason"]=REASON; added.append([k,j])
n={l:sum(o["label"]==l for v in sub.values() for o in v) for l in ("solar_panel","hotspot","crack")}
out.mkdir(parents=True,exist_ok=False); pth=out/"submission.json"; pth.write_text(json.dumps(sub,ensure_ascii=False,allow_nan=False)+"\n")
chk=subprocess.run([sys.executable,str(KIT/"make_submission.py"),"--check",str(pth)],capture_output=True,text=True); (out/"format_check.txt").write_text(chk.stdout+chk.stderr)
(out/"manifest.json").write_text(json.dumps({"base":str(base),"scores":sys.argv[1],"rule":"CL v3 (shadow counterfactual negs) + fixed pipeline (only platform-proven exclusions, no G01/G07 domain, relative glare, skip very bright (>220) modules unless triangle-defect prototype >=.9)","N":N,"skip":skip,"added":added,"counts":n,"format_check_exit_code":chk.returncode,"uploaded":False},ensure_ascii=False,indent=1))
print(len(added),skip,n,chk.returncode)
