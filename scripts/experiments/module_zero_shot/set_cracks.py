"""OUT = BASE with cracks set exactly to: (KEEP_OLD ? BASE cracks : none) + modules listed in MANIFEST 'added' (first N). Usage: BASE OUT MANIFEST N KEEP_OLD(0/1)"""
import json,sys,subprocess
from pathlib import Path
P=Path("/home/M11509204/researchs/Solar-Anomaly-Detection"); KIT=P/"data/raw/evaluation_kit/author-kit_1012"
base,out,man,N,keep=Path(sys.argv[1]),Path(sys.argv[2]),Path(sys.argv[3]),int(sys.argv[4]),sys.argv[5]=="1"
REASON="模組板破裂，破損處的導電路徑受損。"
sub=json.loads((base/"submission.json").read_text()); add=json.loads(man.read_text())["added"][:N]
eo=json.load(open("test_cache_eo_module-maskrcnn-eo-all.json")); mods={k:[m for m in v if m["det_score"]>=0.7] for k,v in eo.items()}
if not keep:
    for v in sub.values():
        for o in v:
            if o["label"]=="crack": o["label"]="solar_panel"; o.pop("reason",None)
miss=0
for k,j in add:
    o={json.dumps(x["points"]):x for x in sub[k]}.get(json.dumps(mods[k][j]["polygon"]))
    if o is None or o["label"]=="hotspot": miss+=1; continue
    o["label"]="crack"; o["reason"]=REASON
n={l:sum(o["label"]==l for v in sub.values() for o in v) for l in ("solar_panel","hotspot","crack")}
out.mkdir(parents=True,exist_ok=False); p=out/"submission.json"; p.write_text(json.dumps(sub,ensure_ascii=False,allow_nan=False)+"\n")
chk=subprocess.run([sys.executable,str(KIT/"make_submission.py"),"--check",str(p)],capture_output=True,text=True)
(out/"format_check.txt").write_text(chk.stdout+chk.stderr)
(out/"manifest.json").write_text(json.dumps({"base":str(base),"crack_source":str(man),"N":N,"keep_old_cracks":keep,"missing":miss,"counts":n,"format_check_exit_code":chk.returncode,"uploaded":False},ensure_ascii=False,indent=1))
print(n,miss,chk.returncode)
