"""Add-only union: OUT = F, plus every hotspot of BASE (same module polygon) relabelled hotspot with BASE's reason. Usage: BASE F OUT"""
import json,sys,subprocess
from pathlib import Path
base,f,out=map(Path,sys.argv[1:4]); KIT=Path("/home/M11509204/researchs/Solar-Anomaly-Detection/data/raw/evaluation_kit/author-kit_1012")
B=json.loads((base/"submission.json").read_text()); F=json.loads((f/"submission.json").read_text())
added=0
for k,objs in B.items():
    hot={json.dumps(o["points"]):o["reason"] for o in objs if o["label"]=="hotspot"}
    for o in F[k]:
        r=hot.get(json.dumps(o["points"]))
        if r and o["label"]=="solar_panel": o["label"]="hotspot"; o["reason"]=r; added+=1
n={l:sum(o["label"]==l for v in F.values() for o in v) for l in ("solar_panel","hotspot","crack")}
out.mkdir(parents=True,exist_ok=False); p=out/"submission.json"; p.write_text(json.dumps(F,ensure_ascii=False,allow_nan=False)+"\n")
chk=subprocess.run([sys.executable,str(KIT/"make_submission.py"),"--check",str(p)],capture_output=True,text=True)
(out/"format_check.txt").write_text(chk.stdout+chk.stderr)
(out/"manifest.json").write_text(json.dumps({"rule":"union of BASE hotspots and F","base":str(base),"f":str(f),"added_from_base":added,"counts":n,"format_check_exit_code":chk.returncode,"uploaded":False},ensure_ascii=False,indent=1))
print(added,n,chk.returncode)
