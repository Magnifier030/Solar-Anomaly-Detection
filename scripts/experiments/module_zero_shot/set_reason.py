"""OUT = BASE with every crack reason replaced (geometry/labels unchanged -> isolates F1_TEXT). Usage: BASE OUT REASON"""
import json,sys,subprocess
from pathlib import Path
KIT=Path("/home/M11509204/researchs/Solar-Anomaly-Detection/data/raw/evaluation_kit/author-kit_1012")
base,out,R=Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3]
sub=json.loads((base/"submission.json").read_text()); n=0
for v in sub.values():
    for o in v:
        if o["label"]=="crack": o["reason"]=R; n+=1
out.mkdir(parents=True,exist_ok=False); p=out/"submission.json"; p.write_text(json.dumps(sub,ensure_ascii=False,allow_nan=False)+"\n")
chk=subprocess.run([sys.executable,str(KIT/"make_submission.py"),"--check",str(p)],capture_output=True,text=True)
(out/"format_check.txt").write_text(chk.stdout+chk.stderr)
(out/"manifest.json").write_text(json.dumps({"base":str(base),"rule":"text probe: all crack reasons replaced","crack_reason":R,"n_crack":n,"format_check_exit_code":chk.returncode,"uploaded":False},ensure_ascii=False,indent=1))
print(n,chk.returncode)
