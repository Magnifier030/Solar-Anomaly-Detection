"""Keep cracks only in test EO frames whose nearest train domain is in KEEP (others -> solar_panel). Usage: BASE OUT G01,G07"""
import json,sys,subprocess,collections
from pathlib import Path
base,out=Path(sys.argv[1]),Path(sys.argv[2]); KEEP=set(sys.argv[3].split(","))
KIT=Path("/home/M11509204/researchs/Solar-Anomaly-Detection/data/raw/evaluation_kit/author-kit_1012")
dom=json.load(open(Path(__file__).parent/"test_eo_domain.json")); sub=json.loads((base/"submission.json").read_text())
c=collections.Counter()
for k,objs in sub.items():
    for o in objs:
        if o["label"]!="crack": continue
        if k in dom and dom[k][0] in KEEP: c["kept"]+=1
        else: o["label"]="solar_panel"; o.pop("reason",None); c["removed_"+(dom[k][0] if k in dom else "nodom")]+=1
out.mkdir(parents=True,exist_ok=False); p=out/"submission.json"; p.write_text(json.dumps(sub,ensure_ascii=False,allow_nan=False)+"\n")
chk=subprocess.run([sys.executable,str(KIT/"make_submission.py"),"--check",str(p)],capture_output=True,text=True)
(out/"format_check.txt").write_text(chk.stdout+chk.stderr)
(out/"manifest.json").write_text(json.dumps({"base":str(base),"rule":"cracks kept only in frames nearest to train domains "+",".join(sorted(KEEP)),"counts":c,"format_check_exit_code":chk.returncode,"uploaded":False},indent=1))
print(dict(c),chk.returncode)
