"""Add-only crack: keep BASE fully; in EO frames with no crack, add DINOv2 top-1 (z floor c) if z>=THR and CLIP whole p>=P.
Usage: BASE OUT CACHE THR C P"""
import json, subprocess, sys
from pathlib import Path
import numpy as np, torch
args=sys.argv[1:]; sys.argv=sys.argv[:1]+["eo"]
import raw_eval as R, clip_crack as C
from rectify import rectify
from transformers import CLIPModel, CLIPProcessor
mm=R.mm; PROJECT=R.PROJECT
KIT=PROJECT/"data/raw/evaluation_kit/author-kit_1012"; TEST=PROJECT/"data/raw/datasets/9_30_test_set"
base,out,cache=Path(args[0]),Path(args[1]),Path(args[2]); thr,c,pg=map(float,args[3:6])
REASON="模組板破裂，破損處的導電路徑受損。"
hub=PROJECT/".hf_cache/hub"
with torch.no_grad():
    model=CLIPModel.from_pretrained("openai/clip-vit-base-patch16",cache_dir=hub).cuda().eval(); proc=CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16",cache_dir=hub)
    t=proc(text=C.NORMAL+C.CRACK,return_tensors="pt",padding=True).to("cuda")
    txt=torch.nn.functional.normalize(C.feat(model.get_text_features(**t)),dim=-1)
    txt=torch.stack([torch.nn.functional.normalize(txt[:4].mean(0),dim=0),torch.nn.functional.normalize(txt[4:].mean(0),dim=0)])
    sub=json.loads((base/"submission.json").read_text()); eo=json.loads(cache.read_text()); added=0
    for key,modules in eo.items():
        objs=sub[key]
        if any(o["label"]=="crack" for o in objs): continue
        mods=[m for m in modules if m["det_score"]>=0.7]
        if len(mods)<3 or not all("raw" in m for m in mods): continue
        s=R.z(np.array([m["raw"] for m in mods]),c); s=np.where(np.array([m["fill"] for m in mods])<R.FILL_MIN,0,s)
        top=int(np.argmax(s))
        if s[top]<thr: continue
        crop=rectify(mm.load_image(TEST/key,"eo"),np.asarray(mods[top]["polygon"],float),224,inset=0.03)
        img=torch.nn.functional.normalize(C.feat(model.get_image_features(pixel_values=proc(images=[crop],return_tensors="pt")["pixel_values"].cuda())),dim=-1)
        if float((100*img@txt.T).softmax(-1)[0,1])<pg: continue
        tp=json.dumps(mods[top]["polygon"]); hit=False
        for o in objs:
            if json.dumps(o["points"])==tp: o["label"]="crack"; o["reason"]=REASON; hit=True; break
        if hit: added+=1
out.mkdir(parents=True,exist_ok=False); path=out/"submission.json"
path.write_text(json.dumps(sub,ensure_ascii=False,allow_nan=False)+"\n")
chk=subprocess.run([sys.executable,str(KIT/"make_submission.py"),"--check",str(path)],capture_output=True,text=True)
(out/"format_check.txt").write_text(chk.stdout+chk.stderr)
(out/"manifest.json").write_text(json.dumps({"base":str(base),"rule":"add-only top1 z>=thr(c) & clip>=P in crack-free EO frames","thr":thr,"c":c,"p":pg,"added":added,"uploaded":False},indent=1))
print("added",added); print(chk.stdout[-200:])
