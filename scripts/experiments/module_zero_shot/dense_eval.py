import json,sys,numpy as np
sys.argv=["x","ir"]
import raw_eval as R, border_eval_lib as B
d=json.load(open("raw_ir_dense.json"))
for K in (2,999):
  for thr in (3.25,5,8):
    smp=B.label_v(d,0.5,thr,"raw+std+brightpk",1.0,K=K)
    line=[]
    for g in sorted({s["group"] for s in smp}):
        ss=[s for s in smp if s["group"]==g]
        nh=sum(len(s["prediction"].get("hotspot",[])) for s in ss); nf=sum(bool(s["prediction"].get("hotspot")) for s in ss)
        line.append(f"{g}:{nf}/{len(ss)}f {nh}h")
    g2=[s for s in smp if s["group"]=="G02"]
    print(f"K={K} thr={thr} G02 TIoU {R.score(g2)[0]:.3f} |"," ".join(line))
