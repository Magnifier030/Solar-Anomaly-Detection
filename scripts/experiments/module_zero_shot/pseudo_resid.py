import importlib.util, numpy as np, cv2
from pathlib import Path
P=Path("/home/M11509204/researchs/Solar-Anomaly-Detection")
spec=importlib.util.spec_from_file_location("mm",P/"scripts/module_maskrcnn.py");mm=importlib.util.module_from_spec(spec);spec.loader.exec_module(mm)
import glob
fs=sorted(glob.glob(str(P/"data/**/test*/**/IR*/**/*.*"),recursive=True))
fs=[f for f in fs if f.lower().endswith(('.jpg','.png','.jpeg'))]
print(len(fs))
res=[];n=0
for f in fs:
    im=cv2.imread(f)
    if np.abs(im[...,0].astype(int)-im[...,2]).mean()<=5: continue
    n+=1
    px=im.reshape(-1,3)[::97].astype(np.float32)
    d=((px[:,None]-mm.INFERNO[None])**2).sum(-1).min(1)**.5
    res.append((np.mean(d),np.percentile(d,99),f))
    if n>=40: break
a=np.array([r[:2] for r in res]);print("pseudo frames sampled",n,"mean resid",a[:,0].mean(),"p99 resid",a[:,1].mean(), "max p99",a[:,1].max())
