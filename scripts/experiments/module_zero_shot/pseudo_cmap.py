import numpy as np, cv2, glob
from pathlib import Path
P=Path("/home/M11509204/researchs/Solar-Anomaly-Detection")
fs=[f for f in sorted(glob.glob(str(P/"data/**/test*/**/IR*/**/*.*"),recursive=True)) if f.lower().endswith(('.jpg','.png','.jpeg'))]
maps={}
g=np.arange(256,dtype=np.uint8)[:,None]
for nm in dir(cv2):
    if nm.startswith("COLORMAP_"):
        try: maps[nm]=cv2.applyColorMap(g,getattr(cv2,nm))[:,0].astype(np.float32)
        except Exception: pass
try:
    import matplotlib; 
    for nm in ["inferno","magma","plasma","hot","afmhot","gist_heat"]:
        maps["mpl_"+nm]=(matplotlib.colormaps[nm](np.arange(256))[:,:3][:,::-1]*255).astype(np.float32)
except Exception as e: print(e)
px=[]
n=0
for f in fs:
    im=cv2.imread(f)
    if np.abs(im[...,0].astype(int)-im[...,2]).mean()<=5: continue
    px.append(im.reshape(-1,3)[::211]); n+=1
    if n>=60: break
px=np.concatenate(px).astype(np.float32);print(px.shape)
for nm,m in maps.items():
    d=((px[:,None]-m[None])**2).sum(-1).min(1)**.5
    print(f"{nm:24s} mean {d.mean():6.2f} med {np.median(d):6.2f} p95 {np.percentile(d,95):6.2f}")
