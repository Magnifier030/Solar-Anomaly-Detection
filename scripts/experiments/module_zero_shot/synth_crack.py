"""Synthetic-anomaly crack detector for cross-site generalisation.
Normal EO modules from non-eval groups (G02,G03,G05,G06,G08-G12) + synthetic crack overlays -> patch-level linear probe
on DINOv2 tokens (labels from the synthetic mask). Module score = mean of top-3 patch probs.
Eval: real cracks in G01/G07 (unseen) and G04, per-group AUC vs CLIP B/16 whole and DINOv2 mutual z."""
import json, sys
from pathlib import Path
import numpy as np, cv2, torch
sys.argv=["raw_eval.py","eo"]
import raw_eval as R, clip_crack as C
from rectify import rectify
from zs_score import _tokens
S=Path(__file__).parent; mm=R.mm; rng=np.random.default_rng(0); SZ=224; G=SZ//14
EVAL={"G01","G07","G04"}
def crack(img):
    h=img.shape[0]; mask=np.zeros((h,h),np.float32)
    for _ in range(rng.integers(1,3)):
        p=rng.uniform(0.15,0.85,2)*h; ang=rng.uniform(0,2*np.pi); pts=[p.copy()]
        for _ in range(rng.integers(8,40)):
            ang+=rng.normal(0,0.35); p=p+rng.uniform(3,7)*np.array([np.cos(ang),np.sin(ang)]); pts.append(p.copy())
            if rng.random()<0.06:
                q=p.copy(); a2=ang+rng.choice([-1,1])*rng.uniform(0.5,1.2); br=[q.copy()]
                for _ in range(rng.integers(4,15)): a2+=rng.normal(0,0.3); q=q+4*np.array([np.cos(a2),np.sin(a2)]); br.append(q.copy())
                cv2.polylines(mask,[np.int32(br)],False,1.0,int(rng.integers(1,3)))
        cv2.polylines(mask,[np.int32(pts)],False,1.0,int(rng.integers(1,4)))
    mask=cv2.GaussianBlur(mask,(0,0),0.7); mask=np.clip(mask,0,1)[...,None]
    a=rng.uniform(0.4,0.9)
    tgt=img*rng.uniform(0.15,0.5) if rng.random()<0.6 else np.full_like(img,rng.uniform(200,255),dtype=np.float32)
    out=img*(1-a*mask)+tgt*(a*mask)
    return np.clip(out,0,255).astype(np.uint8), mask[...,0]
def crops_of(it, thr):
    mods=[m for m in it["modules"] if m["det_score"]>=thr]
    if len(mods)<3: return [],mods
    img=mm.load_image(mm.image_path(mm.TRAIN/"EO"/"image",it["name"]),"eo")
    return [np.ascontiguousarray(rectify(img,np.asarray(m["polygon"],float),SZ,inset=0.03),np.uint8) for m in mods],mods
items=json.load(open(S/"raw_eo.json"))+json.load(open(S/"raw_eo_cal.json"))
# --- train set
norm=[]
for it in items:
    if it["group"] in EVAL: continue
    cr,_=crops_of(it,0.7); norm+=cr
idx=rng.permutation(len(norm))[:3000]; norm=[norm[i] for i in idx]; print("normal crops",len(norm),flush=True)
X=[];Y=[]
with torch.no_grad():
    for i in range(0,len(norm),128):
        batch=norm[i:i+128]; syn=[crack(c.astype(np.float32)) for c in batch]
        tn=_tokens(batch).cpu(); ts=_tokens([s for s,_ in syn]).cpu()
        for j,(s,m) in enumerate(syn):
            lab=cv2.resize(m,(G,G),interpolation=cv2.INTER_AREA).reshape(-1)
            pos=np.where(lab>0.03)[0]; neg=np.where(lab==0)[0]
            X.append(ts[j,pos]); Y+= [1]*len(pos)
            nn=rng.choice(G*G,20,replace=False); X.append(tn[j,nn]); Y+=[0]*20
            if len(neg): k=rng.choice(neg,min(10,len(neg)),replace=False); X.append(ts[j,k]); Y+=[0]*len(k)
X=torch.cat(X).float().cuda(); Y=torch.tensor(Y,dtype=torch.float32).cuda(); print("patches",X.shape,"pos frac",Y.mean().item(),flush=True)
W=torch.zeros(X.shape[1],device="cuda",requires_grad=True); b=torch.zeros(1,device="cuda",requires_grad=True)
opt=torch.optim.Adam([W,b],lr=1e-2); pw=(1-Y.mean())/Y.mean()
for step in range(400):
    opt.zero_grad(); l=torch.nn.functional.binary_cross_entropy_with_logits(X@W+b,Y,pos_weight=pw)+1e-4*(W**2).sum(); l.backward(); opt.step()
print("train loss",l.item(),flush=True)
# --- eval
res={}
with torch.no_grad():
    for it in items:
        if it["group"] not in EVAL: continue
        cr,mods=crops_of(it,0.7)
        if not cr: continue
        mods=[m for m in mods if "raw" in m]
        if len(mods)!=len(cr): continue
        t=_tokens(cr); p=torch.sigmoid(t.float()@W+b); sc=p.topk(3,dim=1).values.mean(1).cpu().numpy()
        y=C.cv_iou_truth(it,[m["polygon"] for m in mods]); zd=R.z(np.array([m["raw"] for m in mods]),0.1)
        r=res.setdefault(it["group"],[[],[],[],[]]); r[0]+=list(y); r[1]+=list(sc); r[2]+=list(zd); r[3]+=[it["name"]]*len(mods)
np.save(S/"synth_crack_eval.npy",{g:[np.array(v) for v in r] for g,r in res.items()},allow_pickle=True)
for g,(y,sc,zd,_) in sorted(res.items()):
    y=np.array(y,bool); print(g,"n",len(y),"pos",y.sum(),"synth AUC %.3f"%C.roc_auc_score(y,sc),"dino-z AUC %.3f"%C.roc_auc_score(y,zd),"sum-rank AUC %.3f"%C.roc_auc_score(y,np.argsort(np.argsort(sc))+np.argsort(np.argsort(zd))))
a=np.load(S/"clip_window.npy"); m=np.load(S/"clip_window_meta.npy",allow_pickle=True)
for g in ("G01","G07","G04"):
    k=m[:,1]==g; print(g,"CLIP whole AUC %.3f"%C.roc_auc_score(a[k,0].astype(bool),a[k,2]))
