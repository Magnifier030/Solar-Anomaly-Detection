"""Supervised-contrastive head on frozen DINOv3-L small-defect descriptors: classes crack / normal / white-spot (synthetic white blobs
on clean train modules + platform-proven non-cracks). Loss = SupCon(tau .1) + BCE(crack vs rest). Compared with linear LR on same data."""
import json,os,numpy as np,torch,torch.nn as nn,torch.nn.functional as Fn
torch.cuda.set_per_process_memory_fraction(0.35)
from clip_crack import roc_auc_score as auc
from crack_sup_lib import fit
O="/home/M11509204/researchs/Solar-Anomaly-Detection/outputs/submissions/"
def load(p): X=np.load(p+".npy").astype(np.float32); X[:,-8:]=np.log(X[:,-8:]+1e-3); return X
X=load("crackbig3_train"); M=json.load(open("crackbig3_train_meta.json")); Z=load("crackbig3_syn"); ZM=json.load(open("crackbig3_syn_meta.json")); T=load("crackbig3_test"); TM=json.load(open("crackbig3_test_meta.json"))
mu,sd=X.mean(0),X.std(0)+1e-6; X,Z,T=[(a-mu)/sd for a in (X,Z,T)]
grp=np.array([m[1] for m in M]); y=np.array([m[2] for m in M]); view=np.array([m[4] for m in M]); zg=np.array([m[1] for m in ZM])
# platform negatives (as in crack_big_fit)
eo=json.load(open("test_cache_eo_module-maskrcnn-eo-all.json")); pos={}
for kk,mods in eo.items():
    for j,m in enumerate([m for m in mods if m["det_score"]>=0.7]): pos[(kk,json.dumps(m["polygon"]))]=j
def cr(f): s=json.load(open(O+f+"/submission.json")); return {(k,json.dumps(o["points"])) for k,v in s.items() for o in v if o["label"]=="crack"}
Gc=cr("ir-v6union-sp7-diode-addcrack-zlow2-20261107"); Hc=cr("G-clipall70-crack-20261107"); dom=json.load(open("test_eo_domain.json"))
idx={(m[0],m[3]):i for i,m in enumerate(TM)}
Hneg=[idx[(k,pos[(k,p)])] for k,p in Hc-Gc if (k,p) in pos]; D17=[idx[(k,pos[(k,p)])] for k,p in Gc if (k,p) in pos and dom[k][0] in ("G01","G07")]
S2=json.load(open("spot_sets.json")); v2=[idx[tuple(x)] for x in S2["v2"]]; v2b=[idx[tuple(x)] for x in S2["v2b"]]
ARCH=os.environ.get("ARCH","lin")
class Head(nn.Module):
    def __init__(s,d):
        super().__init__(); s.c=nn.Linear(128,1)
        s.f=nn.Sequential(nn.Dropout(.5),nn.Linear(d,128)) if ARCH=="lin" else nn.Sequential(nn.Dropout(.5),nn.Linear(d,256),nn.GELU(),nn.Dropout(.5),nn.Linear(256,128))
    def forward(s,x): e=Fn.normalize(s.f(x),dim=-1); return e,s.c(e).squeeze(-1)*10
def supcon(e,lab,tau=.1):
    sim=e@e.T/tau; n=len(lab); eye=torch.eye(n,device=e.device,dtype=torch.bool); sim=sim.masked_fill(eye,-1e9)
    pm=(lab[:,None]==lab[None])&~eye; lp=sim-sim.logsumexp(1,keepdim=True); return -(lp*pm).sum(1).div(pm.sum(1).clamp(min=1)).mean()
def train_cl(Xs,labs,ep=int(os.environ.get('EP',400)),bs=128,seed=0):
    torch.manual_seed(seed); g=np.random.default_rng(seed); h=Head(Xs.shape[1]).cuda(); opt=torch.optim.AdamW(h.parameters(),3e-4,weight_decay=float(os.environ.get('WD',.1)))
    Xt=torch.tensor(Xs,device="cuda"); by=[np.where(labs==c)[0] for c in range(3)]
    for it in range(ep):
        b=np.concatenate([g.choice(i,bs) for i in by]); lab=torch.tensor(labs[b],device="cuda")
        x=Xt[b]+.1*torch.randn_like(Xt[b]); e,lo=h(x)
        loss=supcon(e,lab)+Fn.binary_cross_entropy_with_logits(lo,(lab==1).float()); opt.zero_grad(); loss.backward(); opt.step()
    h.eval(); return lambda A: torch.cat([h(torch.tensor(A[i:i+4096],device="cuda"))[1] for i in range(0,len(A),4096)]).detach().cpu().numpy()
def lr(Xs,labs): w,b=fit(Xs,(labs==1).astype(int)); return lambda A:(torch.tensor(A,device="cuda")@w+b).cpu().numpy()
# labels: 0 normal, 1 crack, 2 white-spot hard negative
lab_tr=y.copy()
print("LOGO (held-out group): AUC crack vs normal | crack vs synthetic white-spot of that group")
for name,trainer,use_syn in ((("LR  no-syn",lr,False),("LR  +syn",lr,True)) if os.environ.get("LRB") else ())+(("CL  +syn",train_cl,True),):
    r=[]
    for gname in ("G01","G04","G07"):
        tr=grp!=gname; Xs=np.vstack([X[tr]]+([Z[zg!=gname]] if use_syn else [])); ls=np.r_[lab_tr[tr],np.full((zg!=gname).sum() if use_syn else 0,2)]
        f=trainer(Xs,ls); k=(grp==gname)&(view==0); s=f(X[k]); sz=f(Z[zg==gname]) if (zg==gname).any() else f(Z)
        yy=y[k]; r.append(f"{gname} {auc(yy,s):.3f}/{auc(np.r_[np.ones(yy.sum()),np.zeros(len(sz))],np.r_[s[yy==1],sz]):.3f}")
    print(" ",name," ".join(r),flush=True)
print("TEST (full train fit). v2 top100 ~40% hit, v2 101-500 ~2%, H-added 0%")
for name,trainer,extra in (("LR +syn",lr,False),("CL +syn",train_cl,False),("CL +syn+pneg",train_cl,True)):
    Xs=np.vstack([X,Z]+([T[Hneg+D17]] if extra else [])); ls=np.r_[lab_tr,np.full(len(Z),2),np.full(len(Hneg+D17) if extra else 0,2)]
    st=trainer(Xs,ls)(T); o=np.argsort(-st); top=lambda n,L:sum(i in set(L) for i in o[:n])
    print(f"  {name}: top100 has v2top100 {top(100,v2)} v2_101-500 {top(100,v2b)} H-added {top(100,Hneg)} | top200: {top(200,v2)} {top(200,v2b)} {top(200,Hneg)}",flush=True)
    np.save("crackcl_test_scores_"+name.replace(" ","").replace("+","_")+".npy",st)
