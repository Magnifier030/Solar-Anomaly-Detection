"""CL v3 = CL v1 + counterfactual tree-shadow negatives (class 3) + optional group-balanced crack sampling (BAL=1).
Reports LOGO and test indicators: v2 101-500 junk in top100, triangle-defect modules (35 known) in top100, G01/G07-domain picks."""
import os,json,numpy as np,torch,torch.nn.functional as Fn
os.environ.setdefault("ARCH","lin"); os.environ.setdefault("EP","1500")
exec(open("crack_cl.py").read().split('# labels: 0 normal')[0])
Sh=load("crackbig3_shadow"); Sh=(Sh-mu)/sd; shg=np.array([m[1] for m in json.load(open("crackbig3_shadow_meta.json"))])
BAL=os.environ.get("BAL","0")=="1"
def train_cl3(Xs,labs,cg,ep=1500,bs=128,seed=0):
    torch.manual_seed(seed); g=np.random.default_rng(seed); h=Head(Xs.shape[1]).cuda(); opt=torch.optim.AdamW(h.parameters(),3e-4,weight_decay=.1)
    Xt=torch.tensor(Xs,device="cuda"); by=[np.where(labs==c)[0] for c in range(4)]
    pc=by[1]; w=np.ones(len(pc))
    if BAL:
        gg=cg[pc]; 
        for u in set(gg): w[gg==u]=1/(gg==u).sum()
    w/=w.sum()
    for it in range(ep):
        b=np.concatenate([g.choice(by[0],bs),g.choice(pc,bs,p=w),g.choice(by[2],bs//2),g.choice(by[3],bs//2)]); lab=torch.tensor(labs[b],device="cuda")
        e,lo=h(Xt[b]+.1*torch.randn_like(Xt[b])); loss=supcon(e,lab)+Fn.binary_cross_entropy_with_logits(lo,(lab==1).float()); opt.zero_grad(); loss.backward(); opt.step()
    h.eval(); return lambda A: torch.cat([h(torch.tensor(A[i:i+4096],device="cuda"))[1] for i in range(0,len(A),4096)]).detach().cpu().numpy()
def assemble(mask_tr,extra_idx,hold=None):
    Xs=np.vstack([X[mask_tr],Z[zg!=hold] if hold else Z,Sh[shg!=hold] if hold else Sh,T[extra_idx]])
    ls=np.r_[y[mask_tr],np.full(len(Z[zg!=hold]) if hold else len(Z),2),np.full(len(Sh[shg!=hold]) if hold else len(Sh),3),np.full(len(extra_idx),2)]
    cg=np.r_[grp[mask_tr],np.full(len(ls)-mask_tr.sum(),"x")]; return Xs,ls,cg
r=[]
for gname in ("G01","G04","G07"):
    tr=grp!=gname; Xs,ls,cg=assemble(tr,[],gname); f=train_cl3(Xs,ls,cg); k=(grp==gname)&(view==0); s=f(X[k]); yy=y[k]
    sz=f(Z[zg==gname]) if (zg==gname).any() else f(Z); ss=f(Sh[shg==gname]) if (shg==gname).any() else f(Sh)
    r.append(f"{gname} {auc(yy,s):.3f}/{auc(np.r_[np.ones(yy.sum()),np.zeros(len(sz))],np.r_[s[yy==1],sz]):.3f}/{auc(np.r_[np.ones(yy.sum()),np.zeros(len(ss))],np.r_[s[yy==1],ss]):.3f}")
print("BAL",BAL,"LOGO crack-vs normal/spot/shadow:"," ".join(r),flush=True)
Xs,ls,cg=assemble(np.ones(len(y),bool),Hneg+D17); st=train_cl3(Xs,ls,cg)(T); o=np.argsort(-st)
tri=set(np.argsort(-np.load("tri_scores.npy"))[:35].tolist()); d=np.array([dom[m[0]][0] for m in TM])
top=lambda n,L:sum(i in set(L) for i in o[:n])
print(f"  test top100: v2top100 {top(100,v2)} | v2 101-500 junk {top(100,v2b)} | triangle {top(100,tri)}/35 | G01/G07-domain {int(np.isin(d[o[:100]],['G01','G07']).sum())} | H-added {top(100,Hneg)}")
np.save(f"crackcl3_bal{int(BAL)}_test_scores.npy",st)
