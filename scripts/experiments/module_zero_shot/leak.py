"""Missing-label check: fit CL head WITHOUT G02/G03/G12 train modules, score those (all labelled solar_panel), show top tiles per group,
and compare score distribution with test. If top ones look like test hits (white triangles/blobs), train labels likely miss cracks."""
import os,json,numpy as np,cv2
os.environ.setdefault("ARCH","lin"); os.environ.setdefault("EP","1500")
exec(open("crack_cl.py").read().split('# labels: 0 normal')[0])
sys_=__import__("sys"); sys_.argv=["raw_eval.py","eo"]
import raw_eval as R
from rectify import rectify
mm=R.mm
HO=("G02","G03","G12"); tr=~np.isin(grp,HO)
Xs=np.vstack([X[tr],Z,T[Hneg+D17]]); ls=np.r_[y[tr],np.full(len(Z)+len(Hneg+D17),2)]
f=train_cl(Xs,ls); st=f(T); s=f(X)
thr100=np.sort(st)[-100]
print("test top-100 threshold",round(float(thr100),2))
for g in HO:
    k=np.where((grp==g)&(view==0))[0]; v=s[k]
    print(g,"modules",len(k),"above test-top100 thr",int((v>=thr100).sum()),"| frames",len({M[i][0] for i in k[v>=thr100]}))
    o=k[np.argsort(-v)][:40]; tiles=[]
    for r,i in enumerate(o):
        n,_,_,j,_=M[i]; shapes=[(l,p) for l,p in mm.load_shapes(n,"eo") if l in ("solar_panel","crack") and len(p)>=4]
        im=mm.load_image(mm.image_path(mm.TRAIN/"EO"/"image",n),"eo"); c=np.ascontiguousarray(rectify(im,shapes[j][1].astype(float),160,inset=0.03),np.uint8)
        if c.ndim==2: c=cv2.cvtColor(c,cv2.COLOR_GRAY2BGR)
        c=c.copy(); cv2.putText(c,f"{r}",(3,15),0,.5,(0,255,255),1); tiles.append(c)
    cv2.imwrite(f"leak_{g}.jpg",np.vstack([np.hstack(tiles[i:i+10]) for i in range(0,40,10)]))
    json.dump([[M[i][0],int(M[i][3]),float(s[i])] for i in o],open(f"leak_{g}.json","w"))
