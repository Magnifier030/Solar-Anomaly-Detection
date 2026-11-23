import json,glob,os,collections,numpy as np,cv2
from shapely.geometry import Polygon
P="/home/M11509204/researchs/Solar-Anomaly-Detection/data/raw/datasets/9_16_train_set/train/"
M=json.load(open("crackbig_train_meta.json")); G={m[0]:m[1] for m in M}
lab=collections.Counter(); issues=collections.Counter(); ex=collections.defaultdict(list)
crack_rows=[]
def poly(p):
    p=np.asarray(p,float)
    if len(p)==2: x0,y0=p.min(0);x1,y1=p.max(0); p=np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]])
    return Polygon(p).buffer(0) if len(p)>=3 else None
for mod in ("EO","IR"):
    fs=sorted(glob.glob(P+mod+"/label/*.json"))
    imgs={os.path.basename(f)[:-4] for f in glob.glob(P+mod+"/image/*.jpg")}
    print(mod,"label files",len(fs),"images",len(imgs),"images w/o label",len(imgs-{os.path.basename(f)[:-5] for f in fs}))
    for f in fs:
        n=os.path.basename(f)[:-5]; d=json.load(open(f)); W,H=d["imageWidth"],d["imageHeight"]
        if os.path.basename(d.get("imagePath",""))[:-4]!=n: issues[mod+" imagePath mismatch"]+=1; ex[mod+" imagePath mismatch"].append(n)
        sh=d["shapes"]; P_=[]
        for s in sh:
            l=s["label"]; lab[(mod,l,s.get("shape_type"))]+=1; p=np.asarray(s["points"],float)
            if l not in("solar_panel","hotspot","crack"): issues[mod+" odd label "+l]+=1
            if (mod=="EO" and l=="hotspot") or (mod=="IR" and l=="crack"): issues[mod+" wrong-modality label "+l]+=1; ex[mod+" wrong-modality"].append(n)
            if (p[:,0]<-2).any() or (p[:,1]<-2).any() or (p[:,0]>W+2).any() or (p[:,1]>H+2).any(): issues[mod+" out of bounds"]+=1
            g=poly(p)
            if g is None or g.area<10: issues[mod+" degenerate poly"]+=1; continue
            if l!="solar_panel" and not (s.get("reason") or "").strip(): issues[mod+" anomaly w/o reason"]+=1
            P_.append((l,g,len(p)))
        # overlaps
        for i in range(len(P_)):
            for j in range(i+1,len(P_)):
                a,b=P_[i][1],P_[j][1]; inter=a.intersection(b).area
                if inter<=0: continue
                iou=inter/a.union(b).area
                if iou>0.5:
                    k=f"{mod} dup IoU>.5 {P_[i][0]}+{P_[j][0]}"; issues[k]+=1; ex[k].append(n)
        if mod=="EO":
            pan=[g.area for l,g,_ in P_ if l=="solar_panel"]
            for l,g,nv in P_:
                if l=="crack": crack_rows.append((n,G.get(n,"?"),g.area/(W*H),g.area/np.median(pan) if pan else np.nan,nv,len(pan)))
print("\nLABELS",dict(lab)); print("\nISSUES"); [print(" ",k,v,ex.get(k,[])[:6]) for k,v in sorted(issues.items())]
c=collections.defaultdict(list)
for r in crack_rows: c[r[1]].append(r)
print("\nEO crack geometry per group: n, area/image median, area/median-panel-in-frame (p10/med/p90), vertices median, frames w/o any panel label")
for g,R in sorted(c.items()):
    a=np.array([r[3] for r in R]); print(" ",g,len(R),f"{np.median([r[2] for r in R]):.4f}",np.round(np.nanpercentile(a,[10,50,90]),2),np.median([r[4] for r in R]),sum(r[5]==0 for r in R))
json.dump(crack_rows,open("audit_crack_rows.json","w"))
