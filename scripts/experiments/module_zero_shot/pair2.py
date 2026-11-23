import json,glob,os,numpy as np
P="/home/M11509204/researchs/Solar-Anomaly-Detection/data/raw/datasets/9_16_train_set/train/"
X=[];Y=[];multi=[]
for f in sorted(glob.glob(P+"EO/label/*.json")):
    n=os.path.basename(f)[:-5]; e=json.load(open(f)); i=json.load(open(P+"IR/label/"+n+".json"))
    cr=[np.mean(s["points"],0)/[e["imageWidth"],e["imageHeight"]] for s in e["shapes"] if s["label"]=="crack"]
    hs=[np.mean(s["points"],0)/[i["imageWidth"],i["imageHeight"]] for s in i["shapes"] if s["label"]=="hotspot"]
    if len(cr)==1 and len(hs)==1: X.append(cr[0]); Y.append(hs[0])
    elif cr and hs: multi.append((cr,hs))
X=np.array(X);Y=np.array(Y); A=np.c_[X,np.ones(len(X))]; W,*_=np.linalg.lstsq(A,Y,rcond=None); r=np.linalg.norm(A@W-Y,axis=1)
print("1-crack/1-hotspot frames",len(X),"| affine EO->IR residual median",round(np.median(r),3),"p90",round(np.percentile(r,90),3),"| frac<.05",round((r<.05).mean(),2))
print("affine",np.round(W.T,3))
# multi frames: count cracks with a hotspot within .05 after mapping, and #cracks vs #hotspots
nc=sum(len(c) for c,_ in multi); nh=sum(len(h) for _,h in multi); hit=0
for c,h in multi:
    for cc in c: hit+=min(np.linalg.norm(np.r_[cc,1]@W-hh) for hh in h)<.06
print("multi frames",len(multi),"cracks",nc,"hotspots",nh,"cracks with mapped hotspot <.06:",hit)
