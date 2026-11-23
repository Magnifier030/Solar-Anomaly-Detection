import numpy as np, raw_eval as R
W,H=640,512
def touches(p,m):
    p=np.asarray(p); return p[:,0].min()<=m or p[:,1].min()<=m or p[:,0].max()>=W-1-m or p[:,1].max()>=H-1-m
def label(items,c,thr,m,extra):
    out=[]
    for it in items:
        mods=[x for x in it["modules"] if x["det_score"]>=0.5]; pred={}
        if len(mods)>=3:
            s=R.z(np.array([x["raw"] for x in mods]),c)+np.abs(R.z(np.array([x["std"] for x in mods]),c))
            s=np.where(np.array([x["fill"] for x in mods])<R.FILL_MIN,0,s)
            if m is not None: s=s-np.array([extra if touches(x["polygon"],m) else 0 for x in mods])
        else: s=np.zeros(len(mods))
        for x,v in zip(mods,s): pred.setdefault("hotspot" if v>=thr else "solar_panel",[]).append(x["polygon"])
        out.append({"ground_truth":it["truth"],"prediction":pred,"group":it["group"]})
    return out
import cv2
def label_area(items,c,thr,m,amin):
    out=[]
    for it in items:
        mods=[x for x in it["modules"] if x["det_score"]>=0.5]; pred={}
        if len(mods)>=3:
            s=R.z(np.array([x["raw"] for x in mods]),c)+np.abs(R.z(np.array([x["std"] for x in mods]),c))
            s=np.where(np.array([x["fill"] for x in mods])<R.FILL_MIN,0,s)
            ar=np.array([cv2.contourArea(np.float32(x["polygon"])) for x in mods]); med=np.median(ar)
            cut=np.array([touches(x["polygon"],m) and a<amin*med for x,a in zip(mods,ar)])
            s=np.where(cut,-1e9,s)
        else: s=np.zeros(len(mods))
        for x,v in zip(mods,s): pred.setdefault("hotspot" if v>=thr else "solar_panel",[]).append(x["polygon"])
        out.append({"ground_truth":it["truth"],"prediction":pred,"group":it["group"]})
    return out
def label_cap(items,c,thr,m,K,rel=0.0):
    out=[]
    for it in items:
        mods=[x for x in it["modules"] if x["det_score"]>=0.5]; pred={}
        if len(mods)>=3:
            s=R.z(np.array([x["raw"] for x in mods]),c)+np.abs(R.z(np.array([x["std"] for x in mods]),c))
            s=np.where(np.array([x["fill"] for x in mods])<R.FILL_MIN,0,s)
            s=np.where(np.array([touches(x["polygon"],m) for x in mods]),-1e9,s)
            keep=np.zeros(len(s),bool); order=np.argsort(-s)[:K]; keep[order]=True
            keep&=s>=thr; keep&=s>=rel*s.max()
        else: keep=np.zeros(len(mods),bool)
        for x,k in zip(mods,keep): pred.setdefault("hotspot" if k else "solar_panel",[]).append(x["polygon"])
        out.append({"ground_truth":it["truth"],"prediction":pred,"group":it["group"]})
    return out
def frame_score(mods,c,variant,a=1.0):
    z=lambda k:R.z(np.array([x[k] for x in mods]),c)
    base=z("raw")+np.abs(z("std"))
    pk=np.maximum(z("pk"),z("pn"))
    s={"base":base,"base+pk":base+a*pk,"raw+pk":z("raw")+a*pk,"raw+std+brightpk":base+a*z("pk"),"pk":pk}[variant]
    s=np.where(np.array([x["fill"] for x in mods])<R.FILL_MIN,0,s)
    return np.where(np.array([touches(x["polygon"],5) for x in mods]),-1e9,s)
def label_v(items,c,thr,variant,a,K=2,rel=0.7):
    out=[]
    for it in items:
        mods=[x for x in it["modules"] if x["det_score"]>=0.5]; pred={}
        keep=np.zeros(len(mods),bool)
        if len(mods)>=3:
            s=frame_score(mods,c,variant,a); keep[np.argsort(-s)[:K]]=True; keep&=(s>=thr)&(s>=rel*s.max())
        for x,k in zip(mods,keep): pred.setdefault("hotspot" if k else "solar_panel",[]).append(x["polygon"])
        out.append({"ground_truth":it["truth"],"prediction":pred,"group":it["group"]})
    return out
