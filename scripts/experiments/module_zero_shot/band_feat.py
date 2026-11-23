"""Band-shape features of IR hotspot GT crops: does a vertical 1/3-width strip spanning the module explain the anomaly (bypass diode)?"""
import glob, json, re, csv, importlib.util, sys
import numpy as np, cv2
from rectify import rectify
import clip_crack as C
P="/home/M11509204/researchs/Solar-Anomaly-Detection/"
spec=importlib.util.spec_from_file_location("mm",P+"scripts/module_maskrcnn.py"); mm=importlib.util.module_from_spec(spec); spec.loader.exec_module(mm)
G={r["frame_id"].split(".")[0]:r["group"] for r in csv.DictReader(open(P+"data/raw/supplement/SolarDefectDataset_supplement/train_groups.csv"))}

def band_features(grey, poly, size=96):
    c=rectify(grey,np.asarray(poly,float),size,inset=0.08).astype(np.float32)
    med=np.median(c); mad=np.median(np.abs(c-med))*1.4826+2.0
    best=None
    for sign in (1,-1):
        d=sign*(c-med)/mad
        col=d.mean(0)                       # column profile (long axis vertical)
        w=size//3
        win=np.convolve(col,np.ones(w)/w,"valid")
        j=int(np.argmax(win)); inside=col[j:j+w].mean(); outside=np.r_[col[:j],col[j+w:]].mean()
        strip=d[:,j:j+w].mean(1)            # per-row mean inside best strip
        cover=(strip>1.0).mean()            # vertical coverage of the strip
        peak=np.percentile(d,99)
        conc=(d>0.5*peak).mean()            # area fraction of the hot part
        ifr=(d[:,j:j+w]>1.0).mean(); ofr=(np.c_[d[:,:j],d[:,j+w:]]>1.0).mean()
        f=(inside-outside, cover, conc, peak, ifr, ofr)
        if best is None or f[0]>best[0]: best=f
    return best

def main():
    rows=[]
    for f in sorted(glob.glob(P+"data/raw/datasets/9_16_train_set/train/IR/label/*.json")):
        n=f.split('/')[-1][:-5]; shapes=json.load(open(f))["shapes"]
        hs=[s for s in shapes if s["label"]=="hotspot"]
        if not hs: continue
        img=mm.load_image(mm.image_path(mm.TRAIN/"IR"/"image",n),"ir"); grey=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        for s in hs:
            r=s.get("reason","")
            k="diode" if ("二極體" in r or "電池串" in r) else ("internal" if re.search("隱裂|微裂|電性|接觸",r) else "board")
            rows.append((G[n],k)+tuple(band_features(grey,s["points"])))
    g=np.array([r[0] for r in rows]); k=np.array([r[1] for r in rows]); a=np.array([r[2:] for r in rows],float)
    np.save("band_feat.npy",a); np.save("band_meta.npy",np.stack([g,k],1))
    y=k=="diode"
    for i,nm in enumerate(["contrast","cover","conc","peak","in_frac","out_frac"]):
        print(nm,"AUC diode-vs-rest",round(C.roc_auc_score(y,a[:,i]),3),"| G04 only",round(C.roc_auc_score(y[g=="G04"],a[g=="G04",i]),3))
    for grp in sorted(set(g)):
        for cat in ("board","internal","diode"):
            m=(g==grp)&(k==cat)
            if m.any(): print(grp,cat,m.sum(),np.round(np.median(a[m],0),2))
if __name__=="__main__": main()
