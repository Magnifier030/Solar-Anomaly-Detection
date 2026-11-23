import json,glob,os,numpy as np
P="/home/M11509204/researchs/Solar-Anomaly-Detection/data/raw/datasets/9_16_train_set/train/"
M=json.load(open("crackbig_train_meta.json")); G={m[0]:m[1] for m in M}
d_all=[];n_frames=0;rand=[]
rng=np.random.default_rng(0)
for f in sorted(glob.glob(P+"EO/label/*.json")):
    n=os.path.basename(f)[:-5]; e=json.load(open(f)); i=json.load(open(P+"IR/label/"+n+".json"))
    cr=[np.mean(s["points"],0)/[e["imageWidth"],e["imageHeight"]] for s in e["shapes"] if s["label"]=="crack"]
    hs=[np.mean(s["points"],0)/[i["imageWidth"],i["imageHeight"]] for s in i["shapes"] if s["label"]=="hotspot"]
    if not cr or not hs: continue
    n_frames+=1
    for c in cr:
        d_all.append(min(np.linalg.norm(c-h) for h in hs)); rand.append(min(np.linalg.norm(rng.uniform(0,1,2)-h) for h in hs))
print("EO-crack & IR-hotspot frames",n_frames,"cracks",len(d_all),"| normalised dist crack→nearest hotspot: median",round(np.median(d_all),3),"frac<.05",round(np.mean(np.array(d_all)<.05),2),"| random point median",round(np.median(rand),3))
