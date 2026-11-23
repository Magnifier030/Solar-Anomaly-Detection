import json,glob,os,numpy as np,cv2,collections
P="/home/M11509204/researchs/Solar-Anomaly-Detection/data/raw/datasets/"
TM=json.load(open("crackbig3_test_meta.json")); dom=json.load(open("test_eo_domain.json"))
add=json.load(open("tmp_crackcl/manifest.json"))["added"]
def th(p): g=cv2.resize(cv2.imread(p,0),(96,54)).astype(np.float32); return ((g-g.mean())/(g.std()+1e-6)).ravel()
M=json.load(open("crackbig_train_meta.json")); G={m[0]:m[1] for m in M}
crf={os.path.basename(f)[:-5] for f in glob.glob(P+"9_16_train_set/train/EO/label/*.json") if any(s["label"]=="crack" for s in json.load(open(f))["shapes"])}
tr=[n for n in crf if G.get(n)=="G01"]; A=np.stack([th(P+"9_16_train_set/train/EO/image/"+n+".jpg") for n in tr]); A/=np.linalg.norm(A,axis=1,keepdims=True)
for r,(k,j) in enumerate(add[:40]):
    t=th(P+"9_30_test_set/"+k); t/=np.linalg.norm(t); s=A@t
    print(r,k[-12:],dom[k][0],"max corr to train G01 crack frames",round(float(s.max()),2),tr[int(s.argmax())])
