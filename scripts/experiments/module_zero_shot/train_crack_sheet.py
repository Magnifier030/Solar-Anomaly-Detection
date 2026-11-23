import json,glob,random,sys
from pathlib import Path
import numpy as np,cv2
from rectify import rectify
P=Path("/home/M11509204/researchs/Solar-Anomaly-Detection/data/raw/datasets/9_16_train_set/train/EO")
import csv
grp={}
for r in csv.DictReader(open("/home/M11509204/researchs/Solar-Anomaly-Detection/data/raw/supplement/SolarDefectDataset_supplement/train_groups.csv")):
    grp[list(r.values())[0]]=list(r.values())[1]
items=[]
for f in sorted(glob.glob(str(P/"label/*.json"))):
    d=json.load(open(f))
    for s in d["shapes"]:
        if s["label"]=="crack" and len(s["points"])>=3: items.append((f,d["imagePath"],s["points"]))
random.seed(0); random.shuffle(items)
tiles=[];seen=0
for f,ip,pts in items[:32]:
    im=cv2.imread(str(P/"image"/Path(ip).name)); 
    if im is None: continue
    c=rectify(im,np.asarray(pts,float),192,inset=0.0); cv2.putText(c,Path(f).stem[:14],(3,14),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,255,255),1); tiles.append(c)
while len(tiles)%8: tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite("train_crack.jpg",np.vstack([np.hstack(tiles[i:i+8]) for i in range(0,len(tiles),8)])); print(len(items),len(tiles))
