import cv2, glob, random, numpy as np
R="/home/M11509204/researchs/Solar-Anomaly-Detection/data/raw/"
random.seed(0)
ims=sorted(glob.glob(R+"external/PV-HSD-2025/extracted/*/images/*.JPG"))
sel=random.sample(ims,8)
tiles=[]
for p in sel:
    im=cv2.imread(p); h,w=im.shape[:2]
    lab=p.replace("/images/","/labels/").rsplit(".",1)[0]+".txt"
    for l in open(lab):
        c,x,y,bw,bh=map(float,l.split())
        cv2.rectangle(im,(int((x-bw/2)*w),int((y-bh/2)*h)),(int((x+bw/2)*w),int((y+bh/2)*h)),(0,255,0),2)
    tiles.append(cv2.resize(im,(320,256)))
t=sorted(glob.glob(R+"datasets/9_30_test_set/test/IR/image/*"))
random.shuffle(t)
for p in t[:8]: tiles.append(cv2.resize(cv2.imread(p),(320,256)))
rows=[np.hstack(tiles[i:i+4]) for i in range(0,16,4)]
cv2.imwrite("ext_sheet.jpg",np.vstack(rows))
