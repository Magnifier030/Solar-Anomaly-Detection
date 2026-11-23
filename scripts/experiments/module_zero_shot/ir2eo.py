import json,glob,os,collections,numpy as np,cv2
from shapely.geometry import Polygon,Point
P="/home/M11509204/researchs/Solar-Anomaly-Detection/data/raw/datasets/"
O="/home/M11509204/researchs/Solar-Anomaly-Detection/outputs/submissions/"
W=np.array([[1.342,0.043,-0.155],[0.148,1.084,-0.161]])  # EO norm -> IR norm (train G04)
Ainv=np.linalg.inv(np.r_[W,[[0,0,1]]])
M=json.load(open("crackbig_train_meta.json")); G={m[0]:m[1] for m in M}
sz=collections.Counter()
for f in glob.glob(P+"9_16_train_set/train/EO/label/*.json")[:2000]:
    d=json.load(open(f)); i=json.load(open(f.replace("/EO/","/IR/"))); sz[(G.get(os.path.basename(f)[:-5]),d["imageWidth"],d["imageHeight"],i["imageWidth"],i["imageHeight"])]+=1
print("train (group, EO WxH, IR WxH):",sorted(sz.items()))
dom=json.load(open("test_eo_domain.json")); tsz=collections.Counter()
for k in dom:
    e=cv2.imread(P+"9_30_test_set/"+k); i=cv2.imread(P+"9_30_test_set/"+k.replace("/EO/","/IR/"))
    tsz[(dom[k][0],e.shape[1],e.shape[0],None if i is None else i.shape[1],None if i is None else i.shape[0])]+=1
print("test (EO-domain, EO WxH, IR WxH):",sorted(tsz.items(),key=lambda x:-x[1])[:15])
