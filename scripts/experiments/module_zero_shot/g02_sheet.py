import sys; sys.argv=["raw_eval.py","ir"]
import raw_eval as R, json, numpy as np, cv2
from rectify import rectify
from border_eval_lib import touches
import clip_crack as C
it=[i for i in json.load(open("raw_ir_cal.json")) if i["group"]=="G02"]
rows=[]
for i in it[:24]:
    gt=i["truth"].get("hotspot",[]); mods=[m for m in i["modules"] if m["det_score"]>=0.5]
    if not gt or len(mods)<3: continue
    s=R.z(np.array([m["raw"] for m in mods]),0.5)+np.abs(R.z(np.array([m["std"] for m in mods]),0.5))
    s=np.where([touches(m["polygon"],5) for m in mods],-1e9,s)
    y=C.cv_iou_truth({"truth":{"crack":gt}},[m["polygon"] for m in mods]) if False else None
    img=R.mm.load_image(R.mm.image_path(R.mm.TRAIN/"IR"/"image",i["name"]),"ir")
    gtc=cv2.resize(rectify(img,np.asarray(gt[0],float),96),(80,120)); cv2.rectangle(gtc,(0,0),(79,119),(0,255,0),3)
    top=[cv2.resize(rectify(img,np.asarray(mods[j]["polygon"],float),96),(80,120)) for j in np.argsort(-s)[:5]]
    rows.append(np.hstack([gtc]+top))
cv2.imwrite("g02_rank.jpg",cv2.cvtColor(np.vstack(rows[:12]),cv2.COLOR_RGB2BGR))
i=it[0]; img=R.mm.load_image(R.mm.image_path(R.mm.TRAIN/"IR"/"image",i["name"]),"ir").copy()
for t in i["truth"]["hotspot"]: cv2.polylines(img,[np.int32(t)],True,(0,255,0),2)
cv2.imwrite("g02_frame.jpg",cv2.cvtColor(img,cv2.COLOR_RGB2BGR))
