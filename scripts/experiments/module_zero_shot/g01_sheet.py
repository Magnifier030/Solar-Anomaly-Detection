import sys; sys.argv=["raw_eval.py","ir"]
import raw_eval as R, json, numpy as np, cv2
from rectify import rectify
it=[i for i in json.load(open("raw_ir.json")) if i["group"]=="G01"]
s=R.label(it,0.5,2.5,True)
fp=[];gt=[];ctx=[]
for x,i in zip(s,it):
    img=R.mm.load_image(R.mm.image_path(R.mm.TRAIN/"IR"/"image",i["name"]),"ir")
    for p in x["prediction"].get("hotspot",[]):
        t=cv2.resize(rectify(img,np.asarray(p,float),96),(80,120))
        if not i["truth"].get("hotspot"): fp.append(t)
        if len(ctx)<6 and not i["truth"].get("hotspot"):
            c=img.copy(); cv2.polylines(c,[np.int32(p)],True,(255,0,0),2); ctx.append(cv2.resize(c,(320,256)))
    for p in i["truth"].get("hotspot",[]): gt.append(cv2.resize(rectify(img,np.asarray(p,float),96),(80,120)))
def grid(ts,n=12):
    ts=ts+[np.zeros_like(ts[0])]*((-len(ts))%n); return np.vstack([np.hstack(ts[r*n:(r+1)*n]) for r in range(len(ts)//n)])
cv2.imwrite("g01_fp.jpg",cv2.cvtColor(np.vstack([grid(fp[:48]),grid(gt)]),cv2.COLOR_RGB2BGR))
cv2.imwrite("g01_ctx.jpg",cv2.cvtColor(grid(ctx,3),cv2.COLOR_RGB2BGR))
