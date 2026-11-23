import sys,glob,random,numpy as np,cv2,torch
sys.argv=["raw_eval.py","ir"]
import raw_eval as R
mm=R.mm
ims=sorted(glob.glob(str(R.PROJECT/"data/raw/external/PV-HSD-2025/extracted/valid/images/*.JPG"))); random.seed(0); random.shuffle(ims)
dev=torch.device("cuda"); model=mm.build_model("ir",pretrained=False)
model.load_state_dict(torch.load(R.PROJECT/"outputs/module-maskrcnn-ir-all/last.pt",map_location="cpu")["model"]); model.to(dev).eval()
tiles=[]
for p in ims[:8]:
    im=cv2.cvtColor(cv2.imread(p),cv2.COLOR_BGR2RGB)
    for variant in ("rgb","grey"):
        x=im if variant=="rgb" else cv2.cvtColor(cv2.cvtColor(im,cv2.COLOR_RGB2GRAY),cv2.COLOR_GRAY2RGB)
        pr=mm.predict(model,x,dev)
        cnt={t:len(mm.select(pr,t,0.5,0.5)) for t in (0.5,0.3,0.1)}
        print(p[-12:],variant,cnt)
        vis=cv2.cvtColor(x,cv2.COLOR_RGB2BGR).copy()
        for d in mm.select(pr,0.3,0.5,0.5): cv2.polylines(vis,[np.int32(d["polygon"])],True,(0,255,0),2)
        tiles.append(cv2.resize(vis,(320,256)))
cv2.imwrite("ext_det.jpg",np.vstack([np.hstack(tiles[i:i+4]) for i in range(0,16,4)]))
