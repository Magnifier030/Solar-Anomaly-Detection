import numpy as np, cv2, re
src=open("crack_big.py").read()
code=src[src.index("def smooth_field"):src.index("@torch.no_grad()")]
SZ=672; rng=np.random.default_rng(1); exec(code)
im=cv2.imread("train_crack.jpg"); c=cv2.resize(im[0:192,768:960],(SZ,SZ)); c2=cv2.resize(im[192:384,0:192],(SZ,SZ))
t=[c]+[augment(c) for _ in range(7)]+[c2]+[augment(c2) for _ in range(7)]
t=[cv2.resize(x,(200,200)) for x in t]
cv2.imwrite("aug_check.jpg",np.vstack([np.hstack(t[:8]),np.hstack(t[8:])]))
