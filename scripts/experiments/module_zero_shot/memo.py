import json,numpy as np,collections
O="/home/M11509204/researchs/Solar-Anomaly-Detection/outputs/submissions/"
X=np.load("crackbig3_train.npy").astype(np.float32); M=json.load(open("crackbig3_train_meta.json"))
T=np.load("crackbig3_test.npy").astype(np.float32); TM=json.load(open("crackbig3_test_meta.json")); dom=json.load(open("test_eo_domain.json"))
def nz(A): A=A[:,:1024]; return A/np.linalg.norm(A,axis=1,keepdims=True)
cl=[i for i,m in enumerate(M) if m[4]==0]
Xc=nz(X[cl]); g=np.array([M[i][1] for i in cl]); y=np.array([M[i][2] for i in cl])
G01c=Xc[(g=="G01")&(y==1)]; G01n=Xc[(g=="G01")&(y==0)]; G04c=Xc[(g=="G04")&(y==1)]; G07c=Xc[(g=="G07")&(y==1)]
intra=(G01c@G01c.T)[np.triu_indices(len(G01c),1)]; print("train G01 crack-crack cos median",round(float(np.median(intra)),3),"| G01 crack vs G01 normal max-cos median",round(float(np.median((G01c@G01n.T).max(1))),3))
idx={(m[0],m[3]):i for i,m in enumerate(TM)}
def picks(f): return [idx[tuple(x)] for x in json.load(open(f))["added"]]
for name,f in (("CL top100","tmp_crackcl/manifest.json"),("DINOv3-L top100","tmp_crackbig3/manifest.json")):
    P=picks(f); Tn=nz(T[P]); a=(Tn@G01c.T).max(1); b=(Tn@G01n.T).max(1); c4=(Tn@G04c.T).max(1); c7=(Tn@G07c.T).max(1)
    same=a>np.median(intra)
    print(name,": near-copy of train G01 crack module (cos > intra median):",int(same.sum()),"| closer to G01 crack than any G01 normal:",int((a>b).sum()),
          "| nearest crack group counts",collections.Counter(np.array(["G01","G04","G07"])[np.argmax(np.c_[a,c4,c7],1)]).most_common(),
          "| domains of near-copies",collections.Counter(dom[TM[i][0]][0] for i,s in zip(P,same) if s).most_common())
