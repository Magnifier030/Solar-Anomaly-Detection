import json, random, sys
import numpy as np
sys.argv=["x"]
exec(open("crack_sheet.py").read().split("random.seed(1)")[0])
random.seed(2)
for lo,hi,fn in ((0.8,0.85,"band80.jpg"),(0.75,0.8,"band75.jpg"),(0.7,0.75,"band70.jpg"),(0.6,0.7,"band60.jpg")):
    b=[r for r in rows if lo<=r[2]<hi and not r[3]]; print(lo,hi,len(b)); sheet(random.sample(b,min(32,len(b))),fn)
