# F1_c ≈ 2t/(P+N); sets: G (258, F .04918: t_G = t_G223 since its 35 G01/G07 cracks have 0 hits), thr30-bigcrack100 = v2top100 + G223 (323, .21345),
# bigcrack500 = G223 + v2top500 (723, .13467), DINOv3-L 100 (.32659), CL 100 (.39206)
import numpy as np
for p101 in (0,.01,.02,.03,.05):
    # solve N: t(v2 101-500) = 400*p101
    # .13467(723+N)/2 - .21345(323+N)/2 = 400 p101
    N=(.13467*723-.21345*323-800*p101)/(.21345-.13467)
    tG=.04918*(258+N)/2; tv2=.21345*(323+N)/2-tG; tL=.32659*(100+N)/2; tC=.39206*(100+N)/2
    print(f"prec(v2 101-500)={p101:.2f}: N≈{N:.0f} | t: G223 {tG:.1f} ({tG/223:.0%}), v2top100 {tv2:.1f}, DINOv3-L {tL:.1f}, CL {tC:.1f} | CL recall {tC/N:.0%}, max F1 if 100 all correct {200/(100+N):.2f}")
