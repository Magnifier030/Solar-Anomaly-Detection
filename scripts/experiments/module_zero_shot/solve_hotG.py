"""usage: solve_hotG.py F_pseudo F_grey  -> G, per-domain T/precision (swap full: F=.33533, P=650 = 449 pseudo + 201 grey)"""
import sys
F,P,P1,P2=.33533,650,449,201; F1,F2=map(float,sys.argv[1:3])
G=(F*P-F1*P1-F2*P2)/(F1+F2-F); T1=F1*(P1+G)/2; T2=F2*(P2+G)/2
print(f"G={G:.0f}  T={T1+T2:.0f} prec={(T1+T2)/P:.2f} rec={(T1+T2)/G:.2f} | pseudo T={T1:.0f} prec={T1/P1:.2f} | grey T={T2:.0f} prec={T2/P2:.2f} | GT not found={G-T1-T2:.0f}")
