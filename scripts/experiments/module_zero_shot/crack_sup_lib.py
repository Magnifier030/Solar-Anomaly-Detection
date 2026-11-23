import json, numpy as np, torch
def fit(X, y, wd=1e-3, ep=300):
    X = torch.tensor(X, device="cuda"); yt = torch.tensor(y, dtype=torch.float32, device="cuda")
    w = torch.zeros(X.shape[1], device="cuda", requires_grad=True); b = torch.zeros(1, device="cuda", requires_grad=True)
    pw = (len(y) - y.sum()) / max(y.sum(), 1)
    opt = torch.optim.LBFGS([w, b], max_iter=ep, line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad(); l = torch.nn.functional.binary_cross_entropy_with_logits(X @ w + b, yt, pos_weight=torch.tensor(pw, device="cuda")) + wd * (w * w).sum(); l.backward(); return l
    opt.step(closure); return w.detach(), b.detach()
