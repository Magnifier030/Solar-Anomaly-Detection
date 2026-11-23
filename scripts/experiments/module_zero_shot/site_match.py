"""Assign each test frame to its nearest train group by DINOv2 CLS embedding (cosine kNN, k=5), per modality."""
import csv, json, sys, collections
from pathlib import Path
import numpy as np, torch, cv2, importlib.util
from transformers import AutoModel
P = Path("/home/M11509204/researchs/Solar-Anomaly-Detection")
spec = importlib.util.spec_from_file_location("mm", P / "scripts/module_maskrcnn.py"); mm = importlib.util.module_from_spec(spec); spec.loader.exec_module(mm)
G = {r["frame_id"].split(".")[0]: r["group"] for r in csv.DictReader(open(P / "data/raw/supplement/SolarDefectDataset_supplement/train_groups.csv"))}
TEST = P / "data/raw/datasets/9_30_test_set"
model = AutoModel.from_pretrained("facebook/dinov2-with-registers-base").cuda().eval().half()
mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda().half(); std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda().half()
@torch.no_grad()
def emb(images):
    out = []
    for i in range(0, len(images), 32):
        x = torch.from_numpy(np.stack([cv2.resize(im, (448, 364)) for im in images[i:i + 32]])).cuda().half().permute(0, 3, 1, 2) / 255
        f = model(pixel_values=(x - mean) / std).last_hidden_state
        out.append(torch.nn.functional.normalize(torch.cat([f[:, 0], f[:, 5:].mean(1)], 1).float(), dim=1).cpu())
    return torch.cat(out).numpy()
res = {}
for mod, folder in (("ir", "IR"), ("eo", "EO")):
    names = sorted(G); tr = emb([mm.load_image(mm.image_path(mm.TRAIN / folder / "image", n), mod) for n in names])
    keys = sorted(k for k in json.load(open(P / "outputs/submissions/ir-v3-peak-top2-diode-clipwin-noglare-20261106/submission.json")) if f"/{folder}/" in k)
    te = emb([mm.load_image(TEST / k, mod) for k in keys])
    g = np.array([G[n] for n in names])
    # leave-one-group-out self similarity reference: nearest train frame from a different group
    sim_tt = tr @ tr.T; np.fill_diagonal(sim_tt, -1)
    same = sim_tt.max(1)
    cross = np.array([sim_tt[i][g != g[i]].max() for i in range(len(g))])
    sim = te @ tr.T; nn = np.argsort(-sim, 1)[:, :5]
    vote = [collections.Counter(g[r]).most_common(1)[0][0] for r in nn]; top = sim.max(1)
    print(f"== {mod}: train nearest same-group sim median {np.median(same):.3f}, nearest other-group median {np.median(cross):.3f}")
    print("test nearest sim quantiles", np.round(np.percentile(top, [5, 25, 50, 75, 95]), 3))
    c = collections.Counter(vote); print("test assigned groups", sorted(c.items()))
    for grp in sorted(c): 
        k = np.array(vote) == grp; print(grp, "n", k.sum(), "median sim", np.round(np.median(top[k]), 3))
    res[mod] = {k: [v, float(t)] for k, v, t in zip(keys, vote, top)}
json.dump(res, open("site_match.json", "w"))
