"""AnomalyDINO-style memory bank of normal module patches, combined with within-frame scoring.

Bank: rectified DINOv2 patch tokens of GT normal (solar_panel) modules, built per training group.
A module's bank score is the mean of its top-10% patch distances to the nearest bank patch,
where the bank excludes the module's own group and the held-out groups (no leakage).
Scores are robust-z'd within the frame and combined with the cached within-frame score.

Uses the cached pipeline_*.json (predicted modules) for held-out and calibration groups.
Usage: python bank_eval.py ir|eo
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT = Path("/home/M11509204/researchs/Solar-Anomaly-Detection")
SCRATCH = Path(__file__).parent
sys.path.insert(0, str(PROJECT / "src"))
spec = importlib.util.spec_from_file_location("module_maskrcnn", PROJECT / "scripts/module_maskrcnn.py")
mm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mm)

from rectify import rectify  # noqa: E402
from solar_ir.metrics.official import f1_tiou  # noqa: E402
from zs_score import SIZE, TOP, _tokens, robust_z  # noqa: E402

MOD = sys.argv[1]
KTH = {"ir": 1, "eo": 2}[MOD]
HELD = {"G01", "G03", "G07"}
ANOM = mm.ANOMALY[MOD]
PER_GROUP = 400  # normal modules per group in the bank
rng = np.random.default_rng(0)


def crops(image, polygons):
    return [np.ascontiguousarray(rectify(image, np.asarray(p, float), SIZE, inset=0.03), np.uint8) for p in polygons]


def build_banks():
    path = SCRATCH / f"bank_{MOD}.pt"
    if path.exists():
        return torch.load(path)
    groups = mm.frames_by_group()
    folder = mm.TRAIN / mm.FOLDER[MOD] / "image"
    by_group = {}
    for name, g in groups.items():
        if g in HELD or not (mm.TRAIN / mm.FOLDER[MOD] / "label" / f"{name}.json").exists():
            continue
        for label, pts in mm.load_shapes(name, MOD):
            if label == "solar_panel" and len(pts) >= 3:
                by_group.setdefault(g, []).append((name, pts))
    banks = {}
    for g, entries in sorted(by_group.items()):
        pick = rng.choice(len(entries), min(PER_GROUP, len(entries)), replace=False)
        chosen = {}
        for i in pick:
            chosen.setdefault(entries[i][0], []).append(entries[i][1])
        toks = []
        for name, polys in chosen.items():
            image = mm.load_image(mm.image_path(folder, name), MOD)
            toks.append(_tokens(crops(image, polys)).half())
        t = torch.cat(toks)
        # keep a random half of patches per module to bound memory
        idx = torch.randperm(t.shape[1])[: t.shape[1] // 2]
        banks[g] = t[:, idx].reshape(-1, t.shape[-1]).cpu()
        print(f"bank {g}: {len(chosen)} frames {t.shape[0]} modules", flush=True)
    torch.save(banks, path)
    return banks


@torch.no_grad()
def bank_score(tokens, bank):
    k = max(1, int(round(tokens.shape[1] * TOP)))
    out = []
    for t in tokens.half():
        best = torch.full((t.shape[0],), -2.0, device="cuda", dtype=torch.float16)
        for s in range(0, bank.shape[0], 200000):
            best = torch.maximum(best, (t @ bank[s:s + 200000].T).max(1).values)
        out.append((2 - 2 * best.float()).topk(k).values.mean())
    return torch.stack(out).cpu().numpy()


def score_items(items, banks):
    folder = mm.TRAIN / mm.FOLDER[MOD] / "image"
    gpu = {g: b.cuda() for g, b in banks.items()}
    for n, item in enumerate(items, 1):
        mods = item["modules"]
        if len(mods) < 3:
            for m in mods:
                m["bank"] = 0.0
            continue
        image = mm.load_image(mm.image_path(folder, item["name"]), MOD)
        bank = torch.cat([b for g, b in gpu.items() if g != item["group"]])
        raw = bank_score(_tokens(crops(image, [m["polygon"] for m in mods])), bank)
        z = robust_z(raw)
        for m, r, zz in zip(mods, raw, z):
            m["bank_raw"], m["bank"] = float(r), float(zz)
        if n % 50 == 0:
            print(f"scored {n}/{len(items)}", flush=True)


def evaluate(items, key, thr, w=1.0):
    samples = []
    for it in items:
        pred = {}
        for m in it["modules"]:
            if m["det_score"] < 0.5:
                continue
            s = {"zs": m["zs"], "bank": m["bank"], "sum": m["zs"] + w * m["bank"]}[key]
            pred.setdefault(ANOM if s >= thr else "solar_panel", []).append(m["polygon"])
        samples.append({"ground_truth": it["truth"], "prediction": pred, "group": it["group"]})
    r = f1_tiou(samples, ("solar_panel", ANOM))
    per = {g: f1_tiou([s for s in samples if s["group"] == g], ("solar_panel", ANOM)).get("TIoU_" + ANOM, 0)
           for g in sorted({s["group"] for s in samples})}
    return r.get("TIoU_" + ANOM, 0), per


def main():
    banks = build_banks()
    sets = {}
    for tag, suffix in (("held", ""), ("cal", "_cal")):
        cache = SCRATCH / f"pipeline_{MOD}_module-maskrcnn-{MOD}-g013_k{KTH}{suffix}.json"
        out = SCRATCH / f"bank_scored_{MOD}{suffix}.json"
        if out.exists():
            sets[tag] = json.loads(out.read_text())
        else:
            sets[tag] = json.loads(cache.read_text())
            score_items(sets[tag], banks)
            out.write_text(json.dumps(sets[tag]))
    grid = np.arange(0.5, 15.01, 0.25)
    for key, w in (("zs", 1.0), ("bank", 1.0), ("sum", 0.5), ("sum", 1.0), ("sum", 2.0)):
        best = max(grid, key=lambda t: evaluate(sets["cal"], key, t, w)[0])
        cal = evaluate(sets["cal"], key, best, w)[0]
        held, per = evaluate(sets["held"], key, best, w)
        oracle = max(evaluate(sets["held"], key, t, w)[0] for t in grid)
        print(f"{key} w={w}: cal thr {best:.2f} cal {cal:.3f} -> held {held:.3f} {per} | oracle {oracle:.3f}", flush=True)


if __name__ == "__main__":
    main()
