"""Qwen2.5-VL yes/no crack verification of EO candidates (per-frame DINOv2 top-1, z>=1.5) plus all GT-crack modules.
Score = P('Yes')/(P('Yes')+P('No')) for the first answer token. Reports AUC vs CLIP (whole+3x3)/2 on the same candidates
and gated top-1 F1 on unseen groups. Usage: python vlm_crack.py [test]"""
import json, sys
from pathlib import Path
import numpy as np, torch
MODE = sys.argv[1] if len(sys.argv) > 1 else "val"
sys.argv = ["raw_eval.py", "eo"]
import raw_eval as R
import clip_crack as C
from rectify import rectify
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
S = Path(__file__).parent; PROJECT = Path("/home/M11509204/researchs/Solar-Anomaly-Detection")
mm = R.mm
PROMPT = ("This is a close-up crop of one photovoltaic module from a drone RGB image. "
          "Is this solar module physically damaged, with cracked or shattered glass or broken cells? "
          "Ignore reflections, sun glare, shadows, dirt and grid lines. Answer Yes or No.")

def load():
    hub = PROJECT / ".hf_cache/hub"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", cache_dir=hub, dtype=torch.bfloat16).cuda().eval()
    proc = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", cache_dir=hub)
    tok = proc.tokenizer
    ids = [tok.encode(w, add_special_tokens=False)[0] for w in ("Yes", "No")]
    return model, proc, ids

@torch.no_grad()
def score(model, proc, ids, crops):
    out = []
    for i in range(0, len(crops), 8):
        batch = crops[i:i + 8]
        msgs = [[{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}] for _ in batch]
        texts = [proc.apply_chat_template(m, add_generation_prompt=True, tokenize=False) for m in msgs]
        x = proc(text=texts, images=[Image.fromarray(c) for c in batch], return_tensors="pt", padding=True, padding_side="left").to("cuda")
        logit = model(**x).logits[:, -1, ids].float()
        out += logit.softmax(-1)[:, 0].cpu().tolist()
    return np.array(out)

def main():
    model, proc, ids = load()
    if MODE == "val":
        rows = []
        for tag, suf in (("held", ""), ("cal", "_cal")):
            for n, it in enumerate(json.loads((S / f"raw_eo{suf}.json").read_text()), 1):
                mods = [m for m in it["modules"] if m["det_score"] >= 0.7 and "raw" in m]
                if len(mods) < 3: continue
                y = C.cv_iou_truth(it, [m["polygon"] for m in mods])
                zd = R.z(np.array([m["raw"] for m in mods]), 0.1); zd = np.where(np.array([m["fill"] for m in mods]) < R.FILL_MIN, 0, zd)
                top = int(np.argmax(zd)); pick = set(np.where(y)[0]) | ({top} if zd[top] >= 1.5 else set())
                if it["group"] == "G04" and n % 4: pick = {top} & pick if zd[top] >= 1.5 else set()  # subsample G04
                if not pick: continue
                image = mm.load_image(mm.image_path(mm.TRAIN / "EO" / "image", it["name"]), "eo")
                for j in sorted(pick):
                    rows.append(dict(tag=tag, group=it["group"], frame=f"{tag}{n}", y=bool(y[j]), top=bool(j == top and zd[top] >= 1.5),
                                     crop=np.ascontiguousarray(rectify(image, np.asarray(mods[j]["polygon"], float), 448, inset=0.03))))
        print("crops", len(rows), "pos", sum(r["y"] for r in rows), flush=True)
        p = score(model, proc, ids, [r["crop"] for r in rows])
        for r, v in zip(rows, p): r["vlm"] = float(v); del r["crop"]
        json.dump(rows, open(S / "vlm_crack_val.json", "w"))
        y = np.array([r["y"] for r in rows]); g = np.array([r["group"] for r in rows]); top = np.array([r["top"] for r in rows])
        for grp in ("G01", "G07", "G04"):
            k = (g == grp)
            if y[k].any() and (~y[k]).any(): print(grp, "AUC all", round(C.roc_auc_score(y[k], p[k]), 3))
        k = top
        print("top-1 candidates", k.sum(), "pos", y[k].sum(), "AUC", round(C.roc_auc_score(y[k], p[k]), 3))
        for grp in ("G01", "G07", "G04", "normal"):
            kk = k & ((g == grp) if grp != "normal" else ~np.isin(g, ["G01", "G03", "G07", "G04"]))
            print(grp, "cands", kk.sum(), "pos", y[kk].sum(), "mean p pos", np.round(p[kk & y].mean() if (kk & y).any() else -1, 3), "neg", np.round(p[kk & ~y].mean(), 3))
        for t in (0.3, 0.5, 0.7, 0.9):
            un = k & np.isin(g, ["G01", "G03", "G07"]); nm = k & ~np.isin(g, ["G01", "G03", "G07", "G04"])
            tp = (un & y & (p >= t)).sum(); fp = (un & ~y & (p >= t)).sum(); fn = (np.isin(g, ["G01", "G07"]) & y).sum() - tp
            print(f"p>={t}: unseen tp {tp} fp {fp} fn {fn} F1 {2*tp/max(2*tp+fp+fn,1):.3f} | normal-group fp {(nm & (p>=t)).sum()} | G04 kept {(k&(g=='G04')&y&(p>=t)).sum()}/{(k&(g=='G04')&y).sum()}")

if __name__ == "__main__":
    main()
