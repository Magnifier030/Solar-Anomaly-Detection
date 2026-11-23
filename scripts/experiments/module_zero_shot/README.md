# 模組級零樣本實驗腳本

2026-11-05 至 2026-11-23 的研究腳本原樣保存，用來追溯實驗，不是可直接安裝的套件。

- 執行：在本資料夾內用專案 `.venv` 執行。腳本彼此以同目錄 import，部分寫死本機專案絕對路徑。
- 中間產物（特徵、快取、抽樣圖）約 3.6GB，未進版控。檔名、大小與 SHA-256 見 [artifact_manifest.csv](../../../docs/experiments/logs/module_zero_shot_20261123/artifact_manifest.csv)。
- 完整結論見 [實驗總整理](../../../docs/experiments/module_zero_shot_20261123.md)。

## IR 熱點（48）

| 腳本 | 用途 |
|---|---|
| `band_feat.py` | Band-shape features of IR hotspot GT crops: does a vertical 1/3-width strip spanning the module explain the anomaly (bypass diode) |
| `band_sheet.py` | （無說明，臨時分析腳本） |
| `band_test.py` | （無說明，臨時分析腳本） |
| `border_area.py` | （無說明，臨時分析腳本） |
| `border_eval.py` | （無說明，臨時分析腳本） |
| `border_eval_lib.py` | （無說明，臨時分析腳本） |
| `border_fast.py` | （無說明，臨時分析腳本） |
| `border_filter.py` | Relabel IR hotspots whose polygon lies within M px of the image border to solar_panel. Usage: SRC OUT M |
| `build_ir_floor.py` | Submission: EO + crack reasons from BASE; IR rebuilt from the test cache with hotspot = z(mutual, c) + /z(std, c)/ >= THR (fill <  |
| `build_ir_v2.py` | IR rebuilt from test cache: s = z(mutual,c)+/z(std,c)/ (fill<0.85 -> 0), modules within 5px of border excluded, hotspot = top-K pe |
| `build_ir_v3.py` | IR rebuilt from test cache: s = z(mutual,c)+/z(std,c)/ (fill<0.85 -> 0), modules within 5px of border excluded, hotspot = top-K pe |
| `build_ir_v4.py` | IR rebuilt from test cache: s = z(mutual,c)+/z(std,c)/ (fill<0.85 -> 0), modules within 5px of border excluded, hotspot = top-K pe |
| `build_ir_v5.py` | IR v5: v3 score; separate (K,thr,rel) for dense (>=NDENSE modules) and sparse frames. Reasons for hotspots already in BASE are cop |
| `build_ir_v6.py` | IR v5: v3 score; separate (K,thr,rel) for dense (>=NDENSE modules) and sparse frames. Reasons for hotspots already in BASE are cop |
| `cap_eval.py` | （無說明，臨時分析腳本） |
| `frame_intensity.py` | Within-frame intensity outlier score per module, combined with the DINOv2 mutual score. Texture features miss uniform brightness c |
| `frame_mutual.py` | Zero-shot within-frame mutual scoring of module crops (MuSc-style, no training). Modules in one frame share panel type, lighting a |
| `g01_sheet.py` | （無說明，臨時分析腳本） |
| `g01nb.py` | （無說明，臨時分析腳本） |
| `g01sheet.py` | （無說明，臨時分析腳本） |
| `g02_sheet.py` | （無說明，臨時分析腳本） |
| `hot_bench.py` | Multi-hotspot benchmark (no test labels, no platform): transplant REAL GT hotspot residuals (G01-G04) into n random modules of rea |
| `hot_build.py` | Hotspot rebuild on top of BASE (EO/crack untouched). swap TD TA KA : keep BASE hotspots with avg-CL logit >= TD, add non-picked mo |
| `hot_cl.py` | Hotspot CL head (SupCon + BCE, linear) on hotcl_* features. Classes: 0 normal (real G01-G04 normals + syn clean), 1 hotspot (real, |
| `hot_err.py` | Validation error decomposition of the uploaded hotspot pipeline (v3 rule + CL swap with LOGO logits) on G01-G04. FN causes: nomod  |
| `hot_feat.py` | Hotspot CL features: DINOv3-L small-defect descriptor (as crack_big3) on IR module crops, contrast-normalised per frame. Crop norm |
| `hot_pool.py` | Pooled LOGO comparison: one global rule for all 4 labelled groups (like test: one threshold for all sites). Uses held-out logits h |
| `hot_real.py` | Real-label check (no transplant, no per-frame cap): G01-G04 GT. Threshold = 1 % FPR on normal modules of the OTHER three groups, a |
| `hot_real2.py` | Real-label check (no transplant, no per-frame cap): G01-G04 GT. Threshold = 1 % FPR on normal modules of the OTHER three groups, a |
| `hot_sheet.py` | usage: hot_sheet.py LOGIT_TAG -> hot_cmp{TAG}.jpg rows: picks lowest CL (3 rows), top CL not picked (3 rows), crops shown in ORIGI |
| `hot_tpl_test.py` | （無說明，臨時分析腳本） |
| `peak_feat.py` | Add per-module local peak contrast to IR caches: on a 64px rectified grey crop (inset 0.1, Gaussian blur), pk = p99.5 - median (br |
| `pk_eval.py` | （無說明，臨時分析腳本） |
| `pseudo_cmap.py` | （無說明，臨時分析腳本） |
| `pseudo_resid.py` | （無說明，臨時分析腳本） |
| `pseudo_sheet.py` | （無說明，臨時分析腳本） |
| `rel_eval.py` | （無說明，臨時分析腳本） |
| `route_diode.py` | Route reasons: IR hotspots with a bypass-diode band shape (vertical 1/3 strip, uniform, spanning the module) get the diode sentenc |
| `route_sim.py` | （無說明，臨時分析腳本） |
| `solve_hotG.py` | usage: solve_hotG.py F_pseudo F_grey -> G, per-domain T/precision (swap full: F=.33533, P=650 = 449 pseudo + 201 grey) |
| `spot.py` | White-spot score per module: grey 256px rectified; residual = g - max(open_hline, open_vline) (keeps compact bright blobs, removes |
| `spot_ana.py` | （無說明，臨時分析腳本） |
| `spot_eval.py` | （無說明，臨時分析腳本） |
| `spot_eval2.py` | （無說明，臨時分析腳本） |
| `spot_feat.py` | Small-scale spot contrast per module: 128px rectified grey (inset .05), tophat = c - medianBlur(c,k); sp = p99.9 of tophat, sp2 =  |
| `union_ir.py` | Add-only union: OUT = F, plus every hotspot of BASE (same module polygon) relabelled hotspot with BASE's reason. Usage: BASE F OUT |
| `zlow2_ir.py` | zlow2 for the IR mutual term: per module store z-scored (c=0.5) mutual score under 6 transforms -> m['zr6']. |
| `zlow2_ir_eval.py` | （無說明，臨時分析腳本） |

## EO 裂縫（56）

| 腳本 | 用途 |
|---|---|
| `big_bright.py` | （無說明，臨時分析腳本） |
| `big_filter.py` | （無說明，臨時分析腳本） |
| `big_sheet.py` | （無說明，臨時分析腳本） |
| `build_eo_add.py` | Add-only crack: keep BASE fully; in EO frames with no crack, add DINOv2 top-1 (z floor c) if z>=THR and CLIP whole p>=P. Usage: BA |
| `build_eo_add2.py` | Add-only crack: keep BASE fully; in EO frames with no crack, add DINOv2 top-1 (z floor c) if z>=THR and CLIP whole p>=P. Usage: BA |
| `build_eo_big.py` | Add-only cracks from the large-model small-defect classifier (crack_big + geo3/bi, platform negatives). Candidates big_cand_f.json |
| `build_eo_clipall.py` | Add-only cracks: every test EO module (det>=0.7) with CLIP B/16 crack prob >= P and crop mean grey <= BRIGHT -> crack. IR/other un |
| `build_eo_clipgate.py` | Submission: IR + reasons from BASE; EO crack = Mask R-CNN label OR (per-frame DINOv2 top-1 with floored z >= THR AND CLIP 'cracked |
| `build_eo_clipwin.py` | Submission: IR + reasons from BASE; EO crack = Mask R-CNN label OR (per-frame DINOv2 top-1 with floored z >= THR AND CLIP 'cracked |
| `build_eo_crack.py` | Rebuild a submission: keep IR (and reasons) from a base submission, replace EO with modules from the test cache where crack = Mask |
| `build_v3.py` | Fixed candidate pipeline. Exclude only platform-proven non-cracks (H-added 386, G01/G07 zero-hit 35) and all G01/G07-domain frames |
| `clipL_crack.py` | WinCLIP-style text-prompt zero-shot crack score for EO modules (held-out + calibration caches). Per rectified module crop: CLIP Vi |
| `clip_crack.py` | WinCLIP-style text-prompt zero-shot crack score for EO modules (held-out + calibration caches). Per rectified module crop: CLIP Vi |
| `clip_crack_m.py` | WinCLIP-style text-prompt zero-shot crack score for EO modules (held-out + calibration caches). Per rectified module crop: CLIP Vi |
| `clip_test_all.py` | CLIP B/16 crack prob for every test EO module (det>=0.7). Output test_eo_clip_all.json {key: [prob per kept module]} + crops cache |
| `clip_window.py` | WinCLIP-style windowed CLIP crack score vs whole-module CLIP score. Per module: rectify to 448, CLIP 'cracked' prob for the whole  |
| `coinc.py` | （無說明，臨時分析腳本） |
| `crack_band.py` | （無說明，臨時分析腳本） |
| `crack_big.py` | Small-defect crack features with a large frozen backbone + geo3 / bi augmentation (from How-to-overcome-domain-shift). Backbone: D |
| `crack_big3.py` | [DINOv3 ViT-L/16 @768 variant of crack_big.py] Small-defect crack features with a large frozen backbone + geo3 / bi augmentation ( |
| `crack_big3h.py` | [DINOv3 ViT-H+/16 @768 variant of crack_big.py] Small-defect crack features with a large frozen backbone + geo3 / bi augmentation  |
| `crack_big_fit.py` | Fit / validate the large-model small-defect crack classifier (features from crack_big.py). LOGO over crack groups (G01, G04, G07); |
| `crack_cl.py` | Supervised-contrastive head on frozen DINOv3-L small-defect descriptors: classes crack / normal / white-spot (synthetic white blob |
| `crack_cl2.py` | CL v2: add REAL white-spot negatives = v2 ranks 101-500 (platform ~2% precision), split by frame into halves A/B: train with half  |
| `crack_cl3.py` | CL v3 = CL v1 + counterfactual tree-shadow negatives (class 3) + optional group-balanced crack sampling (BAL=1). Reports LOGO and  |
| `crack_cl_shadow.py` | [DINOv3 ViT-L/16 @768 variant of crack_big.py] Small-defect crack features with a large frozen backbone + geo3 / bi augmentation ( |
| `crack_cl_syn.py` | [DINOv3 ViT-L/16 @768 variant of crack_big.py] Small-defect crack features with a large frozen backbone + geo3 / bi augmentation ( |
| `crack_domain_only.py` | Keep cracks only in test EO frames whose nearest train domain is in KEEP (others -> solar_panel). Usage: BASE OUT G01,G07 |
| `crack_feat.py` | High-res module features for supervised crack learning. Train: GT polygons (solar_panel=0, crack=1) of every train EO frame. Test: |
| `crack_final.py` | LOGO precision-vs-threshold for dino-abs; final model on all train; score test modules. |
| `crack_sheet.py` | （無說明，臨時分析腳本） |
| `crack_sup.py` | Leave-one-group-out supervised crack classifier on high-res module features. No per-frame cap. Variants: feature set x (absolute / |
| `crack_sup_lib.py` | （無說明，臨時分析腳本） |
| `dom.py` | （無說明，臨時分析腳本） |
| `dom2.py` | （無說明，臨時分析腳本） |
| `dom_filt.py` | （無說明，臨時分析腳本） |
| `dom_sheet.py` | （無說明，臨時分析腳本） |
| `drop_det_crack.py` | Relabel cracks that sit on a Mask R-CNN class-2 module (all 28 scored 0 on the platform) to solar_panel. Usage: SRC OUT |
| `fast_sweep.py` | Parallel sweep of EO crack rules on cached raw features (held-out + calibration). rule: crack if zero-shot floored-z >= thr (plain |
| `g07big.py` | （無說明，臨時分析腳本） |
| `glare_filter.py` | Relabel EO cracks whose rectified module crop has > SAT fraction of pixels >= 245 grey (sun glare) back to solar_panel. Train crac |
| `leak.py` | Missing-label check: fit CL head WITHOUT G02/G03/G12 train modules, score those (all labelled solar_panel), show top tiles per gro |
| `reg.py` | Per-frame EO<->IR registration by ECC on module masks (init = train-G04 affine). Then: do CL crack modules coincide with predicted |
| `relabel_eval.py` | EO crack recall: keep class-agnostic modules, relabel a module as anomaly when any class-2 detection overlapping it (polygon IoU > |
| `set_cracks.py` | OUT = BASE with cracks set exactly to: (KEEP_OLD ? BASE cracks : none) + modules listed in MANIFEST 'added' (first N). Usage: BASE |
| `solveN.py` | （無說明，臨時分析腳本） |
| `synth_crack.py` | Synthetic-anomaly crack detector for cross-site generalisation. Normal EO modules from non-eval groups (G02,G03,G05,G06,G08-G12) + |
| `train_crack_sheet.py` | （無說明，臨時分析腳本） |
| `tri.py` | （無說明，臨時分析腳本） |
| `tri2.py` | （無說明，臨時分析腳本） |
| `v3cmp.py` | （無說明，臨時分析腳本） |
| `v3direct.py` | Direct DINOv3 crack test: same candidate pipeline as v2 (exclude G/H cracks, panel-ness>=.75, low-freq range<=60), PFX scores. ->  |
| `v3hcmp.py` | （無說明，臨時分析腳本） |
| `vlm_crack.py` | Qwen2.5-VL yes/no crack verification of EO candidates (per-frame DINOv2 top-1, z>=1.5) plus all GT-crack modules. Score = P('Yes') |
| `zlow2_eo.py` | zlow2 (How-to-overcome-domain-shift Stage-1 pose search) on within-frame DINOv2 mutual crack scoring. Per module crop: 6 generic t |
| `zlow2_eval.py` | （無說明，臨時分析腳本） |

## Text（5）

| 腳本 | 用途 |
|---|---|
| `set_reason.py` | OUT = BASE with every crack reason replaced (geometry/labels unchanged -> isolates F1_TEXT). Usage: BASE OUT REASON |
| `text_ana.py` | （無說明，臨時分析腳本） |
| `text_ana2.py` | （無說明，臨時分析腳本） |
| `text_cat.py` | Proxy per hidden category (board / internal / diode for hotspot; crack) using leave-self-out max BERTScore. |
| `text_opt.py` | Reason optimisation: train reasons grouped by (modality, group) -> BERTScore (platform settings) matrix; LOO best sentence per clu |

## 共用流程與快取（31）

| 腳本 | 用途 |
|---|---|
| `audit.py` | （無說明，臨時分析腳本） |
| `aug_check.py` | （無說明，臨時分析腳本） |
| `bank_eval.py` | AnomalyDINO-style memory bank of normal module patches, combined with within-frame scoring. Bank: rectified DINOv2 patch tokens of |
| `dense_eval.py` | （無說明，臨時分析腳本） |
| `dist_cmp.py` | （無說明，臨時分析腳本） |
| `dist_site.py` | （無說明，臨時分析腳本） |
| `distinct.py` | （無說明，臨時分析腳本） |
| `dup.py` | （無說明，臨時分析腳本） |
| `ext_cache.py` | PV-HSD-2025 valid split as unseen-site dense IR validation: modules (Mask R-CNN ir-all), v3 feats, hotspot = module containing a G |
| `ext_det_probe.py` | （無說明，臨時分析腳本） |
| `ext_sheet.py` | （無說明，臨時分析腳本） |
| `frames.py` | （無說明，臨時分析腳本） |
| `full_sweep.py` | （無說明，臨時分析腳本） |
| `ir2eo.py` | （無說明，臨時分析腳本） |
| `ir2eo2.py` | （無說明，臨時分析腳本） |
| `ir2eo_vis.py` | （無說明，臨時分析腳本） |
| `make_submission.py` | Test submission: module Mask R-CNN instances + zero-shot within-frame anomaly labels. For each test image (IR and EO scored separa |
| `memo.py` | （無說明，臨時分析腳本） |
| `module_clf_eo.py` | EO (RGB) module-crop crack classifier: frozen DINOv2 features + linear probe, group split. Every EO polygon (solar_panel or crack) |
| `normal_cache.py` | Train IR frames of hotspot-free dense groups (+ all G02): modules from the all-data Mask R-CNN, zero-shot feats, GT. |
| `pair.py` | （無說明，臨時分析腳本） |
| `pair2.py` | （無說明，臨時分析腳本） |
| `pipeline_eval.py` | Full module pipeline on held-out groups: Mask R-CNN module instances + zero-shot anomaly labels. Detections come from the other se |
| `pvhsd_eval.py` | Module-level hotspot selection on PV-HSD (unseen dense sites). GT = modules containing a box centre. |
| `raw_eval.py` | Recompute raw per-module mutual score and grey std on cached predicted modules, then test robust z-scores with a scale floor (scal |
| `rectify.py` | Perspective-rectify a module polygon to an upright size x size crop. Axis-aligned bbox crops of tilted modules carry black corners |
| `removed_sheet.py` | （無說明，臨時分析腳本） |
| `site_match.py` | Assign each test frame to its nearest train group by DINOv2 CLS embedding (cosine kNN, k=5), per modality. |
| `test_cache.py` | Cache per-module test predictions + raw zero-shot features so submissions can be rebuilt cheaply. Per test image: modules from Mas |
| `true_val_tiou.py` | Official F1_TIoU of a DINO-seg IR checkpoint against the ORIGINAL per-module polygons. The training-time validation rasterizes GT  |
| `zs_score.py` | Reusable zero-shot within-frame module scoring (rectified DINOv2 mutual score + intensity std). score_frame(image_rgb, polygons, k |
