# PatchCore 基線實驗紀錄（2026-10-27）

實作：`scripts/patchcore_eval.py`。輸出：`outputs/patchcore/`（每個 run 的 `metrics.json` 含逐張診斷與閾值掃描）。

## 方法

- 骨幹：ImageNet WideResNet-50-2（與 PatchCore 原文相同），取 layer2 + layer3，3×3 局部平均後串接為 1536 維。
- 輸入：IR 灰階、原生 640×512，不縮放（小面板 G02 需要原生解析度）。
- Memory bank：只用**無 hotspot 的訓練影格**。每張影格隨機保留 25% patch，再以 random projection 的 greedy coreset 取 10%。
- 分數：每個 patch 到 bank 的 L2 最近鄰距離 → 上採樣到 512×640 → Gaussian blur（σ=4）。影像分數為最大值。
- 閾值校準：從訓練正常影格中留出 20%（不進 bank），取其影像最大分數的第 99 百分位。**不使用驗證標註**。
- 評分：與 DNA-Net 實驗相同的官方 `F1_TIoU`（只計 hotspot），GT 由 Labelme 多邊形光柵化後取輪廓。
- 切分：與 DNA-Net 相同。Fold 使用 `configs/external/finetune_fold_manifest.json`；照片切分為 `photo_stratified`、seed 7（838/208，與 F02 一致）。

## 結果

| 切分 | 驗證張數／陽性 | 影像 AUROC | 校準閾值 F1_TIoU | oracle 閾值 F1_TIoU | oracle 閾值 pixel IoU |
|---|---:|---:|---:|---:|---:|
| fold-G01 | 350／7 | 0.866 | 0.0000 | 0.0027 | 0.0033 |
| fold-G02 | 269／190 | 0.910 | 0.0003 | 0.0062 | 0.0056 |
| fold-G03 | 74／16 | 0.666 | 0.0000 | 0.0287 | 0.0211 |
| fold-G04 | 353／215 | 0.580 | 0.0000 | 0.0589 | 0.1968 |
| photo | 208／85 | 0.894 | 0.0000 | 0.0390 | 0.1560 |
| photo（隨機 bank，消融） | 208／85 | 0.886 | 0.0000 | 0.0321 | 0.1402 |

DNA-Net 對照（同樣的切分；來源為各 run `metrics.json` 的 `best_metric`，即以驗證集選出的最佳 epoch，偏樂觀）：

| 切分 | DNA-Net（PV-HSD 預訓練＋微調）F1_TIoU |
|---|---:|
| fold-G01 | 0.0065 |
| fold-G02 | 0.0000 |
| fold-G03 | 0.0549 |
| fold-G04 | 0.0000 |
| photo（F02） | 0.7498 |

注意：DNA-Net 數字是驗證集上選最佳 epoch 的結果，PatchCore 的 oracle 數字也是同樣方式選出。校準閾值那一欄才是不使用驗證標註的結果。

## 診斷（photo 切分，隨機 bank 版本，逐張資料見 `outputs/patchcore/photo-random/metrics.json`）

- 有瑕疵影像的最大分數中位數 2.45，正常影像為 2.03。影像層級能分開，但重疊明顯。
- 有瑕疵影像在 oracle 閾值下，預測面積中位數 8.7%，GT 面積中位數 7.1%，但 pixel IoU 中位數只有 0.10。預測位置與多邊形 GT 對不上。
- 校準閾值（約 2.8–2.9）比 oracle 閾值（約 1.5–1.9）高很多，因為留出的訓練正常影格分數比驗證正常影格高。在這個閾值下，幾乎所有影像都沒有預測，F1 為 0。
- coreset 與隨機 bank 結果接近（AUROC 0.894 vs 0.886），bank 取樣方式不是瓶頸。

## 結論

- 距離式方法確實能提供有用的影像層級訊號（fold-G02 AUROC 0.91、照片切分 0.89），但在這個切分下無法轉成合格的多邊形定位。
- 無論用哪個切分，校準閾值下的 F1_TIoU 都是 0。即使用 oracle 閾值，最高也只有 0.059（G04），遠低於 DNA-Net 在照片切分上的 0.750。
- 跨 group 的 fold 上，DNA-Net 在 G02/G04 的最佳分數為 0，PatchCore 的 oracle 分數為 0.006／0.059，略高但仍極低。在 G01 與 G03，PatchCore oracle（0.003／0.029）與 DNA-Net（0.007／0.055）都很低，DNA-Net 略高。
- 這個 benchmark 的標註是整片模組的多邊形，而距離式方法產生的是局部高分區。這個不一致是我們觀察到的主要失敗模式，但尚未用視覺化直接驗證。

## 限制

- 只評估訓練分布內的驗證集。測試集有 200 張偽彩圖（見 `docs/experiments/README.md`），PatchCore 對色彩偏移很敏感，這個風險沒有在這裡測試。
- 只用 1 個 seed。G03 只有 16 個陽性，G01 只有 7 個，其數字不穩定。
- 沒有做面板層級的聚合，也沒有做逐 patch 的正規化或其他校準改進。
