# DNA-Net 設定、原始資料與競賽切分查核（2026-10-20）

本次以[作者官方 repository](https://github.com/YeRen123455/Infrared-Small-Target-Detection)
的 commit `21584be00d19bc3080c78deee0953af250bed99b` 為重現基準。
原始論文為 [Dense Nested Attention Network for Infrared Small Target Detection](https://arxiv.org/abs/2106.00487)。
這是模型與訓練設定的對齊，並非已在競賽資料重現論文分數。

## 原始資料與預訓練

- **NUDT-SIRST**：作者提出的合成紅外小目標資料，1327 張、256×256，有 pixel mask。
  [作者公布的百度下載](https://pan.baidu.com/s/1WdA_yOHDnIiyj4C9SbW_Kg?pwd=nudt)，提取碼 `nudt`。
  官方 repo 只附資料範例及 `50_50/train.txt`、`test.txt`，不是完整資料。
  本次確認作者仍列出此連結，但百度頁面無法由工具取得，**尚未確認完整壓縮檔能成功下載，亦未下載完整 NUDT-SIRST**。
- **SIRST（作者程式稱 NUAA-SIRST）**：[資料原作者 GitHub](https://github.com/YimianDai/sirst)。
  是另一個實拍紅外資料來源，請保留原始資料版本及 DNA-Net 的 split 清單；SIRST v2 是另一版本。
- **NUST-SIRST**：[官方所指向的 MDvsFA_cGAN](https://github.com/wanghuanphd/MDvsFA_cGAN)。
- **不一定要重新預訓練**：官方直接提供
  [NUDT 權重](https://drive.google.com/file/d/1NDvjOiWecfWNPaO12KeIgiJMTKSFS6wj/view?usp=sharing)、
  [SIRST 權重](https://drive.google.com/file/d/1W0jFN9ZlaIdGFemYKi34tmJfGxjUGCRc/view?usp=sharing)。
  本次未驗證這兩個 Drive 檔案能成功下載。
  官方 GitHub 另附 `pretrain_DNANet_model.tar`，本次已成功取得並驗證能嚴格載入目前模型；
  該檔案內未記載訓練資料來源，不能直接當作已確認來源的 NUDT 權重。

技術上可用上述紅外分割資料預訓練後再微調 solar IR。
但原始資料的標註是小目標，競賽補充文件明訂異常標註涵蓋**整個太陽能模組**，不是只標亮點。
背景、目標形狀、尺寸、黑底去識別化也不同。因此預訓練是可驗證的初始化假設，不能視為一定改善。
建議在相同 group folds 比較 scratch、NUDT 初始化、SIRST 初始化，其他設定固定。

本次找到競賽官方頁面，但未取得足以確認「外部資料／外部權重使用條件」的條文，
所以這裡只確認技術可行性，沒有斷言競賽允許。正式使用前須核對完整規則及公告。

## 已對齊的設定

設定檔：`configs/solar_ir_dnanet.yaml`。原先的 checkpoint 不可使用此新設定續訓，請開新 run。

| 項目 | 此前設定 | 對齊作者程式後 |
|---|---|---|
| 模型 | ResNet-18、channel two | ResNet-18、channel three `[16,32,64,128,256]` |
| 輸入 | 灰階 1 channel、0–1 | PIL RGB 3 channels、ImageNet mean/std，仍只讀 IR 資料夾 |
| 尺寸 | 640×512 | base/crop 256×256 |
| 初始化 | PyTorch 預設 | Conv2d Xavier normal，bias 維持預設 |
| Loss | Dice | 官方 batch-global SoftIoU，smooth=1 |
| Deep supervision | 四個輸出 | 四個輸出的 SoftIoU 等權平均 |
| Optimizer | AdamW、1e-4 | Adagrad、0.05，weight decay=0 |
| Epoch / batch | 150 / 2 | 1500 / 4（parser 預設；README 範例使用 16） |
| AMP / gradient clip | 開 / 1 | 關 / 無 |
| Drop last | 否 | train 是、validation 否 |
| Early stop | patience 25 | patience 1500，相當於完整跑完 |
| Checkpoint selection | F1_TIoU | 全 validation 累積 intersection/union 的 mIoU |

仍會回報競賽 F1_TIoU。競賽資料、Labelme polygon→mask 與 group split 是本任務的必要適配；
不能使用 NUDT 的 50/50 split 取代競賽 group 規則。
原始水平翻轉、0.5–2 倍縮放、補黑、隨機裁切、Gaussian blur 均保留。
256 是官方設定，會降低本競賽小物件的像素數；若之後改回較高解析度，請另立實驗並標示此差異。

**Scheduler 的原始碼問題**：`parse_args_train.py` 宣告 CosineAnnealingLR，
但 `train.py` 僅在 Trainer 初始化時呼叫一次 `scheduler.step()`，epoch loop 完全沒有 step。
本設定明寫 `scheduler: upstream_once`，重現此實際行為：
`lr = eta_min + (0.05 - eta_min) * (1 + cos(pi / 1500)) / 2`，之後固定。
沒有默默改成每 epoch cosine，也沒有宣稱原始碼已實際完成 cosine decay。
若要按論文意圖實作逐 epoch cosine，需另外標示為修正版實驗。

SoftIoU 是 `1 - (sum(sigmoid(logits)*mask)+1) / (sum(sigmoid(logits))+sum(mask)-intersection+1)`。
所有 sum 包含 batch 維度，不能替換成逐張 IoU 後平均。

## 官方權重使用方式

取得有確認來源的官方 checkpoint 後，用 PyTorch >=2.6、NumPy >=2 的環境轉為 tensor-only 權重：

```sh
python scripts/convert_dnanet_checkpoint.py path/to/official.tar data/pretrained/dnanet.pt
```

然後在 YAML 的 `task.model` 加 `pretrained_path: data/pretrained/dnanet.pt`，用新 run 微調。
載入採 strict=True，不會靜默忽略缺少的 layer。
一般本專案 `best.pt` 或 tensor-only state_dict 可直接載入，不需上述轉換。
目前成功驗證的 demo 轉換結果在 `outputs/dnanet_audit/upstream_demo_weights.pt`；
它只用於相容性檢查，尚未作為競賽訓練初始化。

## 官方對 group 的要求

本機官方補充文件 `data/raw/supplement/SolarDefectDataset_supplement/README_資料集說明.md`
「分組資訊」段原文：

> group 相同的影格來自同一次連續拍攝，畫面高度相似。自行切分驗證集時，請以 group 為單位，否則近乎重複的影像會讓驗證分數偏高。

**所以是明確請以 group 切分，不只是相同照片不得重複。** 這段是官方建議切分方式，
未提供「不照做即取消資格」的證據。
同名 EO/IR 是同時刻配對，未來多模態訓練也應沿用相同的 frame/group 分割。

## 尺寸 shift 的實測原因

以原始 polygon 的 shoelace 面積除以原圖面積計算，每個異常物件計一次，尚未做 augmentation：

| Split | 影像數 | 有異常影像 | 異常物件 | 物件面積比例中位數 |
|---|---:|---:|---:|---:|
| train | 831 | 238（28.64%） | 254 | 9.1112% |
| validation | 215 | 190（88.37%） | 190 | 0.3806% |

驗證 group 是 **G02/G05/G08**，後兩組沒有異常，因此驗證異常全部來自 G02。
訓練異常來自 G01/G03/G04，其面積中位數分別約 **8.49% / 2.88% / 9.50%**。
不只是尺寸差約 **23.94 倍**，陽性影像比例也嚴重不一致。

目前 `group` splitter 先挑影像數最接近 validation 目標的陽性 group，因此總是選 G02（197 張），
再補正常 group；此路徑**沒有使用 seed**，換 seed 不會換 group。
這不是 augmentation 造成的假象；隨機縮放裁切可能進一步改變模型看到的尺寸，但原始分割本身已有落差。

資料只有四個有異常的 group，而且小尺寸物件集中在 G02；因此不應承諾單一 80/20 group split
能同時讓大小物件均衡。建議逐 group 驗證，至少分別 hold out G01、G02、G03、G04，
將正常 group 分配至各 fold，回報各 fold 的分數、尺寸與陽性比例，以及整體 out-of-fold 結果。
G02 fold 可保留為「小尺度泛化」的壓力測試。
選 fold 應根據固定資料統計事先決定，不要看模型分數後挑最好的一組。

已新增 `task.validation_groups: [G02, G05, G08]` 的明確指定功能；仍要求 `split_strategy: group`，
可用它建立固定 folds。此輪**未偷偷改動原有切分**，也未拆開任何 group。
不要把 `random_stratified` 當成對外泛化表現，它仍可能把同次拍攝的近似影像放在兩邊。

重跑統計：

```sh
python scripts/audit_solar_split.py --config configs/solar_ir_dnanet.yaml --output outputs/split_audit.json
```

本次完整統計：`outputs/dnanet_audit/original_split.json`。
模型已與作者原始程式載入同一份參數比較，四個 forward 輸出最大誤差皆為 0；
官方 demo checkpoint 也通過 strict 載入。尚未進行完整 1500 epoch 訓練或證明預訓練增益。

檢查結果：`pytest -q` 為 **11 passed**；本次修改的 Python 檔案 Ruff 通過，
`git diff --check` 通過；`MYPYPATH=src mypy src/research_training` 通過。
全 `src` 的 strict mypy 仍有 6 個模型 `nn.Module` 回傳 Any 的型別錯誤
（DNANet 4 個、既有 simple_unet 2 個），因此沒有宣稱整個專案型別檢查通過。
