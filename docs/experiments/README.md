# 太陽能板 IR 模組實驗紀錄

記錄日期：2026-10-22。範圍為本專案的 U-Net／DNA-Net 初步訓練、PV-HSD 預訓練、競賽微調、切分比較、凍結對照、Test 提交與診斷。另一個 `How-to-overcome-domain-shift` 專案不納入本紀錄。

本次查核共找到 **14 個 run 目錄：11 個有訓練證據、3 個只有初始化設定**。記錄時本專案沒有訓練程序在執行。完整可追溯數值與設定見 [JSON](results_20261022.json)，適合試算表使用的摘要見 [CSV](results_20261022.csv)。原始 checkpoint、日誌和提交檔保留原位，不搬動、不覆寫。

> **最新進度（2026-11-23）：** 模組級零樣本路線的完整實驗、平台分數、負面結果與省思，見[實驗總整理](module_zero_shot_20261123.md)。所有提交檔的 md5 與分數見[提交帳本](submission_ledger_20261123.csv)。

## 研究目的與任務定義

先完成紅外線異常區域預測，再轉成競賽需要的 polygon。競賽三類為正常模組 `solar_panel`、IR 熱斑 `hotspot`、EO/RGB 裂紋 `crack`；本階段只訓練 hotspot。異常標註涵蓋整片異常模組，不只是局部亮點或裂紋。RGB 與根因文字尚未訓練，四通道／雙分支融合也只有討論，沒有實驗結果。

## 資料與驗證方法

競賽 Train 有 1046 張 IR，428 張有異常。主要正樣本 group 為 G01/G02/G03/G04；G02 的異常面積較小，G04 較大，先前統計面積占比中位數約差 24 倍。依官方補充說明，同 group 來自同次連拍，畫面高度相似，建議以 group 切分。細節見 [group 分析](../solar_group_analysis.md)。

| 驗證方式 | train／validation | 用途與限制 |
|---|---:|---|
| 後期固定 group：G02/G05/G08 作 validation | 831／215 | 跨場景與小目標泛化；與訓練分布落差大 |
| `photo_stratified`：依 group×正負樣本分層抽照片 | 838／208 | 同場景開發；照片 ID 不重疊，實體板子與近似影像尚未確認互斥 |
| 初期 `random_stratified` | 837／209 | 依正負照片抽樣，同樣不能視為跨 group 表現 |

後期照片切分中，G02 正樣本分為 train 152／validation 38；驗證不是完全沒有小瑕疵。所有 split 的完整 frame 清單保存在 checkpoint metadata 或切分查核檔中，不以模型高分反推其為獨立場景。

外部 PV-HSD-2025 原始 4025 張影像，其中兩張可見光影像排除但不刪原檔；處理後 4023 張 640×512、單通道灰階 PNG，YOLO bbox 聯集轉成二值弱 mask。依檔名前綴家族切分 train 3284／validation 739，這不是經確認的物理場站分組。原框面積中位數約 77 像素，因此不能將失敗直接歸因於「預訓練瑕疵都太大」。弱 mask 是局部熱點框，不等於競賽整片模組 polygon。來源、授權與資料使用紀錄見 [PV-HSD 紀錄](../datasets/PV-HSD-2025.md)。

## 模型、Loss 與指標

初期包含 Simple U-Net、DNA-Net ResNet-18/channel_size=two；後期使用 DNA-Net ResNet-18/channel_size=three、單通道輸入、四個 deep-supervision heads。不同階段並非完全相同的受控比較。

後期預訓練／微調共用 batch-global SoftIoU：`1 - (intersection + 1) / (union + 1)`，先 sigmoid，對整個 batch 聚合，四個輸出 Loss 等權平均。預訓練以 `weak_mask_iou` 選模，微調以 IR-only `F1_TIoU` 選模；本機只計 hotspot，平台則還包含 solar_panel、crack，兩者不能直接比總分。

後期設定為 640×512、batch 4、Adagrad、seed 7、AMP 關閉；預訓練 LR=0.05，微調 LR=0.005。`scheduler: upstream_once` 依作者公開程式只在初始化調整一次，不是每 epoch cosine decay。新 cuDNN 設定開啟 benchmark；既有長預訓練未被中途改設定。版本與 backend 見各 run 的 runtime_backend.json／metadata。

**本文件 epoch 一律以 1 起算顯示；JSON 同時保留原始 0 起算值。** 例如預訓練 best_epoch_raw=370，在此記為第 371 epoch；先前對話／部分文件直接寫「370」是沿用原始欄位。`epochs_recorded` 為完成驗證的輪數。完成訓練後 latest.pt 可能保存已回復的最佳 epoch，不能只靠 latest.pt 的 epoch 推算跑過幾輪。

## 初期：未使用外部預訓練

| ID | 模型／設定 | 切分 | 完整輪數／最佳輪 | 最佳 F1_TIoU | 同輪 pixel IoU | 狀態 |
|---|---|---|---:|---:|---:|---|
| E01 | Simple U-Net，256，batch 8，LR 1e-4 | 未顯式保存 split；831/215 | 32／7 | 0.000138739 | 0.004472667 | 完成，early stop |
| E02 | Simple U-Net v2，同上 | 未顯式保存 split；831/215 | 21／1 | 0.000086154 | 0.004751854 | 中斷，無完成報告 |
| E03 | DNA-Net c2 v1，256，batch 8，LR 1e-4 | random_stratified | 38／38 | 0.878357788 | 0.944325265 | 中斷，checkpoint step 4000 |
| E04 | DNA-Net c2 v6，256，batch 8，Dice，LR 1e-4 | group | 32／7 | 0.000749085 | 0.003064155 | 完成，early stop |
| E05 | DNA-Net c2 v7，實際 256，batch 8，Dice | group | 25／18 | 0.000579618 | 0.009094206 | 中斷，checkpoint step 2700 |
| E06 | DNA-Net c2 v8，640×512，batch 2，Dice | group | 30／5 | 0.007966162 | 0.010492299 | 完成，early stop |

初期部分 run 未顯式保存 optimizer、Loss 或 metric aggregation，不能依目前程式的預設值補成歷史事實。不同版本的 pixel IoU 也不保證採相同聚合方式，因此上表不應當成單一參數的消融結論。

E05 有明確設定漂移：現有 resolved_config.json 是 640×512、batch 2，但 latest.pt 內保存的執行設定是 **256、batch 8**，本紀錄採 checkpoint 為準，差異完整保留於 JSON。不能把 E05 的數值當作高解析度結果。

另外三個目錄 `20261019_2305_dnanet-r18-c2-seed7-{v2,v3,vˇ}_solar-ir-dnanet-from-scratch` 只有設定與資料 manifest，沒有 checkpoint／history，記為「只有初始化，無訓練結果」，不是三個零分實驗。使用者另表示做過低 LR；目前未找到可獨立識別的 PV-HSD 低 LR 對照目錄，因此不虛構其數值。

## 外部預訓練與後續微調

| ID | 實驗 | 完整輪數／最佳輪 | 選模指標／最佳值 | 同輪 IoU／Dice | 狀態 |
|---|---|---:|---|---|---|
| P01 | PV-HSD 單通道預訓練 | 372／371 | weak-mask IoU **0.599136476** | Dice **0.749325008** | 依使用者要求停止；step 306200，可續訓 |
| F01 | PV-HSD → 全微調，group | 17／1 | F1_TIoU **0** | IoU 0；Dice 約 4.89e-12 | 中斷，無完成報告 |
| F02 | PV-HSD → 全微調，photo_stratified | 111／61 | F1_TIoU **0.749772498** | **0.938720711／0.968391895** | 完成，50 輪無改善後停止 |
| F03 | PV-HSD → 凍結早期三 stage，group | 53／3 | F1_TIoU **0.000039938** | 0.003573628／0.007121805 | 完成，early stop；後期驗證 IoU=0 |
| F04 | PV-HSD → 全微調，group，重新對照 | 8／1 | F1_TIoU **0** | IoU 0；Dice 約 4.89e-12 | 中斷，step 1700，未完成第 9 輪 |

P01 最大設定為 1500 輪，並未跑滿；第 373 輪中途停止。它的 weak-mask 分數只代表與外部 bbox mask 的一致性，不是競賽分數。F04 與 F01 是同樣訓練政策的重複對照，不是新的方法。

F03 固定 conv0_0、conv1_0、conv2_0 的參數與 BN 統計；169690 個參數凍結、4526878 個可訓練，並非凍結整個 encoder。執行後查核 106 個 frozen state 張量與預訓練來源完全相同。F03 的 10971 筆 iteration loss、F04 在當時查核的 1449 筆 loss 均無 NaN／Inf；梯度範數未記錄，不能宣稱已完整檢查梯度。這支持「凍結確實執行、但這個政策未改善驗證」，不等於所有凍結策略皆無效。

### Run 對照

所有目錄均位於 `outputs/`：

| ID | run 名稱 |
|---|---|
| E01 | 20261019_2330_simple-unet-baseline_solar-ir-polygon-mask-dnanet |
| E02 | 20261019_2330_simple-unet-baseline-v2_solar-ir-polygon-mask-dnanet |
| E03 | 20261019_2305_dnanet-r18-c2-seed7-v1_solar-ir-dnanet-from-scratch |
| E04 | 20261019_2305_dnanet-r18-c2-seed7-v6_solar-ir-dnanet-from-scratch |
| E05 | 20261019_2305_dnanet-r18-c2-seed7-v7_solar-ir-dnanet-from-scratch |
| E06 | 20261019_2305_dnanet-r18-c2-seed7-v8_solar-ir-dnanet-from-scratch |
| P01 | pvhsd-dnanet-1ch-pretrain |
| F01 | solar-ir-dnanet-1ch-pvhsd-finetune-cudnn |
| F02 | solar-ir-dnanet-1ch-pvhsd-finetune-cudnn-photo |
| F03 | solar-ir-dnanet-pvhsd-group-freeze_early |
| F04 | solar-ir-dnanet-pvhsd-group-full |

## 診斷一：預訓練直接測競賽與微調後零分

完整 G02/G05/G08 validation 215 張，微調來源為 F01 固定 step 1035 快照，不是最終 checkpoint。

| 項目 | P01 直接測競賽 | F01 微調快照 |
|---|---:|---:|
| hotspot F1_TIoU | 0.001336068 | 0 |
| pixel IoU | 0.035863903 | 0 |
| 預測前景像素 | 12921 | 0 |
| TP／交集 | 7525 | 0 |
| FP | 5396 | 0 |
| FN | 196900 | 204425 |
| IoU 聯集 | 209821 | 204425 |

微調零分的分式為 **0／204425**，不是因分母太大而四捨五入成零。最高前景機率只有約 0.0000764，threshold 降至 0.05 仍無前景。當時抽查 8 張訓練正樣本 IoU 約 0.8835，僅代表抽查，不是完整 train 分數。[完整診斷](../training/transfer_validation_20261022.md)。

## 診斷二：照片切分高分由哪些 group 構成

F02 最佳權重重算，與保存分數有微小數值差異：

| group | validation 張數／正樣本 | pixel IoU | hotspot TIoU |
|---|---:|---:|---:|
| G01 | 57／1 | 0 | 0 |
| G02 | 39／38 | 0.9366 | 0.7924 |
| G03 | 4／3 | 0.8827 | 0.1217 |
| G04 | 61／43 | 0.9593 | 0.8113 |

G02 小目標在同場景驗證可辨識，不能將失敗概括成「所有小瑕疵都不會」。G01 仍失敗；G03 的高 pixel IoU 沒有轉成高 polygon TIoU，也提示模組級輪廓／實例分離問題。各 group 陽性數量不同，不視為同樣可靠的估計。

## Test 提交與平台分數

提交使用 F02 最佳權重，threshold=0.5，只輸出 IR hotspot；全部 reason=null、EO=[]，不預測 solar_panel 或 crack。696 IR＋696 EO、完整 1392 個 key，官方格式檢查通過。IR 74 張有有效 polygon，合計 141 個 hotspot。

使用者回報提交時間 **2026-10-22 22:00:03**、介面識別字串 `a41425664`、名次顯示 114/114。以下為使用者提供的顯示值，不是本機自行計算的 Test GT 分數；Public/Private subset 未提供：

| 平台欄位 | 分數 |
|---|---:|
| score | **0.0010203** |
| F1_TEXT | 0 |
| F1_TIoU | 0.00145 |
| TIoU_hotspot | **0.00437** |
| TIoU_crack | 0 |
| TIoU_solar_panel | 0 |

未作答其他兩類及文字會限制總分，但 hotspot 本身也很低，因此不能只歸因於 RGB／文字空白。格式正確不代表定位正確；平台與本機照片驗證不具相同分布。

原始提交：`outputs/submissions/dnanet-ir-only-photo-20261022/submission.json`，SHA-256 `2d579ffbbc5fc760dd2a0433c0e3a598094349a927ea5e7b3085338239787cfa`。整包原產物 701 個檔案已備份 Drive，並下載提交檔核對 SHA-256：[雲端 submission](https://drive.google.com/file/d/1gRV04ifgt8XlL3YHj0BA1lk26aQ0m3y8/view)。備份 receipt 的 uploaded_to_competition=false 是備份當時狀態；之後使用者已自行提交，以上述平台回報為準。

## Test 輸出分析

| 來源外觀 | Test 張數 | 有有效 polygon 的張數 | polygon 數 |
|---|---:|---:|---:|
| 灰階 | 496 | 19 | 27 |
| 偽彩 | 200 | 55 | 114 |

外觀分類是 RGB 通道差的啟發式，不是官方 group／正負樣本標籤。141 個 polygon 面積中位數 **51.5 像素²**，86 個小於 100；抽圖可見不少邊角與局部碎片，沒有穩定包住整個模組。不能在沒有 Test GT 時直接把所有小輪廓判為誤報。

同權重重算時，**615／696 張的最大前景機率小於 0.001**。threshold 從 0.5 降至 0.001，有前景像素的影像只從 76 增至 81；76 是「至少一個像素」，74 是原提交「正面積 polygon」，兩者定義不同。重算與原 mask 共差 11 個像素，未發現足以解釋低分的大規模座標或前處理錯誤。

P01 預訓練直接測 Test 有 291 張出現前景，F02 微調後為 76 張，反應明顯減少；這不是預訓練比較準確的證據，也不足以因果證明 catastrophic forgetting。灰階 Train 與偽彩 Test、不同熱異常形態、位置與板子排列，以及局部熱點／整片模組的監督差異，皆是待分離的因素。[完整分析與圖例](../training/test_output_analysis_20261022.md)。

## 功能驗證與尚未執行項目

CPU 小資料預訓練→競賽微調、單通道權重 strict load、640×512 四個輸出 shape、資料雜湊、cuDNN GPU forward/backward、照片切分無 frame 重疊，以及 frozen 參數／BN 不更新均已驗證。這些結果不列入正式效能表；33 項測試是在加入凍結功能時通過，並非每個歷史實驗各自跑過 33 項測試。

SIRST／SIRST-V2 尚未下載或訓練；RGB、正常面板類別、根因文字、四通道輸入、雙分支融合與影像配準尚無模型結果。EO／IR 同名配對不保證像素對齊，四通道不可僅 resize 後假設對齊。

目前最符合證據的解讀是：同場景照片驗證偏樂觀，模型對新場景出現大量背景判斷與模組輪廓失敗；尚不能只歸因於 batch size、目標尺寸或單一資料不足。後續應以固定 group／leave-one-group-out 驗證，分別測外觀增強、不同外部預訓練資料與熱點到模組輪廓的銜接。

## 更新紀錄與證據來源

本次新增紀錄沒有啟動／停止訓練，沒有改參數、權重或原始 submission。新總表與 Test 分析尚未同步 Drive；已確認備份的是先前 submission 原產物。

重新匯出本機保存結果：

```sh
conda activate solar-ir
python scripts/export_experiment_records.py \
  --destination docs/experiments/results_20261022
```

腳本只讀現有 run；若 checkpoint 與 resolved config 不同，以 checkpoint 記錄設定並保留差異。缺少完成報告不推定為訓練成功；缺少 Loss／optimizer 等歷史欄位不依現行程式補值。JSON 同時保存來源路徑、best checkpoint 與 metrics 雜湊、診斷結果及平台回報。若更新後續日期，請使用新檔名保留快照。
