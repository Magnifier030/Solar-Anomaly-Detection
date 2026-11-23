# 單通道 DNA-Net：PV-HSD 預訓練 → 競賽微調

此流程已處理完成並通過 CPU 冒煙測試。正式 PV-HSD 預訓練已由使用者啟動；競賽微調可於預訓練完成後進行。

2026-10-22：依使用者要求以 SIGINT 停止正式預訓練，保留可續訓的 `latest.pt`（epoch 372、step 306200）及微調來源 `best.pt`（epoch 370、weak-mask IoU 0.599136）。停止紀錄位於 `outputs/pvhsd-dnanet-1ch-pretrain/analysis/pretraining_stop.json`。微調建議先採全模型更新，使用現有 cuDNN 微調設定（Adagrad 0.005、batch 4、patience 50、F1_TIoU 選模）；尚未啟動微調。理由是來源為局部熱點框弱標註，競賽則為整片異常模組 polygon，且影像域不同，需要允許各層調適。這是待競賽驗證的實驗選擇，未宣稱優於凍結部分層。

## 檔案結構

```text
configs/pvhsd2025/
  pretrain_dnanet_1ch.yaml          # 外部弱標註預訓練
  finetune_solar_dnanet_1ch.yaml    # 競賽人工 polygon 微調
scripts/pvhsd2025/
  download.py                     # 固定版本下載與雜湊驗證
  audit.py                        # 原始影像、YOLO 框與重複檔查核
src/solar_ir/
  preprocessing/pvhsd2025.py       # 單通道轉換、框 mask、分組切分
  datasets/masks.py               # PNG 配對 loader，共用競賽同步 augmentation
  pretraining.py                  # 預訓練 task，重用 DNA-Net SoftIoU
  task.py                         # 競賽 polygon task
  optimization.py                 # 兩階段共用 optimizer policy
  models/dnanet.py                 # 兩階段共用 DNA-Net

data/raw/external/PV-HSD-2025/      # 不修改官方原檔
  PV-HSD-2025.zip
  extracted/
  source_metadata/
  download_receipt.json
  audit_summary.json
  ...
data/processed/pv_hsd2025/grayscale_box_masks_v1/
  images/*.png                    # 4,023 張 640×512、L mode 單通道圖
  masks/*.png                     # 4,023 張二值 bbox union 弱 mask
  manifest.json                   # 檔案雜湊、來源、切分、弱標註聲明

docs/datasets/PV-HSD-2025.{md,json} # 來源、授權、處理與使用事件
outputs/
  pvhsd-dnanet-1ch-pretrain/        # 正式預訓練啟動後產生
  solar-ir-dnanet-1ch-pvhsd-finetune/ # 正式微調啟動後產生
  verification/pvhsd2025_1ch/      # 小型實際資料測試，不是正式預訓練權重
```

原始來源與處理後影像皆不進 Git；設定、程式與來源紀錄可進版本控制。不要把驗證測試的 checkpoint 複製成正式預訓練結果。

## 處理內容與標註意義

- 移除兩張已人工確認的可見光照片，原始檔仍保留；4,023 張候選 IR 均轉成 `L` 模式 lossless PNG，原生尺寸 640×512。載入後 shape 為 `[B,1,512,640]`，像素除以 255，與競賽灰階 loader 一致。
- 單通道轉換是**偽彩熱影像的灰階亮度**，不是從 R-JPEG 恢復校正後的溫度。來源 metadata 未被改寫；新的 PNG 是明確的訓練衍生資料。
- YOLO 框以半開像素範圍 rasterize，取所有框的聯集，背景 0／框內 255。空標註產生全零 mask。**這是 bbox 弱標註，不是人工面板輪廓，也未使用 SAM。**
- 本流程先預訓練「局部熱點分割」；後續競賽用真正的 Labelme 整片異常模組 polygon 微調整個網路。沒有宣稱兩種標註語意完全相同，亦沒有保證預訓練一定改善分數。
- 預訓練指標叫 `weak_mask_iou`／`weak_mask_dice`，只代表與框 mask 的一致性，不會冒用競賽 `F1_TIoU`。

| 新的外部 split | 圖像 | 有框影像 | 框數 | 檔名家族數 |
|---|---:|---:|---:|---:|
| train | 3,284 | 3,107 | 20,685 | 1,009 |
| validation | 739 | 691 | 4,492 | 224 |

取檔名最後一個底線前的前綴作為家族，以 seed 7 的 SHA-256 將整個家族固定分配，家族不跨 split。這只是避免已知檔名前綴跨切分的啟發式，不是經核實的場站／連拍 group；近似影像仍可能跨不同前綴。原始 split 也保存在每個 manifest sample 內。

## 訓練設定

兩階段採相同架構：DNA-Net ResNet-18、channel_size=three、四個 deep supervision heads、單通道輸入與單類別輸出。Conv2d 採作者 Xavier normal 初始化，Loss 為 batch-global SoftIoU（smooth=1），四個輸出等權平均。

| 設定 | PV-HSD 預訓練 | 競賽微調 |
|---|---|---|
| 輸入／大小 | 單通道／640×512 | 單通道／640×512 |
| 監督 | 熱點 bbox 弱 mask | 人工整片異常模組 polygon |
| 初始化 | Xavier scratch | 嚴格載入第一階段 best.pt |
| Optimizer | Adagrad | 新建 Adagrad，不繼承第一階段 optimizer |
| Learning rate | 0.05 | 0.005 |
| Batch / 最大 epochs | 4 / 1500 | 4 / 1500 |
| Early stopping patience | 1500 | 50 |
| Checkpoint selection | weak_mask_iou | F1_TIoU |
| AMP／gradient clipping | 關／無 | 關／無 |

單通道及 640×512 是本次明確調整，與作者預設 RGB／256×256 不同。保留作者原始程式實際的 `scheduler: upstream_once` 行為：初始化調整一次後固定 LR，不宣稱每 epoch cosine decay。微調採較低 LR 與競賽指標選模，是本任務的微調設定。

競賽 validation 固定 G02/G05/G08，仍 group-disjoint。先前指出的 G02 尺度／場景落差依然存在；預訓練是否改善要靠這個驗證與後續其他 group folds 實測，不因新增資料便視為已解決。

## 開始訓練

所有指令從 repository 根目錄執行，使用具 CUDA 版 PyTorch 的環境。可先安裝專案需求並檢查 GPU：

```sh
python -m pip install -e ".[torch,eval,wandb]"
python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
export TRAINING_OUTPUT_ROOT=outputs
```

現有資料已處理好；若在其他主機準備資料，先依資料紀錄下載、查核，再執行：

```sh
python -m solar_ir.preprocessing.pvhsd2025
```

處理器對相同 signature 為冪等，會沿用已完成的 manifest；不同版本／參數會要求使用新版本目錄，不覆蓋舊實驗資料。

第一階段：

```sh
CUDA_VISIBLE_DEVICES=0 research-train run \
  --config configs/pvhsd2025/pretrain_dnanet_1ch.yaml \
  --device cuda --no-drive
```

1500 epochs 是最大設定，這個正式長訓練不會由本輪自動啟動。若要縮短，請在首次啟動前調整 YAML 的 `max_epochs` 和 `patience`，並保留修改後設定；不要在同一個 run 中途改設定續訓。

第一階段完成後，checkpoint 為 `outputs/pvhsd-dnanet-1ch-pretrain/best.pt`。第二階段：

```sh
CUDA_VISIBLE_DEVICES=0 research-train run \
  --config configs/pvhsd2025/finetune_solar_dnanet_1ch.yaml \
  --device cuda --no-drive
```

微調設定中的 `pretrained_path` 和 `artifacts` 均指向上述權重。若改 `TRAINING_OUTPUT_ROOT`、run name 或使用 `--run-id`，請同步修改微調 YAML 的這兩處路徑；目前指令刻意採固定路徑方便銜接。

訓練預設即時顯示啟動階段、資料 SHA-256 驗證、batch、Loss、LR、GPU 記憶體、驗證結果、checkpoint 保存及 train ETA；每 10 個 batch 或約 10 秒更新一次（於完成 batch 後檢查）。不依賴 W&B，也不受 `--no-drive` 影響。所有狀態同步追加到 run 目錄的 `console.log`。

另一個終端可持續觀看：

```sh
tail -F outputs/pvhsd-dnanet-1ch-pretrain/console.log
```

若要避免 Python 其他輸出被緩衝，也可用模組指令啟動：

```sh
CUDA_VISIBLE_DEVICES=0 python -u -m research_training.cli run \
  --config configs/pvhsd2025/pretrain_dnanet_1ch.yaml \
  --device cuda --no-drive
```

已啟動的舊程序不會自動套用新程式；重啟後會從最近保存的 checkpoint 接續。舊程序目前可用 `tail -F outputs/pvhsd-dnanet-1ch-pretrain/train.log` 觀看每 100 steps／epoch 更新的紀錄。Logging 更動不要求更改訓練設定；請勿使用 `--no-resume` 來恢復既有訓練。

重跑同一指令會自動 Resume。沿用相同的 `TRAINING_OUTPUT_ROOT`，每個 run 同時只啟動一個程序。兩階段均由現有訓練框架保存 best/latest checkpoint、設定、optimizer/RNG 狀態及日誌。

第一階段會在 build 時驗證所有處理後影像與 mask 的雜湊。processed manifest 與微調來源 checkpoint 也列為 `artifacts`，它們的 digest 會進入 config compatibility hash，避免更換資料或初始化權重後誤續訓。

## 使用紀錄與驗證

### 照片切分開發實驗（2026-10-22）

新增模組 `src/solar_ir/datasets/splits.py` 的 `photo_stratified_split`，透過 `task.split_strategy: photo_stratified` 選用。以 `(group, 是否有 hotspot)` 分層，在每層內依 seed 隨機抽照片作 validation；group 僅用來平衡場景，允許同 group 跨兩邊。每層 validation 數約為該層的 20%，至少保留一張訓練照片與一張驗證照片；單張的層留在訓練，因此總比例可能略有差異。相同 seed 與資料可重現切分，照片 ID 不重複，尚未確認實體板子與近似連拍影像互不重複。這是同場景開發驗證，不能當作 group-disjoint 的跨場景分數。

新設定 `configs/pvhsd2025/finetune_solar_dnanet_1ch_cudnn_photo.yaml` 只改 run name、切分策略，並移除原有固定 validation_groups。仍從預訓練 best.pt 開始全模型微調，不接續原 group 微調 checkpoint；其餘模型、Loss、augmentation、LR 0.005、batch 4、seed 7、early stopping、cuDNN 與原微調設定完全相同。

| 切分 | 照片 | 有異常照片 | G02 有異常照片 |
|---|---:|---:|---:|
| train | 838 | 343 | 152 |
| validation | 208 | 85 | 38 |

```sh
conda activate solar-ir
export TRAINING_OUTPUT_ROOT=outputs
CUDA_VISIBLE_DEVICES=0 python -u -m research_training.cli run \
  --config configs/pvhsd2025/finetune_solar_dnanet_1ch_cudnn_photo.yaml \
  --device cuda --no-drive
```

獨立輸出目錄為 `outputs/solar-ir-dnanet-1ch-pvhsd-finetune-cudnn-photo/`。啟動後 checkpoint metadata 保存實際 train_frames / validation_frames；切分查核清單位於 `outputs/verification/photo_split/split_manifest.json`。已通過 24 項測試，另確認實際 1046 張影像的完整覆蓋、照片 ID 無交集、預訓練權重載入、全參數可訓練及 `[B,1,512,640]` loader shape；未自動啟動此長訓練。

### cuDNN 執行設定（2026-10-21）

CUDA 環境已安裝 cuDNN 9.19，原有 PyTorch 預設也已啟用 cuDNN。新增以下設定檔，供後續實驗開啟卷積演算法自動調優：

- `configs/pvhsd2025/pretrain_dnanet_1ch_cudnn.yaml`：新建 `pvhsd-dnanet-1ch-pretrain-cudnn`，從頭預訓練。
- `configs/pvhsd2025/finetune_solar_dnanet_1ch_cudnn.yaml`：新建 `solar-ir-dnanet-1ch-pvhsd-finetune-cudnn`，載入目前原始預訓練 run 的 `outputs/pvhsd-dnanet-1ch-pretrain/best.pt`。

兩者均設定 `runtime.cudnn.enabled: true`、`benchmark: true`、`deterministic: false`；模型、Loss、AMP 與其他訓練設定沿用各自原版。依 [PyTorch 文件](https://docs.pytorch.org/docs/2.11/notes/randomness.html)，benchmark 會測試卷積演算法後選用較快者，首次遇到新 shape 需要調優時間，跨次執行也不保證完全相同的數值結果；本次未宣稱實測加速幅度。

目前正在跑的程序保持運行，原始兩份 YAML 不改動。若該程序之後需要續訓，繼續使用原本的 `pretrain_dnanet_1ch.yaml`，不要改用新設定檔接續同一 run。之後啟動競賽微調可用：

```sh
CUDA_VISIBLE_DEVICES=0 python -u -m research_training.cli run \
  --config configs/pvhsd2025/finetune_solar_dnanet_1ch_cudnn.yaml \
  --device cuda --no-drive
```

新啟動的 run 會於終端顯示 backend 狀態，並將版本與開關保存至 `runtime_backend.json` 和 checkpoint metadata；缺少 cuDNN 的 CUDA 環境會明確報錯。沒有 `runtime` 設定的舊實驗保留原本預設政策（enabled=true、benchmark=false、deterministic=false）。

此項變更已通過 20 項測試、Ruff 與 mypy；另在空閒的 GPU 1（L40S）以 `[2,1,32,32]` 執行 DNA-Net 四個輸出的前向與反向運算，profiler 確認出現 `aten::cudnn_convolution`，Loss 有限。結果保存在 `outputs/verification/cudnn_backend/result.json`。這是功能驗證，未測量完整訓練的速度差異；GPU 0 原有程序未中斷，原始設定的續訓 compatibility hash 也保持一致。

每次 run 的 `resolved_config.json`、`data_manifest.json` 和 checkpoint metadata 會保留外部資料版本、監督方式、manifest 雜湊；微調同時保留預訓練 checkpoint 的雜湊。資料使用總表保存在 `docs/datasets/PV-HSD-2025.json`，本次已新增灰階／框 mask 處理與小型驗證事件。

初期驗證包含：15 項單元測試、資料／mask 校驗、單通道權重 strict load、CPU 實際資料小型預訓練→checkpoint→競賽微調，以及 640×512 輸入的四個輸出 shape。小型驗證刻意使用少量樣本、64×64、1 epoch，**不能代表正式訓練效能**；完整 batch loader 另以正式尺寸檢查。正式 GPU 長訓練已另行啟動。

測試結果及小型 run 在 `outputs/verification/pvhsd2025_1ch/result.json` 與其 `runs/` 目錄。
