# 結合少樣本學習與多模態視覺語言模型之太陽能板瑕疵檢測與根因分析比賽

本 repository 是本競賽的研究訓練框架。第一階段只使用 Thermal IR，先完成瑕疵區域的
像素級預測，再將預測 mask 轉換為競賽需要的 polygon/bounding box；RGB 與根因文字分析
保留為後續階段。

目前官方評測要求瑕疵定位與根因文字分析，定位部分比較預測 polygon 與人工標註 polygon。
因此本專案的內部模型輸出會保留 segmentation mask，不直接把 bbox 當成最終輸出。

> 已確認：訓練資料的 annotation 是 Labelme polygon，因此第一階段可以直接使用真正的
> pixel-level ground truth。IR 的異常類別是 `hotspot`；`crack` 僅出現在 EO/RGB。

模型採插件式設計，詳見 [MODEL_PLUGINS.md](MODEL_PLUGINS.md)。目前使用可替換的 U-Net
baseline；未來切換 DNANet 或其他 segmentation framework 時，只需新增 model factory 並
修改 YAML，不需要改動 dataset、checkpoint 或 Colab 流程。

驗證指標同樣模組化。目前依官方 evaluation kit 實作 `F1_TIoU`，包含官方的 polygon
overlap、recall/precision aggregation 與 duplicate prediction handling；`F1_TEXT` 保留
為後續文字模型插件。IR-only 實驗會以 `F1_TIoU` 作為 checkpoint selection metric。

這是一個從 mmCows 訓練基礎設施抽出的、與研究領域無關的 PyTorch 模板。模型、資料與 objective 是外掛；訓練生命週期、可重現性、斷點續訓、Google Drive 權重同步，以及 Weights & Biases（W&B）前端追蹤由核心統一處理。

## 保留的關鍵能力

- YAML/JSON resolved config，並以行為設定雜湊防止用不同實驗設定誤續訓。
- 單一設定或 `defaults + experiments` queue；queue 共用台北時間 run ID，可用名稱篩選與跨機器續跑。
- 原子化本機 checkpoint；保存 model、optimizer、AMP scaler、Python/NumPy/Torch/CUDA RNG、early stopping、best weights、epoch/batch/step 與版本資訊。
- 每隔 N optimizer steps 與 epoch 結束保存；可由中斷的 batch 繼續。若 task 使用隨機 sampler，應像範例一樣提供 `train_generator`；需要 bitwise resume 時請使用 `num_workers=0`，或讓 worker-side augmentation 本身可重播。
- Google Drive 雙後端：推薦既有登入的 `rclone`，或 Application Default Credentials 的 Drive API。每個 checkpoint 使用不可變版本檔名，再更新含 SHA-256 的 `latest.json` 指標；新機器會下載並驗證後續訓。
- W&B `online`、`offline`、`disabled`；相同 run name 產生固定 run ID 並使用 `resume="allow"`。訓練設定、曲線、系統資源與 final summary 可在 W&B 前端查看。
- 本機 `history.json`、逐事件 `train_log.jsonl`、可直接閱讀的 `train.log`、`metrics.json`、`resolved_config.json`、`data_manifest.json`、`best.pt`，不把雲端服務當成唯一紀錄。
- CPU/CUDA auto selection、AMP、gradient clipping、min/max checkpoint selection、early stopping。
- iteration duration、throughput、learning rate、ETA、PyTorch GPU memory 與 `nvidia-smi` board telemetry。

## 快速開始

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[torch,wandb,dev]"
Copy-Item .env.example .env
research-train run --config configs/smoke.yaml --device cpu --no-drive
python -m pytest
python -m ruff check .
python -m mypy src
```

smoke task 是合成資料，不代表任何研究效能。輸出位於 `outputs/synthetic-smoke/`。

## Colab 與共享 Google Drive

完整操作流程請參考 [DEPLOY.md](DEPLOY.md)。

目前也可以不使用 rclone，直接把 Google Drive 掛載到 Colab，讓每個人用自己的
Google 帳號存取同一個共享資料夾。請先在 Google Drive 建立資料夾並分享給協作者，
不要共用 Google 帳號或密碼；若資料夾位於「與我共用」，可先在 Drive 網頁將它
新增捷徑到自己的「我的雲端硬碟」。

在 Colab 執行：

```python
from google.colab import drive
drive.mount("/content/drive")
```

```python
import os

os.environ["TRAINING_OUTPUT_ROOT"] = (
    "/content/drive/MyDrive/research-training/experiments"
)
```

接著安裝並執行專案：

```bash
!pip install -e ".[torch,wandb,dev]"
!research-train run \
    --config configs/smoke.yaml \
    --device cuda \
    --no-drive
```

`--no-drive` 在這裡代表不啟用 rclone/API backend；輸出仍會直接寫入
`TRAINING_OUTPUT_ROOT` 指定的共享 Drive 目錄，因此另一台主機或另一個 Colab
runtime 指向同一個目錄即可 Resume。使用相同的 run name 或相同的 `--run-id`，
並且同一個 run 同時間只讓一台機器訓練，避免兩個程序同時覆蓋 checkpoint。

掛載 Drive 適合先開始使用，但大型 checkpoint 或非常頻繁的保存可能較慢；可把
`training.checkpoint_every_steps` 調大。之後若需要更高吞吐量，再切回本機暫存加
rclone 同步，不需要更改 task 或 checkpoint 格式。

## 接到新研究

建立一個 class 實作 `ResearchTask`：

1. `build` 建立 model、optimizer、train/validation loaders，並回傳資料版本、split hash、樣本數等 metadata。
2. `training_loss` 定義研究自己的 forward、loss 與少量 scalar diagnostics。
3. `validation_metrics` 在獨立 validation split 計算 metrics，且必須包含 `training.selection_metric`。
4. 提供無參數 factory，例如 `my_project.task:create_task`，在 YAML 的 `task.factory` 指向它。

核心刻意不猜測 batch 結構、模型 signature、metric 或資料切分方式。資料洩漏規則仍由 task 負責：normalization、feature selection 與 threshold 必須只 fit train；test 不應被拿來選 checkpoint。建議把 data/split manifest 的 SHA-256、dataset version、Git revision 與樣本數放進 `TaskBundle.metadata`。

大型 dataset、split manifest 或 pretrained weights 可在頂層宣告。檔案不存在時會依 Drive file ID 下載；無論是否下載都會驗證設定中的 digest，並把實際 digest 寫進 resolved config、`data_manifest.json` 與 checkpoint compatibility hash：

```yaml
artifacts:
  - name: dataset
    path: data/processed/train.parquet
    version: study_v3
    drive_file_id: optional-google-drive-file-id
    sha256: expected-lowercase-sha256
  - name: split_manifest
    path: data/manifests/split.json
    sha256: expected-lowercase-sha256
```

若 task 完成後另產生 predictions、plots 或其他檔案，可選擇實作 `artifacts(config, run_directory) -> list[Path]`；核心會限制檔案必須位於該 run 目錄並一起上傳。

多組實驗可使用 [queue.example.yaml](configs/queue.example.yaml)：

```powershell
research-train run --config configs/queue.example.yaml --run-id 20261004_2300 --no-drive
research-train run --config configs/queue.example.yaml --experiment small-model --run-id 20261004_2300 --no-drive
```

## Google Drive 權重上傳

在 `.env` 設定：

```dotenv
TRAINING_GDRIVE_PARENT_FOLDER_ID=你的資料夾ID
TRAINING_GDRIVE_FOLDER_PATH=research-training/experiments
TRAINING_GDRIVE_BACKEND=rclone
TRAINING_RCLONE_REMOTE=Google
```

先在機器上私下完成 `rclone config`，不要把 token 放進 repo、設定檔或聊天。執行時預設啟用 Drive；`--no-drive` 可做純本機測試。若選 `api`，安裝 `[drive]` extra 並配置 Google Application Default Credentials。

遠端每個 run 會包含 `checkpoint_step_*_progress.pt`／`*_complete.pt`、`latest.json`、`best.pt`、`metrics.json` 與 `resolved_config.json`。同步失敗會停止訓練，但已完成的本機 atomic checkpoint 仍可重試。

## W&B 前端檢視

```powershell
wandb login
research-train run --config configs/smoke.yaml --wandb-mode online --wandb-project my-study --no-drive
research-train dashboard --run-name synthetic-smoke
```

最後一行會印出並開啟該 run 的 W&B 頁面。也可設定 `WANDB_ENTITY` 後用：

```powershell
research-train dashboard --project my-study
```

無網路時用 `--wandb-mode offline`，之後執行 `wandb sync outputs/<run>/wandb/<offline-run-directory>`。API key 只透過 `wandb login` 或私密環境變數提供。

## 建議專案結構

```text
configs/                 experiment definitions
src/research_training/   reusable engine/storage/tracking
src/your_study/          task plugins, models, datasets, metrics
tests/                   leakage, split, resume, and smoke tests
data/raw/                immutable source data (ignored)
outputs/                 ignored run artifacts
```

不同 seeds 應使用不同 run names（例如 `study-seed-7`），並在研究層彙整跨 seed 與獨立樣本單位的 uncertainty；模板不會把相關 rows 當成獨立實驗單位。
