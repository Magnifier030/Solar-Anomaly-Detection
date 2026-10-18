# Colab 部署與共享訓練流程

本文件說明如何在 Google Colab 使用這個專案，並將 checkpoint、權重、log 與
metrics 寫入共享的 Google Drive。

每個人都使用自己的 Google 帳號，不需要共用密碼或 OAuth token。

## 1. 建立共享資料夾

由專案負責人先在 Google Drive 建立資料夾，例如：

```text
research-training/
└── experiments/
```

將 `research-training` 資料夾分享給其他協作者，權限設為「編輯者」。

其他協作者如果只在「與我共用」看到資料夾，請在 Google Drive 網頁上對資料夾
選擇「整理」→「新增捷徑」→「我的雲端硬碟」。

建議所有人使用相同的 Drive 資料夾結構：

```text
My Drive/
└── research-training/
    └── experiments/
```

## 2. 開啟 Colab 並掛載自己的 Drive

在 Colab 新增 Python cell：

```python
from google.colab import drive

drive.mount("/content/drive")
```

依照畫面指示登入自己的 Google 帳號並允許存取。

確認共享資料夾可讀取：

```python
from pathlib import Path

DRIVE_ROOT = Path("/content/drive/MyDrive/research-training")
EXPERIMENT_ROOT = DRIVE_ROOT / "experiments"
EXPERIMENT_ROOT.mkdir(parents=True, exist_ok=True)

print(EXPERIMENT_ROOT)
print(EXPERIMENT_ROOT.exists())
```

如果輸出是 `False`，請先確認共享資料夾已新增捷徑到「我的雲端硬碟」，或調整
`DRIVE_ROOT` 為實際路徑。

## 3. 取得專案程式碼

如果 repository 是公開的：

```bash
!git clone <REPOSITORY_URL> /content/research-training-template
%cd /content/research-training-template
```

如果 repository 是私有的，請使用適合的 Git 認證方式。不要把 GitHub token
直接寫進 notebook、Google Drive 或 commit。

如果 Colab 已經有專案目錄，改用：

```python
%cd /content/research-training-template
!git pull
```

## 4. 安裝套件

```bash
!pip install -e ".[torch,wandb,dev]"
```

如果只需要訓練、不使用 W&B 或開發工具，也可以安裝：

```bash
!pip install -e ".[torch]"
```

確認 CLI 可用：

```bash
!research-train --help
```

## 5. 設定輸出位置

在執行訓練前設定環境變數：

```python
import os

os.environ["TRAINING_OUTPUT_ROOT"] = (
    "/content/drive/MyDrive/research-training/experiments"
)
```

這個設定會讓所有 run 的輸出直接寫入共享 Google Drive。

本模式不使用 rclone，因此執行指令必須加上 `--no-drive`。這裡的
`--no-drive` 只是關閉 rclone/API backend，不代表不使用 Google Drive；輸出仍由
`TRAINING_OUTPUT_ROOT` 指定。

## 6. 執行 smoke test

先用合成資料確認環境正常：

```bash
!research-train run \
    --config configs/smoke.yaml \
    --device cuda \
    --no-drive
```

如果 Colab 沒有 GPU，可以改成：

```bash
!research-train run \
    --config configs/smoke.yaml \
    --device cpu \
    --no-drive
```

完成後，應該會看到：

```text
/content/drive/MyDrive/research-training/experiments/synthetic-smoke/
```

其中包含 `latest.pt`、`best.pt`、`metrics.json`、`history.json`、
`train_log.jsonl` 與 `resolved_config.json`。

## 7. 執行正式實驗

正式實驗請使用清楚且唯一的 run name，例如在 config 中設定：

```yaml
name: study-seed-7
```

執行：

```bash
!research-train run \
    --config configs/my-study.yaml \
    --device cuda \
    --no-drive
```

如果使用 queue：

```bash
!research-train run \
    --config configs/queue.example.yaml \
    --experiment small-model \
    --run-id 20261016_2300 \
    --device cuda \
    --no-drive
```

相同的 `--run-id` 會指向相同的實驗目錄。不同 seed 請使用不同的 run name，
例如 `study-seed-1`、`study-seed-2`、`study-seed-3`。

## 8. Colab 斷線後 Resume

Colab runtime 中斷或關閉後，重新開啟 notebook，依序執行：

1. 掛載自己的 Google Drive
2. 取得或更新 repository
3. 安裝套件
4. 設定相同的 `TRAINING_OUTPUT_ROOT`
5. 使用相同的 config、run name 或 `--run-id`

例如：

```python
from google.colab import drive
drive.mount("/content/drive")

import os
os.environ["TRAINING_OUTPUT_ROOT"] = (
    "/content/drive/MyDrive/research-training/experiments"
)
```

```bash
%cd /content/research-training-template
!research-train run \
    --config configs/my-study.yaml \
    --device cuda \
    --no-drive
```

程式會讀取共享資料夾中的 `latest.pt`，恢復 model、optimizer、AMP scaler、
RNG state、epoch、batch、global step 與 best model 狀態。

## 9. 多人協作規則

- 同一個 run 同一時間只能由一台機器訓練。
- 其他人可以查看 Drive 內容或 W&B，但不要同時啟動相同 run。
- Resume 前要確認前一台機器已停止訓練並完成最後一次 checkpoint。
- 不要任意修改已開始 run 的 training config；設定不同時，程式會拒絕 Resume。
- 程式碼更新後，最好先確認使用相同 Git commit，再接續訓練。
- 不要刪除 `latest.pt`、`best.pt` 或 `resolved_config.json`，除非確定不再需要該 run。

## 10. W&B 選用設定

若要在 W&B 查看訓練曲線，先在 Colab 登入：

```bash
!wandb login
```

執行時指定：

```bash
!research-train run \
    --config configs/my-study.yaml \
    --device cuda \
    --no-drive \
    --wandb-mode online \
    --wandb-project my-study
```

W&B 用於查看曲線與系統狀態；Google Drive 仍是 Resume 所需 checkpoint 的主要
保存位置。

## 11. 常見問題

### 找不到共享資料夾

確認資料夾已分享給目前登入 Colab 的 Google 帳號，並且已新增捷徑到「我的雲端
硬碟」。也可以列出路徑確認：

```bash
!ls -la /content/drive/MyDrive
```

### Drive 寫入很慢

調高 `configs/*.yaml` 中的：

```yaml
training:
  checkpoint_every_steps: 100
```

checkpoint 越大、保存越頻繁，Drive I/O 對訓練速度影響越明顯。

### Resume 被拒絕

通常是 config 的行為設定變更。確認 config、run name、seed、model 參數與資料
版本和原本一致。若要開始新實驗，請使用新的 run name。

