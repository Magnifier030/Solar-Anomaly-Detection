# Google Colab 執行方式

本專案沿用訓練模板的可恢復 checkpoint。Colab runtime 中斷後，只要重新掛載同一個
Google Drive、重新安裝 repository，便可以用相同 run name resume。

## 最快的 baseline

以下流程假設你已將整個 repository 放在自己的 Google Drive，例如：

```text
MyDrive/solar-ir-competition/
```

資料不需要再上傳到其他服務；訓練會直接讀取該 repository 內的 `data/raw/`，輸出則寫入
同一個 Drive 的 `experiments/`。

### 1. 掛載 Drive 並進入專案

```python
from google.colab import drive
drive.mount("/content/drive")
```

```python
from pathlib import Path
import os

PROJECT_DIR = Path("/content/drive/MyDrive/solar-ir-competition")
os.chdir(PROJECT_DIR)
os.environ["TRAINING_OUTPUT_ROOT"] = str(PROJECT_DIR / "experiments")
print(PROJECT_DIR)
print((PROJECT_DIR / "data/raw/datasets/9_16_train_set").exists())
```

### 2. 安裝套件

```bash
!git clone <YOUR_REPOSITORY_URL> /content/solar-ir-competition
%cd /content/solar-ir-competition
!pip install -e ".[torch,wandb,dev,eval]"
```

### 3. 設定 W&B

建議在 Colab 的 Secrets 建立 `WANDB_API_KEY`，不要把 key 寫進 notebook：

```python
from google.colab import userdata
import os

os.environ["WANDB_API_KEY"] = userdata.get("WANDB_API_KEY")
```

如果不使用 Colab Secrets，也可以互動登入：

```python
!wandb login
```

### 4. 執行簡單 baseline

這個設定使用小型 SimpleUNet、5 epochs、IR polygon mask，並以官方 `F1_TIoU` 選最佳
checkpoint：

```bash
!research-train run \
    --config configs/solar_ir_colab_baseline.yaml \
    --device cuda \
    --no-drive \
    --wandb-mode online
```

`--no-drive` 在這裡只代表不使用 rclone/API；因為 `TRAINING_OUTPUT_ROOT` 已經指向
Google Drive，所以 checkpoint、`best.pt`、metrics、logs 仍會直接寫進 Drive。

### 5. 訓練結束後清理 GPU

訓練 engine 在正常結束或例外時都會呼叫 W&B `finish()`。Notebook 最後再執行：

```python
import gc
import torch

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
print("GPU cache released")
```

若要在成功完成後自動釋放 Colab runtime，可在最後執行；這會中斷目前 runtime，請放在
確認輸出已寫入 Drive 之後：

```python
from google.colab import runtime
runtime.unassign()
```

不建議在 debug 階段自動執行 `runtime.unassign()`，否則錯誤訊息可能來不及查看。

若 repository 已存在，使用：

```bash
%cd /content/solar-ir-competition
!git pull
!pip install -e ".[torch,wandb,dev,eval]"
```

將資料與訓練輸出放到 Drive：

```python
import os
os.environ["TRAINING_OUTPUT_ROOT"] = \
    "/content/drive/MyDrive/solar-ir-competition/experiments"
```

執行 smoke test：

```bash
!research-train run --config configs/smoke.yaml --device cuda --no-drive
```

正式訓練會使用相同 CLI；`--no-drive` 只關閉 rclone/API backend，並不會關閉上面指定的
Google Drive 輸出目錄：

```bash
!research-train run \
    --config configs/solar_ir_baseline.yaml \
    --device cuda \
    --no-drive \
    --resume
```

同一個 run 同時只能由一個 Colab runtime 執行。資料下載完成後，會再補上正式的 config、
資料掛載路徑與答案卷產生指令。
