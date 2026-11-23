# IR-only 提交檔（2026-10-22）

競賽：AI CUP 2026 太陽能板瑕疵檢測與根因分析。官方議題頁：<https://www.aidea-web.tw/topic/3c8ec48b-0dff-4b14-9b5e-d91bbb98c699>。提交結構以本機官方 `data/raw/evaluation_kit/author-kit_1012/README.md` 及檢查工具為準。

依使用者要求，696 張 IR 使用照片切分微調的最佳 DNA-Net（顯示 epoch 61、F1_TIoU=0.7497724981）預測，只輸出 `hotspot`；每個物件 `reason: null`。696 張 EO 全部填 `[]`，不產生 `solar_panel` 或 `crack` 預測。官方 kit 明確允許 null reason，但文字分數為 0；空陣列表示無預測，不代表已確認該影像正常。總分包含三類定位及文字，不能把平台總分當成 IR-only 指標。

## 產物

目錄：`outputs/submissions/dnanet-ir-only-photo-20261022/`

- `submission.json`：待使用者上傳的單一 JSON，完整 1392 個 key。
- `format_check.txt`：官方格式檢查通過紀錄。
- `manifest.json`：checkpoint/config/test-list/submission SHA-256、外部預訓練來源、backend、逐圖預測計數，uploaded=false。
- `inference_config.yaml`：實際推論設定。
- `ir_masks/`：696 張原解析度 0/255 預測 mask。
- `preview.jpg`：8 張有預測與 4 張無預測影像的抽樣疊圖；不是正確率驗證。

測試集共 74 張 IR 有可輸出的輪廓，合計 141 個 hotspot。沒有測試答案，無法得知其真實準確率；預覽中也有很小的輪廓，未額外手動調整或依 test 調門檻。

## 重現

```sh
conda activate solar-ir
CUDA_VISIBLE_DEVICES=0 python -u -m solar_ir.inference \
  --config configs/submission/dnanet_ir_only_photo.yaml
```

既有輸出目錄不覆寫；重跑請複製設定並指定新的 output_dir。輸入採和訓練 validation 共用的灰階／除以 255／resize，eval 模式取最後一個 supervision head，threshold=0.5；先將機率 resize 回原圖再二值化。外輪廓沿用 CHAIN_APPROX_SIMPLE，不額外濾除小物件；僅在超過 100 頂點時使用 approxPolyDP 簡化，避免官方退件。零面積輪廓不輸出，超過 500 物件／影像則報錯。

```sh
python data/raw/evaluation_kit/author-kit_1012/make_submission.py --check \
  outputs/submissions/dnanet-ir-only-photo-20261022/submission.json
```

已通過 27 項測試、Ruff、官方格式檢查；另確認所有 key 與官方清單一致、EO 空陣列、IR reason 全為 null、polygon 頂點上限與原圖座標範圍。尚未上傳 AIdea，也沒有取得排行榜成績。外部預訓練資料使用已記錄於 PV-HSD-2025 provenance 文件與提交 manifest。

## Google Drive 備份

2026-10-22 已依使用者要求備份整個提交目錄，共 701 個原始產物，另附 `drive_backup_receipt.json`。雲端路徑為 `research-training/experiments/submissions/dnanet-ir-only-photo-20261022/`。已通過 rclone check 雜湊比對，並下載雲端 submission.json 比對 SHA-256。雲端提交檔：<https://drive.google.com/file/d/1gRV04ifgt8XlL3YHj0BA1lk26aQ0m3y8/view>。這是 Drive 備份，尚未提交 AIdea；manifest.json 的 uploaded=false 仍表示沒有上傳競賽平台。
