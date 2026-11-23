# PV-HSD-2025 資料工具

從專案根目錄執行：

```sh
python scripts/pvhsd2025/download.py --revision 40748bc7c445cf9125f2cd25d115fa35bbeb17af
python scripts/pvhsd2025/audit.py
python -m solar_ir.preprocessing.pvhsd2025
```

原始資料放 `data/raw/external/PV-HSD-2025/`，處理後資料放
`data/processed/pv_hsd2025/grayscale_box_masks_v1/`。
訓練指令與限制見 [兩階段操作文件](../../docs/training/pvhsd2025_dnanet.md)。

比較預訓練直接遷移與目前微調 checkpoint 的完整競賽驗證（CPU，不佔用訓練 GPU）：

```sh
PYTHONPATH=src python -u scripts/pvhsd2025/diagnose_transfer.py \
  --config configs/pvhsd2025/finetune_solar_dnanet_1ch_cudnn.yaml \
  --output outputs/solar-ir-dnanet-1ch-pvhsd-finetune-cudnn/analysis/transfer_validation_20261022
```

輸出包含 checkpoint 快照、SHA-256、逐圖像素 TP/FP/FN、0.05–0.5 門檻的 IoU/Dice，以及官方 F1_TIoU 的 precision/recall 加權分子與物件數分母。沿用訓練驗證的 polygon 轉換；checkpoint 原始 epoch 欄位也保留供核對。每次比較請使用新的輸出目錄。

可加上 `--device cuda` 並用 `CUDA_VISIBLE_DEVICES=1` 指定 GPU。若要保留已有比較基準、只重算結果，使用相同輸出目錄並加 `--reuse-snapshots`；否則會重新取得當下的 checkpoint 快照。
