# 資料目錄

資料已從 macOS Downloads 複製到本專案。原始 Downloads 檔案仍保留，因為工作區沙盒不允許
直接跨目錄刪除或搬移外部檔案。

```text
data/raw/
├── datasets/
│   ├── 9_16_train_set/
│   │   └── train/{IR,EO}/{image,label}/
│   └── 9_30_test_set/
│       └── test/{IR,EO}/image/
├── supplement/
│   └── SolarDefectDataset_supplement/train_groups.csv
├── evaluation_kit/
│   └── author-kit_1012/
└── archive/
    ├── 9_16_train_set.zip
    ├── 9_30_test_set.zip
    ├── SolarDefectDataset_supplement.zip
    └── evaluation_kit.zip
```

目前已確認：

- train：1046 張 IR + 1046 張 EO，兩種模態各自有 1046 個 Labelme JSON
- test：696 張 IR + 696 張 EO
- IR 異常：`hotspot`
- EO 異常：`crack`
- annotation：polygon，可轉成 segmentation mask
- submission：JSON，key 必須符合 `test_images.csv`，points 可為 polygon

## 外部資料：PV-HSD-2025

已下載至 `data/raw/external/PV-HSD-2025/`，保留官方 ZIP 與完整原始檔。
原始 4,025 張影像、25,177 個 YOLO 框；另有排除兩張可見光照片的 4,023 張 IR 候選清單。
單通道 PNG 與 bbox 弱 mask 位於 `data/processed/pv_hsd2025/grayscale_box_masks_v1/`，
已完成小型 CPU 訓練流程驗證，正式預訓練尚未啟動。
完整指令見[兩階段訓練文件](../docs/training/pvhsd2025_dnanet.md)。
來源版本、授權原文、雜湊與使用事件見
[外部資料紀錄](../docs/datasets/PV-HSD-2025.md)及
[機器可讀紀錄](../docs/datasets/PV-HSD-2025.json)。
