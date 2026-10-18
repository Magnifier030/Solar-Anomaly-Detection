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
