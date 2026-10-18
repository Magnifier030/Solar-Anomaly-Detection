# 專案狀態

## 已決定的第一階段範圍

- 輸入：Thermal IR only
- 類別：單一瑕疵類別
- 模型輸出：pixel-level defect mask
- 競賽輸出：由 mask 後處理得到 polygon / bounding box
- 暫不處理：RGB、根因文字、VLM
- 執行環境：Google Colab GPU，checkpoint 與 logs 寫入 Google Drive

## 已確認的資料格式

訓練資料已整理進 `data/raw/`。官方 annotation 是 Labelme JSON，`shapes` 使用 polygon，
因此可以直接產生真正的 pixel-level mask，不需要 bbox pseudo-mask。

- IR 異常類別：`hotspot`
- EO/RGB 異常類別：`crack`
- `solar_panel` 是模組區域，不是異常類別
- IR 與 EO 座標系統不同，不能直接套用另一模態的 mask
- `train_groups.csv` 必須以 group 為單位切分，避免連續拍攝影格造成 validation leakage

## 下一個實作階段

先建立資料清單與可視化檢查，確認：

- Thermal IR 檔案格式、位元深度與數值範圍
- polygon annotation 的座標定義
- train/test image pairing 與 sample ID
- 官方答案卷欄位及 polygon 編碼

確認後再接入 DNANet，並加入 mask-to-polygon/bbox inference 與官方答案卷 exporter。
