# PV-HSD-2025 外部資料取得與使用紀錄

目前狀態：**已完成單通道與 bbox 弱 mask 處理，正式 PV-HSD 預訓練已啟動。**
實際 run：`outputs/pvhsd-dnanet-1ch-pretrain/`；2026-10-20 查核到有效訓練 steps、checkpoint 與驗證結果。
操作文件見[單通道兩階段訓練](../training/pvhsd2025_dnanet.md)。下方原始資料查核章節保留下載當時的發現；最新處理與使用狀態以本段及 JSON 紀錄為準。
下載日期：2026-10-20（台北時間）。用途為評估太陽能板熱紅外預訓練資料。

## 來源與固定版本

- 作者公布的 [Hugging Face repository](https://huggingface.co/datasets/MisakaMikoto128/PV-HSD-2025)。
- 固定 revision：`40748bc7c445cf9125f2cd25d115fa35bbeb17af`。
- [本次下載的固定版本壓縮檔](https://huggingface.co/datasets/MisakaMikoto128/PV-HSD-2025/resolve/40748bc7c445cf9125f2cd25d115fa35bbeb17af/PV-HSD-2025.zip)。
- 來源論文：[PV-HSD-2025 Infrared Hotspot Dataset for Photovoltaic Panels and YOLOv8-P1 Defect Detection Model](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5134282)。報告／發表時須依作者要求引用；此處不編造未取得的正式 BibTeX。
- 壓縮檔大小：3,740,409,982 bytes。
- SHA-256：`3fac0ae8e3e345e9f3d1c45afb4251530f79bfda8d772aacd708799a32416c32`，與官方 LFS metadata 完全相符。
- 壓縮檔、解壓原檔及查核結果占用約 8.7 GiB；未另外下載模型權重或獨立的原始飛行資料包。

## 本機存放與證據

原始根目錄：`data/raw/external/PV-HSD-2025/`。

| 檔案／路徑 | 用途 |
|---|---|
| `PV-HSD-2025.zip` | 未修改的官方壓縮檔 |
| `extracted/` | 完整解壓的原始影像、標註與 metadata |
| `source_metadata/README.upstream.md` | 固定 revision 的作者說明及授權原文 |
| `source_metadata/huggingface_api.json` | 官方 revision、檔案大小、LFS 雜湊等證據 |
| `download_receipt.json` | 下載時間、來源、完整性驗證 |
| `file_manifest.json` | 解壓後全部 8,055 個檔案的路徑、大小及 SHA-256 |
| `audit_summary.json` / `audit_records.json` | 統計、逐影像尺寸與 bbox 查核 |
| `ir_candidate_manifest.json` | 排除兩張已確認可見光照片後的候選 IR 清單 |
| `bbox_samples.jpg` / `bbox_samples.json` | 9 張不同框面積的代表圖與對應檔名 |

原始資料被 `.gitignore` 排除；**本紀錄與同目錄的 `PV-HSD-2025.json`、`PV-HSD-2025-exclusions.json` 可納入版本控制**，避免資料使用事實只留在 ignored 輸出。

## 實際資料內容

所有壓縮檔成員完成解壓，ZIP 讀取時進行 CRC 驗證；所有 4,025 張影像均完整解碼成功。

| 官方 split | 影像 | 有異常框的影像 | 空標註 | 框數 |
|---|---:|---:|---:|---:|
| train | 2,808 | 2,658 | 150 | 17,433 |
| valid | 1,217 | 1,140 | 77 | 7,744 |
| 合計 | 4,025 | 3,798 | 227 | **25,177** |

資料頁宣稱 25,181 框，本次固定版本實際少 4 框；未為了湊數新增或更改標註。
類別為單一 `hot_spot`，每行為 YOLO `class cx cy width height`。未發現無效數值、超界框、遺漏標註對應或孤立標註檔。空標註保留為來源原樣，不等同已確認完全健康。

影像解析度為 4,023 張 640×512，加兩張 4000×3000。經人工檢視，後兩張 `train/images/62_2.JPG`、`train/images/62_3.JPG` 為可見光照片，故只在 IR 候選清單排除，原檔與空標註均保留。候選清單為 train 2,806、valid 1,217，合計 **4,023 張**。

所有 JPEG 的 Pillow mode 均為 RGB；熱影像的偽彩色編碼也會是 RGB，因此 mode 不可用來判別可見光／熱紅外。原始 JPEG 未重新存檔，保留其中可能包含的輻射／相機 metadata；本次尚未驗證全部檔案是否具有可解碼的 16-bit 輻射資料。

## 分割監督與切分限制

代表圖顯示標註框通常只涵蓋**局部熱點**，不是完整面板。框面積占整圖的 P10/P50/P90 約為 0.00732% / 0.02350% / 0.09125%。此處是 bbox 面積，不能與競賽 polygon 面積當作完全相同的量直接比較。

競賽標註涵蓋整片異常模組，後續需辨識熱點所屬面板並補面板輪廓，或明確改採局部熱點的輔助預訓練任務。不能直接填滿熱點框後宣稱符合競賽真實 mask。目前已建立局部熱點 bbox 聯集的弱 mask，用於輔助預訓練；尚未建立面板輪廓 pseudo mask。

全檔 SHA-256 與解碼後 RGB 像素雜湊都未發現完全重複影像；但尚未檢查近似影像或場景隔離。以檔名最後一個底線前的前綴作為啟發式家族，共 1,233 組，其中 745 組跨 train/valid。這不是官方 group 定義，也未證明每組都同源，但足以提醒不能把官方隨機切分直接視為獨立場景驗證。後續應另查場景／近似影像，再決定分割方式。

## 授權原文保留

來源的授權標示有矛盾：Hugging Face README frontmatter 是 `apache-2.0`；正文說僅供學術研究、商業使用需作者授權；下載包 `data.yaml` 則標 `Public Domain`。本次完整保留這些來源證據，未自行宣告哪一項取代其他項。

下載與技術查核不代表已確認競賽允許外部資料。正式訓練／提交時仍需依競賽規則描述使用方式，並保留引用及外部資料使用紀錄。

## 可重現命令與後續使用紀錄

```sh
python scripts/pvhsd2025/download.py --revision 40748bc7c445cf9125f2cd25d115fa35bbeb17af
python scripts/pvhsd2025/audit.py
```

下載器會驗證官方檔案大小及 SHA-256，支援中斷下載續傳；解壓時檢查路徑、拒絕 symlink 並保留來源內容。查核程式只產生清單、統計及代表圖，不改原始影像或框。

機器可讀的取得與使用紀錄為 `docs/datasets/PV-HSD-2025.json`。目前 `training_used=true`、`mask_generation_used=true`（bbox 弱 mask）、`formal_pretraining_started=true`；`experiment_runs` 同時記錄先前 CPU 驗證與正式預訓練 run。未來每次 mask 生成或訓練都應新增事件，記錄來源 revision、衍生清單／mask 雜湊、演算法與模型版本、人工修正方式、split manifest、run 名稱、設定與使用範圍；相同資訊也應進入該次實驗的 metadata。不能將「已下載」寫成「已完成預訓練」。
