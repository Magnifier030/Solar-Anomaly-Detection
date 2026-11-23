# 太陽能板 IR 瑕疵資料查核（2026-10-20）

更新：PV-HSD-2025 已於同日完成官方固定版本下載與實檔查核，見[取得與使用紀錄](datasets/PV-HSD-2025.md)。下文「未下載」為初次搜尋當時的狀態，最新狀態以上述紀錄為準。

本次使用 AnySearch 遠端 MCP 的匿名 search、batch_search，並以資料作者、官方 repository、資料典藏頁及部分 ZIP 檔內容交叉查核。未進行完整資料下載或模型訓練。沒有把搜尋片段直接當成可用資料的證據。

## 優先候選

| 資料 | 規模 | 標註 | 取得方式與適用性 |
|---|---|---|---|
| [PV-HSD-2025](https://huggingface.co/datasets/MisakaMikoto128/PV-HSD-2025) | 4,025 張、25,181 個熱斑框；另提供 11,366 張原始 IR | YOLO boxes | 有 Hugging Face、Roboflow、百度及 Drive；最值得優先檢查小尺度熱斑 |
| [PV Thermal Defects](https://zenodo.org/records/14644158) | ZIP 實查 22,842 張 JPG | 22,842 個 TXT；抽查為單類別 defect 的 YOLO boxes | 可匿名部分下載；適合候選大規模預訓練，須重查來源與分割 |
| [MHHS](https://github.com/zss313/MHHS-Dataset) | 作者列出 9,148 張、640×512、4 種拍攝高度 | LabelImg 熱斑標註 | 需申請；對尺度變化研究有價值 |
| [InfraredSolarModules](https://github.com/RaptorMaps/InfraredSolarModules) | 20,000 張、24×40 | 12 類影像分類；10,000 正常、10,000 異常 | 官方 GitHub ZIP；適合表徵／分類預訓練，沒有原生分割監督 |
| [PVF-10](https://github.com/wangbobby1026/PVF-Dataset) | 5,579 張裁切影像、8 座電站、10 類 | 分類 | 官方有百度／Drive；可補跨電站多樣性，但不是 mask 資料 |
| [ThermoSolar-PV 對應專案](https://github.com/P-Darabi/SolarPanelsDefectDetector) | 原始 2,723 張，增強後約 7,500 張 | YOLO 偵測標註、8 類熱異常 | README 指向作者 Kaggle；不能把增強後張數算成新案例 |

上表是針對目前問題的候選排序與用途判斷，尚未用本競賽 group folds 證明任何一組能改善泛化。

## 原檔查核：PV Thermal Defects

作者 [GitHub](https://github.com/zhaider-psu/PV_Thermal_Defects_dataset) 稱超過 23,000 張，Zenodo 說明為五個來源整合。本次用 HTTP Range 讀取 ZIP central directory、data.yaml 及三個 TXT，沒有下載完整 1.43 GB：

- JPG 22,842，TXT 22,842；train 21,475、valid 896、test 471。
- data.yaml 為 `nc: 1`、`names: [defect]`。
- 三個 TXT 均為 `class x_center y_center width height`，支持「偵測框」判斷；未逐一檢查全部 TXT，不能宣稱已完整驗證標註品質。
- 檔案目錄未見 PNG mask；目前沒有證據可當作完整 polygon/mask 分割資料直接使用。
- 去除檔名 `.rf.<hash>` 尾碼後，48 個同源名稱組跨 split。這是潛在洩漏線索，尚非像素重複或近似程度的完整檢驗。
- 不知道五個來源是否與其他候選重疊，不能直接把所有資料集的張數相加。
- Zenodo API license 為 CC BY 4.0，data.yaml 亦標 CC BY 4.0；GitHub 的程式碼 license 與各原始資料授權須分別看待。

本次查核記錄：`outputs/external_dataset_search/curated_archive_audit.json`。

## PV-HSD-2025 的使用細節

[作者資料頁](https://huggingface.co/datasets/MisakaMikoto128/PV-HSD-2025) 同時提供已標註資料與原始飛行資料。不要把 11,366 張原始 IR 全當成已有標註或互相獨立的案例；原始資料與 4,025 張標註集也可能重疊。

[Roboflow v1](https://universe.roboflow.com/misaka-cpn7n/pv-hsd-2025-orgxy/dataset/1) 列出 2,808 train、1,217 valid、無 test，且未做 augmentation。Hugging Face viewer 的 8,055 rows 不等於 8,055 個獨立有標註樣本。

資料頁的 Apache-2.0 metadata、正文的學術研究限制，以及 Roboflow 的 Public Domain 顯示不一致，使用前需釐清適用版本的條款。尚未完整下載核對影像／標註。

## MHHS 與 PVF-10 的規模辨識

[MHHS 官方](https://github.com/zss313/MHHS-Dataset) 的表格為 28m：1,783、35m：2,323、47m：1,820、60m：3,222，合計 9,148。README 文字中的 28m 張數另寫為 1,738，與表格不符；申請時需以實際檔案核對。作者要求機構信箱、親簽申請表寄回，故本次未取得完整資料，也未代寄申請。

PVF-10 採[作者 repo／論文摘要](https://github.com/wangbobby1026/PVF-Dataset)的 5,579 張，沒有採用其他文章提到的約 8,000 張。GitHub LFS 有流量限制，作者另附百度及 Drive。尚未完整下載確認。

## 小型、可補充的資料

- [sUAS Infrared Image Dataset](https://zenodo.org/records/20860870)：各類公布數相加 3,675 張，8 類，正常 2,835 張、異常 840 張。是模組 crop 分類資料；Zenodo 有 4 MB ZIP。單一正射影像來源的裁切不能直接當成大量獨立電站。
- [PVsegmentation](https://imr.ciirc.cvut.cz/Datasets/PVsegmentation)：搜尋索引列出 136 RGB、74 thermal，合計超過 3,000 個模組 segmentation masks。標的是面板，不可直接等同異常 mask；本次頁面擷取失敗，這項尚未完成原頁／檔案驗證。
- [Thermal PV Panel Detection](https://zenodo.org/records/16420123)：640×512；正文稱 351 張，split 表格合計 353 張及 26,678 個面板標註。是單一電站，需核對實檔與異常類別，不可把面板標註數當作故障案例數。
- [Photovoltaic System Thermal Images](https://data.mendeley.com/datasets/82vzccxb6y/2)：277 張，熱斑與 snail trails。原始研究只涉及 4 片串聯面板；後續研究自行補 segmentation annotations，不代表原始下載就提供同一套 mask。本次未確認可下載的異常分割 mask。

## 不列為主要 IR 預訓練來源

- [PVEL-AD](https://github.com/binyisu/PVEL-AD) 雖有 36,543 張，但為 EL 電致發光影像，不是現場熱紅外溫度影像。
- [PV Multi-Defect 原始論文](https://www.mdpi.com/2079-9292/12/3/561)與後續文獻的模態描述不一致；不因類別叫 hot_spot 就認定為熱紅外，本次不列為已驗證熱 IR 資料。
- [Augmented Infrared Solar Modules](https://github.com/HrachA/augmented-infrared-solar-modules-set) 是既有 InfraredSolarModules 的增強版本，不提供相同張數的新獨立故障。
- [大規模北美電站研究](https://arxiv.org/abs/2503.02128) 涵蓋大量電站，但本次沒有找到可公開下載完整研究資料的可靠入口，不當作已可取得的資料。
- AnySearch 找到的 [CFRP 熱檢測 mask 資料](https://data.mendeley.com/datasets/jrsb4b9yy5/1) 是碳纖維材料，並非太陽能板，已排除。

## 對 DNA-Net 的實際意義

目前較大型的相關資料以分類／bbox 為主，沒有確認一套「上萬張、可直接匿名取得、具有可靠太陽能異常 polygon/mask」的資料。

建議先查 PV-HSD-2025，接著查整合資料的原始來源和去重品質；若需要跨拍攝高度資料，再申請 MHHS。它們比一般非太陽能小目標資料更貼近本任務，但改善幅度仍要以競賽 group folds 驗證。

分類標籤適合預訓練 encoder／表徵，不能直接監督原本 DNA-Net 的像素輸出；bbox 需要框監督、弱監督或經檢驗的 pseudo masks。不能把框填滿就宣稱得到真實分割標註。尤其目前競賽目標是整個異常模組，而外部資料可能只標亮點，必須先核對標註語意。

## AnySearch 連接狀態與永久設定

已實際向 `https://api.anysearch.com/mcp` 完成匿名 MCP initialize、tools/list、search、batch_search，查詢有正常結果。這是本次透過 HTTP 使用的連線，**尚未寫入永久 Codex 設定，也不代表新的原生工具已在當前對話載入**。

若要讓後續 Codex 工作階段自動連接，可在 `~/.codex/config.toml` 合併以下區塊；不要覆蓋既有設定：

```toml
[mcp_servers.anysearch]
url = "https://api.anysearch.com/mcp"
http_headers = { "X-Anysearch-Client" = "mcp/1.0.0" }
```

此為匿名模式。若要用自己的 key，再加 `bearer_token_env_var = "ANYSEARCH_API_KEY"`，並在啟動 Codex 的環境中私下設定該變數；不需把 key 寫在專案或貼到聊天。用 `codex mcp list` 檢查設定；當前工具列表是否刷新需要重新檢查，不假設立即可用。

來源：[AnySearch 官方 README](https://github.com/anysearch-ai/anysearch-mcp-server/blob/main/README.md)、[OpenAI 官方 MCP 說明](https://developers.openai.com/learn/docs-mcp)。
