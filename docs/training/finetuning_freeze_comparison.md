# 跨 group 全微調／部分凍結比較

兩個實驗由同一份 `outputs/pvhsd-dnanet-1ch-pretrain/best.pt` 初始化，使用 G02/G05/G08 作完整 group-disjoint validation，不接續照片切分微調的權重。目的為比較是否保留早期特徵有助於跨場景泛化。從頭訓練與低 LR 已由使用者表示做過，本輪不另新增這兩種實驗。

## 兩份參數檔

- `configs/pvhsd2025/finetune_solar_dnanet_1ch_group_full.yaml`：所有參數與訓練 BN 均可更新。
- `configs/pvhsd2025/finetune_solar_dnanet_1ch_group_freeze_early.yaml`：凍結 `conv0_0`、`conv1_0`、`conv2_0`，包含其 Conv、attention、BN affine 與 running statistics；保留 `conv3_0`、`conv4_0`、nested fusion、projections 與四個輸出頭可訓練。這是早期三個 encoder stage 的凍結，不是整個 encoder 凍結。

除實驗名稱與 `training.finetuning` 外，與既有 cuDNN group 微調設定相同：單通道 640×512、四個 supervision heads、SoftIoU、Adagrad LR=0.005、batch=4、seed=7、原 augmentation、max_epochs=1500、patience=50、以 F1_TIoU 選模、AMP=false。全微調與先前 `finetune_solar_dnanet_1ch_cudnn.yaml` 的訓練政策相同；如果已完成該對照，可沿用既有結果，不必為換實驗名稱重跑。

## 指令

從 repository 根目錄執行；不會由本次修改自動啟動。

```sh
conda activate solar-ir
export TRAINING_OUTPUT_ROOT=outputs

# 部分凍結
CUDA_VISIBLE_DEVICES=0 python -u -m research_training.cli run \
  --config configs/pvhsd2025/finetune_solar_dnanet_1ch_group_freeze_early.yaml \
  --device cuda --no-drive

# 全模型微調（同一張 GPU 上接續執行）
CUDA_VISIBLE_DEVICES=0 python -u -m research_training.cli run \
  --config configs/pvhsd2025/finetune_solar_dnanet_1ch_group_full.yaml \
  --device cuda --no-drive
```

若希望一次依序跑完兩者，可用 `for variant in freeze_early full; do ... --config configs/pvhsd2025/finetune_solar_dnanet_1ch_group_${variant}.yaml ... || break; done`。輸出分別在 `outputs/solar-ir-dnanet-pvhsd-group-freeze_early/`、`outputs/solar-ir-dnanet-pvhsd-group-full/`；同一指令重跑會自動續訓，請勿以不同政策共用同一 run 名稱。

## 實作與查核

可重用的 `src/solar_ir/finetuning.py` 接受 `mode: full` 或 `mode: freeze_modules`，後者搭配 module 名稱清單。model 建立後、optimizer 建立前套用政策；optimizer 只接收可訓練參數。BN 統計不受 requires_grad 控制，因此凍結 subtree 的 forward pre-hook 會在每次執行前重新套用 eval，避免 engine 每個 epoch 的 model.train() 重新打開 frozen BN 更新。未凍結部分照常維持訓練模式，state_dict key 不變，checkpoint 仍可 strict load。

終端與 checkpoint metadata 會記錄凍結模組、可訓練／凍結參數量及 frozen_running_stats。測試包含連續 train()/backward()/optimizer.step() 之後的 frozen 權重與 BN buffers 完全不變、未凍結權重和 BN 仍更新，以及兩個設定只改政策與 run name。全測試共 33 項通過。實際資料 build 也已確認可載入預訓練權重、使用相同 train/validation 清單；查核保存在 `outputs/verification/finetuning_policy/build_check.json`。

## SIRST 後續方向

原始 SIRST：<https://github.com/YimianDai/sirst>；作者亦提供 SIRST-V2：<https://github.com/YimianDai/open-sirst-v2>。這些是通用紅外小目標資料，適合測試小目標表徵，但不是太陽能板整片異常模組監督，不能保證遷移改善。尚未在本輪下載或啟動 SIRST 預訓練。

先前 PV-HSD 原框查核的面積中位數為整圖 0.02350%，約 77 像素（640×512），並非普遍大目標；同時是局部熱點 bbox 弱監督。若本輪對照仍失敗，只能支持「這些微調政策未改善目前跨 group 表現」，還不能排除標註語意、資料外觀、尺度與訓練策略的交互影響。下一步應以固定 group 驗證比較外部資料來源，避免單憑排行榜或同場景照片切分判定。
