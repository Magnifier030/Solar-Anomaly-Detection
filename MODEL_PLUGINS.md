# 模型插件設計

模型不直接寫死在 `research_training` engine，也不讓 dataset 知道模型細節。每個模型只要
提供一個 `create_model(config) -> torch.nn.Module` factory，就能接入同一套訓練流程。

目前 baseline：

```yaml
task:
  model:
    factory: solar_ir.models.simple_unet:create_model
```

未來加入 DNANet 時，可以建立：

```text
src/solar_ir/models/dnanet.py
```

並將設定改成：

```yaml
task:
  model:
    factory: solar_ir.models.dnanet:create_model
    backbone: resnet_18
    deep_supervision: true
```

模型插件只負責：

- 建立模型
- 定義輸入與輸出 tensor shape
- 處理模型專屬參數

評估指標也採相同的插件概念。官方定位指標位於
`src/solar_ir/metrics/official.py`，目前 validation 的 selection metric 是
`F1_TIoU`；之後可以新增其他 metric module，或在 task 中替換 metric factory，而不需要
修改 DNANet。

以下功能應維持共用：

- `datasets/`：影像、Labelme polygon 與 group split
- `metrics/`：IoU、Dice 與未來的 TIoU
- `postprocess/`：mask 轉 polygon / bbox
- `submission/`：官方 JSON exporter 與格式檢查
- `research_training/`：checkpoint、resume、W&B、Drive 與 Colab 執行

因此更換模型時，實驗差異會被限制在 model plugin 與 YAML，不會影響資料切分或既有實驗結果。
