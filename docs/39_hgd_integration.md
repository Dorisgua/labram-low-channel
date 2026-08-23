# HGD 第一版接入说明

## 数据与通道

- 数据集：Schirrmeister2017 High Gamma Dataset（HGD）。
- 任务：右手、左手、静息、双脚四分类。
- 原始文件：14 名被试，每人包含官方 train 和 test EDF。
- 原始信号：500 Hz；每个事件 annotation 持续 4 秒。
- EDF 包含 128 EEG、2 EOG 和 3 EMG。
- 第一版保留与 LaBraM `standard_1020` 位置表严格对应的 78 个 EEG 通道。
- `HGD_78_CHANNELS` 定义在 `Channels_definition.py`，并写入预处理 manifest 的 `selected_channels`。

## 离线预处理

```bash
python dataset_maker/make_HGD.py --dry-run
python dataset_maker/make_HGD.py
```

每个 annotation 按以下顺序处理：

1. 提取事件 onset 后 0--4 秒的完整 128 EEG。
2. 转换到 μV，使用官方 `max(abs(trial)) < 800 μV` 规则判断伪迹。
3. trial 合格后选取 78 个 LaBraM 兼容通道。
4. 使用 `resample_poly(up=2, down=5)` 从 500 Hz 降到 200 Hz。
5. 保存 `[trial, 78, 800]` float32 EEG 和 int64 标签数组。

默认输出：

```text
HGD/processed_data_4s_200hz/
├── manifest.json
├── preprocess_config.json
└── subjects/
    ├── sub01_train_eeg.npy
    ├── sub01_train_labels.npy
    ├── sub01_test_eeg.npy
    ├── sub01_test_labels.npy
    └── ...
```

第一版不额外执行高通或 exponential running standardization。训练读取器默认使用仅由训练 trial 计算的通道级 z-score。

## 数据划分

第一版保留官方记录边界：

- test：全部官方 test EDF 的清洁 trial。
- train/validation：每名被试的官方 train 清洁 trial 内，按类别进行固定 seed 42 的 80%/20% 划分。

14 名被试都会出现在 train、validation 和 test，因此该设置不是 cross-subject。它是一个合并所有被试、同时保留官方 test session 的基线。后续 cross-subject 实验应使用独立名称，不能与本结果混报。

## 第一版训练

```bash
bash scripts/hgd/39Omeanpool.finetune_hgd78_labrambase_freeze_cnn.sh
```

配置为 78 通道、4 秒、mean pooling、冻结 CNN、无 prototype、seed 0，默认 2 epochs 用于 smoke test。
