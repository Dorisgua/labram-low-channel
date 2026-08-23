# Siena 第一版接入说明

## 数据语义

- 任务：癫痫发作检测二分类，非发作 0、发作 1。
- 病人：PN00、PN01、PN03、PN05、PN06、PN07、PN09、PN10、PN11、PN12、PN13、PN14、PN16、PN17。
- 第一阶段来源：复制数据中的 `Siene/processed_data/PNxx/*.pkl`。
- 每个源样本：10 秒、512 Hz、29 通道、`[29, 5120]`、float64、单位 V。
- 源 pickle 已由 AdaBrain-Bench 完成 0.1--75 Hz 带通、50 Hz 陷波、固定通道选择、标准 10 秒切窗和发作窗口增强。
- 总计 51,349 个窗口：非发作 50,665，发作 684。

类别极不平衡，因此第一版使用 balanced accuracy 选择最佳 checkpoint，并同时报告 PR-AUC、ROC-AUC 和 accuracy。不能仅凭 accuracy 判断效果。

## LaBraM 离线转换

```bash
python dataset_maker/make_Siena.py --dry-run
python dataset_maker/make_Siena.py
```

第二阶段仅做 LaBraM 所需的表示转换：

1. 保留完整 10 秒及原标签，不任取前 4 秒。
2. `resample_poly(up=25, down=64)` 将 512 Hz 降到 200 Hz。
3. 将 V 转换为 μV，并以 float32 保存。
4. 每名病人输出 `[samples, 29, 2000]` EEG、标签和逐 trial 通道统计。
5. 生成 `manifest.json`，记录标签、通道和 benchmark 划分。

默认输出为：

```text
Siene/processed_data_10s_200hz/
├── manifest.json
├── preprocess_config.json
└── subjects/
    ├── PN00_eeg.npy
    ├── PN00_labels.npy
    ├── PN00_stats.npy
    └── ...
```

## Cross-subject benchmark

严格复现本地 AdaBrain-Bench 的 Siena JSON：

- train/validation 病人：PN00--PN14 中实际存在的 12 名病人。
- 每名训练病人内按标签使用全局 seed 42 做 80%/20% 划分。
- test 病人：PN16、PN17，完全不参与训练、验证或归一化统计。
- 预期 train：38,128（负 37,652，正 476）。
- 预期 validation：9,542（负 9,419，正 123）。
- 预期 test：3,679（负 3,594，正 85）。

注意：AdaBrain 的 seizure-enhanced 窗口使用 5 秒步长，同一病人 train/validation 随机划分可能包含时间上重叠的增强窗。最终 test 是病人隔离的，因此 test 不存在这种跨病人泄漏；validation 仍应按 benchmark 局限解释。

## 第一版训练

```bash
bash scripts/40Omeanpool.finetune_siena29_labrambase_freeze_cnn.sh
```

配置：29 通道、10 秒、200 Hz、mean pooling、冻结 CNN、无 prototype、seed 0，默认 2 epochs 做 smoke test。
