# PreExp37：EEGMAT 最小接入记录

## 任务

EEGMAT 是心算状态二分类数据集：

- 标签0：心算前的背景 EEG（`SubjectXX_1`）。
- 标签1：连续减法心算期间的 EEG（`SubjectXX_2`）。

当前实验不使用受试者的心算表现好/差分组标签。

## 数据

- SSD 路径：`/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/EEGMAT`
- 36名受试者：`Subject00`–`Subject35`
- 每名受试者30个样本，总计1080个 pickle
- 样本字段：`X`、`Y`
- 原始样本 shape：`[19, 2000]`
- 原始采样率：500Hz
- 每段时长：4秒

复制校验：HDD 源目录和 SSD 目标目录均为1080个文件、328,510,080字节。

## 19通道顺序

```text
FP1, FP2, F3, F4, F7, F8, T3, T4, C3, C4,
T5, T6, P3, P4, O1, O2, FZ, CZ, PZ
```

保留 preprocessing 和 AdaBrain-Bench manifest 使用的旧式 `T3/T4/T5/T6` 命名。

## Cross-subject 协议

复现现有 AdaBrain-Bench EEGMAT 协议：

- `Subject00`–`Subject31`：用于 train/validation。
- 每个训练受试者内部按类别、随机种子42进行80/20拆分。
- `Subject32`–`Subject35`：只用于 test。
- train：768，标签分布 `{0: 384, 1: 384}`。
- validation：192，标签分布 `{0: 96, 1: 96}`。
- test：120，标签分布 `{0: 60, 1: 60}`。

本仓库动态生成的三个 split 与现有 AdaBrain-Bench JSON 逐样本比较，symmetric difference 均为0。

## 输入处理

- 使用 Fourier resampling 从500Hz降采样到200Hz。
- 输出 shape：`[19, 800]`。
- 使用 train split 统计量做逐通道 z-score。
- validation/test 复用训练集统计量。
- 动态计算的 mean/std 与现有 AdaBrain-Bench train JSON 完全一致，最大绝对差为0。
- loader 已做 shape、标签、NaN/Inf 和通道顺序检查。

## 第一版模型设置

- 全部19通道。
- `completion_scope=none`。
- `classifier_mode=mean_pool`。
- 冻结 CNN/patch_embed。
- 训练 Transformer、`fc_norm` 和二分类 Linear head。
- LaBraM patch size：200。
- `num_t=4`。
- 最佳 checkpoint 指标：validation accuracy。

运行脚本：

```text
scripts/37Omeanpool.finetune_eegmat19_labrambase_freeze_cnn.sh
```

脚本默认2个 epoch，作为短跑检查。正式实验需要显式设置正式 epoch 数，例如：

```bash
GPU_IDS=0 EPOCHS=50 SEED=0 \
  bash scripts/37Omeanpool.finetune_eegmat19_labrambase_freeze_cnn.sh
```

## Smoke test

已完成 seed 0、1 epoch smoke test：

- 训练12 iterations，forward/backward 正常。
- validation 和 test 正常完成。
- checkpoint、`checkpoint-best`、`log.txt`、terminal log 和 TensorBoard 均已生成。
- smoke test 的 validation/test accuracy 均为50%；该结果只用于验证管线，不用于效果结论。

输出组：

```text
outputs/preexp37_eegmat19_mean_pool_cross_subject
```

下一步应先完成全通道 mean-pooling 正式 seed 0，再考虑 seeds 1/2、AdaBrain 分类头、少通道或 prototype。
