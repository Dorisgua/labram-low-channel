# ERP Core 实验效果汇总

整理日期：2026-10-09。结果来源：`outputs/erpcore/` 下的训练日志。

## 统计口径

- 分类实验统一取验证集 Balanced Accuracy 最佳 epoch 对应的测试集指标，不取测试集指标的最大值。
- Epoch 从 0 开始。
- A/O/N 汇总为 seed 0、1、2 的均值 ± 样本标准差；D stage2 和 recon_only stage2 各只有 seed 0。
- Acc、BA（Balanced Accuracy）、F1 均以百分数表示；F1 为 weighted F1，Kappa 为 Cohen's Kappa。
- Stage1 按验证集 Loss 最小值选取最佳 epoch。

## 整体比较

| 实验 | Test Acc ↑ | Test BA ↑ | Test F1 ↑ | Test Kappa ↑ |
|---|---:|---:|---:|---:|
| A | **60.74 ± 0.61** | 39.45 ± 1.54 | 55.69 ± 0.60 | 0.4987 ± 0.0070 |
| O | 60.18 ± 0.65 | **45.66 ± 0.36** | **59.21 ± 0.57** | **0.5097 ± 0.0072** |
| N | 59.65 ± 1.48 | 38.94 ± 1.63 | 55.67 ± 0.27 | 0.4906 ± 0.0116 |
| D stage2 | 56.85 | 37.43 | 52.88 | 0.4566 |
| recon_only stage2 | 60.44 | 37.98 | 55.76 | 0.4926 |

按主要选模指标 BA，O 表现最好，且三个 seed 的 BA 标准差最小。O 的平均 BA 比 A 高 6.22 个百分点，比 N 高 6.72 个百分点。A 的平均 Acc 最高。D 和 recon_only 仅有单个 seed，无法评估跨 seed 波动。

## 各 seed 详细结果

| 实验 | Seed | 最佳 epoch | Val BA (%) | Test Acc (%) | Test BA (%) | Test F1 (%) | Test Kappa |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 0 | 1 | 38.46 | 61.38 | 40.29 | 56.27 | 0.5065 |
| A | 1 | 4 | 36.49 | 60.16 | 40.38 | 55.74 | 0.4967 |
| A | 2 | 1 | 38.58 | 60.68 | 37.67 | 55.07 | 0.4930 |
| O | 0 | 26 | 44.95 | 60.55 | 45.39 | 59.42 | 0.5129 |
| O | 1 | 8 | 46.47 | 59.43 | **46.07** | 58.57 | 0.5015 |
| O | 2 | 28 | 46.10 | 60.57 | 45.52 | 59.64 | 0.5147 |
| N | 0 | 1 | 38.67 | 60.59 | 39.73 | 55.62 | 0.5008 |
| N | 1 | 1 | 38.73 | 60.41 | 37.07 | 55.43 | 0.4931 |
| N | 2 | 6 | 37.46 | 57.95 | 40.03 | 55.96 | 0.4780 |
| D stage2 | 0 | 4 | 37.78 | 56.85 | 37.43 | 52.88 | 0.4566 |
| recon_only stage2 | 0 | 1 | 37.77 | 60.44 | 37.98 | 55.76 | 0.4926 |

## Stage1 损失

Stage1 日志记录损失，没有下游分类指标。Test Loss 为验证集最佳 epoch 对应的测试损失。

| Stage1 目录 / 运行 | 最佳 epoch | Val Loss ↓ | Test Loss | Test 重建损失 `loss_missing` |
|---|---:|---:|---:|---:|
| `erp_core_D_stage1` 根目录日志 | 46 | 6.1032 | 6.6171 | 0.004078 |
| `seed0_20260929_102523_63633` | 15 | 184.5376 | 207.3026 | 0.008694 |
| `seed0_20260929_103212_81175` | 11 | 185.9034 | 207.9663 | 0.256010 |
| `recon_only/seed0_20260929_233627_2214963/stage1` | 49 | 2.1976 | 2.0220 | 0.004044 |

根目录 `erp_core_D_stage1/log.txt` 包含多次运行的追加记录，此处取最后一次运行的最佳 epoch。不同 Stage1 配置的总损失项和权重可能不同，总 Loss 不宜直接横向比较。

## 日志来源

以下路径均相对于仓库根目录：

- A/O/N seed 0：`outputs/erpcore/aon/seed0_20260930_000705_2308507/{A,O,N}/log.txt`
- A/O/N seed 1：`outputs/erpcore/aon/seed1_20260930_000949_2318130/{A,O,N}/log.txt`
- A/O/N seed 2：`outputs/erpcore/aon/seed2_20260930_000954_2319386/{A,O,N}/log.txt`
- D stage2：`outputs/erpcore/erp_core_D_stage2/checkpoints/erp_core_D_stage2_seed0_20260929_113250/log.txt`
- D stage1：`outputs/erpcore/erp_core_D_stage1/log.txt`，以及两个 seed0 运行目录中的 `log.txt`
- recon_only：`outputs/erpcore/recon_only/seed0_20260929_233627_2214963/{stage1,stage2}/log.txt`
