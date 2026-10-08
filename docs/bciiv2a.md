# BCI-IV-2A 实验结果

数据协议：CBraMod 跨被试划分。以下结果均为实际 seed 0 的测试集 balanced accuracy（BAcc）；分类 checkpoint 按验证集 BAcc 选择，报告该 epoch 对应的测试结果，未按测试集选点。

## A/O/N/D 对照

| 数据集 | A：静态补全 | O：全导联 | N：少导联 | D：动态补全 |
|---|---:|---:|---:|---:|
| BCI-IV-2A（CBraMod 划分） | 47.40% | 47.92% | 46.35% | 46.18% |

D 列使用 2026 年 9 月 29 日原始配置结果，未将后续消融实验替换为主结果。A/O/N 为 2026 年 10 月 8 日完成的运行。各方法只有单种子结果，不能据此判断统计显著性；比较前还应核对实际训练配置一致性。

| 方法 | 运行目录（相对 outputs/bciiv2a_cbramod） |
|---|---|
| A | `bciiv2a_cbramod_A_freeze_cnn/checkpoints/bciiv2a_cbramod_A_freeze_cnn_seed0_20261008_153137` |
| O | `bciiv2a_cbramod_O_freeze_cnn/checkpoints/bciiv2a_cbramod_O_freeze_cnn_seed0_20261008_153525` |
| N | `bciiv2a_cbramod_N_freeze_cnn/checkpoints/bciiv2a_cbramod_N_freeze_cnn_seed0_20261008_153549` |
| D | `D_pipeline/seed0_20260929_154546_823179` |

## D 的配置对照与消融

以下五次运行均为实际 seed 0，Stage1 为 20 轮、Stage2 为 50 轮，均在 2026 年 9 月 30 日新增 corrector 时间编码之前启动。

| 运行目录（相对 D_pipeline） | 配置 | 测试 BAcc |
|---|---|---:|
| `seed0_20260929_154546_823179` | 原始 D 配置 | 46.18% |
| `seed0_20260930_110510_716019` | 原配置重跑：重建＋subject 对比＋交换约束 | 47.31% |
| `seed0_20260930_111038_742795` | 重建＋subject/task 双分支对比＋交换约束 | 47.31% |
| `seed0_20260930_144901_1214809` | 关闭对比和交换，以重建为主 | 47.48% |
| `seed0_20260930_145642_1240189` | 关闭重建和交换，保留双分支对比及正则 | 44.18% |

## 47.48% 对应的设置

运行目录：`outputs/bciiv2a_cbramod/D_pipeline/seed0_20260930_144901_1214809`。

| 参数 | 设置 |
|---|---:|
| 缺失导联重建 `missing_weight` | 50.0 |
| subject summary 对比 `subject_summary_contra_weight` | 0 |
| task summary 对比 `task_summary_contra_weight` | 0 |
| subject correction 对比 `subject_correction_contra_weight` | 0 |
| task correction 对比 `task_correction_contra_weight` | 0 |
| subject 交换约束 `permute_sub_weight` | 0 |
| task 交换约束 `permute_task_weight` | 0 |
| 实际 seed | 0 |
| Stage1 / Stage2 训练轮数 | 20 / 50 |
| 新增 corrector 时间编码 | 本次 Stage1 在该修改前启动 |

该实验保留动态补全结构，关闭对比损失与交换约束，主要通过缺失导联重建训练；这里称为“以重建为主”，不表示已确认所有正则项均关闭。

Stage2 按验证集 BAcc 选中 epoch 30（从 0 开始，即第 31 轮）：验证集 BAcc 为 46.27%，对应测试集 BAcc 为 47.48%。

原始指标：验证集 BAcc = 0.46267361111111116，测试集 BAcc = 0.47482638888888884。

## 结果来源

- [BCI 实验输出目录](../outputs/bciiv2a_cbramod/)
- [47.48% 实验的 Stage1 日志](../outputs/bciiv2a_cbramod/D_pipeline/seed0_20260930_144901_1214809/stage1/log.txt)
- [47.48% 实验的 Stage2 日志](../outputs/bciiv2a_cbramod/D_pipeline/seed0_20260930_144901_1214809/stage2/log.txt)
- [原始运行的解耦诊断](disentanglement_comparison/BCI解耦诊断_20260930.md)
