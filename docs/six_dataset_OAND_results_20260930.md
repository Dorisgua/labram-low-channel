# 六个数据集 O/A/N/D 阶段性结果（2026-09-30）

## 统计口径

- O：全部真实导联；N：少量真实导联；A：少导联加静态 prototype；D：Stage1 学习动态补全，Stage2 训练分类器。
- 分类 checkpoint 按各运行的验证集 `best_metric` 选择，再读取该 epoch 的测试集 BAcc；测试集不参与选点。O/A/N 的“均值 ± 标准差”来自实际 `--seed 0/1/2`，标准差为样本标准差。
- D 目前主要只有实际 seed 0。表中单次结果不能与 O/A/N 的三种子均值当作同等证据。BAcc 单位为百分比，MSE 越小越好。

## 分类效果总表：测试集 BAcc（%）

| 数据集 | O：全导联 | N：少导联 | A：静态补全 | D：动态补全 | D 的实际 seed |
|---|---:|---:|---:|---:|---|
| [ERP-Core](#erp-core-和-bci-iv-2a) | — | — | — | 37.43 | 0 |
| [TUEV](#tuev) | 61.23 ± 2.64 | **62.45 ± 1.12** | 58.28 ± 2.21 | 52.12† | 0 |
| [SEED](#seed) | 53.37 ± 0.74 | 55.04 ± 0.74 | **55.57 ± 0.99** | 54.90 | 0 |
| [SEED-V](#seed-v) | **41.95 ± 0.05** | 38.04 ± 0.32 | 38.09 ± 0.28 | 37.31 | 0 |
| [BCI-IV-2A（CBraMod 划分）](#erp-core-和-bci-iv-2a) | — | — | — | 46.18 | 0 |
| [SHU](#shu) | **58.07 ± 0.74** | 55.45 ± 1.23 | 54.78 ± 0.72 | 56.11 | 0 |

“—”表示当前 `outputs` 中没有同协议的 O/A/N 日志，不能填零。D 列对有多次 seed 0 重跑的数据集取最早完整运行；其他重跑见下表。† TUEV D 是较早的 Stage2 运行，见[结果限制](#结果限制)。

## SHU

| 方法 | seed 0 测试 BAcc | seed 1 测试 BAcc | seed 2 测试 BAcc | 三种子均值 |
|---|---:|---:|---:|---:|
| O（32 导联） | 57.93 | 58.87 | 57.42 | **58.07** |
| N（13 导联） | 56.46 | 55.82 | 54.08 | 55.45 |
| A（13→32 静态补全） | 54.88 | 55.45 | 54.02 | 54.78 |
| D（13→32 动态补全） | 56.11 | — | — | — |

D 的另外两次重跑目录分别叫 `seed1`、`seed2`，但命令实际都是 `--seed 0`；测试 BAcc 分别为 **56.61%** 和 **54.76%**，不能计入 seed 1/2。三次 Stage1 的验证集缺失导联 MSE 分别为 0.001684、0.001691、0.001691。真实 seed 0 的第一次 D 比 A 高 1.23 个百分点，比 N 低 0.35 个百分点。

来源：[O/A/N 三种子日志](../outputs/shu/aon/)、[D 三次运行](../outputs/shu/D_pipeline/)。

## SEED

| 方法 | seed 0 测试 BAcc | seed 1 测试 BAcc | seed 2 测试 BAcc | 三种子均值 |
|---|---:|---:|---:|---:|
| O（62 导联） | 53.70 | 52.52 | 53.88 | 53.37 |
| N（23 导联） | 55.17 | 55.70 | 54.24 | 55.04 |
| A（23→62 静态补全） | 54.57 | 56.56 | 55.57 | **55.57** |
| D（23→62 动态补全） | 54.90 | — | — | — |

SEED D 有第二次实际 seed 0 重跑，测试 BAcc 为 **55.74%**；两次 Stage1 最佳验证集缺失导联 MSE 均约 0.002481。来源：[O/A/N](../outputs/seed/aon/)、[D](../outputs/seed/D_pipeline/)。

## SEED-V

| 方法 | seed 0 测试 BAcc | seed 1 测试 BAcc | seed 2 测试 BAcc | 三种子均值 |
|---|---:|---:|---:|---:|
| O（62 导联） | 41.93 | 42.01 | 41.91 | **41.95** |
| N（23 导联） | 38.20 | 38.25 | 37.68 | 38.04 |
| A（23→62 静态补全） | 37.80 | 38.36 | 38.12 | 38.09 |
| D（23→62 动态补全） | 37.31 | — | — | — |

SEED-V 的分类 checkpoint 按验证集 accuracy 选择，表中仍统一报告该 epoch 的测试 BAcc。D Stage1 最佳验证集缺失导联 MSE 为 **0.006832**。来源：[O/A/N](../outputs/seedv/aon/)、[D](../outputs/seedv/D_pipeline/)。

## TUEV

| 方法 | seed 0 测试 BAcc | seed 1 测试 BAcc | seed 2 测试 BAcc | 三种子均值 |
|---|---:|---:|---:|---:|
| O（23 导联） | 58.20 | 63.06 | 62.43 | 61.23 |
| N（13 导联） | 62.79 | 63.36 | 61.21 | **62.45** |
| A（13→23 静态补全） | 55.73 | 59.56 | 59.56 | 58.28 |
| D（13→23 旧运行） | 52.12† | — | — | — |

TUEV 的旧 D 分类 checkpoint 按验证集 Cohen’s kappa 选择，对应测试 BAcc **52.12%**；其配对 Stage1 最佳验证集缺失导联 MSE 为 **0.004485**。来源：[O/A/N](../outputs/tuev/aon/)、[Stage1](../outputs/tuev/tuev_D_stage1/seed0_20260929_113930_282524/log.txt)、[Stage2](../outputs/tuev/tuev_D_stage2/checkpoints/tuev_D_stage2_seed0_20260929_142126/log.txt)。

## ERP-Core 和 BCI-IV-2A

| 数据集与运行 | 验证集 BAcc | 测试集 BAcc | 配对 Stage1 验证集缺失导联 MSE | Stage2 选择指标 |
|---|---:|---:|---:|---|
| ERP-Core D（12→28） | 37.78 | **37.43** | 0.009664 | 验证集 BAcc |
| BCI-IV-2A CBraMod D（13→22） | 45.66 | **46.18** | 0.002128 | 验证集 BAcc |

ERP-Core 的 Stage2 命令实际加载 `erp_core_D_stage1/seed0_20260929_102523_63633/checkpoint-best.pth`；上表 MSE 只取这一配对 Stage1，不能用其他 ERP-Core Stage1 运行替代。来源：[ERP-Core Stage1](../outputs/erpcore/erp_core_D_stage1/seed0_20260929_102523_63633/log.txt)、[ERP-Core Stage2](../outputs/erpcore/erp_core_D_stage2/checkpoints/erp_core_D_stage2_seed0_20260929_113250/log.txt)、[BCI-IV-2A D](../outputs/bciiv2a_cbramod/D_pipeline/seed0_20260929_154546_823179/)。

## 结果限制

1. SHU 的 D 目录名 `seed1`、`seed2` 与实际训练参数不符；这三次是 seed 0 重跑，不能当作三种子均值。
2. TUEV D Stage2 于 2026-09-29 14:21 启动，早于此 checkout 后续补入 `tuev13_with_tuev23` 动态补全前向分支。旧运行没有记录可验证的代码提交，因此 **52.12% 只作为历史记录，不作为已验证的动态补全分类效果**。
3. Stage1 缺失导联 MSE 与 Stage2 分类 BAcc 是不同指标；MSE 下降不等于分类提升。ERP-Core、BCI-IV-2A 缺少本目录内同协议的 O/A/N 结果，无法计算相对基线增益。
4. 文档只统计此 checkout 的 `outputs` 中可核验的日志；`recon_only` 消融和历史归档没有并入主表。
