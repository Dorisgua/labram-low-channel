# LaBraM 少导联 EEG 特征补全

本项目研究：在仅有部分 EEG 导联的情况下，通过特征空间补全，减少缺失导联带来的分类性能损失。

## 1. 方法与对照

| 方法 | 输入方式 |
|---|---|
| O：全导联 | 真实全导联 EEG |
| N：少导联 | 仅使用真实观测导联 |
| A：静态补全 | 观测特征 + 训练集平均 Prototype |
| D：动态补全 | 观测特征 + 根据当前样本修正后的 Prototype |

D 的补全公式：

预测缺失特征 = Prototype + subject 修正量 + task 修正量

subject/task 是两个分支的设计目标，是否实现有效解耦仍需实验验证。

### 两阶段训练

| 阶段 | 处理流程 | 训练内容 |
|---|---|---|
| Stage1 | 少导联 EEG → 冻结 CNN → 动态补全模块 | 通过重建、对比、交换及正则约束训练补全模块，各项权重可配置 |
| Stage2 | 观测特征与补全特征 → LaBraM Transformer → 分类头 | 冻结 CNN 和补全模块，训练 Transformer 与分类头 |

Stage1 使用真实全导联特征提供监督。推理时，补全模块只读取观测导联。

当前正式基线不使用额外 corrector 时间编码。LaBraM 主 Transformer 原有的时间编码保留。

## 2. 当前 AOND 主结果

以下均为实际 seed0，报告验证集选模对应 epoch 的测试成绩。

ERP-Core、TUEV、SEED、SEED-V、SHU 分类训练为 30 轮；
BCI-IV-2A 分类训练为 50 轮，D Stage1 为 20 轮。

| 数据集 | 测试指标 | O：全导联 | N：少导联 | A：静态补全 | D：动态补全 | D−N（百分点） | D−A（百分点） |
|---|---|---:|---:|---:|---:|---:|---:|
| ERP-Core | BAcc（%） | 45.39 | 39.73 | 40.29 | 37.43 | −2.30 | −2.86 |
| TUEV | BAcc（%） | 58.20 | 62.79 | 55.73 | 57.32 | −5.47 | +1.59 |
| SEED | BAcc（%） | 53.70 | 55.17 | 54.57 | 54.90 | −0.27 | +0.33 |
| SEED-V | Acc（%） | 42.04 | 38.89 | 38.44 | 38.20 | −0.69 | −0.24 |
| SHU | BAcc（%） | 57.93 | 56.46 | 54.88 | 56.11 | −0.35 | +1.23 |
| BCI-IV-2A（CBraMod 划分） | BAcc（%） | 47.92 | 46.35 | 47.40 | 46.18 | −0.17 | −1.22 |

D−N 为正表示动态补全优于直接使用少导联；
D−A 为正表示动态补全优于静态补全。

| 数据集 | best checkpoint 选择依据 | Batch size | LR | Classifier mode |
|---|---|---:|---:|---|
| BCI-IV-2a | Val Balanced Accuracy | 64 | `5e-4` | `adabrain_all_token`，scope=`real` |
| ERP-Core | Val Balanced Accuracy | 64 | `5e-4` | `adabrain_all_token`，scope=`real` |
| TUEV | Val Balanced Accuracy | 64 | `5e-4` | `adabrain_all_token`，scope=`real` |
| PhysioNet | Val Balanced Accuracy | 64 | `5e-4` | `adabrain_all_token`，scope=`real` |
| SEED | Val Balanced Accuracy | 64 | `5e-4` | `adabrain_all_token`，scope=`real` |
| SEED-V | Val Accuracy | 64 | `5e-4` | `adabrain_all_token`，scope=`real` |
| EEGMAT | Val Balanced Accuracy | 64 | `5e-4` | `adabrain_all_token`，scope=`real` |
| HGD | Val Balanced Accuracy | 64 | `5e-4` | `adabrain_all_token`，scope=`real` |
| Siena | Val Balanced Accuracy | 64 | `5e-4` | `adabrain_all_token`，scope=`real` |
| Attention | Val Balanced Accuracy | 64 | `5e-4` | `adabrain_all_token`，scope=`real` |
| AAD | Val Balanced Accuracy | 64 | `5e-4` | `adabrain_all_token`，scope=`real` |
| FACED | Val Balanced Accuracy | 64 | `5e-4` | `adabrain_all_token`，scope=`real` |
| Zuo2025 | Val Balanced Accuracy | 64 | `5e-4` | `adabrain_all_token`，scope=`real` |

