# Attention 历史结果

以下结果来自 `eeg-main` 中已有运行日志的 `Best test metrics`；不同 seed/任务参数不作均值合并。

| 历史实验 | O/N/A | 输入/目标导联 | completion | classifier | 训练方式 | Test Acc | Test BAcc | Test Kappa | Test F1 |
|---|---|---|---|---|---|---:|---:|---:|---:|
| 44Omeanpool，attention26，seed0 | O | 26 | `none` | `mean_pool` | full finetune | 81.48% | **76.89%** | 0.4465 | 82.87% |
| 44Omeanpool，attention26，seed1 | O | 26 | `none` | `mean_pool` | full finetune | 81.85% | 75.78% | 0.4411 | 83.03% |
| 44Omeanpool，attention26，seed2 | O | 26 | `none` | `mean_pool` | full finetune | 81.11% | 69.11% | 0.3598 | 81.64% |
| 44Omeanpool，attention10，seed0 | O | 10 | `none` | `mean_pool` | full finetune | 81.85% | 69.11% | 0.3691 | 82.16% |
| 44Omeanpool，attention10，seed1 | O | 10 | `none` | `mean_pool` | full finetune | 84.63% | 69.44% | 0.4127 | 84.14% |
| 44Omeanpool，attention10，seed2 | O | 10 | `none` | `mean_pool` | full finetune | 85.00% | 74.56% | 0.4763 | 85.22% |
| 44Omeanpool，attention26 | O | 26 | `none` | `mean_pool` | freeze CNN | 74.81% | 68.44% | 0.2892 | 77.11% |
| 44Omeanpool，attention26，seed1 | O | 26 | `none` | `mean_pool` | freeze CNN | 73.33% | 65.33% | 0.2421 | 75.71% |
| 44Omeanpool，attention26，seed2 | O | 26 | `none` | `mean_pool` | freeze CNN | 73.33% | 63.56% | 0.2202 | 75.50% |
| 44Nmeanpool，attention10 | N | 10 | `none` | `mean_pool` | freeze CNN | 73.70% | 63.78% | 0.2255 | 75.79% |
| 44Nmeanpool，attention10，seed1 | N | 10 | `none` | `mean_pool` | freeze CNN | 79.26% | 68.89% | 0.3360 | 80.32% |
| 44Nmeanpool，attention10，seed2 | N | 10 | `none` | `mean_pool` | freeze CNN | 77.41% | 71.33% | 0.3441 | 79.29% |
| 44Ameanpool，attention10→26 | A | 10 → 26 | `attention10_with_attention26` | `mean_pool` | freeze CNN | 76.30% | 64.00% | 0.2471 | 77.58% |
| 44Ameanpool，attention10→26，seed1 | A | 10 → 26 | `attention10_with_attention26` | `mean_pool` | freeze CNN | 78.52% | 61.33% | 0.2267 | 78.52% |
| 44Ameanpool，attention10→26，seed2 | A | 10 → 26 | `attention10_with_attention26` | `mean_pool` | freeze CNN | 80.00% | 65.78% | 0.3047 | 80.34% |
