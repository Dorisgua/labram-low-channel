| 历史实验 | O/N/A | 输入/目标导联 | completion | classifier | 训练方式 | 有效 seeds | Val BAcc 均值 | Test Acc 均值 | Test BAcc 均值 | Test Kappa 均值 | Test F1 均值 |
|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|
| 35N | N | 23 | `none` | `mean_pool` | freeze CNN | 0/1/2 | 37.22% | 39.18% | 39.10% | 0.2381 | 39.40% |
| 35Ah | A | 23 → 62 | `seedv23_with_seedv62`，high pool | `mean_pool` | freeze CNN | 0/1/2 | 37.27% | 39.63% | 39.51% | 0.2438 | 40.03% |
| 35Al | A | 23 → 62 | `seedv23_with_seedv62`，low pool | `mean_pool` | freeze CNN | 0/1/2 | 36.89% | 39.46% | 39.44% | 0.2424 | 39.72% |
| 35O | O | 62 | `none` | `mean_pool` | freeze CNN | 0/1/2 | **39.51%** | 40.18% | 40.53% | 0.2537 | 40.45% |
| 35O | O | 62 | `none` | `mean_pool` | full finetune | 0/1/2 | 39.36% | **40.81%** | **41.08%** | **0.2610** | **41.28%** |
