| 历史实验 | O/N/A | 输入/目标导联 | completion | classifier | 训练方式 | 有效 seeds | Val BAcc 均值 | Test Acc 均值 | Test BAcc 均值 | Test Kappa 均值 | Test F1 均值 |
|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|
| 17N | N | 13 | `none` | `mean_pool` | freeze CNN，50 epochs | 0 | 66.60% | 79.87% | 60.62% | 0.5958 | 80.34% |
| 17Ah | A | 13 → 23 | `tuev13_with_tuev23`，high pool | `mean_pool` | freeze CNN，50 epochs | 0/1/2 | 63.60% | 81.30% | 61.48% | 0.6319 | 81.74% |
| 17O | O | 23 | `none` | `mean_pool` | freeze CNN，50 epochs | 0/1/2 | **68.03%** | **81.68%** | **63.90%** | **0.6391** | **82.01%** |
