| 历史实验 | O/N/A | 输入/目标导联 | completion | classifier | 训练方式 | 有效 seeds | Val BAcc 均值 | Test Acc 均值 | Test BAcc 均值 | Test Kappa 均值 | Test F1 均值 |
|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|
| 36Oada | O | 62 | `none` | `adabrain_all_token`，scope=`all` | freeze CNN，50 epochs | 0/1/2 | **90.57%** | 53.75% | 53.41% | 0.3058 | 53.52% |
| 36Nada | N | 23 | `none` | `adabrain_all_token`，scope=`all` | freeze CNN，50 epochs | 0/1/2 | 87.42% | **55.70%** | **55.24%** | **0.3341** | **54.50%** |
| 36Omeanpool | O | 62 | `none` | `mean_pool` | freeze CNN，50 epochs | 0/1/2 | 89.02% | 52.81% | 52.50% | 0.2918 | 52.60% |
| 36Nmeanpool | N | 23 | `none` | `mean_pool` | freeze CNN，50 epochs | 0/1 | 85.81% | 54.56% | 54.16% | 0.3175 | 53.64% |
| 36Ahmeanpool | A | 23 → 62 | `seed23_with_seed62`，high pool | `mean_pool` | freeze CNN，50 epochs | 0/2 | 84.61% | 54.90% | 54.47% | 0.3224 | 53.72% |
| 36Almeanpool | A | 23 → 62 | `seed23_with_seed62`，low pool | `mean_pool` | freeze CNN，50 epochs | 1 | 85.76% | 55.09% | 54.67% | 0.3253 | 54.05% |
