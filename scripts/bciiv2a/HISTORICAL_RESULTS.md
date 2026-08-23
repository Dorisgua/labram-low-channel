| 历史实验 | O/N/A | 输入/目标导联 | completion | classifier | 训练方式 | 最佳 epoch | Val BAcc | Test Acc | Test BAcc | Test Kappa | Test F1 |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|
| 33Aada | A | 13 → 22 | `bciiv2a13_with_bciiv2a22`，high pool | `adabrain_all_token` | freeze CNN | 8 | 49.54% | 49.23% | 49.23% | 0.3230 | 49.09% |
| 33Arealada | A | 13 → 22 | `bciiv2a13_with_bciiv2a22`，high pool | `adabrain_all_token`，scope=`real` | freeze CNN | 9 | 53.32% | 54.71% | 54.71% | 0.3961 | 54.58% |
| 33Nada | N | 13 | `none` | `adabrain_all_token` | freeze CNN | 21 | 52.70% | 52.62% | 52.62% | 0.3683 | 52.49% |
| 33Nada | N | 13 | `none` | `adabrain_all_token` | full finetune | 9 | 52.93% | 52.31% | 52.31% | 0.3642 | 52.29% |
| 33Oada | O | 22 | `none` | `adabrain_all_token` | freeze CNN | 36 | 57.56% | 55.48% | 55.48% | 0.4064 | 55.50% |
| 33Oada | O | 22 | `none` | `adabrain_all_token` | full finetune | 36 | **63.12%** | **57.95%** | **57.95%** | **0.4393** | **57.92%** |
