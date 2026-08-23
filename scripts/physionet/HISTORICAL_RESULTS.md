| 历史实验 | O/N/A | 输入/目标导联 | completion | classifier | 训练方式 | 有效 seeds | Val BAcc 均值 | Test Acc 均值 | Test BAcc 均值 | Test Kappa 均值 | Test F1 均值 |
|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|
| 34Neegfm-23 | N | 23 | `none` | `adabrain_all_token`，scope=`all` | freeze CNN，30 epochs | 0/1/2 | 48.79% | 51.49% | 51.50% | 0.3532 | 51.17% |
| 34Aeegfm-23 | A | 23 → 64 | `physionet23_with_physionet64`，high pool | `adabrain_all_token`，scope=`real` | freeze CNN，30 epochs | 0/1/2 | 47.52% | 51.22% | 51.23% | 0.3497 | 51.26% |
| 34Neegfm-32 | N | 32 | `none` | `adabrain_all_token`，scope=`all` | freeze CNN，30 epochs | 0/1/2 | 52.00% | 54.97% | 55.00% | 0.3998 | 54.65% |
| 34Aeegfm-32 | A | 32 → 64 | `physionet32_with_physionet64`，high pool | `adabrain_all_token`，scope=`real` | freeze CNN，30 epochs | 0/1 | 51.41% | 55.26% | 55.27% | 0.4034 | 55.62% |
| 34Oeegfm | O | 64 | `none` | `adabrain_all_token`，scope=`all` | freeze CNN，30 epochs | 0/1/2 | **58.04%** | **63.22%** | **63.23%** | **0.5095** | **63.28%** |
| 34Oeegfm | O | 64 | `none` | `adabrain_all_token`，scope=`all` | full finetune，50 epochs | 0 | 57.96% | 63.07% | 63.09% | 0.5077 | 63.11% |
