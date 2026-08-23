# EEGMAT 历史结果

以下结果来自 `eeg-main` 中已有运行日志的 `Best test metrics`。

| 历史实验 | O/N/A | 输入/目标导联 | completion | classifier | 训练方式 | Test Acc | Test BAcc | Test Kappa | Test F1 |
|---|---|---|---|---|---|---:|---:|---:|---:|
| 37Oada，eegmat19 | O | 19 | `none` | `adabrain_all_token` | freeze CNN | **83.33%** | **83.33%** | 0.6667 | 83.29% |
| 37Oada，eegmat19 | O | 19 | `none` | `adabrain_all_token` | full finetune | 76.67% | 76.67% | 0.5333 | 76.66% |
| 37Omeanpool，eegmat19 | O | 19 | `none` | `mean_pool` | full finetune | 69.17% | 69.17% | 0.3833 | 69.06% |
| 37Omeanpool，eegmat19 | O | 19 | `none` | `mean_pool` | freeze CNN | 70.00% | 70.00% | 0.4000 | 69.46% |
| 37Nada，eegmat8，seed0 | N | 8 | `none` | `adabrain_all_token` | freeze CNN | 75.83% | 75.83% | 0.5167 | 75.70% |
| 37Nada，eegmat8，seed1 | N | 8 | `none` | `adabrain_all_token` | freeze CNN | 82.50% | 82.50% | 0.6500 | 82.50% |
| 37Nada，eegmat8，seed2 | N | 8 | `none` | `adabrain_all_token` | freeze CNN | 75.83% | 75.83% | 0.5167 | 75.70% |
| 37Nmeanpool，eegmat8 | N | 8 | `none` | `mean_pool` | freeze CNN | 60.00% | 60.00% | 0.2000 | 57.33% |
| 37Aada，eegmat8→19，seed0 | A | 8 → 19 | `eegmat8_with_eegmat19` | `adabrain_all_token` | freeze CNN | 75.83% | 75.83% | 0.5167 | 75.70% |
| 37Aada，eegmat8→19，seed1 | A | 8 → 19 | `eegmat8_with_eegmat19` | `adabrain_all_token` | freeze CNN | 78.33% | 78.33% | 0.5667 | 78.33% |
| 37Aada，eegmat8→19，seed2 | A | 8 → 19 | `eegmat8_with_eegmat19` | `adabrain_all_token` | freeze CNN | 74.17% | 74.17% | 0.4833 | 74.04% |
| 37Ameanpool，eegmat8→19 | A | 8 → 19 | `eegmat8_with_eegmat19` | `mean_pool` | freeze CNN | 70.00% | 70.00% | 0.4000 | 68.00% |
