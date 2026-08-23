# FACED 历史结果

以下结果来自 `eeg-main` 中已有运行日志的 `Best test metrics`。

| 历史实验 | O/N/A | 输入/目标导联 | completion | classifier | 训练方式 | Test Acc | Test BAcc | Test Kappa | Test F1 |
|---|---|---|---|---|---|---:|---:|---:|---:|
| 42Oada，faced32 | O | 32 | `none` | `adabrain_mlp_token` | full finetune | **54.09%** | **54.19%** | 0.4820 | 54.39% |
| 42Oada，faced32 | O | 32 | `none` | `adabrain_mlp_token` | freeze CNN | 29.61% | 29.36% | 0.2056 | 29.39% |
| 42Omeanpool，faced32，run1 | O | 32 | `none` | `mean_pool` | freeze CNN | 14.29% | 11.11% | 0.0000 | 3.57% |
| 42Omeanpool，faced32，run2 | O | 32 | `none` | `mean_pool` | freeze CNN | 18.12% | 17.12% | 0.0719 | 16.20% |
