# Siena 历史结果

以下结果来自 `eeg-main` 中已有运行日志的 `Best test metrics`。Siena 的 Accuracy 与 BAcc 差距较大，不能只报告 Accuracy。

| 历史实验 | O/N/A | 输入/目标导联 | completion | classifier | 训练方式 | Test Acc | Test BAcc | Test ROC-AUC | Test PR-AUC |
|---|---|---|---|---|---|---:|---:|---:|---:|
| 40Oada，siena29 | O | 29 | `none` | `adabrain_all_token` | full finetune | **97.66%** | **79.28%** | 0.9091 | 0.4337 |
| 40Oada，siena29 | O | 29 | `none` | `adabrain_all_token` | freeze CNN | 97.53% | 51.64% | — | — |
| 40Omeanpool，siena29 | O | 29 | `none` | `mean_pool` | freeze CNN | 97.69% | 50.00% | — | — |
| 40Aada，siena13→29 | A | 13 → 29 | `siena13_with_siena29` | `adabrain_all_token` | freeze CNN | 97.31% | 50.38% | — | — |
| 40Nada，siena13 | N | 13 | `none` | `adabrain_all_token` | freeze CNN | 97.01% | 49.65% | — | — |
