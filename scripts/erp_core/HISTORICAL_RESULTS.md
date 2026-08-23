# ERP-Core 历史结果

当前 `LaBraM-unified-AON/scripts/erp_core/` 没有 Bash 实验脚本。本文件只记录 `eeg-main` sibling 仓库中日志可核对的历史结果，不能视为 unified-AON 已运行结果。

| 历史实验 | O/N/A | 输入/目标导联 | completion | classifier | 训练方式 | Test Acc | Test BAcc | Test Kappa | Test F1 |
|---|---|---|---|---|---|---:|---:|---:|---:|
| 45Oada，erpcore28 | O | 28 | `none` | `adabrain_all_token`，scope=`all` | full finetune | **65.29%** | **47.83%** | 0.5656 | 62.85% |
| 45Nada，erpcore12 | N | 12 | `none` | `adabrain_all_token`，scope=`all` | full finetune | 63.31% | 42.20% | 0.5326 | 58.07% |
| 45Aada，erpcore12→28 | A | 12 → 28 | prototype completion | `adabrain_all_token`，scope=`all` | freeze CNN | 60.05% | 39.37% | 0.4959 | 56.62% |
| 45Nmeanpool，erpcore12，seed0 | N | 12 | `none` | `mean_pool` | full finetune | 59.12% | 43.99% | 0.4948 | 58.41% |
| 45Nmeanpool，erpcore12，seed1 | N | 12 | `none` | `mean_pool` | full finetune | 62.53% | 44.38% | 0.5326 | 58.76% |
| 45Nmeanpool，erpcore12，seed2 | N | 12 | `none` | `mean_pool` | full finetune | 60.77% | 43.30% | 0.5096 | 58.26% |
| 45Omeanpool，erpcore28 | O | 28 | `none` | `mean_pool` | full finetune | 62.46% | 47.30% | 0.5374 | 61.56% |
| 45Omeanpool，erpcore28 | O | 28 | `none` | `mean_pool` | freeze CNN | 57.26% | 43.03% | 0.4787 | 56.91% |
| 45Omeanpool，erpcore28，seed0 | O | 28 | `none` | `mean_pool` | freeze CNN | 56.29% | 42.78% | 0.4720 | 56.14% |
| 45Omeanpool，erpcore28，seed1 | O | 28 | `none` | `mean_pool` | freeze CNN | 54.01% | 41.59% | 0.4475 | 54.15% |
| 45Omeanpool，erpcore28，seed2 | O | 28 | `none` | `mean_pool` | freeze CNN | 57.25% | 43.01% | 0.4785 | 56.89% |
| 45Ahmeanpool，erpcore12→28，seed0 | A | 12 → 28 | prototype completion | `mean_pool` | freeze CNN | 54.71% | 38.48% | 0.4414 | 53.80% |
| 45Ahmeanpool，erpcore12→28，seed1 | A | 12 → 28 | prototype completion | `mean_pool` | freeze CNN | 53.52% | 34.06% | 0.4185 | 49.89% |
| 45Ahmeanpool，erpcore12→28，seed2 | A | 12 → 28 | prototype completion | `mean_pool` | freeze CNN | 56.33% | 39.03% | 0.4605 | 54.20% |
