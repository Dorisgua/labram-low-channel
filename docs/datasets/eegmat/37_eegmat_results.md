# EEGMAT Results

结论：当前 EEGMAT 完整 3 seeds 的 freeze_cnn AdaBrain-style 结果中，`37Nada_EEGMAT8_freeze_cnn` 高于 `37Aada_EEGMAT8_with_EEGMAT19_prototype_freeze_cnn`。单 seed 结果里，`37Oada_EEGMAT19_freeze_cnn` 的 Test Balanced Acc 最高，但目前只有 seed0，不能直接和 3 seeds 均值严格比较。

| 配置 | Seed | 最佳 Epoch | Val Acc | Val κ | Test Acc | Test Balanced Acc | Test κ | Test Weighted F1 | Log |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 37Omeanpool_EEGMAT19_freeze_cnn | 0 | 38 | 69.27% | 0.3854 | 70.00% | 70.00% | 0.4000 | 69.46% | `outputs/preexp37_eegmat19_mean_pool_cross_subject/run_logs/37Omeanpool.finetune_eegmat19_labrambase_freeze_cnn_20260721_074113.log` |
|  | 均值±SD（1 seed） | - | 69.27% | 0.3854 | 70.00% | 70.00% | 0.4000 | 69.46% | seed 0 |
| 37Nmeanpool_EEGMAT8_freeze_cnn | 0 | 0 | 63.54% | 0.2708 | 60.00% | 60.00% | 0.2000 | 57.33% | `outputs/preexp37_eegmat8_mean_pool_cross_subject/run_logs/37Nmeanpool.finetune_eegmat8_labrambase_freeze_cnn_20260721_074525.log` |
|  | 均值±SD（1 seed） | - | 63.54% | 0.2708 | 60.00% | 60.00% | 0.2000 | 57.33% | seed 0 |
| 37Ameanpool_EEGMAT8_with_EEGMAT19_prototype_freeze_cnn | 0 | 21 | 69.27% | 0.3854 | 70.00% | 70.00% | 0.4000 | 68.00% | `outputs/preexp37_eegmat8_with_eegmat19_mean_pool_cross_subject/run_logs/37Ameanpool.finetune_eegmat8_with_eegmat19_prototype_labrambase_freeze_cnn_20260721_075441.log` |
|  | 均值±SD（1 seed） | - | 69.27% | 0.3854 | 70.00% | 70.00% | 0.4000 | 68.00% | seed 0 |
| 37Omeanpool_EEGMAT19_full_finetune | 0 | 24 | 72.40% | 0.4479 | 69.17% | 69.17% | 0.3833 | 69.06% | `outputs/preexp37_eegmat19_mean_pool_full_finetune_cross_subject/run_logs/37Omeanpool.finetune_eegmat19_labrambase_full_finetune_20260721_082805.log` |
|  | 均值±SD（1 seed） | - | 72.40% | 0.4479 | 69.17% | 69.17% | 0.3833 | 69.06% | seed 0 |
| 37Oada_EEGMAT19_full_finetune | 0 | 11 | 80.73% | 0.6146 | 76.67% | 76.67% | 0.5333 | 76.66% | `outputs/preexp37_eegmat19_adabrain_full_finetune_cross_subject/run_logs/37Oada.finetune_eegmat19_labrambase_full_finetune_20260721_083446.log` |
|  | 均值±SD（1 seed） | - | 80.73% | 0.6146 | 76.67% | 76.67% | 0.5333 | 76.66% | seed 0 |
| 37Oada_EEGMAT19_freeze_cnn | 0 | 19 | 79.69% | 0.5938 | 83.33% | 83.33% | 0.6667 | 83.29% | `outputs/preexp37_eegmat19_adabrain_freeze_cnn_cross_subject/run_logs/37Oada.finetune_eegmat19_labrambase_freeze_cnn_20260721_112349.log` |
|  | 均值±SD（1 seed） | - | 79.69% | 0.5938 | 83.33% | 83.33% | 0.6667 | 83.29% | seed 0 |
| 37Nada_EEGMAT8_freeze_cnn | 0 | 16 | 75.00% | 0.5000 | 75.83% | 75.83% | 0.5167 | 75.70% | `outputs/preexp37_eegmat8_adabrain_freeze_cnn_cross_subject/run_logs/37Nada.finetune_eegmat8_labrambase_freeze_cnn_seed0_task01_20260721_164132.log` |
|  | 1 | 38 | 75.00% | 0.5000 | 82.50% | 82.50% | 0.6500 | 82.50% | `outputs/preexp37_eegmat8_adabrain_freeze_cnn_cross_subject/run_logs/37Nada.finetune_eegmat8_labrambase_freeze_cnn_seed1_task02_20260721_164632.log` |
|  | 2 | 11 | 75.52% | 0.5104 | 75.83% | 75.83% | 0.5167 | 75.70% | `outputs/preexp37_eegmat8_adabrain_freeze_cnn_cross_subject/run_logs/37Nada.finetune_eegmat8_labrambase_freeze_cnn_seed2_task03_20260721_165133.log` |
|  | 均值±SD（3 seeds） | - | 75.17±0.30% | 0.5035±0.0060 | 78.06±3.85% | 78.06±3.85% | 0.5611±0.0770 | 77.96±3.93% | seed 0/1/2 |
| 37Aada_EEGMAT8_with_EEGMAT19_prototype_freeze_cnn | 0 | 9 | 73.44% | 0.4688 | 75.83% | 75.83% | 0.5167 | 74.54% | `outputs/preexp37_eegmat8_with_eegmat19_adabrain_freeze_cnn_cross_subject/run_logs/37Aada.finetune_eegmat8_with_eegmat19_prototype_labrambase_freeze_cnn_seed0_task04_20260721_165633.log` |
|  | 1 | 8 | 73.44% | 0.4688 | 78.33% | 78.33% | 0.5667 | 78.18% | `outputs/preexp37_eegmat8_with_eegmat19_adabrain_freeze_cnn_cross_subject/run_logs/37Aada.finetune_eegmat8_with_eegmat19_prototype_labrambase_freeze_cnn_seed1_task05_20260721_170134.log` |
|  | 2 | 46 | 73.44% | 0.4688 | 74.17% | 74.17% | 0.4833 | 73.76% | `outputs/preexp37_eegmat8_with_eegmat19_adabrain_freeze_cnn_cross_subject/run_logs/37Aada.finetune_eegmat8_with_eegmat19_prototype_labrambase_freeze_cnn_seed2_task06_20260721_170634.log` |
|  | 均值±SD（3 seeds） | - | 73.44±0.00% | 0.4688±0.0000 | 76.11±2.10% | 76.11±2.10% | 0.5222±0.0419 | 75.49±2.36% | seed 0/1/2 |
