# SEED Results

结论：在当前已完成的 SEED 实验中，完整 3 seeds 结果按 Test Balanced Acc 排序为 `36Nada_SEED23_freeze_cnn > 36Oada_SEED62_freeze_cnn > 36Omeanpool_SEED62_freeze_cnn`。`36Nmeanpool`、`36Ahmeanpool`、`36Almeanpool` 目前不是完整 3 seeds，只作为参考。

| 配置 | Seed | 最佳 Epoch | Val Acc | Val κ | Test Acc | Test Balanced Acc | Test κ | Test Weighted F1 | Log |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 36Oada_SEED62_freeze_cnn | 0 | 43 | 90.78% | 0.8617 | 54.98% | 54.56% | 0.3237 | 54.16% | `outputs/preexp36_seed_cross_subject/run_logs/36Oada.finetune_seed62_labrambase_freeze_cnn_seed0_task01_20260718_144900.log` |
|  | 1 | 47 | 90.64% | 0.8595 | 52.19% | 51.94% | 0.2831 | 52.51% | `outputs/preexp36_seed_cross_subject/run_logs/36Oada.finetune_seed62_labrambase_freeze_cnn_seed1_task02_20260718_144900.log` |
|  | 2 | 46 | 90.39% | 0.8559 | 54.08% | 53.73% | 0.3107 | 53.88% | `outputs/preexp36_seed_cross_subject/run_logs/36Oada.finetune_seed62_labrambase_freeze_cnn_seed2_task03_20260718_153337.log` |
|  | 均值±SD（3 seeds） | - | 90.60±0.20% | 0.8590±0.0029 | 53.75±1.42% | 53.41±1.34% | 0.3058±0.0207 | 53.52±0.89% | seed 0/1/2 |
| 36Nada_SEED23_freeze_cnn | 0 | 47 | 87.59% | 0.8138 | 55.28% | 54.81% | 0.3276 | 54.05% | `outputs/preexp36_seed23_cross_subject/run_logs/36Nada.finetune_seed23_labrambase_freeze_cnn_seed0_task04_20260718_153757.log` |
|  | 1 | 48 | 87.41% | 0.8112 | 56.17% | 55.75% | 0.3417 | 55.17% | `outputs/preexp36_seed23_cross_subject/run_logs/36Nada.finetune_seed23_labrambase_freeze_cnn_seed1_task05_20260718_161615.log` |
|  | 2 | 47 | 87.43% | 0.8114 | 55.64% | 55.15% | 0.3330 | 54.29% | `outputs/preexp36_seed23_cross_subject/run_logs/36Nada.finetune_seed23_labrambase_freeze_cnn_seed2_task06_20260718_162209.log` |
|  | 均值±SD（3 seeds） | - | 87.48±0.10% | 0.8121±0.0014 | 55.70±0.45% | 55.24±0.48% | 0.3341±0.0071 | 54.50±0.59% | seed 0/1/2 |
| 36Omeanpool_SEED62_freeze_cnn | 0 | 47 | 88.84% | 0.8325 | 52.68% | 52.38% | 0.2897 | 52.58% | `outputs/preexp36_seed62_mean_pool_cross_subject/run_logs/36Omeanpool.finetune_seed62_labrambase_freeze_cnn_seed0_task07_20260718_165912.log` |
|  | 1 | 46 | 89.43% | 0.8414 | 54.04% | 53.72% | 0.3103 | 53.76% | `outputs/preexp36_seed62_mean_pool_cross_subject/run_logs/36Omeanpool.finetune_seed62_labrambase_freeze_cnn_seed1_task08_20260718_170602.log` |
|  | 2 | 47 | 88.93% | 0.8340 | 51.71% | 51.40% | 0.2752 | 51.45% | `outputs/preexp36_seed62_mean_pool_cross_subject/run_logs/36Omeanpool.finetune_seed62_labrambase_freeze_cnn_seed2_task09_20260718_174250.log` |
|  | 均值±SD（3 seeds） | - | 89.07±0.32% | 0.8360±0.0048 | 52.81±1.17% | 52.50±1.17% | 0.2918±0.0176 | 52.60±1.16% | seed 0/1/2 |
| 36Nmeanpool_SEED23_freeze_cnn | 0 | 47 | 85.84% | 0.7876 | 53.41% | 53.00% | 0.3002 | 52.39% | `outputs/preexp36_seed23_mean_pool_cross_subject/run_logs/36Nmeanpool.finetune_seed23_labrambase_freeze_cnn_seed0_task10_20260718_175049.log` |
|  | 1 | 49 | 85.90% | 0.7885 | 55.72% | 55.31% | 0.3349 | 54.89% | `outputs/preexp36_seed23_mean_pool_cross_subject/run_logs/36Nmeanpool.finetune_seed23_labrambase_freeze_cnn_seed1_task11_20260718_183058.log` |
|  | 均值±SD（2 seeds） | - | 85.87±0.04% | 0.7880±0.0006 | 54.56±1.63% | 54.16±1.64% | 0.3175±0.0245 | 53.64±1.76% | seed 0/1 |
| 36Ahmeanpool_SEED23_with_SEED62_proto_high | 0 | 49 | 84.47% | 0.7670 | 55.61% | 55.20% | 0.3332 | 54.57% | `outputs/preexp36_seed23_with_seed62_mean_pool_high_cross_subject/run_logs/36Ahmeanpool.finetune_seed23_with_seed62_prototype_labrambase_freeze_cnn_seed0_task13_20260718_191456.log` |
|  | 2 | 48 | 84.89% | 0.7734 | 54.20% | 53.75% | 0.3116 | 52.87% | `outputs/preexp36_seed23_with_seed62_mean_pool_high_cross_subject/run_logs/36Ahmeanpool.finetune_seed23_with_seed62_prototype_labrambase_freeze_cnn_seed2_task15_20260718_200534.log` |
|  | 均值±SD（2 seeds） | - | 84.68±0.30% | 0.7702±0.0045 | 54.90±1.00% | 54.47±1.03% | 0.3224±0.0153 | 53.72±1.20% | seed 0/2 |
| 36Almeanpool_SEED23_with_SEED62_proto_low | 1 | 49 | 85.83% | 0.7874 | 55.09% | 54.67% | 0.3253 | 54.05% | `outputs/preexp36_seed23_with_seed62_mean_pool_low_cross_subject/run_logs/36Almeanpool.finetune_seed23_with_seed62_prototype_labrambase_freeze_cnn_seed1_task17_20260718_205152.log` |
|  | 均值±SD（1 seed） | - | 85.83% | 0.7874 | 55.09% | 54.67% | 0.3253 | 54.05% | seed 1 |
