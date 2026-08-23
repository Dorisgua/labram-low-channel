# HGD 历史结果

以下结果来自 `eeg-main` 中已有运行日志的 `Best test metrics`。

| 历史实验 | O/N/A | 输入/目标导联 | completion | classifier | 训练方式 | Test Acc | Test BAcc | Test Kappa | Test F1 |
|---|---|---|---|---|---|---:|---:|---:|---:|
| 39Omeanpool，hgd78 | O | 78 | `none` | `mean_pool` | freeze CNN | **74.35%** | **74.35%** | 0.6580 | 74.42% |
| 39Nmeanpool，hgd20 | N | 20 | `none` | `mean_pool` | freeze CNN | 71.49% | 71.49% | 0.6199 | 71.58% |
| 39Ameanpool，hgd20→78 | A | 20 → 78 | `hgd20_with_hgd78` | `mean_pool` | freeze CNN | 71.40% | 71.40% | 0.6187 | 71.72% |
| 39Ameanpool，smoke | A | 20 → 78 | `hgd20_with_hgd78` | `mean_pool` | freeze CNN，1 epoch smoke | 25.02% | 25.00% | 0.0000 | 10.02% |
