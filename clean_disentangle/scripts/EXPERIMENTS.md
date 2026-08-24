# A/B/C 正式实验对照表

| 实验 | Wrapper | 输入 / Scope | Missing Fill | Output Base | 普通组合 Prediction | Target | Subject Contrastive | Task Contrastive | Subject Swap Recon | Task Swap Recon | Ordinary Recon | Missing MSE |
|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|
| A — Full + D only | `train_full_d_only.sh` | `x_full` / `FULL` | N/A | `NONE` | `D_sub_full + D_task_full` | `H_full` | 1 | 1 | 1 | 1 | 0 | 0 |
| B — Full + Prototype + D | `train_full_prototype_d.sh` | `x_full` / `FULL` | N/A | `PROTOTYPE` | `P_full + D_sub_full + D_task_full` | `H_full` | 1 | 1 | 1 | 1 | 0 | 0 |
| C — Missing + Prototype + D | `train_missing_prototype_d.sh` | `x_obs` / `MISSING` | `PROTOTYPE` | `PROTOTYPE` | `P_miss + D_sub_miss + D_task_miss` | `H_miss` | 1 | 1 | 1 | 1 | 0 | 0 |

三个实验统一使用 `ComponentMode=IDENTITY`、`CompositionMode=SUM`、`SAMPLING=cslpae`。默认训练参数相同：seed 0、50 epochs、batch size 64、temperature 0.2、AdamW、LR 5e-4、min LR 1e-6、weight decay 0.05、warmup 5 epochs。

表中的“普通组合 Prediction”用于说明各实验的 base 与 D 组合关系；由于 `RECON_WEIGHT=0`，不会对普通 self-combination 计算 MSE。训练中的 reconstruction 来自 subject/task 两个 swap reconstruction loss。

运行命令：

```bash
bash clean_disentangle/scripts/train_full_d_only.sh
bash clean_disentangle/scripts/train_full_prototype_d.sh
bash clean_disentangle/scripts/train_missing_prototype_d.sh
```
