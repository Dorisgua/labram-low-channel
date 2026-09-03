# 三阶段解耦、恢复与 Transformer 修改方案

## 目标

```text
阶段一：全 28 导联学习 subject/task 解耦
阶段二：用解耦信息恢复冻结 CNN 的缺失导联 target
阶段三：把恢复后的完整 tokens 接入 Transformer 做分类
```

先只在 ERP CORE 的 `12 -> 28` 导联设置上修改。

## 阶段一：全导联解耦

输入和输出：

```text
x_full [B,28,A,200]
  -> frozen patch_embed/CNN
h_full [B,28,A,200]
  -> shared/subject/task encoder
z_sub, z_task, d_sub, d_task
```

修改位置：

- `modeling_dynamic_stage1.py`
  - 保留 `fullchannel_disentangle` 分支。
  - 只生成 subject/task 解耦特征，不做缺失导联恢复。
- `losses_dynamic.py`
  - 本阶段只使用 subject/task contrastive loss。
  - `missing/reg/permute/correction contrastive` 权重全部设为 `0`。
- `scripts/erp_core/full-disentangle/stage1.sh`
  - 使用 `CHANNEL_SUBSET=erpcore28`。
  - 增加 `--fullchannel_disentangle`。
  - CNN 始终冻结。

运行：

```bash
bash scripts/erp_core/full-disentangle/stage1.sh
```

输出：第一阶段解耦器的 `checkpoint-best.pth`。

## 阶段二：恢复冻结 CNN target

输入和监督：

```text
x_obs [B,12,A,200] + missing prototypes
  -> 加载第一阶段的 disentangler
  -> z_sub/z_task 或 d_sub/d_task
  -> trainable recovery/cross-attention
h_pred_miss [B,16,A,200]

target = frozen CNN(x_full) 的缺失导联特征
h_miss_target [B,16,A,200]
```

需要修改：

- `modeling_dynamic_stage1.py`
  - 新增 `freeze_disentangler()`。
  - 冻结 `shared_encoder`、`subject_encoder`、`task_encoder` 及对应 norm。
  - 不冻结 recovery/cross-attention。
- `run_dynamic_stage1.py`
  - 新增 `--freeze_disentangler` 参数并调用上面的方法。
- 新增 `scripts/erp_core/full-disentangle/stage2.sh`
  - `FINETUNE` 指向阶段一的 `checkpoint-best.pth`。
  - 使用 `CHANNEL_SUBSET=erpcore12`。
  - 不传 `--fullchannel_disentangle`。
  - 传入 `--freeze_disentangler`，只训练恢复模块。

本阶段不开 Transformer 分类，只优化：

```text
loss_recovery = SmoothL1(h_pred_miss, h_miss_target)
```

判断恢复是否有效：

```text
MSE(pred, target) < MSE(prototype, target)
Cosine(pred, target) > Cosine(prototype, target)
```

输出：恢复模块的 `checkpoint-best.pth`。

## 阶段三：接入 Transformer

数据流：

```text
真实 12 导联 CNN tokens + 恢复出的 16 导联 tokens
  -> 按 28 导联顺序排列
  -> LaBraM Transformer
  -> classifier
```

需要修改：

- 新增 `scripts/erp_core/full-disentangle/stage3.sh`。
- 复用现有 `scripts/erp_core/D/stage2.sh`。
- `STAGE1_CHECKPOINT` 改为阶段二的 `checkpoint-best.pth`。
- 冻结 CNN、disentangler 和 recovery，只训练 Transformer 与分类头。

最后比较：

| 实验 | Transformer 输入 |
|---|---|
| O | 真实 28 导联 |
| N | 真实 12 导联 |
| A | 12 导联 + 静态 prototype |
| D | 12 导联 + 解耦信息恢复 token |

只有阶段二的恢复指标优于 prototype，并且阶段三的 `D > A`，才能说明解耦信息对恢复和下游分类都有帮助。
