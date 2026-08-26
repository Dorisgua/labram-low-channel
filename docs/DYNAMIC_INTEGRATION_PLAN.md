# LaBraM-unified-AON Dynamic 集成计划

## 1. 目标

以 `LaBraM-unified-AON-dynamic` 为工作版本，在不破坏 AON 原有 N/O/A 实验的前提下，逐步加入 cleanup 中的 Dynamic Prototype 方法。

最终希望所有实验共享：

- 同一套 LaBraM backbone；
- 同一套数据划分、导联定义和 `input_scale`；
- 同一套分类头、评价指标和 checkpoint 选择规则；
- 同一套输出和 README 记录方式。

这里的“一个框架”不要求 Stage 1 和 Stage 2 只有一个 Python 文件，而是要求它们共享统一的数据、配置、模型接口和评价协议。

## 2. 实验定义

| 实验 | 输入 | 补全 | 训练方式 | 作用 |
|---|---|---|---|---|
| N-End2End | 12 导联 | 无 | CNN、Transformer、分类头全部训练 | 主要 baseline |
| O-End2End | 28 导联 | 无 | CNN、Transformer、分类头全部训练 | 完整输入上限 |
| A-End2End | 12 → 28 | 静态 Prototype | 沿用已有 AON 设置 | 静态补全对照 |
| D-2stage | 12 → 28 | Stage 1 动态补全 | Stage 1 冻结后进行分类 | 主要研究方法 |

`freeze_cnn` 是独立的训练策略，不能和 N/O/A/D 的输入含义混淆。

## 3. 代码来源和迁移关系

cleanup 中的代码只作为 Dynamic 实现参考，不能整体复制成第二套框架：

| cleanup 文件 | AON 中的目标位置 | 用途 |
|---|---|---|
| `modeling_general_prototype_cslp.py` | `modeling_dynamic_stage1.py` | Stage 1 corrector 和 forward |
| `engine_for_preexp14_cslp.py` | `losses_dynamic.py`、`engine_for_dynamic_stage1.py` | 多项 loss 和训练循环 |
| `run_preexp16_erpcore_cslp.py` | `run_dynamic_stage1.py` | ERP Core Stage 1 启动入口 |
| cleanup 的数据字段 | AON `data_processor/erpcore.py` | 统一 batch 接口 |

不直接复制 cleanup 的 Stage 2 入口。Stage 2 最终应接回 AON 的 `run_class_finetuning.py`。

## 4. 目标目录结构

```text
LaBraM-unified-AON-dynamic/
├── run_class_finetuning.py          # N/O/A，以及最终的 Dynamic Stage 2
├── run_dynamic_stage1.py            # Dynamic Stage 1 训练
├── modeling_finetune.py             # 原有 LaBraM backbone 和分类模型
├── modeling_dynamic_stage1.py       # Dynamic Stage 1 模型
├── losses_dynamic.py                # Stage 1 各项 loss
├── engine_for_dynamic_stage1.py     # Stage 1 训练、验证、日志【为什么一定要一个新的engine呢？】【stage2不需要新的engine么】
├── data_processor/erpcore.py        # 统一 ERP Core 数据接口
├── scripts/erp_core/【还是保持N的文件夹里有两个，1个freeze cnn 一个full finetune】
│   ├── N...sh
│   ├── O...sh
│   ├── A...sh
│   ├── D_stage1...sh
│   └── D_stage2...sh
└── docs/
    ├── DYNAMIC_INTEGRATION_PLAN.md
    └── ERP_CORE_EXPERIMENTS.md
```

## 5. 统一数据接口

ERP Core dataset 统一返回字典：

```python
{
    "x": ...,          # 当前实验的主输入
    "x_obs": ...,      # 观测导联
    "x_full": ...,     # 完整导联，仅供 Stage 1 生成重建目标
    "label": ...,      # 下游任务标签
    "subject": ...,    # subject 标签，供 auxiliary/CSLP 使用
    "task": ...,       # task 标签，供 auxiliary/CSLP 使用
}
```

各实验只读取需要的字段：

| 阶段 | 允许使用的字段 |
|---|---|
| N | `x_obs`、`label` |
| O | `x_full`、`label` |
| A | `x_obs`、`label`、Prototype |
| Dynamic Stage 1 | `x_obs`、`x_full`、`subject`、`task`，必要时使用 `label` |
| Dynamic Stage 2 | `x_obs`、`label`、冻结的 Stage 1 |

Dynamic Stage 2 绝不能使用真实的 `x_full`，否则会造成信息泄漏。
【我觉得太麻烦了，不如都输出，只是接收端拿走需要的】

## 6. Stage 1 模型和 loss 的职责

### 6.1 `modeling_dynamic_stage1.py`

模型 forward 只产生中间结果，不组合总 loss：

```text
x_obs
→ patch embedding
→ h_obs
→ prototype tokens
→ corrector
→ d_sub / d_task
→ h_pred_miss
→ auxiliary outputs
```

建议返回：

```text
h_obs
p_miss
d_sub
d_task
h_pred_miss
z_sub
z_task
shared_feature
sub_logits
task_logits
shared_sub_logits
shared_task_logits
h_miss_target【这是什么意思：取出的“缺失导联 latent target”】
```

### 6.2 `losses_dynamic.py`

负责计算并加权：

- missing reconstruction MSE；
- correction regularization；
- subject/task auxiliary loss；
- shared auxiliary loss；
- subject/task contrastive loss；
- close/permute loss；
- CSLP loss；【这是什么意思？】
- `total_loss`。

loss 权重和 CSLP ramp 不应写进模型 forward。【ramp是什么？】

### 6.3 `engine_for_dynamic_stage1.py`

负责：

```text
batch
→ model.forward()
→ compute_dynamic_losses()
→ total_loss.backward()
→ optimizer.step()
→ 记录每一项 loss
→ 保存 checkpoint
```

第一版可以先启用 `missing` 和 `reg`，其余 loss 保留参数并默认关闭；基础链路稳定后再逐项打开 auxiliary 和 CSLP。

## 7. 分阶段实施步骤

### 阶段 0：保护现有基准

- 保留当前 AON N/O/A 代码和结果；
- 在复制版本上工作；
- 不改变现有脚本默认参数；
- 记录当前 branch、commit 和结果。

验收标准：N/O/A 的代码入口和结果解释不受 Dynamic 改动影响。

### 阶段 1：统一 ERP Core dataset

- 让 ERP Core dataset 支持完整 batch 字段；
- 保留原有分类脚本的兼容性；
- 检查 12/28 导联顺序；
- 检查 train/val/test subject split；
- 检查 `input_scale` 和归一化。

验收标准：N/O/A 仍能运行，batch shape 和旧版一致。

### 阶段 2：接入 Stage 1 最小版本

- 加入 Dynamic Stage 1 model；
- 先实现 `missing MSE + reg`；
- 保存 checkpoint 和完整 config；
- 先只跑 ERP Core seed 0。

验收标准：能得到有限的 `h_pred_miss`，`missing_mse` 正常下降，checkpoint 可以重新加载。

### 阶段 3：逐项加入其他 loss

顺序建议为：

```text
missing MSE
→ reg
→ subject/task auxiliary
→ shared auxiliary
→ contrastive
→ close/permute
→ CSLP
```

每次只打开一类 loss，并记录 loss 权重、验证结果和 reconstruction MSE。

### 阶段 4：接入 Stage 2 Dynamic

- 在 `run_class_finetuning.py` 增加 Dynamic 模式；
- 加载 Stage 1 checkpoint/config；
- 冻结 Stage 1 参数并切换 `eval()`；
- 使用 `x_obs` 生成补全 token；
- 接入 AON 原有 Transformer 和分类头；
- 不使用真实 `x_full`。

验收标准：Dynamic Stage 2 能和 N/A 使用相同分类流程，并能记录 Test Acc、Test BAcc、κ、Weighted F1。

### 阶段 5：公平性核对

逐项比较 AON N/O/A 与 cleanup Dynamic：

- 数据划分；
- seed；
- `input_scale`；
- LaBraM checkpoint；
- channel order；
- 分类头；
- Transformer 训练范围；
- CNN 和 Stage 1 冻结策略；
- best Val BAcc / best Val Acc；
- Test 指标读取方式。

只有差异能够解释时，才进行方法结论比较。

### 阶段 6：补齐多 seed 和结果表

- 先验证 seed 0；
- 再跑 seed 1/2；
- 分别整理 best Val BAcc 和 best Val Acc；
- 先计算各实验组 Mean ± SD；
- 再按照 Test Acc、Test BAcc 从高到低排序；
- 在每张结果表下写对应 bash 和日志目录。

## 8. 输出和配置要求

每次实验至少保存：

```text
run.log
config.json
metrics.csv 或 log.txt
checkpoint-best.pth
checkpoint-last.pth
```

配置中记录：

- dataset、seed、split；
- observed/target channel names 和顺序；
- `input_scale`；
- LaBraM checkpoint；
- Prototype checkpoint；
- completion mode；
- Stage 1 checkpoint；
- loss weights；
- best metric；
- trainable/frozen parameter 范围。

## 9. 当前不做的事情

- 不立即合并 Git-Diff-clean-upload 的整个目录；
- 不立即改所有数据集；
- 不把 Stage 1 和 Stage 2 强行写进同一个训练循环；
- 不改变现有 N/O/A 的历史结果；
- 不用 Test 集最高分代替验证集 checkpoint 选择；
- 不把数据、checkpoint 和大规模 outputs 提交到代码仓库。

## 10. 完成标准

满足以下条件后，才认为 Dynamic 已经正式进入 AON：

1. AON 原有 N/O/A 可以正常运行；
2. Stage 1 可以独立训练并保存可追溯 checkpoint；
3. Stage 2 可以加载冻结的 Stage 1；
4. Dynamic Stage 2 只使用观测导联；
5. Dynamic 与 N/O/A 使用统一的分类和评价流程；
6. seed 0 结果能够复现或解释；
7. seed 1/2 能够使用同一套脚本运行；
8. README 能说明每组实验的目的、命令、结果和结论。

