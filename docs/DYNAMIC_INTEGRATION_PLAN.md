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
├── engine_for_finetuning.py         # N/O/A 和 Dynamic Stage 2 分类 engine
├── engine_for_dynamic_stage1.py     # Dynamic Stage 1 训练和重建验证
├── data_processor/erpcore.py        # 统一 ERP Core 数据接口
├── scripts/erp_core/
│   ├── N/
│   │   ├── freeze_cnn.sh
│   │   └── full_finetune.sh
│   ├── O/
│   │   ├── freeze_cnn.sh
│   │   └── full_finetune.sh
│   ├── A/
│   │   ├── freeze_cnn.sh
│   │  
│   └── D/
│       ├── stage1.sh
│       └── stage2.sh
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

统一 dataset 可以返回全部字段；由不同训练阶段的接收端选择需要的字段。为了避免信息泄漏，Dynamic Stage 2 的前向和 loss 代码只能读取 `x_obs`，不能读取 `x_full`。`x_full` 只用于 Stage 1 生成 `h_miss_target`，或用于离线审计。

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
h_obs                  # 必需：观测导联的 latent token
p_miss                 # 必需：缺失导联的 prototype token
d_sub                  # 必需：subject correction
d_task                 # 必需：task correction
h_pred_miss            # 必需：预测的缺失导联 latent
h_miss_target          # 必需：真实完整 EEG 的缺失导联 latent target

z_sub                  # 仅在 subject auxiliary/CSLP 开启时需要
z_task                 # 仅在 task auxiliary/CSLP 开启时需要
shared_feature         # 仅在 shared auxiliary 开启时需要
sub_logits             # 仅在 sub_aux_weight > 0 时需要
task_logits            # 仅在 task_aux_weight > 0 时需要
shared_sub_logits      # 仅在 shared_sub_aux_weight > 0 时需要
shared_task_logits     # 仅在 shared_task_aux_weight > 0 时需要
```

`h_miss_target` 是真实 `x_full` 经过同一个 patch embedding 后，取出缺失导联位置得到的 latent target。它只用于计算缺失导联重建误差，不是分类标签，也不能作为 Dynamic Stage 2 的输入。

### 6.2 `losses_dynamic.py`

负责计算并加权：

- missing reconstruction MSE；
- correction regularization；
- subject/task auxiliary loss；
- shared auxiliary loss；
- subject/task contrastive loss；
- close/permute loss；
- CSLP loss；CSLP 是 **Contrastive Split-Latent Permutation**，即“对比学习 + 拆分 latent + 置换重建”。它约束 `d_sub` 主要表示 subject 信息、`d_task` 主要表示 task 信息，并通过交换这两类 correction 检查表示是否可组合；
- `total_loss`。

核心公式（符号与 cleanup 实现对应）如下：

```text
h_pred_miss = p_miss + d_sub + d_task

L_missing = MSE(h_pred_miss, h_miss_target)
L_reg     = mean(d_sub^2) + mean(d_task^2)

L_sub_aux    = CE(sub_logits, subject)
L_task_aux   = CE(task_logits, task)
L_shared_aux = CE(shared_sub_logits, subject)
              + CE(shared_task_logits, task)

L_sub_contra_s  = InfoNCE(z_sub^i, z_sub^j)     # 同 subject 配对
L_task_contra_t = InfoNCE(z_task^i, z_task^j)    # 同 task 配对
L_d_sub_close   = InfoNCE(d_sub^i, d_sub^j)
L_d_task_close  = InfoNCE(d_task^i, d_task^j)

L_latent_permute_s = 1/2 * [
    MSE(p_miss^i + d_sub^j + d_task^i, h_miss_target^i)
  + MSE(p_miss^j + d_sub^i + d_task^j, h_miss_target^j)]

L_latent_permute_t = 1/2 * [
    MSE(p_miss^i + d_sub^i + d_task^j, h_miss_target^i)
  + MSE(p_miss^j + d_sub^j + d_task^i, h_miss_target^j)]

L_total = Σ_k w_k L_k
```

其中 `i、j` 是按照相同 subject 或相同 task 组成的样本对；实际启用哪些项由对应的 loss weight 决定。当前 cleanup 中 close/permute 的具体实现和权重以迁移后的测试结果为准，不能只根据公式推断效果。

CSLP-AE 原始论文命令启用的核心 loss 是四项：`sub_contra_s`、`task_contra_t`、`latent_permute_s` 和 `latent_permute_t`，分别对应 subject/task latent 的对比损失和 subject/task latent 置换重建损失。原始命令没有启用 `recon`、`restored_permute`、`content`、监督交叉熵或 quadruplet permutation。当前 cleanup 版本在这四项基础上又提供了 `missing MSE`、`reg`、`d_sub_close`、`d_task_close`、`d_sub_permute` 和 `d_task_permute`，因此不能把 cleanup 的完整 loss 组合直接等同于原始 CSLP-AE 的四项组合。

loss 权重和 CSLP ramp 不应写进模型 forward。`ramp` 指“渐进启用”：训练前若干 epoch 将 CSLP 相关 loss 的系数从 0 逐步增加到配置值，避免模型一开始同时受到分类、重建和对比约束的强烈扰动。例如 `start_epoch=5、ramp_epochs=10` 时，第 5 个 epoch 开始启用，第 14 个 epoch 达到完整权重；若 `ramp_epochs <= 0`，则直接使用完整权重。

### 6.3 分类与 Dynamic Stage 1 engine

采用两个独立的 engine 文件：保留原有 `engine_for_finetuning.py`，避免影响 N/O/A，并新增 `engine_for_dynamic_stage1.py` 承担 Stage 1 的字典 batch、多项 loss 和重建验证。两个文件共享训练协议和现有工具，但不强制共享同一个 epoch 函数。

#### 推荐目录职责

```text
engine_for_finetuning.py
├── train_class_batch()
├── train_one_epoch()
└── evaluate()
    用于：N/O/A + Dynamic Stage 2

engine_for_dynamic_stage1.py
├── move_dynamic_batch()
├── train_dynamic_stage1_one_epoch()
└── evaluate_dynamic_stage1()
    用于：Dynamic Stage 1

losses_dynamic.py
├── compute_dynamic_losses()
├── missing / reg / auxiliary
└── contrastive / permute / CSLP

run_dynamic_stage1.py
└── 组织 dataset、model、optimizer、engine 和 checkpoint
```

调用关系：

```text
N/O/A
run_class_finetuning.py
→ engine_for_finetuning.train_one_epoch()
→ model.forward()
→ classification loss

Dynamic Stage 1
run_dynamic_stage1.py
→ engine_for_dynamic_stage1.train_dynamic_stage1_one_epoch()
→ model.forward_stage1()
→ losses_dynamic.compute_dynamic_losses()
→ total_loss

Dynamic Stage 2
run_class_finetuning.py
→ engine_for_finetuning.train_one_epoch()
→ DynamicModel.forward()
→ classification loss
```

两个 engine 需要保持一致的公共训练行为包括 AMP、梯度累积、LR/weight decay 调度、梯度裁剪、NaN/Inf 检查、日志格式和分布式同步。Stage 1 独立处理字典 batch、Dynamic forward、多项 loss 和 reconstruction 指标；Stage 2 继续使用原分类 engine，并且只读取 `x_obs` 和 `label`。

第一版 Stage 1 先启用 `missing` 和 `reg`，其余 loss 保留参数并默认关闭；基础链路稳定后再逐项打开 auxiliary 和 CSLP。

【那么原来的labram的forward入口放在哪里？】

#### 回答

原来的 LaBraM `forward()` 继续放在 `modeling_finetune.py` 的 `NeuralTransformer` 中，供 N/O/A 使用。简化伪代码如下：

```python
class NeuralTransformer:
    def forward(self, x, input_chans=None):
        feature = self.forward_features(x, input_chans)
        return self.head(feature)

    def forward_features(self, x, input_chans=None):
        h = self.patch_embed(x)
        if self.completion_scope != "none":
            h = static_prototype_complete(h)  # A-End2End
        h = add_cls_channel_time_embedding(h)
        return self.transformer(h)
```

Dynamic 使用独立入口，但复用同一个 LaBraM backbone：

```python
class DynamicModel:
    def forward_stage1(self, batch):
        h_obs = self.backbone.patch_embed(batch["x_obs"])
        h_pred_miss = self.corrector(h_obs, self.prototype)
        h_miss_target = make_target(batch["x_full"])
        return h_pred_miss, h_miss_target

    def forward(self, x_obs):  # Stage 2 分类入口
        h_obs = self.backbone.patch_embed(x_obs)
        h_pred_miss = self.frozen_corrector(h_obs, self.prototype)
        h_complete = merge(h_obs, h_pred_miss)
        feature = self.backbone.forward_from_tokens(h_complete)【forward_from_tokens是什么意思？】
        return self.classification_head(feature)
```

其中 `forward_from_tokens()` 是计划新增的 token-level helper，表示跳过第二次 `patch_embed`，直接复用原 LaBraM 的 embedding、Transformer blocks 和 normalization。原来的 `NeuralTransformer.forward()` 不需要删除或替换。

【然后 engine 里怎么做？】

#### 回答

engine 保留原来的分类训练函数，再增加一个 Stage 1 专用函数。两条路径分别调用不同入口：

```python
# Dynamic Stage 1
def train_dynamic_stage1_one_epoch(model, data_loader, optimizer):
    model.train()

    for batch in data_loader:
        batch = move_batch_to_device(batch)

        with autocast():#【】
            outputs = model.forward_stage1(batch)
            losses = compute_dynamic_losses(outputs, batch)
            loss = losses["total_loss"]

        backward_and_optimizer_step(loss)
        log_each_dynamic_loss(losses)
```

Stage 2 继续使用原来的分类 engine：

```python
# N/O/A 和 Dynamic Stage 2
def train_class_batch(model, samples, labels, criterion):
    logits = model(samples)  # 自动调用 model.forward()
    loss = criterion(logits, labels)
    return loss, logits
```

因此整体关系是：

```text
Stage 1 engine → forward_stage1() → dynamic losses
Stage 2 engine → forward()        → classification loss
```

Stage 1 batch 使用 `x_obs`、`x_full`、`subject` 和 `task`；Stage 2 只取 `x_obs` 和 `label`。当前分类 engine 按 `(samples, labels)` 读取 batch，统一 dataset 改成字典后，需要增加一个兼容的 batch 解包函数，但 AMP、梯度累积、optimizer step 和日志逻辑可以继续复用。

【`train_one_epoch()` 和 `train_class_batch()` 的关系是什么？】

下面是按照当前 `engine_for_finetuning.py` 忠实简化后的伪代码。`train_class_batch()` 只负责一个 batch 的模型前向和分类 loss：

```python
def train_class_batch(model, samples, target, criterion, ch_names):
    outputs = model(samples, ch_names)
    loss = criterion(outputs, target)
    return loss, outputs
```

这里的参数名虽然写成 `ch_names`，但 `train_one_epoch()` 实际传入的是转换后的 `input_chans` 索引。

`train_one_epoch()` 是外层训练循环。当前实现同时支持 DeepSpeed 和 PyTorch `NativeScaler` 两条更新路径：

```python
def train_one_epoch(..., loss_scaler, update_freq, ch_names, input_scale):
    # 将导联名称转换成 LaBraM 使用的位置索引。
    input_chans = utils.get_input_chans(ch_names)

    # 切换到训练模式，并在 epoch 开始前清空梯度。
    model.train(True)
    clear_gradients(model, optimizer, loss_scaler)

    for data_iter_step, (samples, targets) in enumerate(data_loader):
        # 将 EEG 和标签移动到设备；同时完成 input_scale、
        # [B, N, A*T] → [B, N, A, T] reshape 和二分类标签处理。
        samples, targets = prepare_class_batch(
            samples, targets, device, input_scale, is_binary
        )
        # targets 是分类真值标签，只传给 criterion 和准确率计算，
        # 不作为 EEG 输入传给模型；它不是 Dynamic 的 h_miss_target。

        if loss_scaler is None:
            # DeepSpeed 路径：输入转成 FP16，DeepSpeed 管理混合精度。
            loss, outputs = train_class_batch(
                model, samples.half(), targets, criterion, input_chans
            )
        else:
            # 普通 PyTorch 路径：使用 autocast 执行混合精度前向。
            with autocast():
                loss, outputs = train_class_batch(
                    model, samples, targets, criterion, input_chans
                )

        # 在反向传播前检查 loss，并根据 update_freq 做梯度累积缩放。
        check_loss_is_finite(loss)
        loss = loss / update_freq
        is_update_step = (data_iter_step + 1) % update_freq == 0

        if loss_scaler is None:      # DeepSpeed
            # DeepSpeed 在 model.step() 内部判断是否到达参数更新边界。
            model.backward(loss)
            model.step()
        else:                        # PyTorch + NativeScaler
            # NativeScaler 只在 is_update_step=True 时更新模型参数。
            loss_scaler(loss, optimizer, update_grad=is_update_step)
            if is_update_step:
                optimizer.zero_grad()

        # 记录当前 batch 的 loss、分类准确率、学习率等训练信息。
        log_training_stats(loss, outputs, targets)
```

【`check_loss_is_finite`、`loss / update_freq` 和 `is_update_step` 是什么意思？】

这三行用于检查异常 loss，并控制梯度累积：

```python
# NaN 或 Inf 无法正常反向传播，因此发现后立即停止训练。
check_loss_is_finite(loss)

# 连续 update_freq 个小 batch 的梯度会累加；每个 loss 先除以
# update_freq，最终得到的是这些小 batch 的平均梯度。
loss = loss / update_freq

# 只有累积到第 update_freq 个小 batch 时才真正更新参数。
is_update_step = (data_iter_step + 1) % update_freq == 0
```

例如 `update_freq=4`：

```text
batch 1：backward，累积梯度，不更新参数
batch 2：backward，累积梯度，不更新参数
batch 3：backward，累积梯度，不更新参数
batch 4：backward，累积梯度，optimizer.step()，然后清空梯度
```

因此，它相当于用 4 个小 batch 模拟一个更大的 batch。PyTorch `NativeScaler` 通过 `update_grad=is_update_step` 控制何时更新；DeepSpeed 则在每次调用 `model.step()` 时由内部的梯度累积配置判断是否真正更新参数。

两者的关系可以概括为：

```text
train_one_epoch()
└── 对每个 batch 调用 train_class_batch()
    ├── model.forward()
    └── criterion() → classification loss
```

因此，之前写成统一的 `backward() → optimizer.step()` 只是概念表达，并不等同于当前代码；上面的两个分支才对应当前 engine 的实际结构。

#### model.forward_pretrain()讨论结论

这里先明确 `model.forward_stage1()` 的接口职责，暂不直接新增实现代码。

`forward_stage1()` 只负责 Dynamic Stage 1 的前向计算，不负责计算 loss、组合 loss 权重或执行 optimizer。建议调用链如下：

```text
x_obs
→ patch_embed
→ h_obs
→ p_obs / p_miss
→ corrector
→ z_sub / z_task / d_sub / d_task / shared_feature
→ h_pred_miss = p_miss + d_sub + d_task
```

建议返回以下中间结果：

- `h_obs`：观测导联的 latent token；
- `p_miss`：缺失导联的 prototype token；
- `d_sub`、`d_task`：subject/task correction；
- `h_pred_miss`：缺失导联的预测 latent；
- `z_sub`、`z_task`、`shared_feature`：供 auxiliary、contrastive 和 CSLP loss 使用；
- 可选的 `sub_logits`、`task_logits`、`shared_sub_logits`、`shared_task_logits`；
- 可选的 `h_miss_target`：由完整 `x_full` 经过同一个 `patch_embed` 后，取出缺失导联位置得到的 latent target。

其中，`h_miss_target` 只用于计算 reconstruction loss，生成时应使用 `torch.no_grad()`，不能让 target 分支参与反向传播。`forward_pretrain()` 不应读取或计算任何 loss weight，也不应执行 Transformer 分类头。

当前仓库需要先处理以下接口前提：

1. `data_processor/erpcore.py` 当前仍返回 `(eeg, label)`，还没有返回计划中定义的 `x_obs`、`x_full`、`subject`、`task` 字典，因此不能直接接入该前向接口。
2. 现有 `modeling_finetune.py` 的 `forward_features()` 已包含静态 prototype 补全和 Transformer 流程。Dynamic Stage 1 建议作为独立 model/wrapper，复用 backbone 的 `patch_embed`，避免改变 N/O/A 的原有入口。
3. ERP Core 当前配置是 12 个观测导联、28 个目标导联、每个导联 `num_t=1`，因此 Stage 1 的张量形状应保持为：

   ```text
   h_obs          [B, 12, 1, D]
   p_miss         [B, 16, 1, D]
   h_pred_miss    [B, 16, 1, D]
   h_miss_target  [B, 16, 1, D]
   ```

cleanup 版本的 corrector 虽然接收 `p_obs`，但内部目前没有实际使用它，只使用 `h_obs` 和 `p_miss`。第一版建议保留这个参数以维持接口一致，是否让 corrector 显式使用 `p_obs` 可以在最小链路验证后再决定。

【engine里是怎么做的？】

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
