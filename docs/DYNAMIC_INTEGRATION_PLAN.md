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

ERP Core dataset 统一返回固定顺序的 tuple：

```python
(
    x,        # 当前实验的主输入
    label,    # 下游任务标签
    x_obs,    # 观测导联
    x_full,   # 完整导联，仅供 Stage 1 生成重建目标
    subject,  # subject 标签，供 auxiliary/CSLP 使用
    task,     # task 标签，供 auxiliary/CSLP 使用
)
```

其中 `x` 根据实验设置：N/A/Dynamic Stage 2 使用 `x_obs`，O 使用 `x_full`；Dynamic Stage 1 虽然也令 `x=x_obs`，但会显式读取后面的 `x_obs` 和 `x_full`。

各实验只读取需要的字段：

| 阶段 | 允许使用的字段 |
|---|---|
| N | `x_obs`、`label` |
| O | `x_full`、`label` |
| A | `x_obs`、`label`、Prototype |
| Dynamic Stage 1 | `x_obs`、`x_full`、`subject`、`task`，必要时使用 `label` |
| Dynamic Stage 2 | `x_obs`、`label`、冻结的 Stage 1 |

统一 dataset 返回全部字段，由不同训练阶段直接解包需要的部分。分类 engine 的训练和验证都使用 `samples, targets, *_ = batch`，因此同时兼容原来的二字段 tuple 和新的完整 tuple；不能再用 `batch[-1]` 取标签，因为完整 tuple 的最后一个字段是 `task`。Dynamic Stage 1 使用完整解包。为了避免信息泄漏，Dynamic Stage 2 只把第一个字段 `x=x_obs` 传给模型，不能把 `x_full` 传入前向。`x_full` 只用于 Stage 1 生成 `h_miss_target`，或用于离线审计。

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

以下先都不加
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

采用两个独立的 engine 文件：保留原有 `engine_for_finetuning.py`，并新增 `engine_for_dynamic_stage1.py` 承担 Stage 1 的完整 tuple、多项 loss 和重建验证。两个文件共享训练协议和现有工具，但不强制共享同一个 epoch 函数。

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

`run_dynamic_stage1.py` 以 `run_class_finetuning.py` 为模板，简化伪代码如下：

```python
def main(args):
    train_set, val_set, test_set = get_dataset(args)               # 复用 ERP Core dataset 构建逻辑
    model = get_dynamic_stage1_model(args)                         # 改 model
    optimizer, scaler, scheduler = build_training_tools(model)     # 保持原流程
    resume_if_needed(model, optimizer, scaler, args)               # 保持原流程

    for epoch in range(args.start_epoch, args.epochs):
        train_stats = train_dynamic_stage1_one_epoch(model, train_set, ...)  # 改 engine
        val_stats = evaluate_dynamic_stage1(model, val_set, ...)             # 改 evaluate
        if val_stats["missing_mse"] < best_missing_mse:
            save_checkpoint_best(model, optimizer, scaler)                   # 越低越好
```

这里不新增重复的 `get_dynamic_dataset()`。`run_dynamic_stage1.py` 也使用 `get_dataset(args)`，底层统一调用 `data_processor/erpcore.py`，各阶段都接收相同顺序的完整 tuple，再由 engine 解包需要的字段。`train/val/test` 仍然必须是三个独立 split，避免训练数据与验证、测试数据混用。

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

两个 engine 需要保持一致的公共训练行为包括 AMP、梯度累积、LR/weight decay 调度、梯度裁剪、NaN/Inf 检查、日志格式和分布式同步。Stage 1 完整解包 tuple 并处理 Dynamic forward、多项 loss 和 reconstruction 指标；Stage 2 继续使用原分类 engine，并且只读取 tuple 前两个字段 `x=x_obs` 和 `label`。

第一版 Stage 1 先启用 `missing` 和 `reg`，其余 loss 保留参数并默认关闭；基础链路稳定后再逐项打开 auxiliary 和 CSLP。

【那么原来的labram的forward入口放在哪里？】

#### 回答

原来的 LaBraM `forward()` 继续放在 `modeling_finetune.py` 的
`NeuralTransformer` 中，供 N/O/A 使用。同时在这个类里新增
`forward_features_from_tokens()`，作为已经得到 latent token 时的特征入口：

```python
class NeuralTransformer(nn.Module):
    # 新增：输入已经 patch-embedded 的 latent token。
    def forward_features_from_tokens(self, h, input_chans=None, **kwargs):
        h = add_cls_channel_time_embedding(h)
        h = run_transformer_blocks(h)
        return norm_and_pool_like_original_forward_features(h, **kwargs)

    # 原入口：输入原始 EEG，仍然先经过 patch_embed。
    def forward_features(self, x, input_chans=None):
        h = self.patch_embed(x)
        if self.completion_scope != "none":
            h = static_prototype_complete(h)  # A-End2End
        return self.forward_features_from_tokens(h, input_chans)

    # 原分类入口保持不变。
    def forward(self, x, input_chans=None):
        feature = self.forward_features(x, input_chans)
        return self.head(feature)
```

Dynamic 使用独立的包装模型，不继承 `NeuralTransformer`。模型内部持有
`self.backbone`，并明确提供 Stage 1、Stage 2 特征提取和分类入口：

```python
class DynamicModel(nn.Module):
    def forward_stage1(self, x_obs, x_full):
        h_obs = self.backbone.patch_embed(x_obs)
        h_full = self.backbone.patch_embed(x_full)
        h_pred_miss = self.corrector(h_obs, self.prototype)
        h_miss_target = select_missing_tokens(h_full)
        return h_pred_miss, h_miss_target

    def forward_features(self, x_obs, input_chans=None, **kwargs):
        # Stage 2：x_obs 仍需经过 patch_embed。
        h_obs = self.backbone.patch_embed(x_obs)
        h_pred_miss = self.frozen_corrector(h_obs, self.prototype)
        h_complete = merge(h_obs, h_pred_miss)

        # h_complete 已是 latent token，从 Transformer 后半段继续执行。
        return self.backbone.forward_features_from_tokens(
            h_complete,
            input_chans=self.target_input_chans_index,
            pool_token_indices=self.obs_indices if self.pooling_scope == "low" else None,
            **kwargs,
        )

    def forward(self, x_obs, input_chans=None, **kwargs):
        feature = self.forward_features(
            x_obs,
            input_chans=input_chans,
            **kwargs,
        )
        return self.backbone.head(feature)
```

其中 `forward_features_from_tokens()` 是计划新增到 `NeuralTransformer` 的 token-level helper。它接收已经完成 12→28 动态补全的 latent token，跳过 `patch_embed` 和静态 prototype 补全，只执行原 LaBraM 的 CLS/channel/time embedding、Transformer blocks、normalization 和 pooling。下面是该方法在 `NeuralTransformer` 类内部的详细伪代码：

```python
def forward_features_from_tokens(
    self,
    h_complete,
    input_chans,
    pool_token_indices=None,
    return_patch_tokens=False,
    return_all_tokens=False,
):
    # h_complete: [B, target_channels, num_t, embed_dim]
    batch_size, target_channels_num, input_time_window, _ = h_complete.shape
    x = h_complete.flatten(1, 2)

    # 后半段与原 forward_features() 保持一致。
    x = prepend_cls_token(x)
    x = add_channel_position_embedding(x, input_chans)
    x = add_time_embedding(x, num_t=h_complete.shape[2])

    for block in self.blocks:
        x = block(x)
    x = self.norm(x)

    # 以下返回逻辑与原 forward_features() 保持一致。
    if self.fc_norm is not None:
        if return_all_tokens:
            return self.fc_norm(x)

        patch_tokens = x[:, 1:, :]
        if return_patch_tokens:
            return self.fc_norm(patch_tokens)

        if pool_token_indices is not None:
            patch_tokens = patch_tokens.reshape(
                batch_size,
                target_channels_num,
                input_time_window,
                self.embed_dim,
            )
            patch_tokens = patch_tokens[:, pool_token_indices, :, :]
            return self.fc_norm(patch_tokens.flatten(1, 2).mean(1))

        return self.fc_norm(patch_tokens.mean(1))

    if return_all_tokens:
        return x
    if return_patch_tokens:
        return x[:, 1:]
    return x[:, 0]
```

这里不能再次调用 `patch_embed`，因为 `h_complete` 已经是 patch embedding 后的 latent。原来的 `NeuralTransformer.forward()` 不删除也不替换，继续供 N/O/A 使用；Dynamic Stage 2 则显式调用自己的 `DynamicModel.forward()`。

【然后 engine 里怎么做？】

#### 回答

engine 保留原来的分类训练函数，再增加一个 Stage 1 专用函数。`forward_stage1()`
只在 `engine_for_dynamic_stage1.py` 的 `train_dynamic_stage1_one_epoch()`
训练循环中显式调用，不会由普通的 `model(...)` 自动调用。两条路径分别调用不同入口：

```python
# Dynamic Stage 1
def train_dynamic_stage1_one_epoch(model, data_loader, optimizer):
    model.train()

    for batch in data_loader:
        batch = move_batch_to_device(batch)
        x, label, x_obs, x_full, subject, task = batch

        with autocast():#【什么意思】
            outputs = model.forward_stage1(x_obs, x_full)
            losses = compute_dynamic_losses(outputs, label, subject, task)
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

Stage 1 完整解包 `x/label/x_obs/x_full/subject/task`，实际使用 `x_obs`、`x_full`、`subject` 和 `task`；Stage 2 使用 `samples, targets, *_ = batch`，只取 `x=x_obs` 和 `label`。AMP、梯度累积、optimizer step 和日志逻辑可以继续复用。

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

    for data_iter_step, batch in enumerate(data_loader):
        # 兼容原来的 (samples, targets) 和新的完整 tuple。
        samples, targets, *_ = batch

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

#### `model.forward_stage1()` 讨论结论

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
- 可选的 `sub_logits`、`task_logits`、`shared_sub_logits`、`shared_task_logits`；（这些都先不加）
- 可选的 `h_miss_target`：由完整 `x_full` 经过同一个 `patch_embed` 后，取出缺失导联位置得到的 latent target。

其中，`h_miss_target` 只用于计算 reconstruction loss，生成时应使用 `torch.no_grad()`，不能让 target 分支参与反向传播。`forward_stage1()` 不应读取或计算任何 loss weight，也不应执行 Transformer 分类头。

当前仓库需要先处理以下接口前提：

1. `data_processor/erpcore.py` 当前仍返回 `(eeg, label)`，还没有返回计划中的完整 tuple `(x, label, x_obs, x_full, subject, task)`，因此不能直接接入 Stage 1 前向接口。
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


## 7. 具体修改计划与新增伪代码

实施顺序：

```text
保护 N/O/A 基线
→ 增加 Dynamic 数据接口
→ 跑通 Stage 1 最小链路
→ 逐项增加扩展 loss
→ 抽取 token-level backbone 接口
→ 接入 Dynamic Stage 2
→ 公平性检查和多 seed 实验
```

### 11.1 阶段 0：保护现有 N/O/A 基线

#### 修改位置

- 不修改现有模型和训练逻辑；
- 新增 `docs/ERP_CORE_BASELINE.md`，记录当前 branch、commit、配置、命令和结果；
- 保存 N/O/A 的 dry-run 或单 epoch 日志。

#### 需要记录的调用链

```text
scripts/erp_core/{N,O,A}/*.sh
→ scripts/erp_core/base.sh
→ scripts/base.sh
→ run_class_finetuning.py
→ engine_for_finetuning.py
→ modeling_finetune.py
```

#### 验收条件

- N 输入保持为 12 导联；
- O 输入保持为 28 导联；
- A 使用 12 导联输入和静态 28 导联 prototype；
- 三条路径至少可以完成 dataset 加载和一次模型前向；
- 后续 Dynamic 改动可以与这一基线进行对照。

### 11.2 阶段 1：统一固定顺序的 tuple 数据接口

#### 修改位置

- 修改 `data_processor/erpcore.py`，让 ERP Core dataset 统一返回第 5 节确定的六字段 tuple；
- 修改 `engine_for_finetuning.py` 的训练和验证 batch 解包方式，使二字段旧 tuple 和六字段新 tuple 都能兼容；
- Dynamic Stage 1 对六个字段进行完整解包；
- 增加 12/28 导联映射检查。


#### 计划修改对照伪代码

先补充 12/28 导联定义，并让 loader 同时保存“当前实验输入导联”“观测导联”和“完整目标导联”的索引。下面红色 `-` 是当前代码，绿色 `+` 是计划修改：

```diff
-from Channels_definition import ERPCORE_30_CHANNELS
+from Channels_definition import (
+    ERPCORE_12_CHANNELS,  # 观测空间：模型实际可见的 12 个 EEG 导联。
+    ERPCORE_28_CHANNELS,  # 目标空间：去掉 HEOG/VEOG 后的 28 个 EEG 导联。
+    ERPCORE_30_CHANNELS,  # 原始空间：simple_data.pt 中保存的 30 个通道。
+)

 class ERPCOREPtLoader(Dataset):
     def __init__(
         self,
         payload,
         indices,
         split,
         mean,
         std,
         normalize_method,
         channel_names,
         target_samples,
     ):
         ...
         self.channel_names = [str(name).strip().upper() for name in channel_names]
-        self.channel_indices = np.asarray(
-            [self.manifest_channel_names.index(name) for name in self.channel_names],
-            dtype=np.int64,
-        )
+        self.observed_channel_names = list(ERPCORE_12_CHANNELS)  # 例如第 1 个是 FP1，第 2 个是 FP2。
+        self.target_channel_names = list(ERPCORE_28_CHANNELS)    # 顺序与补全后的 28 导联 token 顺序一致。
+        self.target_channel_indices = np.asarray(               # 记录目标 28 导联在原始 30 通道中的位置。
+            [self.manifest_channel_names.index(name)            # 例如 FP1→0、FP2→14、O2→27。
+             for name in self.target_channel_names],            # 逐个遍历目标 28 导联，保持目标顺序不变。
+            dtype=np.int64,                                     # 作为 PyTorch/NumPy 通道索引使用整数类型。
+        )
+        self.observed_indices_in_target = np.asarray(           # 记录观测 12 导联在目标 28 导联中的位置。
+            [self.target_channel_names.index(name)              # 例如 FP1→0、FP2→14、F3→1。
+             for name in self.observed_channel_names],          # 顺序严格跟随 ERPCORE_12_CHANNELS。
+            dtype=np.int64,                                     # 用于执行 x_full[observed_indices_in_target]。
+        )
+        # 本例 target_channel_indices = [0, 1, 2, ..., 27]，因为 HEOG/VEOG 正好位于原始 30 通道末尾。
+        # 本例 observed_indices_in_target = [0, 14, 1, 16, 2, 17, 4, 21, 6, 23, 10, 27]。
```

`__getitem__()` 按照当前实现进行下面的替换：

```diff
     def __getitem__(self, index):
         global_index = int(self.indices[index])  # 将当前 split 内下标转换成 simple_data.pt 的全局样本下标。
-        eeg = self.data[global_index, self.channel_indices, :].float()
-        eeg = self._normalize(eeg)
-        if eeg.shape[-1] != self.target_samples:
-            eeg = F.interpolate(
-                eeg.unsqueeze(0),
+
+        # 先从同一个原始样本取得并归一化完整 28 导联。
+        x_full = self.data[                    # self.data 的原始形状是 [样本数, 30, 256]。
+            global_index,                      # 选择当前样本，例如第 100 个全局样本。
+            self.target_channel_indices,       # 从原始 30 通道取出目标 28 导联。
+            :,                                 # 保留当前导联的全部时间采样点。
+        ].float()                              # 转为 float，得到 [28, 256]。
+        x_full = self._normalize(x_full)       # 使用训练集 28 导联统计量进行归一化。
+
+        if x_full.shape[-1] != self.target_samples:  # 当前源数据为 256 点，LaBraM ERP Core 通常需要 200 点。
+            x_full = F.interpolate(                  # 沿时间维进行线性重采样。
+                x_full.unsqueeze(0),                 # [28, 256] → [1, 28, 256]，增加 batch 维。
                 size=self.target_samples,            # 将时间长度调整到 200。
                 mode="linear",                      # 使用一维线性插值。
                 align_corners=False,                 # 与现有 loader 的插值配置保持一致。
             ).squeeze(0)                             # [1, 28, 200] → [28, 200]。
-        if eeg.shape != (len(self.channel_names), self.target_samples):
-            raise ValueError(f"Unexpected ERP CORE sample shape: {tuple(eeg.shape)}")
-        if not torch.isfinite(eeg).all():
+
+        x_obs = x_full[self.observed_indices_in_target]  # 按 12 导联既定顺序取得 [12, 200]。
+        x = (                                            # x 是分类 engine 实际读取的第一个字段。
+            x_full                                      # O：使用完整 28 导联作为分类输入。
+            if self.channel_names == self.target_channel_names
+            else x_obs                                  # N/A/Dynamic Stage 2：使用观测 12 导联。
+        )
+
+        if x_full.shape != (28, self.target_samples):
+            raise ValueError(
+                f"Unexpected ERP CORE x_full shape: {tuple(x_full.shape)}"
+            )
+        if x_obs.shape != (12, self.target_samples):
+            raise ValueError(
+                f"Unexpected ERP CORE x_obs shape: {tuple(x_obs.shape)}"
+            )
+        if not torch.isfinite(x_full).all():
             raise ValueError(f"NaN or Inf in normalized ERP CORE sample {global_index}")
-        return eeg.contiguous(), int(self.labels[index])
+
+        label = int(self.labels[index])                       # 下游 12 类分类标签，已经经过 TASK_REMAP。
+        subject = int(self.subject_values[global_index])      # 原始 subject 编号，供 auxiliary/CSLP 使用。
+        task = int(self.tasks[global_index])                  # 原始 task 编号；是否重映射需与 cleanup 配置一致。
+        return (
+            x.contiguous(),       # 第 0 项：当前实验主输入，分类 engine 使用 samples 读取。
+            label,                # 第 1 项：下游标签，分类 engine 使用 targets 读取。
+            x_obs.contiguous(),   # 第 2 项：固定 12 导联观测输入。
+            x_full.contiguous(),  # 第 3 项：固定 28 导联完整目标，只允许 Stage 1 使用。
+            subject,              # 第 4 项：subject 标签。
+            task,                 # 第 5 项：task 标签，因此不能再用 batch[-1] 读取 label。
+        )
```

因为 `_normalize()` 现在处理的是完整 28 导联，训练集统计量也要从“当前实验导联”改为“目标 28 导联”计算：

```diff
-    channel_indices = np.asarray(
-        [ERPCORE_30_CHANNELS.index(name) for name in channel_names],
-        dtype=np.int64,
-    )
+    target_channel_indices = np.asarray(                           # 目标 28 导联在原始 30 通道中的位置。
+        [ERPCORE_30_CHANNELS.index(name)                            # 例如 FP1→0、FP2→14、O2→27。
+         for name in ERPCORE_28_CHANNELS],                          # HEOG/VEOG 不在目标列表中，因此不会被选中。
+        dtype=np.int64,                                             # _training_statistics() 需要整数索引。
+    )
     ...
-    mean, std = _training_statistics(
-        payload["data"], indices["train"], channel_indices
-    )
+    mean, std = _training_statistics(
+        payload["data"],          # 原始数据：[样本数, 30, 256]。
+        indices["train"],          # 只使用训练 subject 的样本，避免验证/测试信息泄漏。
+        target_channel_indices,    # 只为目标 28 导联分别计算 mean/std。
+    )
```

```diff
+def validate_erpcore_sample(sample, observed_indices_in_target):  # 检查统一 tuple 和导联映射。
+    x, label, x_obs, x_full, subject, task = sample               # 按固定六字段顺序完整解包。
+    assert x_obs.shape == (12, 200)                               # 观测输入必须是 12 导联、200 点。
+    assert x_full.shape == (28, 200)                              # 完整目标必须是 28 导联、200 点。
+    assert torch.equal(                                           # 验证 x_obs 确实来自当前 x_full。
+        x_obs,
+        x_full[observed_indices_in_target],                        # 使用例子中的 12 个 target-space 索引。
+    )
+    assert torch.isfinite(x_obs).all()                             # 观测输入不能包含 NaN/Inf。
+    assert torch.isfinite(x_full).all()                            # 完整目标不能包含 NaN/Inf。
```

分类 engine 的训练循环改为：

```diff
+# engine_for_finetuning.py: train_one_epoch()
-for data_iter_step, (samples, targets) in enumerate(
-    metric_logger.log_every(data_loader, print_freq, header)
-):
+for data_iter_step, batch in enumerate(
+    metric_logger.log_every(data_loader, print_freq, header)
+):
+    samples, targets, *_ = batch
+    # 后续分类训练逻辑保持不变。
```

分类 engine 的验证循环改为：

```diff
+# engine_for_finetuning.py: evaluate()
 for step, batch in enumerate(metric_logger.log_every(data_loader, 10, header)):
-    EEG = batch[0]
-    target = batch[-1]
+    EEG, target, *_ = batch
+    # 不能再使用 batch[-1]，因为完整 tuple 的最后一个字段是 task。
```

#### 验收条件

- ERP Core dataset 固定返回 `(x, label, x_obs, x_full, subject, task)`；
- 分类 engine 使用 `samples, targets, *_ = batch`，同时兼容旧二字段 tuple；
- Dynamic Stage 1 完整解包六个字段；
- `x_obs` 必须能够从同一个 `x_full` 按固定索引精确取得；
- N/A/Dynamic Stage 2 的 `x` 等于 `x_obs`，O 的 `x` 等于 `x_full`；
- Dynamic Stage 2 只把第一个字段 `x=x_obs` 传入模型；
- train/val/test split、归一化、采样率和时间长度保持一致；
- subject/task 映射保存在配置中，可以随 checkpoint 一起追溯。

### 11.3 阶段 2：实现 Dynamic Stage 1 最小链路

#### 新增文件

```text
modeling_dynamic_stage1.py
losses_dynamic.py
engine_for_dynamic_stage1.py
run_dynamic_stage1.py
scripts/erp_core/D/stage1.sh
```

#### `modeling_dynamic_stage1.py`

负责观测 token、corrector 和 reconstruction target，不在模型内部组合 loss。

```diff
+class DynamicStage1Model(nn.Module):
+    def patch_tokens(self, x):
+        # x: [B, C, 200] → [B, C, 1, D]
+        x = x.unsqueeze(2)
+        tokens = self.backbone.patch_embed(x)
+        return tokens.reshape(x.shape[0], x.shape[1], 1, -1)
+
+    def forward_stage1(self, x_obs, x_full):
+        h_obs = self.patch_tokens(x_obs)
+
+        # target 分支不参与反向传播。
+        with torch.no_grad():
+            h_full = self.patch_tokens(x_full)
+            h_miss_target = h_full[:, self.missing_indices]
+
+        p_obs = expand_prototype(
+            self.prototype[self.observed_indices],
+            batch_size=x_obs.shape[0],
+        )
+        p_miss = expand_prototype(
+            self.prototype[self.missing_indices],
+            batch_size=x_obs.shape[0],
+        )
+
+        corrector_outputs = self.corrector(
+            h_obs=h_obs,
+            p_obs=p_obs,
+            p_miss=p_miss,
+        )
+        d_sub = corrector_outputs["d_sub"]
+        d_task = corrector_outputs["d_task"]
+        h_pred_miss = p_miss + d_sub + d_task
+
+        return {
+            "h_obs": h_obs,
+            "p_miss": p_miss,
+            "d_sub": d_sub,
+            "d_task": d_task,
+            "h_pred_miss": h_pred_miss,
+            "h_miss_target": h_miss_target,
+            **corrector_outputs,
+        }
```

#### `losses_dynamic.py`

第一版只实现 `missing MSE + correction regularization`。

```diff
+def compute_dynamic_losses(outputs, config, epoch):
+    missing_mse = F.mse_loss(
+        outputs["h_pred_miss"],
+        outputs["h_miss_target"],
+    )
+    reg_loss = (
+        outputs["d_sub"].square().mean()
+        + outputs["d_task"].square().mean()
+    )
+    total_loss = (
+        config.missing_weight * missing_mse
+        + config.reg_weight * reg_loss
+    )
+
+    return {
+        "total_loss": total_loss,
+        "missing_mse": missing_mse,
+        "reg_loss": reg_loss,
+    }
```

#### `engine_for_dynamic_stage1.py`

```diff
+def train_dynamic_stage1_one_epoch(
+    model,
+    data_loader,
+    optimizer,
+    device,
+    epoch,
+    loss_scaler,
+    update_freq,
+    loss_config,
+):
+    model.train(True)
+    optimizer.zero_grad()
+
+    for data_iter_step, batch in enumerate(data_loader):
+        x, label, x_obs, x_full, subject, task = batch
+        x_obs = x_obs.float().to(device, non_blocking=True)
+        x_full = x_full.float().to(device, non_blocking=True)
+        subject = subject.to(device, non_blocking=True)
+        task = task.to(device, non_blocking=True)
+
+        with torch.cuda.amp.autocast():
+            outputs = model.forward_stage1(x_obs, x_full)
+            losses = compute_dynamic_losses(
+                outputs,
+                loss_config,
+                epoch,
+            )
+            loss = losses["total_loss"]
+
+        check_loss_is_finite(loss)
+        loss = loss / update_freq
+        is_update_step = (data_iter_step + 1) % update_freq == 0
+        backward_with_scaler(
+            loss,
+            optimizer,
+            loss_scaler,
+            update_grad=is_update_step,
+        )
+        log_dynamic_losses(losses)
```

#### `run_dynamic_stage1.py`

```diff
+def main(args):
+    train_set, val_set = build_erpcore_dataset(args)
+    backbone = load_labram_backbone(args.finetune)
+    prototype = load_channel_prototype(args.channel_prototype_path)
+
+    model = DynamicStage1Model(
+        backbone=backbone,
+        prototype=prototype,
+        observed_indices=args.observed_indices,
+        missing_indices=args.missing_indices,
+    )
+    apply_stage1_freeze_policy(model, args)
+    optimizer = build_optimizer_for_trainable_parameters(model, args)
+
+    for epoch in range(args.epochs):
+        train_stats = train_dynamic_stage1_one_epoch(...)
+        val_stats = evaluate_dynamic_stage1(...)
+        save_stage1_checkpoint_and_config(...)
```

#### Bash 到 Python 的参数映射

```text
scripts/erp_core/D/stage1.sh
├── DATA_PATH               → --data_path                → args.data_path
├── FINETUNE                → --finetune                 → args.finetune
├── CHANNEL_PROTOTYPE_PATH  → --channel_prototype_path   → args.channel_prototype_path
├── MISSING_WEIGHT          → --missing_weight           → args.missing_weight
├── REG_WEIGHT              → --reg_weight               → args.reg_weight
├── EPOCHS                  → --epochs                   → args.epochs
├── SEED                    → --seed                     → args.seed
└── OUTPUT_DIR              → --output_dir               → args.output_dir
```

#### 验收条件

- `h_obs` 为 `[B, 12, 1, D]`；
- `h_pred_miss` 和 `h_miss_target` 为 `[B, 16, 1, D]`；
- 所有输出和 loss 都是有限值；
- corrector 的可训练参数能够获得梯度；
- 冻结参数不进入 optimizer；
- 短训练中 `missing_mse` 有下降趋势；
- `checkpoint-best.pth` 可以重新加载。

### 11.4 阶段 3：逐项增加扩展 loss

#### 修改位置

- 扩展 `modeling_dynamic_stage1.py` 的返回字段；
- 扩展 `losses_dynamic.py`；
- 修改 `engine_for_dynamic_stage1.py`，把完整解包得到的 `subject` 和 `task` 传给 loss；
- 在 `run_dynamic_stage1.py` 增加 loss 参数；
- 在 `scripts/erp_core/D/stage1.sh` 增加对应环境变量。

#### 增加顺序

```text
subject/task auxiliary
→ shared auxiliary
→ subject/task contrastive
→ d_sub/d_task close
→ latent permutation
→ CSLP ramp
```

#### 新增伪代码

```diff
+def compute_dynamic_losses(outputs, subject, task, config, epoch):
+    losses = compute_missing_and_reg(outputs, config)
+
+    if config.sub_aux_weight > 0:
+        losses["sub_aux"] = compute_subject_aux(outputs, subject)
+    if config.task_aux_weight > 0:
+        losses["task_aux"] = compute_task_aux(outputs, task)
+    if config.shared_aux_weight > 0:
+        losses["shared_aux"] = compute_shared_aux(outputs, subject, task)
+    if config.contrastive_weight > 0:
+        losses.update(compute_contrastive_losses(outputs, subject, task))
+    if config.permute_weight > 0:
+        losses.update(compute_permutation_losses(outputs, subject, task))
+
+    effective_weights = apply_cslp_ramp(config, epoch)
+    losses["total_loss"] = weighted_sum(losses, effective_weights)
+    return losses
```

```diff
+# engine_for_dynamic_stage1.py
+losses = compute_dynamic_losses(
+    outputs,
+    subject,
+    task,
+    loss_config,
+    epoch,
+)
```

#### 验收条件

- 所有扩展 loss 默认权重为 `0`；
- 每次实验只新增一类 loss；
- batch 中不存在有效 pair 时返回安全的零 loss，不产生 NaN；
- 日志记录每个 epoch 的实际 loss 权重；
- 原始 CSLP 四项和 cleanup 扩展项使用不同的配置名称。

### 11.5 阶段 4：抽取 token-level backbone 接口

#### 修改位置

- 修改 `modeling_finetune.py`；
- 从现有 `forward_features()` 中抽取 patch embedding 之后的公共逻辑；
- 原有 `forward()` 和 N/O/A 调用方式保持不变。

#### 新增伪代码

```diff
+class NeuralTransformer(nn.Module):
+    def patch_tokens(self, x):
+        batch_size, channels, time_windows, _ = x.shape
+        tokens = self.patch_embed(x)
+        return tokens.reshape(
+            batch_size,
+            channels,
+            time_windows,
+            self.embed_dim,
+        )
+
+    def forward_from_tokens(
+        self,
+        tokens,
+        token_input_chans_index,
+        pool_channel_indices=None,
+    ):
+        # tokens: [B, target_channels, time_windows, D]
+        x = tokens.flatten(1, 2)
+        x = self.add_cls_channel_time_embeddings(
+            x,
+            token_input_chans_index,
+            target_channels=tokens.shape[1],
+            time_windows=tokens.shape[2],
+        )
+        x = self.run_transformer_blocks(x)
+        return self.pool_tokens(x, pool_channel_indices)
```

#### 回归检查

```diff
+def test_original_forward_matches_token_forward(model, x, input_chans):
+    model.eval()
+    with torch.no_grad():
+        old_output = model.forward_features(x, input_chans=input_chans)
+        tokens = model.patch_tokens(x)
+        new_output = model.forward_from_tokens(tokens, input_chans)
+    torch.testing.assert_close(old_output, new_output)
```

#### 验收条件

- `forward_from_tokens()` 不重复调用 `patch_embed`；
- N/O/A 的公开入口和脚本不变；
- 相同输入和 checkpoint 下，重构前后的 N/O/A 输出一致；
- position embedding、time embedding 和 pooling 使用正确的 28 导联布局。

### 11.6 阶段 5：接入 Dynamic Stage 2

#### 修改位置

- 新增 Dynamic 分类 wrapper；
- 修改 `run_class_finetuning.py`，增加 Dynamic 模式；
- 继续复用 `engine_for_finetuning.py`；
- 新增 `scripts/erp_core/D/stage2.sh`。

#### Dynamic 分类 wrapper 伪代码

```diff
+class DynamicCompletionClassifier(nn.Module):
+    def forward(self, x_obs, input_chans=None):
+        h_obs = self.backbone.patch_tokens(x_obs)
+
+        with torch.no_grad():
+            corrector_outputs = self.corrector(
+                h_obs=h_obs,
+                p_obs=self.p_obs,
+                p_miss=self.p_miss,
+            )
+            h_pred_miss = (
+                self.p_miss
+                + corrector_outputs["d_sub"]
+                + corrector_outputs["d_task"]
+            )
+
+        h_complete = self.complete_tokens(h_obs, h_pred_miss)
+        feature = self.backbone.forward_from_tokens(
+            h_complete,
+            token_input_chans_index=self.target_input_chans_index,
+            pool_channel_indices=self.pool_channel_indices,
+        )
+        return self.classification_head(feature)
+
+    def train(self, mode=True):
+        super().train(mode)
+        # 分类 engine 调用 model.train(True) 后，corrector 仍保持 eval。
+        self.corrector.eval()
+        return self
```

```diff
+def complete_tokens(self, h_obs, h_pred_miss):
+    h_complete = h_obs.new_empty(
+        h_obs.shape[0],
+        28,
+        h_obs.shape[2],
+        h_obs.shape[3],
+    )
+    h_complete[:, self.observed_indices] = h_obs
+    h_complete[:, self.missing_indices] = h_pred_miss
+    return h_complete
```

#### `run_class_finetuning.py` 新增逻辑

```diff
+if args.completion_mode == "dynamic":
+    if not args.stage1_checkpoint:
+        raise ValueError("Dynamic Stage 2 requires --stage1_checkpoint")
+
+    stage1_checkpoint = torch.load(
+        args.stage1_checkpoint,
+        map_location="cpu",
+    )
+    validate_stage1_checkpoint_config(
+        stage1_checkpoint,
+        observed_ch_names=ch_names,
+        target_ch_names=ERPCORE_28_CHANNELS,
+    )
+    model = build_dynamic_completion_classifier(
+        backbone=model,
+        stage1_checkpoint=stage1_checkpoint,
+    )
+    freeze_corrector_and_auxiliary_heads(model)
```

#### Bash 到 Python 的参数映射

```text
scripts/erp_core/D/stage2.sh
├── DATA_PATH          → --data_path          → args.data_path
├── FINETUNE           → --finetune           → args.finetune
├── STAGE1_CHECKPOINT  → --stage1_checkpoint  → args.stage1_checkpoint
├── COMPLETION_MODE    → --completion_mode    → args.completion_mode
├── FREEZE_CORRECTOR   → --freeze_corrector   → args.freeze_corrector
├── BEST_METRIC        → --best_metric        → args.best_metric
├── SEED               → --seed               → args.seed
└── OUTPUT_DIR         → --output_dir         → args.output_dir
```

#### 验收条件

- Stage 2 只接收 `x_obs` 和 `label`；
- Stage 2 前向不读取 `x_full`；
- corrector 参数不进入 optimizer；
- corrector 在分类训练期间始终保持 `eval()`；
- `h_complete` 为 `[B, 28, 1, D]`；
- logits 为 `[B, 12]`；
- Stage 2 使用原有分类 engine 和统一评价指标。

### 11.7 阶段 6：公平性检查和多 seed 实验

#### 修改位置

- 新增或补充 `docs/ERP_CORE_EXPERIMENTS.md`；
- 不再修改核心模型接口；
- 固定 seed 0 验证后的正式配置。

#### 需要核对的实验字段

```text
dataset split
seed
input_scale
normalization
observed/target channel order
LaBraM checkpoint
prototype checkpoint
Stage 1 checkpoint
CNN/Transformer/corrector 冻结范围
optimizer 和 learning rate
epochs
best_metric
Test 指标读取方式
```

#### seed 对应关系

```text
Stage 1 seed 0 → Stage 2 seed 0
Stage 1 seed 1 → Stage 2 seed 1
Stage 1 seed 2 → Stage 2 seed 2
```

#### 验收条件

- 主比较中 N/O/A/D 使用相同分类协议；
- checkpoint 只由验证集 `balanced_accuracy` 选择；
- Test 集不参与 checkpoint 选择；
- seed 0/1/2 使用相同超参数；
- 每个结果可以追溯到 Bash、config、日志和 checkpoint；
- 最终报告每个 seed 的结果及 Mean ± SD。
