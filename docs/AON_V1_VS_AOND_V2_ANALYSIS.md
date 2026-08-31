# AON v1 与 AOND v2 端点差异分析

## 版本、范围与方法

本报告比较远程 low-channel（Dorisgua/labram-low-channel）上的两个分支末端：

- 旧版：low-channel/develop/aon-v1
- 新版：low-channel/develop/aond-v2
- 旧版 SHA：438adfed780c7c14d1f762327d49ddc06b279a83
- 新版 SHA：1f1337b62b41fffd122276fb1b789db3a3d77dd6

分析使用两个端点的完整差异，即：

~~~text
git diff 438adfed780c7c14d1f762327d49ddc06b279a83 1f1337b62b41fffd122276fb1b789db3a3d77dd6
~~~

没有用共同祖先到新版的差异替代。low-channel 配置为 git@github.com:Dorisgua/labram-low-channel.git；origin 指向另一个仓库，未用于本次版本定位。此前已执行 git fetch low-channel；fetch 只更新了 low-channel/main，两个目标分支及上述 SHA 保持不变。

实际端点差异为 14 个路径，用户预期清单少列了 docs/modeling_finetune.diff：

~~~text
M data_processor/erpcore.py
M docs/README.md
D docs/modeling_finetune.diff
M docs/prototypes/01_generate_erpcore_cnn_patch_prototypes.py
A engine_for_dynamic_stage1.py
M engine_for_finetuning.py
A losses_dynamic.py
A modeling_dynamic_stage1.py
M run_class_finetuning.py
A run_dynamic_stage1.py
A scripts/bash_stage1.sh
A scripts/erp_core/D/stage1.sh
A scripts/erp_core/D/stage2.sh
~~~

总量约为 3149 行新增、361 行删除。以下行号均以各分支端点文件自身为准；本报告文件不属于上述分支端点差异。

## 一、整体变化概览

AOND v2 将旧版静态 AON 分类/补全链路扩展为 ERP CORE 12→28 通道的两阶段链路：

1. 数据处理从返回二元组，变为同时返回 28 通道完整目标、12 通道观测输入，以及 subject/task 元数据。
2. 新增动态 corrector。它在 16 个缺失通道的 latent token 上预测 subject/task 条件修正量。
3. 新增 Stage 1 重建训练、Stage 1 专用损失和入口；Stage 1 的目标是让 12 通道输入恢复到 28 通道 latent 表示。
4. Stage 2 复用 Stage 1 corrector，并在分类训练前冻结 corrector；分类入口和分类引擎兼容新的六字段样本。
5. 增加 ERP CORE 脚本和结果表。
6. 删除一个保存历史代码差异的 docs/modeling_finetune.diff；这不改变运行时代码，但降低了迁移记录的可追溯性。

最重要的配置问题是 correction_scale：

- Stage 1 脚本 scripts/erp_core/D/stage1.sh 第 31 行显式设为 0.02。
- Stage 1 模型 modeling_dynamic_stage1.py 第 387 行保存 Python float correction_scale，第 497–503 行用它缩放 subject/task 修正。
- Stage 2 run_class_finetuning.py 没有 correction_scale 参数，scripts/erp_core/D/stage2.sh 也没有传入它。
- 因而 Stage 2 使用模型构造函数默认值 1.0。
- 该值不是 Parameter，也不是 persistent buffer，不会进入 checkpoint model state_dict。

所以，按实际代码，每个 subject/task 修正分支在 Stage 1 的绝对值上限约为 0.02，两者相加后相对 prototype 的理论偏移上限约为 0.04；Stage 2 重新构造模型后分别变为 1.0 和约 2.0，单分支理论尺度相差 50 倍。Stage 2 虽然加载了 Stage 1 权重，但不会自动恢复 Stage 1 的 correction_scale。这是代码事实；是否应该保持 0.02、改为 1.0，或把它保存进 checkpoint，需要作者确认。

### 当前工作区的修正

针对上面的不一致，当前工作区已增加显式传参，但这部分不属于两个远程分支端点的原始 diff：

- run_class_finetuning.py 第 246–247 行增加 --correction_scale，第 291 行传给动态模型构造函数。
- scripts/base.sh 第 51 行读取 CORRECTION_SCALE，第 120 行统一追加 --correction_scale。
- scripts/erp_core/D/stage2.sh 第 19 行默认 export CORRECTION_SCALE=0.02。

因此当前 D Stage 1 和 D Stage 2 的默认值一致。若 Stage 1 使用了自定义值，Stage 2 也应使用同一个环境变量，例如：

~~~bash
CORRECTION_SCALE=0.01 scripts/erp_core/D/stage1.sh
CORRECTION_SCALE=0.01 scripts/erp_core/D/stage2.sh
~~~

checkpoint 虽然由 utils.save_model 保存了 args 对象，但 Stage 2 当前仍不自动读取 checkpoint.args；显式传参仍是实际生效的来源。

## 二、数据处理

### data_processor/erpcore.py（修改）

用途：读取 ERP CORE 的 epoch 数据、完成通道选择、标准化、重采样、标签映射和训练/验证/测试划分。

旧版关键逻辑：

~~~python
# 旧版约第 133–148 行
eeg = eeg[self.channel_indices]
eeg = (eeg - eeg.mean(axis=1, keepdims=True)) / ...
eeg = resample(eeg, 200)
return torch.from_numpy(eeg), label
~~~

旧版由 channel_indices 决定最终通道，输出主要是 eeg 和 label。

新版关键逻辑位于第 94–143、161–254、257–359 行：

~~~python
self.observed_channel_names = ERPCORE_12_CHANNELS
self.full_channel_names = ERPCORE_28_CHANNELS
...
x_full = normalized_and_resampled_eeg[full_indices]
x_obs = x_full[observed_indices]
return x_full, label, x_obs, x_full, subject, task
~~~

实际返回约为：

- x_full：[28, 200]
- x_obs：[12, 200]
- 完整样本：[x_full, label, x_obs, x_full, subject, task]

N/O 数据集的 Stage 2 使用第一个字段 x_full；A/D 数据集的 Stage 1 使用第三、第四字段 x_obs 和 x_full。引擎随后增加 batch 和 patch/embed 维度，形成 [B, 12, 1, 200]、[B, 28, 1, 200] 等模型输入。

第 222–254 行新增按 subject/task 分组的 pair sampler，使 Stage 1 可以取得同一 subject/task 条件下的样本对。训练与验证使用同一类采样接口。

行为影响：

- 输入由旧版单一通道张量变为同时携带观测/完整信号及元数据的六字段结构。
- 通道数由配置选择扩展到固定 12→28 ERP CORE 结构。
- TASK_REMAP 和数据 split 逻辑没有因此改变。
- 随机 pair 采样可能使验证结果具有随机性；单样本分组会重复自身样本。这些是根据代码推断，未运行验证。
- engine_for_dynamic_stage1.py 实际只解包张量，不直接使用 subject/task 字段，因此 pair 的条件作用主要通过数据组织体现，而不是显式传入模型。

依据：

- 明确依据：新版文件第 94–143、161–254、257–359 行；旧版第 133–148 行。
- 明确依据：提交 83de0a3（add stage2）引入当前数据接口。
- 依据未知：为什么选择这些 12 个观测通道、为什么验证也随机配对，代码和提交说明没有给出性能或论文证明。

### docs/prototypes/01_generate_erpcore_cnn_patch_prototypes.py（修改）

用途：训练并保存 ERP CORE CNN patch prototype。

旧版约第 105 行按二元组解包：

~~~python
for step, (samples, _) in enumerate(data_loader):
~~~

新版第 125–129 行兼容六字段：

~~~python
for step, batch in enumerate(data_loader):
    samples, _, *_ = batch
~~~

prototype 的训练算法没有改变：仍使用 28 通道输入、CNN/TemporalConv 提取 patch 表示，并保存约为 [28, D] 的 prototype；它不是新增 Transformer corrector。

影响：

- 只改变 loader tuple 的读取方式，不改变 prototype 的输入通道、输出维度或优化目标。
- 该 prototype 供新版动态模型作为完整通道的静态参考。
- 由于模型中的 prototype buffer 是 non-persistent，通常不在 checkpoint state_dict 中，Stage 2 需要按脚本再次指定 prototype 路径。

依据：

- 明确依据：新文件第 125–129 行与旧版约第 105 行。
- 提交 83de0a3。
- 依据未知：prototype 训练参数和结果是否足以支持 12→28 补全，代码没有给出证明。

## 三、模型

### modeling_dynamic_stage1.py（新增）

用途：提供动态 Stage 1 corrector 和 Stage 2 分类模型。文件约 902 行；它大量复制旧版 modeling_finetune.py 的 Transformer、位置编码、patch embedding、分类头等结构，同时加入动态通道逻辑。因此“文件新增”不等于全部功能全新。

旧版已有同等基础功能：

- modeling_finetune.py 第 263–270 行已有 DynamicNeuralTransformer 类框架。
- 第 344–348 行已有 ERP 静态 prototype buffer。
- 第 365–387、441–515 行附近已有 corrector、冻结和 token 编码相关静态 AON 机制。
- 第 616–689 行已有静态 ERP completion 分支和 Transformer 前向。

新版新增/改写的关键位置：

1. corrector 初始化：第 365–387 行

~~~python
self.subject_encoder = ...
self.task_encoder = ...
self.correction_scale = float(correction_scale)
~~~

2. 冻结 corrector：第 441–444 行

~~~python
def freeze_corrector(self):
    for p in self.subject_encoder.parameters():
        p.requires_grad = False
    for p in self.task_encoder.parameters():
        p.requires_grad = False
~~~

3. 动态通道索引和 latent 编码：第 451–515 行

~~~python
obs_indices, missing_indices = self._dynamic_channel_indices(num_channels)
h_obs = self._encode_observed_tokens(x_obs)
p_miss = self._prototype_tokens(missing_indices)
z_sub = self.subject_encoder(h_obs, p_miss)
z_task = self.task_encoder(h_obs, p_miss)
d_sub = self.correction_scale * torch.tanh(self.subject_head(z_sub))
d_task = self.correction_scale * torch.tanh(self.task_head(z_task))
h_pred_miss = p_miss + d_sub + d_task
~~~

4. Stage 1 前向：第 517–525 行，返回缺失通道预测和完整目标 latent。
5. Stage 2 ERP 分支：第 616–689 行，将 16 个预测缺失通道写回 28 通道空间，然后送入 Transformer；AdaBrain all-token 模式默认会形成 29 个 token，并输出 [B, 12] 分类 logits。
6. 模型工厂：第 880–902 行注册 dynamic_neural_transformer 和相关变体。

张量行为：

- Stage 1 观测输入约为 [B, 12, 1, 200]。
- 缺失通道数为 16；prototype、subject/task 修正、预测和目标 latent 约为 [B, 16, 1, D]。
- Stage 2 将预测 latent 与 12 个 observed latent 合成为 [B, 28, 1, D]，再进入 Transformer。
- corrector 的输出是 prototype latent 的增量，不是原始 EEG 波形。

默认/可选行为：

- 默认模型名由 run_dynamic_stage1.py 第 98 行设为 dynamic_neural_transformer。
- Stage 1 运行入口第 930–935 行总是调用 freeze_cnn。
- 但代码没有在 Stage 1 同时明确冻结 Transformer、position embedding 或 classification head；它们虽然不一定参与 Stage 1 输出，却可能被加入 optimizer。这是代码检查结论，未运行参数梯度核验。
- p_obs 在动态 forward 中被计算但没有参与最终缺失预测，属于可疑的未使用量。
- 动态 ERP 路径实际硬编码依赖 ERP prototype；parser 虽然暴露其他 dataset/scope 选项，但不能据此推断所有数据集都实现了动态 12→28 逻辑。

检查点兼容性：

- 新模型与旧静态模型并非完全 state_dict 兼容；新增 subject/task encoder/head 等参数，旧 checkpoint 可能出现 missing/unexpected keys。
- correction_scale 是普通 Python float，不会从 checkpoint 恢复。
- prototype buffer 使用 persistent=False，不能把 prototype 路径省略理解为 checkpoint 已包含 prototype。

依据：

- 明确依据：新文件第 365–387、441–525、616–689、880–902 行；旧版 modeling_finetune.py 对应基础结构。
- 明确依据：提交 88bcfe4（add 4 files）、d16a175（add engine）、83de0a3（add stage2）。
- 明确依据：历史设计文档 83de0a3:docs/DYNAMIC_INTEGRATION_PLAN.md 第 816–863、1044–1204 行。
- 代码与计划的差异：计划曾建议单独 wrapper 和 forward_features_from_tokens；实际实现把 corrector 直接放进 DynamicNeuralTransformer，未实现该 token helper。
- 依据未知：subject/task encoder 的具体结构和 correction_scale 的数值没有提交说明或论文依据证明为最优。

## 四、损失函数

### losses_dynamic.py（新增）

用途：计算 Stage 1 的缺失通道 latent 重建、正则、对比学习和 subject/task 置换损失；文件第 1–213 行。

主要模块：

- 第 11–20 行 reconstruction_mse：检查 shape/finite 后计算 MSE。
- 第 23–133 行 compute_stage1_losses：汇总 missing MSE、regularization、summary contrastive、correction contrastive、subject/task permutation。
- 第 146–175 行 swap_subject_task：构造 subject/task 置换样本。
- 第 178–213 行 symmetric_info_nce：双向 InfoNCE。

Stage 1 目标：

~~~text
total =
    missing_weight * missing_mse
  + reg_weight * regularization
  + subject_contrastive_weight * subject_contrastive
  + task_contrastive_weight * task_contrastive
  + permutation_weight * permutation_loss
~~~

脚本 scripts/erp_core/D/stage1.sh 实际启用：

- missing weight = 20
- reg weight = 0.001
- subject/task correction contrastive = 0.005
- permutation = 5.0

因此历史设计文档 83de0a3:docs/DYNAMIC_INTEGRATION_PLAN.md 第 865–889、1036–1042 行所说的“初始只启用 missing+reg、扩展项默认关闭”并不是当前 D Stage 1 脚本的实际配置。CSLP ramp 也没有在当前新增损失文件中实现。

影响：

- 训练目标不再只有重建 MSE；subject/task 对比和 permutation 会改变 corrector latent 的梯度。
- target 在引擎中 detach，但预测分支和 corrector 参数仍获得损失梯度。
- 该文件本身不负责 optimizer.step；更新发生在 Stage 1 engine。
- 没有实测不能断言这些额外损失改善了实验结果。

依据：

- 明确依据：losses_dynamic.py 第 11–20、23–133、146–213 行；stage1.sh 第 23–31 行。
- 明确依据：提交 83de0a3。
- 依据未知：损失权重为何取这些值、是否来自论文或网格搜索，当前提交说明没有证据。

## 五、训练引擎

### engine_for_dynamic_stage1.py（新增）

用途：执行 Stage 1 的训练 epoch、pair 数据准备、梯度缩放、优化器更新和评估；文件第 1–344 行。

主要流程：

- 第 26–44 行：解包 batch，取 x_obs/x_full，调用 model.forward_stage1。
- 第 47–69 行：准备同组/置换所需 pair。
- 第 105–280 行：训练循环；计算 losses_dynamic，NativeScaler/DeepSpeed backward，optimizer step，更新 metric。
- 第 286–344 行：评估并返回验证指标。

关键解包：

~~~python
_, _, x_obs, x_full, _, _ = batch
x_obs = x_obs.unsqueeze(2)
x_full = x_full.unsqueeze(2)
~~~

target 在传入损失前 detached，预测 latent 用于反向传播。

影响：

- 参数更新目标是 Stage 1 loss，而非分类 accuracy。
- 张量主要在这里从 [B, C, T] 变成模型所需的 [B, C, 1, T]。
- 使用 model.train() 后，corrector 的 dropout 等训练行为仍然打开；这与 Stage 1 是否应确定性建模有关，需要作者确认。
- 验证若复用随机 pair sampler，val loss 可能不完全可复现。
- subject/task 元数据未直接传入模型；不能把它解释成显式 ID-conditioned forward。

依据：

- 明确依据：新文件第 26–69、105–280、286–344 行。
- 提交 d16a175、83de0a3。
- 代码推断：具体梯度是否流向每个模块需要运行 autograd 或查看 optimizer 参数才能验证，本次没有启动训练。

### engine_for_finetuning.py（修改）

用途：分类 Stage 2 的训练和评估。

新版变化：

- 第 66–68 行读取新六字段样本的第一个字段和第二个字段作为 data/label。
- 第 187–192 行评估也统一使用 batch 前两个字段，不再使用旧版 batch[0] 与 batch[-1]。
- 第 40–49 行在 model.train() 后让冻结的 patch_embed/corrector 保持 eval，避免冻结模块的 dropout/batchnorm 状态漂移。

行为影响：

- 兼容 erpcore 新六字段返回值，避免误把最后一个 task 字段当分类 label。
- Stage 2 仍由分类 loss/accuracy 驱动，corrector 在 run_class_finetuning.py 中先冻结。
- eval 模式和冻结模块状态更稳定，但未运行训练，未验证数值影响。

依据：

- 明确依据：新版第 40–49、66–68、187–192 行与旧版对应位置。
- 提交 83de0a3。
- 依据未知：冻结模块保持 eval 是否为作者专门设计，提交说明未解释。

## 六、入口与脚本

### run_dynamic_stage1.py（新增）

用途：Stage 1 命令行入口，负责 parser、dataset/model/optimizer/checkpoint、训练和验证。

关键位置：

- 第 98 行：动态模型默认值。
- 第 226 行：ERP CORE 数据集。
- 第 239、242 行：erpcore12 与 12→28 相关选项。
- 第 259–275 行：dynamic loss、correction_scale 等参数。
- 第 285 行：Stage 1 以 loss 作为 best metric。
- 第 930–935 行：调用 freeze_cnn。
- 第 1131–1146 行：训练。
- 第 1160–1190 行：验证。
- 第 1202–1205 行：按较小验证 loss 保存/选择最佳 checkpoint。

影响：

- Stage 1 的模型选择指标是 val loss，不是 val accuracy。
- checkpoint 主要保存模型权重和训练状态，但没有自动保存/恢复 correction_scale 这一普通 float。
- parser 中的通用模型/数据参数不代表动态 ERP 分支已支持全部组合。
- 脚本没有显式传入 abs_pos_emb；parser 默认 false，而旧版某些 base 脚本曾传入它。
- 默认 model_filter_name 为 gzpt 相关的 student 过滤逻辑（第 848–856 行）；若基础 checkpoint 不包含匹配键，可能出现空或部分加载，需运行日志确认。
- AdaBrain classifier 模式如果包裹了模型，Stage 1 engine 仍要求 wrapper 暴露 forward_stage1；默认 mean_pool 路径更符合当前实现。此兼容性风险未运行验证。

依据：

- 明确依据：新文件上述行号。
- 提交 88bcfe4、83de0a3。
- 代码推断：实际加载比例和 missing/unexpected keys 需要运行入口才能确认，本次没有运行训练。

### run_class_finetuning.py（修改）

用途：Stage 2 分类入口。

新版第 34 行导入 modeling_dynamic_stage1，以注册动态模型；第 895–896 行在 ERP scope 下冻结 corrector，然后继续执行 freeze_cnn、AdaBrain wrapper 和分类训练。

影响：

- Stage 2 可以加载 dynamic model 和 Stage 1 checkpoint，并在分类前冻结动态 corrector。
- 没有 correction_scale CLI 参数，也没有从 Stage 1 checkpoint 恢复该配置。
- 因而 Stage 2 默认构造 correction_scale=1.0；即使 corrector 参数被冻结，其前向数值尺度仍和 Stage 1 的 0.02 不同。
- 这会影响 Stage 2 送入 Transformer 的缺失通道 latent 分布，属于检查点加载后仍然存在的行为不兼容。

依据：

- 明确依据：新版第 34、895–896 行；DynamicNeuralTransformer 构造默认值；stage2.sh 未传 correction_scale。
- 提交 83de0a3。
- 依据未知：是否有意让 Stage 2 使用 1.0，没有提交说明证明。

### scripts/bash_stage1.sh（新增）

用途：通用 Stage 1 shell wrapper，将环境变量映射成 run_dynamic_stage1.py 参数。

关键行为：

- 默认 batch size 64、epochs 20、lr 5e-4、weight decay 0.05、sampling 200、z-score、seed 0。
- 固定传入 disable_rel_pos_bias、disable_qkv_bias、no_auto_resume。
- GPU_IDS 映射到 CUDA_VISIBLE_DEVICES。
- 第 83–95 行默认 RUN_BACKGROUND=1，使用 nohup；设为 0 才前台运行。

影响：

- 默认后台启动不会自动串接 Stage 2，脚本结束不等于训练已完成。
- 未显式传 abs_pos_emb，行为依赖入口默认值。
- 未显式清空 model_filter_name，行为依赖入口默认值。
- 没有检查 Stage 1 checkpoint 是否成功生成。

依据：

- 明确依据：新文件第 1–100 行。
- 提交 88bcfe4、83de0a3。
- 代码推断：后台日志、退出码和实际 checkpoint 需要运行后才能确认。

### scripts/erp_core/D/stage1.sh（新增）

用途：ERP CORE D 方案的 Stage 1 配置。

关键配置约在第 8–31 行：

~~~bash
DATASET=erpcore
CHANNELS=12to28
EPOCHS=20
SEED=1
MISSING_WEIGHT=20
REG_WEIGHT=0.001
SUBJECT_CONTRASTIVE_WEIGHT=0.005
TASK_CONTRASTIVE_WEIGHT=0.005
PERMUTATION_WEIGHT=5.0
CORRECTION_SCALE=0.02
~~~

影响：

- 具体启用了额外 subject/task 对比和 permutation 损失。
- correction_scale=0.02 直接限定 Stage 1 修正量的尺度。
- 没有显式解决 base checkpoint 过滤、abs position embedding 或 Stage 2 配置传递问题。

依据：

- 明确依据：新文件第 8–31 行。
- 提交 83de0a3。
- 依据未知：权重和 scale 的选择依据未见代码/文档证据。

### scripts/erp_core/D/stage2.sh（新增）

用途：ERP CORE D 方案的 Stage 2 分类配置。

关键配置约在第 8–33 行：

~~~bash
--model dynamic_neural_transformer
--finetune <STAGE1_CHECKPOINT>
--model_filter_name ""
--freeze_cnn
--ada_brain all_token
~~~

影响：

- Stage 2 复用 Stage 1 checkpoint，关闭 model_filter_name 过滤，并冻结 CNN/corrector。
- 未传 correction_scale，因此使用模型默认 1.0。
- 没有 Stage 1 checkpoint 存在性检查、没有自动等待 Stage 1、没有传 abs_pos_emb。
- 默认通用 wrapper 仍可能后台启动，不能把两个脚本连续写在一起理解为严格流水线。

依据：

- 明确依据：新文件第 1–72 行；run_class_finetuning.py 第 895–896 行。
- 提交 83de0a3。
- 依据未知：Stage 2 使用不同 scale 是否是有意的实验变量。

## 七、文档

### docs/README.md（修改）

用途：项目说明和实验结果记录。

新版第 464–475 行增加结果表，声明按 val acc 选择 best checkpoint，并列出 Dynamic D 的 val/test accuracy 和 balanced accuracy，约为：

~~~text
val acc 60.27%
test acc 59.75%
balanced accuracy 34.65%
~~~

影响：

- 只改变文档，不改变训练、数据、模型或 checkpoint 行为。
- 结果表没有同时给出 seed、checkpoint 路径、prototype 文件、完整命令、数据版本和日志，因此仅凭 README 不能独立复现实验，也不能证明性能提升。
- “按 val acc 选 best checkpoint”与 Stage 1 入口按 val loss 选择 best 的语义不同；表格看起来更像 Stage 2 分类结果，不能反推 Stage 1 的保存规则。

依据：

- 明确依据：新版 docs/README.md 第 464–475 行。
- 提交 1f1337b（add val acc erpcore）。
- 依据未知：结果来源和统计过程未在当前文档中完整给出。

### docs/modeling_finetune.diff（删除）

用途：旧版保存的一份 modeling_finetune.py 历史文本 diff，不是运行时 import 的 Python 模块。

旧版文件约 338 行，记录了静态 AON 代码变化；运行时对应逻辑仍在 modeling_finetune.py 中，所以删除它不会直接改变程序行为。

影响：

- 不改变输入输出、训练目标、梯度、checkpoint 或实验结果。
- 删除后，旧静态 AON 迁移过程的文档证据减少；若需要追踪设计演化，应依靠提交历史和实际 Python 文件。
- 删除依据仅能明确追溯到提交 1f1337b；没有理由把它解释为功能删除。

## 八、实际端到端流程

默认 ERP CORE D Stage 1：

1. stage1.sh 设置 ERP CORE、12→28、损失权重、correction_scale=0.02，并调用 bash_stage1.sh。
2. erpcore.py 读取样本，得到 x_obs [12,200]、x_full [28,200]，并返回六字段 tuple。
3. dynamic Stage 1 engine 读取第三、第四字段，增加维度为 [B,12,1,200] 和 [B,28,1,200]。
4. modeling_dynamic_stage1.py 编码 12 个 observed latent，取 16 个 missing prototype，经过 subject/task encoder/head 计算修正。
5. correction 为 correction_scale 乘 tanh head 输出；Stage 1 预测为 prototype 加 subject correction 加 task correction。
6. losses_dynamic.py 计算 missing MSE、正则、subject/task 对比和 permutation 项。
7. engine 反向传播并执行 optimizer 更新；Stage 1 最佳模型按较小 val loss 选择。

默认 ERP CORE D Stage 2：

1. stage2.sh 把 Stage 1 checkpoint 传给 run_class_finetuning.py。
2. 分类入口构造动态模型，但不接收 Stage 1 的 correction_scale；因此默认使用 1.0。
3. 加载 checkpoint 后冻结 corrector，并冻结 CNN 相关模块。
4. 动态模型重新计算缺失 latent，把 16 个预测通道写回 28 通道 token 空间，经过 Transformer、pooling/AdaBrain 和分类头。
5. engine_for_finetuning.py 读取六字段 batch 的前两个字段作为 data/label，计算分类损失并更新未冻结参数。

可选功能包括 subject/task 对比、permutation loss、AdaBrain all-token、前后台启动方式及通用模型过滤。它们不是全部默认启用；D Stage 1 脚本明确启用了额外损失，D Stage 2 脚本明确启用了 all-token。

## 九、建议作者确认的问题

1. Stage 2 是否应继续使用 Stage 1 的 correction_scale=0.02？如果是，应将它作为显式 Stage 2 参数，并保存到 checkpoint/config；仅冻结 corrector 不能解决尺度差异。
2. 如果 Stage 2 有意使用 1.0，是否应把它记录为实验变量，并重新解释 Stage 1 checkpoint 的可比性？
3. Stage 1 是否应冻结未参与重建的 Transformer、position embedding 和分类头，避免不必要的 optimizer 状态？
4. 验证集随机 pair、singleton 重复采样是否是有意设计？
5. subject/task 条件是否应该由显式 metadata ID 传入，而不是只依赖 pair/permutation 数据组织？
6. prototype 是否应进入 checkpoint，或由 checkpoint 保存其路径、哈希和生成配置？
7. README 结果能否补充 seed、checkpoint、prototype、命令和日志出处？
8. 设计文档与实现的差异是否需要更新：计划中的独立 wrapper、token helper、默认关闭扩展损失，目前都不是当前实现的准确描述？

## 十、结论边界

本报告结论来自固定 SHA 的代码、提交历史和项目文档静态检查。没有启动训练、没有加载 checkpoint 做运行时验证，也没有声称额外损失或新模型一定提升性能。凡标注“代码推断”或“依据未知”的地方，都需要通过作者确认、实验日志或额外设计文档补足。
