# 新数据集接入流程：先完成，再完美

## 核心原则

接入新数据集时，先完成一个最小、正确、可重复运行的训练闭环，再增加复杂设计。

不要在第一版中同时加入少通道、通道原型补全、不同 pooling、AdaBrain 分类头、多 seed 和大规模调参。一次只改变一个主要变量，确保每次出现结果变化时，都能解释变化来自哪里。

## 接入前必须弄清楚的最小设置

不需要先理解项目中的全部参数，但以下设置会直接影响实验是否正确，必须在正式训练前确认：

1. **数据单位与输入缩放**
   - 原始信号使用 V、mV 还是 μV。
   - 数据是否已经归一化。
   - `input_scale` 是否与 LaBraM 预训练时的输入量级一致。

2. **采样率、片段长度与 token 数**
   - 原始采样率和目标采样率。
   - 每个样本包含多少秒 EEG。
   - `patch_size` 和 `num_t` 是否与实际输入一致。

3. **通道名称、数量与顺序**
   - loader 输出的通道顺序必须固定。
   - 通道名称必须能正确映射到 LaBraM 的位置编码。
   - dataset、模型配置和 prototype 文件中的通道顺序必须一致。

4. **标签定义**
   - 类别数量和每个标签的语义。
   - 标签是否从 0 开始且连续。
   - train、validation 和 test 中的标签映射必须一致。

5. **数据划分**
   - 明确使用 within-subject 还是 cross-subject。
   - 同一受试者的数据不能意外同时出现在训练集和测试集。
   - 所有 prototype 和归一化统计量只能由训练集生成，避免数据泄漏。

6. **分类头与最佳模型指标**
   - 明确使用 `mean_pool` 还是 `adabrain_all_token`。
   - 明确使用哪个 validation 指标选择 `checkpoint-best`。
   - 指标应与数据集常用 benchmark 保持一致，或在报告中说明差异。

优先级：**数据划分、信号量级和通道映射高于分类头和调参。**

## 标准接入顺序

每个新数据集按照以下顺序推进：

```text
实现数据读取
→ 全通道输入
→ 不使用 prototype
→ 冻结 CNN/patch_embed
→ mean pooling
→ seed 0 短跑 1–2 epochs
→ seed 0 正式训练
→ seeds 0/1/2
→ AdaBrain all-token classifier
→ 少通道输入
→ prototype 通道补全
→ 其他消融与调参
```

后面的阶段不能阻塞前面阶段的“完成”。例如，只有 seed 0 的全通道 mean-pooling 基线已经完成，也应当先记录为一个有效阶段成果。

## 第一版推荐配置

新数据集的第一个脚本应尽量简单：

```text
全通道
completion_scope=none
classifier_mode=mean_pool
freeze_cnn=true
seed=0
epochs=1 或 2（首次 smoke test）
```

选择 mean pooling 作为第一版，是因为它不要求分类头预先绑定固定的展平 token 数，更适合优先检查数据、shape、通道、标签和训练流程。mean pooling 跑通以后，再加入 AdaBrain flatten classifier，形成一次单变量对比。

## Smoke test 检查清单

第一次短跑必须检查：

- [ ] train、validation、test 均能成功构建。
- [ ] 输入 shape 与预期一致，例如 `[batch, channels, temporal_patches, patch_size]`。
- [ ] 通道数量、名称和顺序正确。
- [ ] `num_t` 与 Transformer 实际输出 token 数一致。
- [ ] 类别数量、标签最小值和最大值正确。
- [ ] 一个 batch 能完成 forward、loss 和 backward。
- [ ] loss 为有限数值，没有 NaN 或 Inf。
- [ ] 训练至少完成 1 个 epoch。
- [ ] validation 和 test 指标能够正常计算。
- [ ] terminal log、`log.txt`、TensorBoard 和 checkpoint 写入预期目录。
- [ ] 日志保存了完整启动命令、seed、数据路径和关键配置。

## “数据集接入完成”的定义

满足以下条件后，第一阶段即可标记为完成：

- loader 能稳定、可重复地读取数据。
- train、validation、test 划分已经确认且不存在受试者泄漏。
- 输入单位、采样率、通道映射和标签映射已记录。
- seed 0 smoke test 正常结束。
- seed 0 的全通道 mean-pooling 正式训练正常结束。
- 能够从 `checkpoint-best` 得到 validation 和 test 指标。
- 有一个独立 `.sh` 可以重复运行该实验。
- 输出目录中存在完整 terminal log 和训练日志。

三 seeds、AdaBrain、少通道、prototype 和超参数优化属于后续增强，不属于第一阶段完成条件。

## 后续实验的单变量顺序

完成基础版本后，建议按以下顺序扩展：

1. **重复性**：保持配置不变，运行 seeds 0、1、2。
2. **分类头**：只把 `mean_pool` 改为 `adabrain_all_token`。
3. **通道消融**：保持分类头和训练配置不变，只减少真实通道。
4. **通道补全**：在相同少通道配置上增加训练集 prototype。
5. **Pooling scope**：分别比较仅真实通道和全部补全通道。
6. **解冻与调参**：最后再比较 full finetune、学习率和其他优化参数。

每一步只改变一个主要因素，并保留对应的全通道基线作为参照。

## 推荐的实验命名语义

沿用当前实验的语义可以减少混淆：

- `O`：原始全通道输入。
- `N`：少通道输入，不做补全。
- `A`：少通道输入，使用 prototype 补全。
- `Ah`：补全后对全部目标通道做 high pooling。
- `Al`：补全后只对真实通道做 low pooling。
- `ada`：AdaBrain all-token flatten classifier。
- `meanpool`：LaBraM mean-pooling classifier。
- `FC`：冻结 CNN/patch_embed。
- `FT`：完整微调。

实验编号、数据集、通道数、分类头和 seed 应同时出现在脚本名称或运行日志中，不能只依赖输出目录的时间戳判断配置。

## 新数据集记录模板

```markdown
### 数据集名称

- 数据路径：
- 数据来源/benchmark：
- 任务与类别：
- 数据单位：
- 原始采样率：
- 模型输入采样率：
- 样本时长：
- 全通道数量与顺序：
- 少通道方案：
- train/validation/test 划分：
- 是否 cross-subject：
- input_scale：
- patch_size：
- num_t：
- 最佳 checkpoint 指标：
- 第一版运行脚本：
- 第一版输出目录：
- 已知问题：
```

## 工作纪律

- 不通过复制整个仓库创建新实验版本；使用 Git commit、branch 或 tag。
- 新实验开始前，先保存当前可运行版本。
- 不覆盖历史输出，不修改已有实验日志。
- 数据、checkpoint、TensorBoard 和 outputs 不提交到 Git。
- 每次训练都保存完整启动命令和代码版本。
- 遇到新想法时，先记录到待办，不立即插入正在完成的最小闭环。

最终目标不是一次做出最复杂的实验，而是持续产出一系列正确、可重复、能够相互比较的已完成实验。
