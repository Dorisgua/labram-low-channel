明白，是 **A、O、N、D 四组放在一起比较**。

下面均为 **实际 seed0、验证集选模对应的测试结果**；原有五个数据集分类训练 30 轮，新增 BCI-IV-2A 分类训练 50 轮。D 使用无额外 corrector 时间编码的基线版本。

| 数据集 | 测试指标 (%) | A | O | N | D |
|---|---|---:|---:|---:|---:|
| ERP-Core | BAcc | 40.29 | **45.39** | 39.73 | 37.43 |
| TUEV¹ | BAcc | 55.73 | 58.20 | **62.79** | 57.32 |
| SEED² | BAcc | 54.57 | 53.70 | **55.17** | 54.90 / 55.74 |
| SEED-V | Acc | 38.44 | **42.04** | 38.89 | 38.20 |
| SHU³ | BAcc | 54.88 | **57.93** | 56.46 | 56.11 |
| BCI-IV-2A（CBraMod 划分）⁴ | BAcc | 47.40 | **47.92** | 46.35 | 46.18 |

- **¹ TUEV**：已从 D 的逐 epoch 日志按验证 BAcc 重新选模，最佳 epoch=16（从 0 开始），Test BAcc=57.32%；A/O/N/D 现统一按验证 BAcc 选模。原 checkpoint-best.pth 仍对应 Kappa 最佳的 epoch=0，epoch=16 的 checkpoint 未单独保存。
- **² SEED**：D 有两次 seed0 运行，均列出，未根据测试成绩择优。
- **³ SHU**：D 使用最早的 seed0 运行；目录标为 seed1、seed2 的运行实际也是 seed0，未混入多 seed 统计。
- **⁴ BCI-IV-2A**：A/O/N 使用 2026-10-08 完成的 seed0 运行；D 使用 `seed0_20260929_154546_823179` 原始运行。四组分类均训练 50 轮，按验证 BAcc 选模；D Stage1 训练 20 轮。47.48% 为后续以重建为主的消融结果，未替换此处 D 基线。详见 [BCI 实验说明](bciiv2a.md)。

纠正前面的判断：**脚本默认轮数不同，但这里已有的 A/O/N 实验实际都跑了 30 轮，与这些 D 结果一致。** 数据路径对应的内容和划分尚未核实，所以这张表是现有结果对照，还不能作为完全统一条件的最终结论。

---

如果只保留刚才表格对应的 **A/O/N/D 基线结果**，保留下面这些目录。路径均相对于 `outputs/`。

**A/O/N：保留 seed0 这一批，目录内的 A、O、N 都保留**

```text
erpcore/aon/seed0_20260930_000705_2308507/
tuev/aon/seed0_20260930_000705_2308507/
seed/aon/seed0_20260930_000705_2308507/
seedv/aon/seed0_20260930_000705_2308507/
shu/aon/seed0_20260930_000705_2308507/
aon_serial/seed0_20260930_000705_2308507/
```

**D：保留这些无额外 corrector 时间编码的运行**

```text
erpcore/erp_core_D_stage1/seed0_20260929_102523_63633/  # Stage1：最佳 epoch=15，Val Loss=184.5376，Test 重建损失（loss_missing）=0.008694；作为下方 Stage2 的初始化
erpcore/erp_core_D_stage2/  # Stage2：最佳 epoch=4（验证 BAcc 选模），Test Acc=56.85%，BAcc=37.43%，Kappa=0.4566，加权 F1=52.88%

tuev/tuev_D_stage1/seed0_20260929_113930_282524/  # Stage1：最佳 epoch=26，Val Loss=231.6276，Test 重建损失（loss_missing）=0.004922；作为下方 Stage2 的初始化
tuev/tuev_D_stage2/  # Stage2：按逐 epoch 日志重新取验证 BAcc 最佳 epoch=16，Test Acc=72.96%，BAcc=57.32%，Kappa=0.5120，加权 F1=75.29%；现有 checkpoint-best.pth 仍为 Kappa 最佳 epoch=0，未保存 epoch=16 权重

seed/D_pipeline/seed0_20260929_172810_1061101/  # Stage2：最佳 epoch=29（验证 BAcc 选模），Test Acc=55.29%，BAcc=54.90%，Kappa=0.3284，加权 F1=54.66%
seed/D_pipeline/seed0_20260929_174452_1112332/  # Stage2：最佳 epoch=19（验证 BAcc 选模），Test Acc=56.13%，BAcc=55.74%，Kappa=0.3410，加权 F1=55.51%

seedv/D_pipeline/seed0_20260929_180714_1194118/  # Stage2：最佳 epoch=13（验证 Acc 选模），Test Acc=38.20%，BAcc=37.31%，Kappa=0.2203，加权 F1=38.12%

shu/D_pipeline/seed0_20260929_181806_1239449/  # Stage2：最佳 epoch=11（验证 BAcc 选模），Test Acc=56.10%，BAcc=56.11%，Kappa=0.1223，加权 F1=55.92%
```

**如果还要保留 BCI 的 D 基线，额外保留**

```text
bciiv2a_cbramod/D_pipeline/seed0_20260929_154546_823179/
```

每个运行内部至少保留 **`checkpoint-best.pth`、`log.txt`、运行日志和 `pipeline.log`（如有）**。需要续训则再保留 `checkpoint.pth`。

这是保留清单，目前没有删除任何内容。

| 检查项 | 要确认什么 |
|---|---|
| 数据划分 | 同一数据集 A/O/N/D 的 train、val、test 样本 ID 是否一致 |
| 预处理 | 归一化、单位、采样率、导联顺序是否一致 |
| 初始化 | CNN 来源是否相同，Prototype 是否来自对应冻结 CNN，且仅用训练集生成 |
| 分类训练 | 实际 30 轮、seed0、优化器、冻结范围和分类器设置是否一致 |
| D 阶段交接 | Stage2 实际加载的是哪一个 Stage1 checkpoint |


TUEV 选模口径复核：Stage2 的 log.txt 完整记录 epoch 0–29。验证 BAcc 唯一最高值为 68.6158%（epoch=16），该轮 Test Acc=72.9624%、BAcc=57.3240%、Kappa=0.512049、加权 F1=75.2945%。结果可直接从日志提取，无需重跑训练；若需要导出该轮模型，现有文件中没有 epoch=16 checkpoint。

---

## SHU corrector 时间编码：动机、实现与结果

**为什么加：** 当前 SHU 每个样本在 200 Hz 下有 800 个采样点，即 4 秒；LaBraM 的 patch_size=200，因此每个导联被分成 **4 个时间片**，每片 1 秒。CNN 输出的观测特征具有时间片差异，但缺失导联的 Prototype 是按导联保存的平均特征，在这 4 个时间片上重复使用。corrector 原本没有显式时间位置编码：展平后的 token 顺序本身并不等于向注意力模块提供位置标记。因此，希望加入时间编码，让 corrector 能区分第 1、2、3、4 个时间片，为时序补全提供位置线索。这不意味着原始 CNN 特征完全没有时间相关信息。

**怎么加：** 加法形式参考了 LaBraM 本身的时间编码。LaBraM 定义可学习参数 `self.time_embed`，形状为 `[1, 16, embed_dim]`。前向计算时截取当前时间片数对应的编码，并复制到各导联上，再按与 token 相同的顺序展平。在补全完成、拼接 CLS 和加入通道位置编码之后，进入 LaBraM 主 Transformer 之前，执行：

```python
# LaBraM 原有时间编码：同一时间片的各导联共享同一个时间向量。
time_embed = self.time_embed[:, :input_time_window, :]
time_embed = time_embed.unsqueeze(1).expand(
    batch_size, target_channels_num, -1, -1
).flatten(1, 2)
x[:, 1:, :] += time_embed  # 跳过 CLS token
```

corrector 的新增实现沿用“可学习时间向量 + token 特征、跨导联共享”的方式，但使用独立的 `nn.Embedding(16, embed_dim)`，不复用 LaBraM 的 `self.time_embed`。对 SHU 取时间索引 0、1、2、3，将时间向量广播成 `[1, 1, 4, embed_dim]`，在展平、拼接 token 并进入 corrector 之前执行：

```python
time_pos = self.corrector["time_embedding"](
    torch.arange(num_t, device=h_obs.device)
).to(dtype=h_obs.dtype)[None, None, :, :]
obs_tokens = (h_obs + time_pos).flatten(1, 2)
miss_tokens = (p_miss + time_pos).flatten(1, 2)
tokens = torch.cat((obs_tokens, miss_tokens), dim=1)
```

**为什么这么加：** 观测 token 和缺失 Prototype token 在同一时间片使用相同标记，方便位置对齐；所有导联共享时间向量，使编码表达时间位置。新增 Embedding 初始化为零，初始时不改变 corrector 输入；这与 LaBraM 原有时间编码的截断正态初始化不同。编码只加到 corrector 输入，CNN 重建目标和最终补全公式中的原始 Prototype 不变。最终仍为 `h_pred_miss = p_miss + d_sub + d_task`。LaBraM 主 Transformer 前原有的时间编码继续保留。

**加了结果如何：** 以下均为实际 seed0，按验证 BAcc 最佳 epoch 读取测试结果，epoch 从 0 开始。

| 运行目录（相对于 outputs/shu/D_pipeline/） | corrector 额外时间编码 | 最佳 epoch | Val BAcc (%) | Test BAcc (%) |
|---|---|---:|---:|---:|
| seed0_20260929_181806_1239449 | 无 | 11 | 65.12 | 56.11 |
| seed1_20260929_220420_1975098 | 无 | 18 | 65.48 | 56.61 |
| seed2_20260929_220507_1976650 | 无 | 9 | 65.97 | 54.76 |
| seed0_20260930_154409_1340884 | 有 | 29 | 65.06 | 53.15 |
| seed0_20260930_161757_1423316 | 有 | 13 | 64.69 | 52.51 |

现有结果没有显示收益，带额外时间编码的两次测试 BAcc 更低。目录名中的 seed1、seed2 不代表实际种子，这些运行的 Stage1 和 Stage2 日志均为 seed0。

**是否加对了：** 从代码位置、广播维度和观测/缺失 token 的对齐方式看，实现符合“给 corrector 提供时间位置”的目的，并参考了 LaBraM 的加法形式。但实现合理不等于有效；目前不能确认下降完全由时间编码造成，仍需核对其他配置并进行受控对照。当前正式基线保留无额外 corrector 时间编码的版本。
