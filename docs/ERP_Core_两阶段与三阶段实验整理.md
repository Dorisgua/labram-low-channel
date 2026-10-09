# ERP-Core：两阶段与三阶段解耦补全实验整理

整理日期：2026-10-09。用途：PPT 素材与中文汇报讲稿。仅整理既有代码、日志、文档和图片；本次未修改代码或脚本，未启动训练。暂时跳过 FiLM。

**可讲的主线：** 固定 Prototype 在少导联基线上收益有限，因此尝试用当前样本生成 subject/task 双分支修正。两阶段联合学习补全及分支约束，但分类收益和解耦证据尚未达到预期；三阶段进一步尝试先在全导联上学习分支，再转到少导联补全。已有三阶段结果显示重建权重调整会改变 MSE，尚不能证明阶段拆分本身改善解耦或稳定提升分类。

> 阅读约定：所有路径相对于当前仓库根目录，`../` 表示相邻本地仓库。相邻仓库结果单独报告，不冒充当前仓库运行。Acc/BAcc 单位为 %，epoch 从 0 开始；每个分类运行按验证 BAcc 选模，再读取该 epoch 的测试成绩。补全 checkpoint 按验证加权总 Loss 选模，不按测试 MSE 选最好的一轮。表中列出全部已发现的正式三阶段配置，不按 Test 成绩挑一次作为代表。`_3stage_uni` 中同名历史输出未作为独立重复实验计数。

## 1. 方法与动机

| 方法 | 实际输入与处理 | 研究作用 |
|---|---|---|
| O：全导联 | 28 个真实导联 | 全导联参照；不是理论上限 |
| N：少导联 | 12 个真实导联，不补全 | 缺失导联基线 |
| A：固定 Prototype | 12 个真实导联的 CNN 特征 + 16 个训练集平均特征 | 补充对应导联的典型特征，不依赖当前样本 |
| D：动态补全 | 观测特征与缺失 Prototype 输入 corrector，双分支输出修正量 | 希望补入当前样本相关信息 |

来源：`docs/README.md` §1；`modeling_finetune.py::forward_features`（固定补全）；`modeling_dynamic_stage1.py::_encode_dynamic_tokens`（动态补全）；Prototype 生成脚本 `docs/prototypes/01_generate_erpcore_cnn_patch_prototypes.py`。

补全公式：

```text
h_pred_miss = p_miss + d_sub + d_task
预测缺失特征 = Prototype + subject 修正量 + task 修正量
```

subject/task 是设计目标，命名不构成解耦证据。当前实现为一个 shared Transformer encoder，后接 subject/task 各一个 Transformer encoder 与 LayerNorm；取缺失位置输出作为 correction，缺失位置均值为 `z_sub/z_task`。当前两个仓库的前向均直接相加，没有 tanh 或 correction_scale 乘法；即使日志保留 `correction_scale=1.0` 参数，也不能按参数名解释为实际缩放。

**参考方法核对：** 本地 `../CSLP-AE/split_model.py` 有 `subject_task_encode/decode`、对比损失与 latent permutation；`../CSLP-AE/train.py:100` 附近明确列出 `sub_contra_s/task_contra_t/latent_permute_s/latent_permute_t`。本项目 `losses_dynamic.py::compute_stage1_losses`、`data_processor/pair_sampling.py::sample_cslpae_pair_batch` 对应借鉴同属性配对、subject/task 双分支、双向对比与交换后重建约束。不能称为完整复现 CSLP-AE：本项目在冻结 CNN 特征空间做 Prototype 加性修正，使用 Transformer corrector，InfoNCE 温度固定 0.2；CSLP 本地代码有编码/解码网络与可学习 logit scale。论文版本、具体引用页码和与原论文逐项对应关系仍待确认，不纳入外部论文成绩。

固定补全收益有限的依据是第 3 节当前同协议 seed0：A 比 N 的 BAcc 高 0.56 个百分点，仍比 O 低 5.10 个百分点；三个 seed 平均收益约 0.50 个百分点。只能作为探索动态补全的动机，不能称为固定方法无效或动态方法必然更好。

## 2. 两阶段实际实现

| 阶段 | 输入与监督 | 实际训练/冻结 |
|---|---|---|
| Stage1 | 主输入 `x_obs=[B,12,1,200]`；`x_full=[B,28,1,200]` 只提供同一冻结 CNN 的真实缺失 token 监督；subject/task ID 用于配对 | 强制冻结 `patch_embed`；Prototype 为 buffer；更新 corrector 的 shared/subject/task encoder 与对应 LayerNorm。主 LaBraM Transformer/分类头不进入该阶段损失前向 |
| Stage2 | 推理及分类只读 12 个观测导联；补到 28 个 token 后输入主 Transformer，使用 12 类标签监督 | 日志 `freeze_cnn=True`；代码 `freeze_corrector()`；训练主 Transformer 与 AdaBrain 分类读出 |

来源：`modeling_dynamic_stage1.py:370`、`:499`、`:549`、`:670`；`run_dynamic_stage1.py:976`；`run_class_finetuning.py:953`、`:968`；`engine_for_dynamic_stage1.py:150`、`:324`。

冻结细节：Stage1 只显式将 CNN 的 `requires_grad=False`；其余未参与 forward 的 backbone 参数不等于逐个显式冻结，但没有该损失梯度。不能把日志 `n_parameters` 直接当作实际更新的 corrector 参数数。

Stage2 的 `classifier_mode=adabrain_all_token`、`classifier_token_scope=real` 表示分类头读取 Transformer 输出的真实导联位置；补全 token 仍参与 Transformer 注意力，可间接影响真实导联输出。不能说分类头直接拼接全部 28 个位置，也不能说补全 token 没有作用。

### 2.1 分支约束如何工作

- **重建：** `L_missing = mean((p_miss+d_sub+d_task-h_miss_target)^2)`，只覆盖缺失 CNN token。
- **对比：** 同 subject 样本比较 `d_sub`，同 task 样本比较 `d_task`；展平、L2 normalize 后做双向对角正样本 InfoNCE。也支持对均值 summary `z` 做对比，但主运行未启用。
- **交换：** 同 subject 配对交换 `d_sub`，保留各自 `d_task`；同 task 配对交换 `d_task`，保留各自 `d_sub`。左右两个交换预测分别对各自目标计算 MSE，再取平均。
- **正则：** 支持 `mean(d_sub²)+mean(d_task²)`，当前主运行权重为 0。

配对只保证同 subject 或同 task，不强制另一属性不同；因此存在同 subject 同 task 的配对，约束并不能直接保证独立性。交换 loss 是训练约束，不是独立、受控的交换成功实验。

### 2.2 实际启用权重，以运行日志为准

| 运行 | 重建 | 正则 | subject/task summary 对比 | subject/task correction 对比 | subject/task 交换 | 轮数 | 角色 |
|---|---:|---:|---|---|---|---:|---|
| `erp_core_D_stage1/seed0_20260929_102523_63633` | 500 | 0 | 0 / 0 | 50 / 50 | 1 / 1 | 50 | 当前 D 分类真正加载的 Stage1 |
| `erp_core_D_stage1/seed0_20260929_103212_81175` | 0 | 0 | 0 / 0 | 50 / 50 | 1 / 1 | 50 | 已有线性探针/t-SNE 的消融；不能替代上一行 |
| `recon_only/seed0_20260929_233627_2214963/stage1` | 500 | 0 | 0 / 0 | 0 / 0 | 0 / 0 | 50 | 仅重建消融 |

来源：上述目录各自 `run_logs/*.log` 的首个 `Namespace(...)`；D 分类 `outputs/erpcore/erp_core_D_stage2/run_logs/erp_core_D_stage2_seed0_20260929_113250.log` 的 `finetune` 明确指向 `102523_63633/checkpoint-best.pth`。总损失不是等权求和，不能横向比较不同权重配置的 total loss。

**时间编码：** 当前两阶段和相邻 ERP 三阶段的 corrector 直接拼接 token 后进入 shared encoder，未加额外 corrector 时间编码；主 LaBraM Transformer 的原有 `time_embed` 保留。ERP-Core 当前每导联仅 1 个 200 点 patch。`docs/answer.md` 讨论的额外时间编码及其他数据集结果不能移植到本实验。

## 3. ERP-Core 分类结果

### 3.1 当前仓库：30 轮、冻结 CNN 的主对照

运行日志共同记录：同一个 `simple_data.pt`；排除 N170 后 12 类；训练/验证/测试分别 49,943 / 7,161 / 14,485 个 trial，按被试划分。验证被试 `(4,7,27,33)`，测试被试 `(5,14,15,20,22,23,26,29)`。256 Hz 来源重采样到 200 点，使用训练集逐导联统计量 z-score，模型输入 scale=1.0。分类 batch=64、LR=5e-4、warmup=5、weight_decay=0.05、label smoothing=0.1，关闭主 Transformer relative position bias/qkv bias，启用绝对位置编码，读出 scope=real。两阶段 D 的 corrector 来自 50 轮 Stage1。

| 方法 | 观测/目标导联数 | 实际 seed | 分类训练轮数 | 验证集选模指标 | 最佳 epoch | Test Acc (%) | Test BAcc (%) | 运行目录 |
|---|---|---:|---:|---|---:|---:|---:|---|
| O | 28/28 | 0 | 30 | Val BAcc | 26 | 60.55 | 45.39 | `outputs/erpcore/aon/seed0_20260930_000705_2308507/O` |
| N | 12/12 | 0 | 30 | Val BAcc | 1 | 60.59 | 39.73 | `outputs/erpcore/aon/seed0_20260930_000705_2308507/N` |
| A | 12/28 | 0 | 30 | Val BAcc | 1 | 61.38 | 40.29 | `outputs/erpcore/aon/seed0_20260930_000705_2308507/A` |
| O | 28/28 | 1 | 30 | Val BAcc | 8 | 59.43 | 46.07 | `outputs/erpcore/aon/seed1_20260930_000949_2318130/O` |
| N | 12/12 | 1 | 30 | Val BAcc | 1 | 60.41 | 37.07 | `outputs/erpcore/aon/seed1_20260930_000949_2318130/N` |
| A | 12/28 | 1 | 30 | Val BAcc | 4 | 60.16 | 40.38 | `outputs/erpcore/aon/seed1_20260930_000949_2318130/A` |
| O | 28/28 | 2 | 30 | Val BAcc | 28 | 60.57 | 45.52 | `outputs/erpcore/aon/seed2_20260930_000954_2319386/O` |
| N | 12/12 | 2 | 30 | Val BAcc | 6 | 57.95 | 40.03 | `outputs/erpcore/aon/seed2_20260930_000954_2319386/N` |
| A | 12/28 | 2 | 30 | Val BAcc | 1 | 60.68 | 37.67 | `outputs/erpcore/aon/seed2_20260930_000954_2319386/A` |
| 两阶段 D | 12/28 | 0 | 30 | Val BAcc | 4 | 56.85 | 37.43 | `outputs/erpcore/erp_core_D_stage2/checkpoints/erp_core_D_stage2_seed0_20260929_113250` |

表 3.1 来源：每行运行目录的 `log.txt`，最后一行 `best_epoch` 指向的 epoch 行；配置为对应 `run_logs/*.log` 的 `Command/Namespace/ERP CORE PT audit/Using ... classifier`。D 的 run log 位于 `outputs/erpcore/erp_core_D_stage2/run_logs/`。模型路径及冻结见第 2 节。seed0 的 O/N/A/D 可作单次同分类协议比较，但不保证分类头初始化逐参数相同：当前代码用于重置头初始化 RNG 的 `torch.manual_seed(seed)` 已被注释，创建 corrector 会消耗 RNG。

三个 seed 的 O/N/A 平均 BAcc 分别 45.66±0.36、38.94±1.63、39.45±1.54（样本标准差）。D 只有 seed0，不能和三 seed 均值进行显著性比较。seed0 D 比 N 低 2.30、比 A 低 2.86 个百分点，目前未观察到动态补全分类收益。

### 3.2 相邻三阶段仓库：50 轮分类，单 seed 配置探索

以下共同为 seed0、12→28、Stage1 50 轮全导联训练、Stage2 20 轮少导联补全、Stage3 50 轮分类。Stage2 保留 correction 对比各 1、交换各 1，只改变直接重建权重；三组 `run_from_stage1` 均继承 `run_20260920_103722_87486/stage1/checkpoint-best.pth`。不是多 seed 重复实验。

| 方法 | 观测/目标导联数 | 实际 seed | 分类训练轮数 | 验证集选模指标 | 最佳 epoch | Test Acc (%) | Test BAcc (%) | 运行目录 |
|---|---|---:|---:|---|---:|---:|---:|---|
| 三阶段 D，重建×500 | 12/28 | 0 | 50 | Val BAcc | 4 | 58.61 | 37.27 | `../LaBraM-unified-AON-dynamic_general_3stage/outputs/erpcore/D_3stage/run_20260920_103722_87486/stage3` |
| 三阶段 D，重建×2500 | 12/28 | 0 | 50 | Val BAcc | 42 | 54.39 | 36.98 | `../LaBraM-unified-AON-dynamic_general_3stage/outputs/erpcore/D_3stage/run_from_stage1_recon2500_20260923_104153_48323/stage3` |
| 三阶段 D，重建×5000 | 12/28 | 0 | 50 | Val BAcc | 3 | 60.06 | 38.43 | `../LaBraM-unified-AON-dynamic_general_3stage/outputs/erpcore/D_3stage/run_from_stage1_recon5000_20260920_144018_549254/stage3` |
| 三阶段 D，重建×50000 | 12/28 | 0 | 50 | Val BAcc | 6 | 57.94 | 39.54 | `../LaBraM-unified-AON-dynamic_general_3stage/outputs/erpcore/D_3stage/run_from_stage1_recon50000_20260923_103921_43084/stage3` |
| A，50 轮参照 | 12/28 | 0 | 50 | Val BAcc | 1 | 61.38 | 40.29 | `../LaBraM-unified-AON-dynamic_general_3stage/outputs/erpcore/erp_core_A_freeze_cnn/checkpoints/erp_core_A_freeze_cnn_seed0_20260923_130138` |

表 3.2 来源：各行 `stage3/log.txt` 及 `stage3/run_logs/*.log`，A 的 run log 在 `erp_core_A_freeze_cnn/run_logs/`。重建配置见各 `stage2/run_logs/*.log`。同划分、分类 batch/LR/冻结与读出模式已从日志核对；但分类轮数从 30 改为 50，学习率日程相应不同，Stage1 输入/权重和 corrector 初始化也变化，必须与表 3.1 分开。

50 轮 A 的选中 epoch 和指标与当前 A seed0 相同；这不代表 30/50 轮训练协议等价。当前没有发现三阶段目录中与其完全配套的 O/N 50 轮正式运行，填写待核实，不能用历史 full finetune 的 O/N 拼成严格三阶段对照。

### 3.3 消融与审计，另表保留

| 方法 | 观测/目标导联数 | 实际 seed | 分类训练轮数 | 验证集选模指标 | 最佳 epoch | Test Acc (%) | Test BAcc (%) | 运行目录 |
|---|---|---:|---:|---|---:|---:|---:|---|
| 两阶段 D，仅重建消融 | 12/28 | 0 | 30 | Val BAcc | 1 | 60.44 | 37.98 | `outputs/erpcore/recon_only/seed0_20260929_233627_2214963/stage2` |
| 三阶段 uni，采样审计（非正式效果） | 12/28 | 0 | 1 | Val BAcc | 0 | 56.25 | 30.93 | `../LaBraM-unified-AON-dynamic_general_3stage_uni/outputs/erpcore/D_3stage/sampling_audit_20260928/stage3` |

表 3.3 来源：每行 `log.txt` 与 `run_logs/*.log`。仅重建分类仍训练 30 轮，BAcc 37.98% 没超过 A/N seed0。uni 审计 Stage2/3 各 1 轮且用于采样检查，不能作为三阶段性能结论。历史 14/21 导联及 full finetune 另见旧文档，它们改变观测集合、seed 或冻结策略，本故事不混入主表。

## 4. 补全结果

### 4.1 MSE 定义与可比范围

`losses_dynamic.py:11` 的 `F.mse_loss` 对 `[B,16,1,200]` 全部元素取平均，即冻结 CNN 缺失导联特征 MSE，**不是 EEG 波形 MSE**。`engine_for_dynamic_stage1.py:335` 用样本数加权累计评估 batch，覆盖整个指定 Val/Test split。ERP 加载器在训练 split 计算 28 导联 mean/std，所有 split 共用；先规范化 full EEG，再切 observed，输入缩放 1.0。来源：`data_processor/erpcore.py:185`、`:303`，`run_dynamic_stage1.py` ERP 配置。

12 观测集合：`FP1 FP2 F3 F4 F7 F8 C3 C4 P3 P4 O1 O2`。

缺失集合 M16：`FC3 C5 P7 PO7 PO3 OZ PZ CPZ FZ FC4 FCZ CZ C6 P8 PO8 PO4`（按目标 28 导联顺序）。目标：`FP1 F3 F7 FC3 C3 C5 P3 P7 PO7 PO3 O1 OZ PZ CPZ FP2 FZ F4 F8 FC4 FCZ CZ C4 C6 P4 P8 PO8 PO4 O2`。来源：主运行日志 `ERPCORE channels/Prototype channels`。

本次 SHA256 校验当前与三阶段仓库的 base checkpoint 完全相同（`7c50583826afac76c4ab18f43d958df40496c8229accc09ed6a227c9bb57c37c`），Prototype 文件也相同（`f204cf453a1fa94c0ee187d4c79b73b1ab6a7865d9200ab1341e1f21d9a56092`）。同 split、M16、z-score、scale 与 MSE reduction 支持以下数值对照；仍需承认日志不保存完整运行时源码快照，历史实现一致性不能仅凭当前代码百分之百保证。

| 方法 | 缺失导联集合 | 评估划分 | 缺失特征 MSE | checkpoint / 运行目录 |
|---|---|---|---:|---|
| A：固定 Prototype | M16 | 完整 Test（14,485）；Val（7,161） | Test 0.007600625；Val 0.008762095 | `outputs/erpcore/prototype_mse_baseline/metrics.json`；使用主 D 的冻结 CNN，固定 Prototype 直接填入缺失位置 |
| 两阶段 D（主分类来源） | M16 | 完整 Test（14,485）；Val（7,161） | Test 0.008694414；Val 0.009664306 | `outputs/erpcore/erp_core_D_stage1/seed0_20260929_102523_63633/checkpoint-best.pth`；选中 epoch 15 |
| 两阶段 D（关闭直接重建的诊断消融） | M16 | 完整 Test（14,485）；Val（7,161） | Test 0.256009557；Val 0.261097122 | `outputs/erpcore/erp_core_D_stage1/seed0_20260929_103212_81175/checkpoint-best.pth`；选中 epoch 11 |
| 两阶段 D（仅重建） | M16 | 完整 Test（14,485）；Val（7,161） | Test 0.004044031；Val 0.004395205 | `outputs/erpcore/recon_only/seed0_20260929_233627_2214963/stage1/checkpoint-best.pth`；选中 epoch 49 |
| 三阶段 D，重建×500 | M16 | 完整 Test（14,485）；Val（7,161） | Test 0.007064448；Val 0.008051906 | `../LaBraM-unified-AON-dynamic_general_3stage/outputs/erpcore/D_3stage/run_20260920_103722_87486/stage2/checkpoint-best.pth`；选中 epoch 17 |
| 三阶段 D，重建×2500 | M16 | 完整 Test（14,485）；Val（7,161） | Test 0.004113958；Val 0.004484244 | `../LaBraM-unified-AON-dynamic_general_3stage/outputs/erpcore/D_3stage/run_from_stage1_recon2500_20260923_104153_48323/stage2/checkpoint-best.pth`；选中 epoch 17 |
| 三阶段 D，重建×5000 | M16 | 完整 Test（14,485）；Val（7,161） | Test 0.003931219；Val 0.004312771 | `../LaBraM-unified-AON-dynamic_general_3stage/outputs/erpcore/D_3stage/run_from_stage1_recon5000_20260920_144018_549254/stage2/checkpoint-best.pth`；选中 epoch 19 |
| 三阶段 D，重建×50000 | M16 | 完整 Test（14,485）；Val（7,161） | Test 0.003896447；Val 0.004262441 | `../LaBraM-unified-AON-dynamic_general_3stage/outputs/erpcore/D_3stage/run_from_stage1_recon50000_20260923_103921_43084/stage2/checkpoint-best.pth`；选中 epoch 17 |

表 4.1 来源：各 checkpoint 同目录 `log.txt` 的选中 epoch，数值为未乘权重的 `test_loss_missing/val_loss_missing`；配置为各 `run_logs/*.log`。三阶段全导联 Stage1 覆盖全部 28 个位置，其 Test MSE 0.011902423 不能作为 M16 补全误差和本表相减。

在已核对的相同缺失范围下，三阶段重建×500 的 Test MSE 比两阶段主 D 低 0.001629966（18.75%），但分类 BAcc 为 37.27%，没有超过两阶段的 37.43%。三阶段×50000 比×500 的 MSE 低 0.003168001（44.84%），其 BAcc 39.54% 高于×500 的 37.27%；两者只是单 seed 权重探索，且×2500 的 MSE 更低但 BAcc 36.98% 低于×500，不能概括为 MSE 越低分类越好。×50000 也未超过同为 50 轮的 A（40.29%）。

**固定 Prototype 基线已补齐（2026-10-09）：** 使用 `tools/evaluate_erpcore_prototype_mse.py`，在主 D 的冻结 CNN 空间，以相同训练集 z-score、input_scale=1.0、M16 范围评估完整 Val/Test；CNN 前向使用 CUDA autocast，误差以 float32 计算、按样本数加权。结果及 checkpoint/Prototype 哈希保存于 `outputs/erpcore/prototype_mse_baseline/metrics.json`。三阶段×500 的 Test MSE 比固定 Prototype 低约 7.05%，×50000 低约 48.74%；两阶段主 D 反而比固定 Prototype 高约 14.39%。这些重建结果不等于分类优势。

**历史局部材料：** 旧文档 `docs/尝试从12导联变成14导联看AND情况_备份.md` §0.2 报告 Worst 5% Train 子集的 C5/C6/P8/P7/PO7 上 Dynamic/Prototype MSE，如 C5 0.02319/0.02689；这不是完整 Test 的 M16 平均 MSE。相邻旧 dynamic 仓库 `test_c5_first_trial_by_subject/samples_and_metrics.csv` 也只比较每被试首个 C5 trial。可作为局部补得更像的示例，不能替代 A 的全范围基线，更不能和表 4.1 相减。

## 5. 两阶段解耦诊断

### 5.1 已有图和线性探针

当前可追溯的少导联诊断目录：

`outputs/erpcore/erp_core_D_stage1/seed0_20260929_103212_81175/disentanglement_best/`

图片：`comparison_z_d_tsne.png`（z/d 双行四列）、`comparison_z_d_tsne.pdf`、`comparison_d_tsne.png`。元数据 `metadata.json` 明确 Test 抽样 n=2400、sample_seed=42、checkpoint SHA256=`3ad5765d856abbe2d6ac15f51060cda24d3ae49c0a58c7044d1be9549b0c75b9`。checkpoint 的 missing_weight=0，**不是主 D 分类的 Stage1**。

图中 subject 分支部分区域呈被试颜色聚集，task 分支有某些类别区域，但多种颜色仍混合，task 分支也有被试结构。图内旧标题写“pre-tanh/actual d=1.0*tanh”，与当前代码和 metadata 的“no tanh/no correction_scale”相矛盾；PPT 必须裁掉或注明旧标签，不能用图标题认定前向公式。

下表从既有 `linear_probe.csv` 按 probe seed 42/43/44 计算均值±样本标准差，指标为 BAcc（%）：

| 表征 | subject 探针 BAcc | task 探针 BAcc |
|---|---:|---:|
| `z_sub` | 50.08 ± 1.31 | 19.37 ± 0.44 |
| `z_task` | 47.51 ± 1.71 | 18.38 ± 0.54 |
| `d_sub` | 49.92 ± 0.95 | 20.25 ± 0.77 |
| `d_task` | 47.43 ± 1.68 | 19.46 ± 1.29 |

表 5.1 来源：上述 `linear_probe.csv`，字段 `target/feature/seed/balanced_accuracy/n_train/n_eval`；每次 n_train=1680、n_eval=720。三个 probe seed 是同一网络 checkpoint 的探针划分/重复，不能当作三次模型训练。生成此 CSV 的探针脚本、标准化、正则、划分粒度与交叉验证方式未找到可完整追溯的实现，须补充。

**观察：** subject 信息不只存在于 subject 分支，task 分支同样可预测 subject；task 探针上 task 分支并未明显强于 subject 分支。按标签数，均匀猜测 BAcc 参考值为 8 被试的 12.5%、12 类的 8.33%，但探针本身是已见被试内评估，不代表未见被试泛化。现有证据不支持“两个分支已分别只保留目标信息”。

### 5.2 分支抵消：本次只读既有特征的补充统计

对上述目录的 `d_sub.npy/d_task.npy`（同 2400 个 Test 样本，各展平为 3200 维），本次只读计算：

| 统计量 | 值 | 观察与局限 |
|---|---:|---|
| `mean(d_sub²)` | 0.241718 | 分支幅值，不是重建误差 |
| `mean(d_task²)` | 0.236086 | 分支幅值，不是重建误差 |
| `mean((d_sub+d_task)²)` | 0.243961 | 和的能量低于两分支能量之和 |
| 样本级余弦相似度均值 | −0.533410 | 存在较强反向分量 |
| 总和能量 / 两分支能量之和 | 0.510588 | 加性修正存在部分抵消；不是完全抵消 |

表 5.2 来源：既有两个 `.npy`，以 float64 对全部样本/特征计算均方能量；余弦逐样本计算再平均。没有重新提取特征或启动训练。由于这是关闭直接重建的消融，不能推广为主 D checkpoint 的行为；反向分量也可能是加性拟合的结果，不能证明“抵消导致分类失败”。

### 5.3 全导联/少导联与交换证据缺口

当前两阶段模型 `_encode_dynamic_tokens` 以 12 观测+16 Prototype 编码，未发现同一主 D checkpoint 的全导联/少导联配套诊断图。相邻旧 `../LaBraM-unified-AON-dynamic/outputs/erpcore/erp_core_D_stage1/disentanglement_best/comparison_z_d_tsne.png` 是另一个 20 轮、不同损失权重/correction_scale 的 checkpoint，应作为历史材料另注，不能拼成当前模型的全导联对照。

未找到主 D 与三阶段使用同探针协议的 leakage 汇总、受控交换测试、单分支去除分类或抵消分析。训练日志的交换 MSE 不等于在固定另一因素后验证可交换性。不凭 t-SNE 聚类宣布成功，也不把“多目标冲突导致失败”当作已证实原因。

## 6. 为什么尝试三阶段

**已观察的问题：** 当前联合 Stage1→分类的 D seed0 未超过 N/A；诊断消融的分支仍有交叉信息，分支命名和结构没有直接保证信息分离。主分类 checkpoint 的诊断尚不齐全。

**提出的假设：** 在少导联输入上同时适应缺失补全和分支约束，可能让学习分支结构更困难；先用真实全导联建立表征，再适配少导联，可能更有利。这只是可检验假设，不是对现有失败的因果解释。

**三阶段怎样检验：** 相邻仓库让第一阶段使用全导联，第二阶段继承其 checkpoint 并转为少导联，最后训练分类。若要检验假设，须比较同样本、同范围的分支探针与受控交换，并测重建后结构是否保留。已有流程还同时改变权重、训练总轮数，故只能作为探索证据。

| 三阶段实际阶段 | 输入及监督范围 | 训练和冻结 | 实际损失/选模 |
|---|---|---|---|
| Stage1（50 轮） | `fullchannel=True`：读取 x_full 的 28 个真实 CNN token；对所有 28 个位置组成 `Prototype+d_sub+d_task` 并计算交换目标 | CNN 冻结；shared/subject/task encoder 与对应 norm 更新；不调用主分类 Transformer/head | 直接重建=0、正则=0、summary=0；correction 对比各1、交换各1；Val total loss 最小，epoch29 |
| Stage2（20 轮） | `fullchannel=False`：12 个真实特征+16 Prototype；只对 M16 缺失特征监督 | 继承 Stage1 best，加载过滤名为空；CNN 冻结，**整个 corrector 继续训练**，并非冻结两分支只训 decoder | 重建=500/2500/5000/50000；correction 对比各1、交换各1，其他0；Val total loss 最小 |
| Stage3（50 轮） | 12 个真实导联，动态补到28个位置，12类分类监督 | 继承 Stage2 best；冻结 CNN/corrector，训练主 Transformer/分类读出 | 分类损失，按 Val BAcc 最大选模 |

表 6.1 来源：`../LaBraM-unified-AON-dynamic_general_3stage/docs/erp_core_minimal_3stage.md`；每阶段 `run_logs/*.log` 的 `fullchannel/finetune/epochs/*weight`；相邻 `modeling_dynamic_stage1.py:492`、`:512`、`:554`，`run_dynamic_stage1.py:944`，分类入口 `freeze_corrector`。三阶段不是“先完全解耦，再只优化重建”：Stage1 虽关闭直接重建，交换项仍包含重建；Stage2 也保留分支约束。未看到额外独立 decoder 或只训补全头的实现。

历史时间也需要说明：已找到的三阶段运行在 9 月20–23日，当前两阶段主分类与诊断在 9 月29日。以上是逻辑上组织汇报的实验故事，不声称三阶段是看完这次 9 月29日诊断后才启动；真实决策时间及早期诊断证据待补。

## 7. 三阶段结果与当前结论

### 7.1 表征区分是否改善

已有三阶段少导联适配后的图位于：

- `../LaBraM-unified-AON-dynamic_general_3stage/outputs/erpcore/D_3stage/run_20260920_103722_87486/stage2/disentanglement_full_few/comparison_d_tsne.png`
- 同目录 `comparison_z_tsne.png`、`metadata.json`、`sample_alignment.csv`。
- 重建×5000 的对应图：`../LaBraM-unified-AON-dynamic_general_3stage/outputs/erpcore/D_3stage/run_from_stage1_recon5000_20260920_144018_549254/stage2/disentanglement_full_few/`。

metadata 明确使用**同一个冻结 Stage2 checkpoint**与全部 14,485 个 Test 样本，三行分别为 full28（全真实输入全部28位置）、full16（全真实输入中选M16位置）、few16（12真实+16Prototype 的缺失位置）。这不是三个训练阶段的图，也不是 Stage1→Stage2 演变。

已查看×500 的 d 图：full16/few16 的 subject 分支有被试色块，task 分支也有被试色块；task 分类中可见局部类别区域但多类仍混合。输入变化确实改变可视结构，不能确认交叉信息减少。各特征独立拟合 t-SNE，轴/距离不能跨图比较；KL divergence 是 t-SNE 拟合指标，不是解耦评分。当前缺少三阶段与主两阶段同协议探针，以及三阶段 Stage1/Stage2 的配套前后诊断，**表征区分改善待核实**。

### 7.2 缺失特征 MSE 是否下降

第4节的同 M16、完整 Test 日志支持：三阶段配置的 MSE 可以低于当前两阶段主 D；增大重建权重总体降低此组配置的 MSE。但这同时改变了对比相对权重、全导联预训练和训练时长，不能归因于“三阶段”这个单一因素。固定 Prototype 完整 Test MSE 为 0.007600625，当前四组三阶段配置的 MSE 均更低；这仍是单 seed 配置对照，不能据此声称稳定分类优势。

### 7.3 下游分类是否提升

三阶段×500/2500/5000/50000 的 BAcc 分别 37.27/36.98/38.43/39.54%。相比两阶段37.43%，既有较低也有较高；相邻50轮 A 为40.29%，三阶段四组均未超过 A。分类训练从30改50轮，且仅 seed0，**没有成立的受控证据表明拆阶段本身提升分类**。

| 问题 | 已观察结果 | 解释假设 | 仍待验证 |
|---|---|---|---|
| 分支是否分开 | 部分局部结构与显著交叉信息并存 | 先全导联可能有助于形成结构 | 同协议 leakage、Stage1/2 前后探针与交换 |
| 补全是否更像 | 三阶段若干配置 M16 MSE 更低 | 更强重建权重可能更重视拟合目标 | 同设置多 seed（固定 Prototype 完整同范围基线已补齐） |
| 分类是否更好 | 某些三阶段配置高于两阶段，仍低于50轮 A | 重建信息未必充分服务判别读出 | 分类轮数/初始化/冻结/权重完全一致的两阶段 vs 三阶段 |

表 7.1 来源：表3、4、5以及三阶段诊断 metadata/图片；解释列不是已证实结论。

## 8. PPT 展示与可直接讲的中文稿

### 第1页：从固定均值到动态修正

素材：第1节方法表、补全公式；已有 `docs/ppt/fixed_prototype_completion.png`，或 `docs/ppt/fixed_prototype_completion_scalp.png`；少导联布局 `docs/figures/erpcore_channel_layouts/erpcore_12_observed.png`。

讲稿：“在 ERP-Core 的当前少导联设置中，固定 Prototype 相比直接用少导联只有较小的 BAcc 收益，和全导联仍有差距。因此我尝试用当前样本产生两个修正分支，希望分别承载 subject 和 task 相关信息；这两个名字是设计目标，是否分开需要后续验证。”

### 第2页：两阶段联合学习补全与分支约束

素材：第2节阶段表与实际损失权重表；可据公式画 shared→subject/task→相加的流程图。

讲稿：“第一阶段冻结 CNN，用真实全导联的缺失特征作为监督，同时加入同属性对比和分支交换约束。第二阶段冻结补全模块，训练主 Transformer 和分类头；当前主实验启用重建500、两个 correction 对比各50、交换各1。”

### 第3页：ERP-Core 分类：动态收益尚未出现

素材：表3.1优先只放 seed0 的 O/N/A/D 四行，角落标注30轮、Val BAcc选模；备份页放全部seed。

讲稿：“相同分类协议的 seed0 中，O、N、A、D 的测试 BAcc 分别是45.39%、39.73%、40.29%和37.43%。固定补全收益有限，而当前两阶段动态补全没有超过 A 或 N，因此需要继续检查补全质量和分支行为。”

### 第4页：重建更像，不一定分类更好

素材：表4.1删去无关行，保留两阶段主D与三阶段四配置；若放旧长尾图 `docs/26b857c9-7006-40bc-bda9-f95c6f3e1871.png`，必须标“历史12→28 Train诊断，非当前受控对照”。

讲稿：“这里的误差是缺失 CNN 特征的 MSE，不是原始 EEG 波形误差。已有三阶段配置可以把同范围 MSE 降低，但分类没有同步单调提高；固定 Prototype 的完整测试 MSE 已补齐，为 0.007600625；当前三阶段四组均更低，但不能将重建排序等同于分类排序。”

### 第5页：解耦诊断：目标信息仍然交叉

素材：第5节 `comparison_z_d_tsne.png`，去除旧 tanh 标签；表5.1线性探针。标注这是 missing_weight=0 消融、Test抽样2400，非主分类 checkpoint。

讲稿：“这组诊断里，subject 信息在两个分支中都能被探针预测，task 分支对 task 的预测也没有明显强于另一分支。局部聚类只能说明有结构，不能证明成功解耦；而且这组 checkpoint 关闭了直接重建，不能直接用它解释主分类运行。”

### 第6页：为什么尝试先全导联、再少导联

素材：表6.1三阶段实际流程、全/少导联输入区别；标注Stage2仍训练整个corrector并保留对比/交换。

讲稿：“我提出一个假设：先让分支看到完整信息，再适配缺失输入，可能比直接在少导联上联合训练更容易形成结构。因此尝试全导联分支训练、少导联补全适配、最后分类三步；它仍是联合约束的适配流程，不是把第二阶段完全变成只重建。”

### 第7页：三阶段诊断：观察输入变化，尚缺前后验证

素材：第7.1节 `comparison_d_tsne.png`，展示full16/few16两行；标“同一Stage2 checkpoint、同Test样本，非Stage1→2”。

讲稿：“这张图比较的是同一个训练后模型在全导联和少导联输入下的表示，输入变化会改变局部结构。两个分支仍都能看到被试结构，现在缺少统一探针和训练前后对照，所以不能把这张图当作三阶段解耦改善的证明。”

### 第8页：三阶段结果与下一步

素材：表3.2与表4.1同配置并排，表7.1作为备份。

讲稿：“四组三阶段配置的 BAcc 在36.98%到39.54%之间，均未超过同为50轮的固定补全40.29%。下一步应先统一分类轮数、初始化、损失和seed，再补齐主 checkpoint 的探针、受控交换及固定 Prototype 完整 MSE 对照，分别回答表征、重建和分类有没有改善。”

## 9. 来源、冲突与待核实清单

### 9.1 来源定位索引

| 表/论点 | 日志或产物 | 文档/代码关键定位 |
|---|---|---|
| 方法与两阶段流程 | 分类/Stage1 `run_logs` 的 `Namespace`、`Freeze CNN`、`Using ... classifier` | `docs/README.md` §1；`modeling_dynamic_stage1.py::_encode_dynamic_tokens/forward_stage1`；`run_class_finetuning.py:953/968` |
| Stage1实际权重 | 第2.2节三个目录 `run_logs/*.log` 的 `Namespace` | `losses_dynamic.py::compute_stage1_losses`；`data_processor/pair_sampling.py`；旧权重文档仅作历史 |
| 表3.1当前分类 | 每行目录 `log.txt` 的 best_epoch与对应epoch；D集中式 `run_logs` 的 finetune | `docs/README.md` §2；`docs/erpcore_results_summary.md`；分类选模入口 `run_class_finetuning.py` |
| 表3.2三阶段分类 | 四组 `stage3/log.txt`、各 `stage2/3/run_logs`；相邻 A `log.txt` | 相邻 `docs/erp_core_minimal_3stage.md`；`scripts/erp_core/D/3stage*.sh`（只读） |
| 表3.3消融/审计 | recon_only 两阶段log；uni `sampling_audit_20260928/stage2/3/log.txt` | 分别核对 `epochs/finetune/*weight`，不当作正式主实验 |
| 表4.1补全MSE | 对应 `log.txt` 选中epoch的 `test_loss_missing/val_loss_missing` | `losses_dynamic.py:11`；`engine_for_dynamic_stage1.py:335`；`data_processor/erpcore.py:185/303` |
| 表5.1探针 | 当前 `103212_81175/disentanglement_best/linear_probe.csv`、metadata | `tools/plot_erpcore_dynamic_stage1_disentanglement.py` 可追踪图；探针生成脚本待确认 |
| 表5.2抵消 | 同目录 `d_sub.npy/d_task.npy`，本次只读计算 | 文内给出统计定义；不对主分类 checkpoint 外推 |
| 表6.1三阶段模块 | `run_20260920_103722_87486/stage1/2/3/run_logs`、其余Stage2日志 | 相邻 `modeling_dynamic_stage1.py:512/554` 与 `run_dynamic_stage1.py:944` |
| 三阶段图与表7.1 | 两个 `stage2/disentanglement_full_few/metadata.json`、PNG、sample_alignment | 同目录元数据 `note`；全导联/少导联均是一个Stage2 checkpoint |
| CSLP本地参考 | `../CSLP-AE/split_model.py`、`train.py` | `subject_task_encode/decode/contrastive_loss`、`train.py:100` 的损失映射；论文细节待确认 |

### 9.2 旧文档与日志的冲突，保留而不覆盖

1. `docs/AON_V1_VS_AOND_V2_ANALYSIS.md` 写重建20、正则0.001、correction对比0.005、交换5及缩放/tanh语境；与当前主D日志500/0/50/1及当前直接相加实现不一致。相邻旧 dynamic 的 metadata 有对应20轮历史配置，不能把旧表称为当前主配置。
2. 当前诊断PNG内的 tanh 标签与 metadata、当前代码不一致，作为旧标签明确注记，不能静默照搬到PPT。
3. 相邻三阶段 `docs/erp_core_minimal_3stage.md` 的“没有启动正式训练，没有实验结果”是适配时的验证边界；后来实际已产生多组50/20/50轮日志，本文报告后续日志，不把该旧句作为目前没有结果的证据。
4. `scripts/erp_core/HISTORICAL_RESULTS.md` 写 unified-AON 没有 Bash 脚本，属于旧目录语境；当前仓库已有脚本。其 full finetune、scope=all、mean_pool 等历史结果不能替换当前 freeze CNN、scope=real 主表。
5. 旧导联扩展文档有D12 Test MSE 0.005573、D14 0.004440，以及历史其他配置；本文主D checkpoint对应0.008694414。不是把一个数改成另一个，而是运行来源不同；14导联也改变缺失集合，不用来推导三阶段收益。
6. `docs/erpcore_results_summary.md` 曾汇总 root `erp_core_D_stage1/log.txt` 的最后运行；该文件70条记录包含追加运行，其0.004078479不能替换分类实际加载的独立 `102523_63633` 结果。需要按运行边界与checkpoint来源精确匹配。

### 9.3 待核实与需要补的证据

- **固定Prototype完整MSE：** 已评估完整 Val/Test，结果与哈希见 `outputs/erpcore/prototype_mse_baseline/metrics.json`。
- **主两阶段解耦证据：** 需 `102523_63633/checkpoint-best.pth` 的全/少导联配套特征、探针、泄漏、交换/去分支诊断。当前线性探针属于关闭重建消融。
- **三阶段是否改善分支：** 需Stage1 best与Stage2 best同Test样本、共同M16范围的同协议probe；再与主两阶段对照。现有 full/few 图不能回答训练前后问题。
- **探针协议：** 需生成 `linear_probe.csv` 的代码与标准化/正则/划分记录，明确是trial随机划分、session隔离还是其他协议；不要把probe seed当训练seed。
- **三阶段同条件分类：** 需O/N的50轮配套日志，以及相同阶段预算、权重、分类器初始化和多个训练seed的两阶段/三阶段对照；本文不启动补实验。
- **历史代码版本：** 当前源码可解释结构，不能保证完全复原每个历史运行；需运行时git commit/diff或源码快照，尤其tanh/缩放、配对采样、RNG初始化与冻结细节。
- **三阶段uni来源：** 同名历史文件不计作新增seed；需运行谱系/复制记录以及正式训练日志，1轮audit不作效果。
- **参考方法与时间线：** 需CSLP-AE论文具体版本、结构/损失对应引用和早期解耦诊断记录；9月20日三阶段与9月29日当前两阶段不能按叙述顺序冒充真实先后因果。
- **因果解释：** 多目标冲突、少导联信息不足、分支抵消或分类读出限制都只是候选解释，需要受控消融；尚未证实任何一种是失败原因。
- **FiLM：** 本文件不依赖FiLM结果，后续有完整配置与日志再单独整理。
