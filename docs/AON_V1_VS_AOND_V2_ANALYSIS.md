# 缺失通道预测与 Stage 1 训练流程

| 模块 | 步骤 | 输入或来源 | 当前处理逻辑 | 输出／目的／当前权重 | 修改位置 |
|---|---|---|---|---|---|
| **缺失通道预测** | **1. 提取观测特征** | `x_obs`：`[B,12,1,200]` | 12 通道 EEG 经过冻结的 CNN／patch encoder：`h_obs = self._patch_tokens(x_obs)` | 观测特征 `h_obs`：`[B,12,1,200]` | `modeling_dynamic_stage1.py` → `_patch_tokens()` |
| 缺失通道预测 | **2. 读取 prototype** | `erpcore28_channel_prototypes`：`[28,200]` | 扩展到 batch 和时间维度，并结合通道索引划分观测、缺失部分 | `p_obs`：`[B,12,1,200]`；`p_miss`：`[B,16,1,200]`。**当前 `p_obs` 被计算，但未用于后续计算** | `modeling_dynamic_stage1.py` → `_encode_dynamic_tokens()` |
| 缺失通道预测 | **3. 确定通道位置** | `real_input_chans_index`、`target_input_chans_index` | 将 12 个真实通道映射到完整 28 通道空间，确定剩余缺失通道 | `obs_indices`：12 个位置；`miss_indices`：16 个位置 | `modeling_dynamic_stage1.py` → `_dynamic_channel_indices()` |
| 缺失通道预测 | **4. 组成 corrector 输入** | `h_obs`、`p_miss` | 整理维度后，将观测特征与缺失 prototype 沿 token 维拼接 | `[B,12,200]` ＋ `[B,16,200]` → **`[B,28,200]`** | `modeling_dynamic_stage1.py` → `_encode_dynamic_tokens()` |
| 缺失通道预测 | **5. 提取共享表示** | `tokens`：`[B,28,200]` | 经过 shared encoder 和 norm，让观测 token 与缺失 prototype token 交互 | `shared_tokens`：`[B,28,200]` | `modeling_dynamic_stage1.py` → `corrector["shared_encoder"]`、`corrector["shared_norm"]` |
| 缺失通道预测 | **6. Subject／Task 分支处理** | `shared_tokens` | 分别进入并行的 subject 和 task corrector 分支 | `subject_tokens`、`task_tokens`：均为 `[B,28,200]` | `modeling_dynamic_stage1.py` → subject／task encoder 及对应 norm |
| 缺失通道预测 | **7. 提取缺失位置输出** | `subject_tokens`、`task_tokens` | 在当前“观测在前、缺失在后”的拼接顺序下，取后 16 个 token，并恢复时间维度 | `subject_missing`、`task_missing`：均为 `[B,16,1,200]` | `modeling_dynamic_stage1.py` → `_encode_dynamic_tokens()` |
| 缺失通道预测 | **8. 生成受限 correction** | `subject_missing`、`task_missing` | 对两个分支分别执行 **`0.02 × tanh(...)`** | `d_sub`、`d_task`：均为 `[B,16,1,200]`；各自逐元素幅度不超过 **0.02** | `_encode_dynamic_tokens()`；配置入口 `CORRECTION_SCALE` |
| 缺失通道预测 | **9. 得到最终预测** | `p_miss`、`d_sub`、`d_task` | **`h_pred_miss = p_miss + d_sub + d_task`** | 输出 `[B,16,1,200]`；相对 prototype 的逐元素总修正幅度不超过 **0.04** | `modeling_dynamic_stage1.py` → `_encode_dynamic_tokens()` |
| **Stage 1 训练目标** | **1. 构造完整 target** | `x_full`：`[B,28,1,200]` | 完整 28 通道经同一个冻结的 patch encoder，按 `miss_indices` 提取 16 个缺失通道特征；目标路径置于 `no_grad()` 中 | `h_miss_target`：`[B,16,1,200]`；**目标路径不反向传播** | `modeling_dynamic_stage1.py` → `forward_stage1()` |
| Stage 1 训练目标 | **2. 主重建损失** | `h_pred_miss`、`h_miss_target` | `L_missing = MSE(h_pred_miss, h_miss_target)` | 使预测特征接近真实目标特征；**权重 20.0** | `losses_dynamic.py` → `compute_stage1_losses()`；`MISSING_WEIGHT` |
| Stage 1 训练目标 | **3. Correction 正则** | `d_sub`、`d_task` | `L_reg = mean(d_sub²) + mean(d_task²)` | 惩罚过大的修正量；**权重 0.001** | `losses_dynamic.py` → `compute_stage1_losses()`；`REG_WEIGHT` |
| Stage 1 训练目标 | **4. Subject 样本配对** | Dataset 的 subject 索引 | 为同一 subject 随机选取 left／right 两个不同样本；**不强制 task 不同** | 两个样本批次分别调用 `forward_stage1()`，得到 correction 和 target | `data_processor/erpcore.py` → `sample_cslpae_pair_batch("subject", ...)` |
| Stage 1 训练目标 | **5. Subject summary 对比** | 配对的 `z_sub_left`、`z_sub_right` | 对 subject 汇总表示计算双向 InfoNCE | **权重 0.0：不贡献当前总损失** | `losses_dynamic.py` → `symmetric_info_nce()`；`SUBJECT_SUMMARY_CONTRA_WEIGHT` |
| Stage 1 训练目标 | **6. Subject correction 对比** | 配对的 `d_sub_left`、`d_sub_right` | 将完整 `d_sub` 展平，再计算双向 InfoNCE | 鼓励配对的 subject correction 接近；**权重 0.005** | `losses_dynamic.py` → `compute_stage1_losses()`；`SUBJECT_CORRECTION_CONTRA_WEIGHT` |
| Stage 1 训练目标 | **7. Subject 交换重建** | 同 subject 的 left／right 输出 | 交换 `d_sub`，保留各自 `d_task`：`left_pred = p_left + d_sub_right + d_task_left`；right 对称处理，并与各自 target 比较 | 约束同 subject 的 correction 在样本间可交换；**权重 5.0** | `losses_dynamic.py` → `swap_sub_reconstruction()`；`PERMUTE_SUB_WEIGHT` |
| Stage 1 训练目标 | **8. Task 样本配对** | Dataset 的 task 索引 | 为同一 task 随机选取 left／right 两个不同样本；**不强制 subject 不同** | 两个样本批次分别调用 `forward_stage1()`，得到 correction 和 target | `data_processor/erpcore.py` → `sample_cslpae_pair_batch("task", ...)` |
| Stage 1 训练目标 | **9. Task summary 对比** | 配对的 `z_task_left`、`z_task_right` | 对 task 汇总表示计算双向 InfoNCE | **权重 0.0：不贡献当前总损失** | `losses_dynamic.py` → `symmetric_info_nce()`；`TASK_SUMMARY_CONTRA_WEIGHT` |
| Stage 1 训练目标 | **10. Task correction 对比** | 配对的 `d_task_left`、`d_task_right` | 将完整 `d_task` 展平，再计算双向 InfoNCE | 鼓励配对的 task correction 接近；**权重 0.005** | `losses_dynamic.py` → `compute_stage1_losses()`；`TASK_CORRECTION_CONTRA_WEIGHT` |
| Stage 1 训练目标 | **11. Task 交换重建** | 同 task 的 left／right 输出 | 交换 `d_task`，保留各自 `d_sub`：`left_pred = p_left + d_sub_left + d_task_right`；right 对称处理，并与各自 target 比较 | 约束同 task 的 correction 在样本间可交换；**权重 5.0** | `losses_dynamic.py` → `swap_task_reconstruction()`；`PERMUTE_TASK_WEIGHT` |
| Stage 1 训练目标 | **12. 汇总总损失** | 上述各子损失 | `20L_missing + 0.001L_reg + 0.005L_sub_corr + 0.005L_task_corr + 5L_permute_sub + 5L_permute_task` | 得到标量 `total_loss`；两个 summary 对比项权重为 0 | `losses_dynamic.py` → `compute_stage1_losses()`；权重配置见 `scripts/erp_core/D/stage1.sh` |
| Stage 1 训练目标 | **13. 前向与反向传播** | 主 batch、subject pair、task pair | 主 batch 前向 1 次；subject-left／right 2 次；task-left／right 2 次；合并损失后反向传播 | 当前常规流程为 **5 次 forward、1 次 backward、1 次 optimizer step**；需结合梯度累积等实际设置确认 | `engine_for_dynamic_stage1.py` → `train_dynamic_stage1_batch()` |
| Stage 1 训练目标 | **14. 参数更新范围** | `total_loss` | CNN 冻结；target 不反向传播；prototype 为 buffer；LaBraM 主 Transformer blocks 不在 Stage 1 前向路径中 | **主要更新 corrector**：shared、subject、task encoder 及对应 LayerNorm | `run_dynamic_stage1.py` 的冻结／优化器逻辑；`modeling_dynamic_stage1.py` 的 `corrector` |
| Stage 1 训练目标 | **15. 选择 best checkpoint** | Validation 加权总损失 | 每个 epoch 验证，保存 validation total loss 更低的模型 | **`checkpoint-best.pth` 对应验证集加权总损失最低**，不是分类准确率最高，也不等同于单独重建损失最低 | `run_dynamic_stage1.py` 的 best checkpoint 选择逻辑 |

**读表说明：**

- `B` 为 batch size；上表对应每通道 **1 个时间 patch、200 维特征**的当前配置。
- 补全对象是 **token 特征，不是原始 EEG 波形**。
- 步骤用于解释数据依赖，不代表所有操作都严格串行；例如通道索引服务于 prototype 划分，target 与预测属于两条路径。
- InfoNCE 的负样本组成及标签处理需要核对实现，不能仅凭“同 subject／task 配对”就断定所有负样本来自不同 subject／task。
- 当前实现与权重依据提供材料整理，正式交付前应与指定代码版本核对。