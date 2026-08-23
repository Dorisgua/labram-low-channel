# BCI-IV-2a PreExp33 实验结果

更新时间：2026-07-15（UTC）

## 结果口径

- 数据集：BCI Competition IV 2a multisession，9 名被试。
- 划分：train 2592、validation 1296、test 1296。
- 所有已记录运行均使用 `seed=0`、batch size 64、50 epochs、LaBraM base checkpoint 和 AdaBrain all-token 分类头。33A-real 仍让 Transformer 处理全部 89 个 token，但分类头只展开 CLS 和 13 个真实通道对应的 53 个 token。
- 表中的 test 指标来自“验证集 balanced accuracy 最优 epoch”对应的 test 结果，不是从 test 集挑选最优 epoch。
- BAcc、F1 和 Kappa 均来自每份日志末尾的 `Best epoch summary`。

## 脚本与运行状态

| 实验 | 脚本 | 真实输入通道 | completion 后通道 | CNN | completion | 状态 |
|---|---|---:|---:|---|---|---|
| 33O full | `33Oada.finetune_bciiv2a_labrambase_full_finetuen.sh` | 22 | 22 | 训练 | 无 | 已完成 |
| 33O freeze | `33Oada.finetune_bciiv2a_labrambase_freeze_cnn.sh` | 22 | 22 | 冻结 | 无 | 已完成 |
| 33N full | `33Nada.finetune_bciiv2a_labrambase_full_finetuen.sh` | 13 | 13 | 训练 | 无 | 已完成 |
| 33N freeze | `33Nada.finetune_bciiv2a_labrambase_freeze_cnn.sh` | 13 | 13 | 冻结 | 无 | 已完成 |
| 33A full | `33Aada.finetune_bciiv2a_labrambase_full_finetuen.sh` | 13 | 22 | 训练 | 9 个缺失通道使用 prototype | 尚无运行日志 |
| 33A freeze（all readout） | `33Aada.finetune_bciiv2a_labrambase_freeze_cnn.sh` | 13 | 22 | 冻结 | 9 个缺失通道使用 prototype；分类头读取全部 89 tokens | 已完成 |
| 33A-real freeze | `33Arealada.finetune_bciiv2a_labrambase_freeze_cnn.sh` | 13 | 22 | 冻结 | 9 个缺失通道使用 prototype；分类头只读取 53 个真实 tokens | 已完成 |

## 已完成实验结果

| 实验 | 最优 epoch | Val BAcc | Test BAcc | Val F1 | Test F1 | Val Kappa | Test Kappa | 
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 33O full（22→22） | 36 | **0.6312** | **0.5795** | 0.6302 | 0.5792 | 0.5082 | 0.4393 | 
| 33O freeze（22→22） | 36 | 0.5756 | 0.5548 | 0.5751 | 0.5550 | 0.4342 | 0.4064 | 
| 33N full（13→13） | 9 | 0.5293 | 0.5231 | 0.5291 | 0.5229 | 0.3724 | 0.3642 | 
| 33N freeze（13→13） | 21 | 0.5270 | 0.5262 | 0.5261 | 0.5249 | 0.3693 | 0.3683 | 
| 33A freeze（13→22，all readout） | 8 | 0.4954 | 0.4923 | 0.4940 | 0.4909 | 0.3272 | 0.3230 | 
| 33A-real freeze（13→22，real readout） | 9 | **0.5332**¹ | **0.5471**¹ | 0.5314 | 0.5458 | 0.3776 | 0.3961 |

¹ 13 个真实输入通道相关实验中的最高值；全体实验最高值仍为 33O full。

## 对比

| 对比 | Val BAcc 变化 | Test BAcc 变化 | 说明 |
|---|---:|---:|---|
| 33O freeze 相对 33O full | -5.56 个百分点 | -2.47 个百分点 | 22 通道情况下，冻结 CNN 会带来可见下降 |
| 33N freeze 相对 33N full | -0.23 个百分点 | +0.31 个百分点 | 13 通道 direct 情况下，冻结 CNN 与 full fine-tune 基本持平 |
| 33A all-readout 相对 33N freeze | -3.16 个百分点 | -3.40 个百分点 | 把 36 个 prototype token 直接展开进分类头会降低效果 |
| 33A-real 相对 33A all-readout | **+3.78 个百分点** | **+5.48 个百分点** | Transformer 仍使用 prototype，但分类头排除 prototype token 后显著改善 |
| 33A-real 相对 33N freeze | **+0.62 个百分点** | **+2.08 个百分点** | 相同 13 通道输入和冻结 CNN 时，prototype 通过 Transformer attention 带来正收益 |
| 33N full 相对 33O full | -10.19 个百分点 | -5.63 个百分点 | 从 22 通道减少到 13 通道后的 full fine-tune 差距 |

当前 `seed=0` 结果支持“prototype 本身可以提供帮助，但不应把低信息 prototype token 全部直接展开进 AdaBrain 分类头”的判断。若要确认稳定性，仍应使用多个 seed 报告均值和标准差。

## 运行日志

- 33O full：`outputs/preexp33_bciiv2a_multisession/run_logs/33Oada.finetune_bciiv2a_labrambase_full_finetuen_20260715_075940.log`
- 33O freeze：`outputs/preexp33_bciiv2a_multisession/run_logs/33Oada.finetune_bciiv2a_labrambase_freeze_cnn_train_transformer_head_eval_20260715_084805.log`
- 33N full：`outputs/preexp33_bciiv2a_multisession/run_logs/33Nada.finetune_bciiv2a_labrambase_full_finetuen_20260715_075940.log`
- 33N freeze：`outputs/preexp33_bciiv2a_multisession/run_logs/33Nada.finetune_bciiv2a_labrambase_freeze_cnn_20260715_084151.log`
- 33A freeze（all readout）：`outputs/preexp33_bciiv2a_multisession/run_logs/33Aada.finetune_bciiv2a_labrambase_freeze_cnn_train_transformer_20260715_083643.log`
- 33A-real freeze：`outputs/preexp33_bciiv2a_multisession/run_logs/33Arealada.finetune_bciiv2a_labrambase_freeze_cnn_20260715_093109.log`
