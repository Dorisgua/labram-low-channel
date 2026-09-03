# 三阶段解耦、恢复与 Transformer 修改方案

## 目标

先在 ERP CORE `12 -> 28` 导联上完成：

```text
阶段一：用完整 28 导联学习 CSLP-AE 式 subject/task 解耦
阶段二：用解耦 latent 恢复冻结 CNN 的缺失导联 feature
阶段三：把恢复后的 28 导联 tokens 接入 LaBraM Transformer
```

## 使用 CSLP-AE 结构

复用 CSLP-AE 的以下结构：

```text
StridedConvolutionalEncoder
subject_encoder_out_net / task_encoder_out_net
subject_decoder_in_net / task_decoder_in_net
TransposedConvolutionalDecoder
```

不要直接使用当前的 `mean(dim=1)` 作为恢复信息，也不要直接 flatten
`[B,28,A,200]`。

CNN feature 先转换为：

```text
h [B,28,A,200]
  -> reshape/permute
h_seq [B,200,28*A]
```

然后进入修改后的 CSLP-AE：

```text
h_seq [B,200,28*A]
  -> Conv1d: 200 -> 128
  -> 4 层 stride=2 encoder
  -> subject/task head: 128 -> 64
d_sub, d_task [B,64,L]
```

其中：

```text
L = (28*A)/16
z_sub  = d_sub.flatten(1)
z_task = d_task.flatten(1)
```

当 `A=4` 时：

```text
直接 flatten：28*4*200 = 22400
CSLP latent：L=7，每个 z 为 7*64 = 448
```

`z_sub/z_task` 只用于 contrastive loss；解码恢复必须使用保留序列结构的
`d_sub/d_task [B,64,L]`。

解码过程：

```text
d_sub + d_task
  -> CSLP decoder mixin
  -> 4 层 transposed convolution
  -> Conv1d: 128 -> 200
h_hat_seq [B,200,28*A]
  -> reshape
h_hat [B,28,A,200]
```

## 阶段一：完整 28 导联解耦

```text
x_full [B,28,A,200]
  -> frozen CNN
h_full [B,28,A,200]
  -> CSLP encoder
d_sub, d_task
  -> CSLP decoder
h_hat_full [B,28,A,200]
```

训练范围：

- 冻结 LaBraM CNN/`patch_embed`。
- 训练 CSLP encoder、subject/task heads 和 decoder。
- 不进入 LaBraM Transformer，不使用分类 loss。

损失先保持简单：

```text
loss_stage1 = reconstruction(h_hat_full, h_full)
            + subject_contrastive(z_sub)
            + task_contrastive(z_task)
```

输出：完整导联 CSLP feature autoencoder 的 `checkpoint-best.pth`。

## 阶段二：12 导联恢复 CNN target

先对真实 12 导联运行同一个冻结 CNN，然后按 28 导联固定位置构造输入：

```text
12 个真实 CNN tokens + 16 个 prototype/mask tokens
  -> [B,28,A,200]
  -> 加载阶段一的 CSLP encoder
  -> d_sub/d_task
  -> CSLP decoder/completion
h_pred_full [B,28,A,200]
```

监督 target：

```text
h_full_target = frozen CNN(x_full)
h_pred_miss   = h_pred_full 中的 16 个缺失导联
h_miss_target = h_full_target 中的 16 个缺失导联
```

训练范围：

- 加载阶段一 `checkpoint-best.pth`。
- 先冻结 CNN 和 CSLP encoder、subject/task heads。
- 只训练 decoder/completion，判断固定解耦信息是否足以恢复 target。
- 不进入 LaBraM Transformer。

主要损失：

```text
loss_stage2 = SmoothL1(h_pred_miss, h_miss_target)
```

必须与静态 prototype 比较：

```text
MSE(pred, target) < MSE(prototype, target)
Cosine(pred, target) > Cosine(prototype, target)
```

输出：恢复模型的 `checkpoint-best.pth`。

## 阶段三：接入 LaBraM Transformer

```text
12 个真实 CNN tokens + 16 个恢复 tokens
  -> 按完整 28 导联顺序排列
  -> LaBraM channel/time position embedding
  -> LaBraM Transformer
  -> classifier
```

第一轮实验冻结：

- CNN/`patch_embed`
- CSLP encoder
- subject/task heads
- CSLP decoder/completion

只训练 LaBraM Transformer 和分类头，避免恢复模块被分类 loss 改变。

最后比较：

| 实验 | Transformer 输入 |
|---|---|
| O | 真实 28 导联 CNN tokens |
| N | 真实 12 导联 CNN tokens |
| A | 12 导联 + 静态 prototype |
| D | 12 导联 + CSLP 恢复 tokens |

只有阶段二优于 prototype，并且阶段三 `D > A`，才能说明解耦信息既能恢复
CNN target，也能改善 Transformer 下游效果。

## 需要修改的文件

- 新增 `modeling_cslp_feature.py`
  - 从 CSLP-AE 提取并适配 encoder、双 latent heads 和 decoder。
  - 输入/输出改为 CNN feature `[B,28,A,200]`。
- 修改 `modeling_dynamic_stage1.py`
  - 使用 `modeling_cslp_feature.py`，替换当前 mean/cross-attention 恢复路径。
- 修改 `losses_dynamic.py`
  - 分开 `stage1 full reconstruction` 和 `stage2 missing reconstruction`。
- 修改 `run_dynamic_stage1.py`、`engine_for_dynamic_stage1.py`
  - 增加阶段选择和对应冻结逻辑。
- 保留并补齐脚本：
  - `scripts/erp_core/full-disentangle/stage1.sh`
  - `scripts/erp_core/full-disentangle/stage2.sh`
  - `scripts/erp_core/full-disentangle/stage3.sh`

不要直接从另一个 CSLP-AE 目录做运行时 import；把所需模块放入当前仓库，保证
checkpoint 和代码可以独立复现。
