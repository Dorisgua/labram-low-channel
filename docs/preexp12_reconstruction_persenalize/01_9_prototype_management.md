# 01.9 prototype 管理说明

这一节专门说明 `01_reconstruction_steps.md` 里的第 9 节：prototype 生成文档和脚本应该怎么组织、每个 prototype 文件里应该保存什么、训练时怎么知道每一行 prototype 对应哪个通道。

核心原则：

```text
prototype 不是一个单独的 tensor。
prototype = channel_prototypes + 通道名顺序 + LaBraM 位置索引 + 生成来源 + 使用场景。
```

如果只保存：

```text
channel_prototypes: [K, embed_dim]
```

只能知道有 K 个 prototype，但不知道第 0 行、第 1 行分别对应哪个通道。这样后续补通道时很容易把 prototype 放错位置。

## 1. prototype 文件放哪里

所有 prototype 相关文件统一放到：

```text
docs/prototypes/
```

当前约定的文件结构是：

```text
docs/prototypes/
  prototype_record.md
  01_generate_tuev23_cnn_patch_embed_mean.py
  01_tuev23_cnn_patch_embed_mean.pth
  02_generate_seedv62_cnn_patch_embed_mean.py
  02_seedv62_cnn_patch_embed_mean.pth
  03_build_tuev23_with_seedv62_extra_mean.py
  03_tuev23_with_seedv62_extra_mean.pth
```

说明：

```text
01_* 对应 TUEV-23 prototype。
02_* 对应 SEEDV-62 prototype。
03_* 对应 TUEV-23 + SEEDV-extra 的 70 通道 prototype。
```

`prototype_record.md` 是总记录表，用来说明每个 `.pth` 的来源、shape、通道顺序、生成命令和对应的 `completion_scope`。

## 2. 每个文件负责什么

### 2.1 `01_generate_tuev23_cnn_patch_embed_mean.py`

作用：

```text
从 TUEV 训练集生成 TUEV-23 的 CNN patch_embed mean prototype。
```

输出：

```text
docs/prototypes/01_tuev23_cnn_patch_embed_mean.pth
```

对应的训练设置：

```text
completion_scope=tuev13_with_tuev23
```

含义：

```text
真实输入可以是 TUEV-13。
目标空间是 TUEV-23。
缺失的 TUEV 通道用 TUEV-23 prototype 补。
```

### 2.2 `02_generate_seedv62_cnn_patch_embed_mean.py`

作用：

```text
从 SEEDV 训练集生成 SEEDV-62 的 CNN patch_embed mean prototype。
```

输出：

```text
docs/prototypes/02_seedv62_cnn_patch_embed_mean.pth
```

对应的训练设置：

```text
completion_scope=seedv23_with_seedv62
```

含义：

```text
真实输入可以是 SEEDV-23。
目标空间是 SEEDV-62。
缺失的 SEEDV 通道用 SEEDV-62 prototype 补。
```

### 2.3 `03_build_tuev23_with_seedv62_extra_mean.py`

作用：

```text
构造 TUEV-23 + SEEDV-extra 的 70 通道 target prototype。
```

输出：

```text
docs/prototypes/03_tuev23_with_seedv62_extra_mean.pth
```

对应的训练设置：

```text
completion_scope=tuev23_with_seedv62_extra
```

含义：

```text
目标空间不是单纯 TUEV-23，也不是单纯 SEEDV-62。
目标空间是 TUEV-23 加上 SEEDV-62 中额外可用的通道，共 70 个通道位置。
真实 TUEV 输入通道的位置后续会被真实 patch_embed feature 覆盖。
其他 extra 位置使用 SEEDV prototype。
```

## 3. `.pth` 文件里必须保存什么

每个 `.pth` 不能只保存 `channel_prototypes`。建议至少保存：

```python
{
    "channel_prototypes": Tensor[K, embed_dim],
    "ch_names": List[str],
    "input_chans": List[int],
    "source_dataset": str,
    "prototype_type": "cnn_patch_embed_mean",
}
```

字段含义：

```text
channel_prototypes:
  prototype 向量本身，shape 是 [K, embed_dim]。

ch_names:
  每一行 prototype 对应的通道名。
  必须满足 channel_prototypes[i] 对应 ch_names[i]。

input_chans:
  每个通道在 LaBraM position embedding 里的位置索引。
  必须满足 input_chans[i] 对应 ch_names[i]。

source_dataset:
  这个 prototype 从哪个数据集统计出来。

prototype_type:
  prototype 的生成方式。这里是 cnn_patch_embed_mean。
```

对于 70 通道的 `03_tuev23_with_seedv62_extra_mean.pth`，建议额外保存：

```python
{
    "target_ch_names": List[str],
    "target_input_chans": List[int],
    "source_seedv_ch_names": List[str],
    "source_seedv_input_chans": List[int],
}
```

原因：

```text
70 通道 target 是人为构造出来的混合空间。
必须明确 70 行 target 的最终顺序，否则只看到 [70, embed_dim] 无法判断第 i 行对应哪个通道。
```

## 4. 为什么必须保存 `ch_names` 和 `input_chans`

只看到：

```text
channel_prototypes.shape = [23, 200]
```

只能知道它有 23 行，每行是 200 维。但不知道：

```text
第 0 行是 FP1 还是 FP2
第 1 行是 F3 还是 F4
第 2 行是 C3 还是 CZ
```

如果行顺序错了，模型会把某个通道的 prototype 补到另一个通道位置。这样实验可以跑完，但结果不可信。

所以必须保证三者一一对应：

```text
channel_prototypes[i] <-> ch_names[i] <-> input_chans[i]
```

也就是说：

```text
第 i 行 prototype 是哪个通道，由 ch_names[i] 说明。
这个通道在 LaBraM position embedding 里的位置，由 input_chans[i] 说明。
```

## 5. 训练时怎么用 prototype

以：

```text
completion_scope=tuev13_with_tuev23
```

为例。

训练时流程：

```text
1. run_class_finetuning.py 读取 docs/prototypes/01_tuev23_cnn_patch_embed_mean.pth。
2. 从 .pth 里取出 channel_prototypes、ch_names、input_chans。
3. 检查当前真实输入通道是否是 ch_names 的子集。
4. 把 channel_prototypes 和 input_chans 传给 modeling_finetune.py。
5. forward_features() 里创建 x_full = [B, 23, T, embed_dim]。
6. x_full 初始填 TUEV-23 prototype。
7. 用真实输入通道的 patch_embed feature 覆盖对应位置。
8. flatten 成 Transformer token。
```

概念例子：

```text
TUEV-13 输入通道：
  FP1, FP2, F3, F4, C3, C4, P3, P4, O1, O2, T3, T4, CZ

TUEV-23 target 通道：
  FP1, FP2, F3, F4, C3, C4, P3, P4, O1, O2, F7, F8, T3, T4, T5, T6, FZ, CZ, PZ, ...
```

模型根据 `ch_names` 找到：

```text
FP1 在 TUEV-23 target 的第几行
FP2 在 TUEV-23 target 的第几行
C3 在 TUEV-23 target 的第几行
...
```

然后：

```text
真实输入通道位置 = 真实 patch_embed feature
缺失目标通道位置 = prototype feature
```

## 6. `prototype_record.md` 应该记录什么

建议表格：

```markdown
| name | file | shape | ch_names | input_chans | source_dataset | completion_scope | generator |
|---|---|---|---|---|---|---|---|
| tuev23_cnn_patch_embed_mean | 01_tuev23_cnn_patch_embed_mean.pth | [23, 200] | TUEV_23_CHANNELS | get_input_chans(TUEV, tuev23) | TUEV | tuev13_with_tuev23 | 01_generate_tuev23_cnn_patch_embed_mean.py |
| seedv62_cnn_patch_embed_mean | 02_seedv62_cnn_patch_embed_mean.pth | [62, 200] | SEEDV_62_CHANNELS | get_input_chans(SEEDV, seedv62) | SEEDV | seedv23_with_seedv62 | 02_generate_seedv62_cnn_patch_embed_mean.py |
| tuev23_with_seedv62_extra_mean | 03_tuev23_with_seedv62_extra_mean.pth | [70, 200] | TUEV_23 + SEEDV-extra | target_input_chans | SEEDV/TUEV mapping | tuev23_with_seedv62_extra | 03_build_tuev23_with_seedv62_extra_mean.py |
```

每次生成或更新 prototype，都要同步更新 `prototype_record.md`。否则后面只看到 `.pth` 文件，很难知道它来自哪里、怎么生成、该用于哪个实验。

## 7. 最小检查项

生成每个 `.pth` 后，至少检查：

```text
1. channel_prototypes.shape[0] == len(ch_names)
2. channel_prototypes.shape[0] == len(input_chans)
3. ch_names 的顺序就是 target channel 顺序
4. input_chans 的顺序和 ch_names 一一对应
5. prototype_type 是 cnn_patch_embed_mean
6. 如果 .pth 里保存了 optional completion_scope，则检查它和 prototype_record.md 记录一致
```

可以写一个简单检查脚本或在生成脚本最后打印：

```text
prototype file: docs/prototypes/01_tuev23_cnn_patch_embed_mean.pth
channel_prototypes: [23, 200]
len(ch_names): 23
len(input_chans): 23
prototype_type: cnn_patch_embed_mean
```
