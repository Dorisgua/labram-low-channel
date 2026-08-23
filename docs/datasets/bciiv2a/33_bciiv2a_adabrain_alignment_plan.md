# 33 系列 BCI-IV-2a 与 AdaBrain 对齐改造计划

## 1. 目标

当前实验的目标是：

1. 先使用完整 22 通道建立与 AdaBrain 尽可能一致的 BCI-IV-2a multisession baseline。
2. 将 baseline 复现到接近 AdaBrain 的水平后锁定模型、数据划分和训练配置。
3. 再开展 `33N` 和 `33Ah`，只改变输入通道和通道补全方式，使性能变化能够归因于删通道或补通道。

当前 BCI-IV-2a 数据读取已经基本对齐 AdaBrain，包括：

- Train：2592 条，trial 1–288，每个 subject 288 条。
- Validation：1296 条，trial 289–432，每个 subject 144 条。
- Test：1296 条，trial 433–576，每个 subject 144 条。
- 9 个 subject 均出现在三个 split 中。
- 250 Hz 重采样到 200 Hz。
- 使用 train statistics 进行 z-score。
- 22 通道顺序与 AdaBrain manifest 一致。
- 四分类标签为 0–3。

目前主要差距不在数据，而在分类 head、模型 flags 和优化器行为。

## 2. 当前 LaBraM 与 AdaBrain 的核心差异

当前 LaBraM 分类路径：

```text
EEG
 → CNN patch_embed
 → Transformer
 → 对所有 EEG patch token 做 mean pooling
 → 200 维
 → Linear(200, 4)
```

AdaBrain 分类路径：

```text
EEG
 → CNN patch_embed
 → Transformer
 → 保留 CLS token 和全部 EEG patch token
 → 展平
 → LinearWithConstraint(17800, 4, max_norm=1)
```

对于 BCI-IV-2a：

```text
22 channels × 4 temporal patches + 1 CLS token = 89 tokens
89 tokens × 200 embedding dimensions = 17800 features
```

因此 AdaBrain 的分类 head 是：

```text
17800 → 4
```

而当前 LaBraM 的分类 head 是：

```text
200 → 4
```

其他差异：

| 部分 | 当前实现 | AdaBrain 对齐实现 |
|---|---|---|
| Transformer 输出 | EEG token mean pooling | CLS + 全部 EEG token |
| 分类 head | 普通 `Linear(200, 4)` | `LinearWithConstraint(17800, 4, max_norm=1)` |
| QKV bias | 当前 shell 关闭 | AdaBrain 开启，新初始化 Q/V bias |
| Relative position bias | 当前 shell 关闭 | AdaBrain 开启 |
| Absolute position embedding | 开启 | 开启 |
| Layer decay | 当前 `0.65` | AdaBrain `1.0` |
| Embedding weight decay | position/CLS/time embedding 被排除 | AdaBrain 对齐实现中参与 weight decay |
| CNN | `33Ofull` 中可训练 | full fine-tuning 中可训练 |
| 模型选择 | Validation BAcc | Validation BAcc |
| Test 使用 | 当前每个 epoch 评估 | 严格方案在 validation 选定后测试 |

## 3. 推荐实现：新增 AdaBrain wrapper

推荐新增 wrapper，而不是把 AdaBrain 特有分类逻辑直接塞进通用 `NeuralTransformer`。

结构：

```text
AdaBrainLaBraMWrapper
├── backbone：现有 NeuralTransformer
└── task_head：AdaBrain all-token constrained head
```

这样可以保证：

- 当前 TUEV `17O/17N/17Ah` 不受影响。
- 原有 mean-pooling LaBraM 路径保持不变。
- BCI-IV-2a 可以通过显式参数选择 AdaBrain head。
- 后续 `33N/33Ah` 可以复用同一 wrapper。

## 4. 需要新增的文件

### 4.1 `modeling_adabrain.py`

建议新增：

```text
modeling_adabrain.py
```

文件中实现两个类。

#### `LinearWithConstraint`

用于复现 AdaBrain 的 max-norm 分类 head：

```python
class LinearWithConstraint(nn.Linear):
    def __init__(self, *args, max_norm=1.0, flatten=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_norm = max_norm
        self.flatten = flatten

    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data,
            p=2,
            dim=0,
            maxnorm=self.max_norm,
        )
        if self.flatten:
            x = x.flatten(start_dim=1)
        return super().forward(x)
```

实际实现时应以参考 AdaBrain 仓库中的 `LinearWithConstraint` 为准。

#### `AdaBrainLaBraMWrapper`

核心结构：

```python
class AdaBrainLaBraMWrapper(nn.Module):
    def __init__(
        self,
        backbone,
        num_channels,
        num_t,
        num_classes,
    ):
        super().__init__()
        self.backbone = backbone
        self.backbone.head = nn.Identity()

        input_dim = (num_channels * num_t + 1) * backbone.embed_dim
        self.task_head = LinearWithConstraint(
            input_dim,
            num_classes,
            max_norm=1,
            flatten=True,
        )

    def forward(self, x, input_chans=None):
        tokens = self.backbone(
            x,
            input_chans=input_chans,
            return_all_tokens=True,
        )
        return self.task_head(tokens)
```

Wrapper 还需要代理当前训练入口依赖的方法：

```python
def get_num_layers(self):
    return self.backbone.get_num_layers()

def no_weight_decay(self):
    return set()
```

`no_weight_decay()` 返回空集合，用于使 position/CLS/time embedding 参与 weight decay，与本地 AdaBrain 对齐实验保持一致。

## 5. 需要修改 `run_class_finetuning.py` 的位置

### 5.1 导入 wrapper

在现有模型 import 附近增加：

```python
from modeling_adabrain import AdaBrainLaBraMWrapper
```

### 5.2 增加分类 head 模式参数

在参数解析区域增加：

```python
parser.add_argument(
    "--classifier_mode",
    default="mean_pool",
    choices=["mean_pool", "adabrain_all_token"],
)
```

默认值必须保留为 `mean_pool`，避免影响已有 TUAB、TUEV 和 `17` 系列实验。

### 5.3 显式配置 `num_t`，并用真实样本校验

AdaBrain 原始实现从 `dataset_config/Classification.json` 读取
`"num_t": 4`，PreExp33 脚本则直接使用 `num_t=4`。当前改造同样显式声明
BCI-IV-2a 每个通道包含 4 个 temporal patches，但额外使用真实样本做一致性
校验，避免配置与数据静默偏离。

在 `DATASET_CONFIGS["bciiv2a"]` 中增加：

```python
"num_t": 4,
```

在 `get_dataset(args)` 中传递：

```python
args.num_t = cfg.get("num_t")
```

dataset 加载完成、backbone 创建完成后进行校验：

```python
sample_x = dataset_train[0][0]

if sample_x.ndim != 2:
    raise ValueError(
        f"Expected dataset sample shaped [channels, time], "
        f"got {tuple(sample_x.shape)}"
    )

if sample_x.shape[0] != len(ch_names):
    raise ValueError(
        f"Dataset/channel-name mismatch: sample has {sample_x.shape[0]} "
        f"channels, but ch_names has {len(ch_names)}"
    )

expected_length = args.num_t * model.patch_size
if sample_x.shape[-1] != expected_length:
    raise ValueError(
        f"BCI-IV-2a sample length mismatch: got {sample_x.shape[-1]}, "
        f"expected num_t({args.num_t}) * patch_size({model.patch_size}) "
        f"= {expected_length}"
    )
```

BCI-IV-2a 当前真实样本为 `[22, 800]`，backbone patch size 为 200，因此：

```text
expected_length = num_t × patch_size = 4 × 200 = 800
```

应输出审计信息：

```python
print(
    "Validated AdaBrain temporal layout: "
    f"sample_length={sample_x.shape[-1]}, "
    f"num_t={args.num_t}, patch_size={model.patch_size}"
)
```

这里读取 `dataset_train[0]` 会额外打开并预处理一个 pkl，但只发生一次，开销
可以忽略。显式 `num_t` 让模型结构和实验记录更清楚，真实样本校验则保证以后
改变裁窗长度、重采样结果或 patch size 时能够立即报错。

### 5.4 在 checkpoint 加载后包装 backbone

必须先把预训练 checkpoint 加载到原始 `NeuralTransformer`，再套 wrapper。

推荐位置：

```text
checkpoint 加载完成
→ completion/prototype 配置完成
→ 可选 freeze_cnn 完成
→ 创建 AdaBrain wrapper
→ model.to(device)
```

对应逻辑：

```python
if args.classifier_mode == "adabrain_all_token":
    model.head = torch.nn.Identity()

    token_channels = len(ch_names)
    model = AdaBrainLaBraMWrapper(
        backbone=model,
        num_channels=token_channels,
        num_t=args.num_t,
        num_classes=args.nb_classes,
    )
```

不能在 checkpoint 加载前创建 wrapper。原因是预训练 checkpoint key 当前是：

```text
blocks.*
patch_embed.*
```

套 wrapper 后模型 key 会变成：

```text
backbone.blocks.*
backbone.patch_embed.*
```

如果先包装，会导致当前 checkpoint 加载逻辑无法匹配。

### 5.5 为未来 `33N/33Ah` 预留 token channel 数

`num_t` 只描述每个通道的 temporal patch 数，删通道不会改变它。只要三个实验
都使用 800 点输入和 200 点 patch，`33O/33N/33Ah` 的 `num_t` 都是 4。

需要单独决定的是 Transformer 最终输出的通道 token 数 `token_channels`。

完整 22 通道时可以直接使用：

```python
token_channels = len(ch_names)
```

但后续必须根据 Transformer 实际输出 token 空间确定：

```text
33O  → 22 个输出通道
33N  → 低通道数量
33Ah → 补全后的 22 个输出通道
```

因此建议后续增加显式配置，例如：

```python
token_channels = cfg["token_channels"]
```

或根据 `completion_scope` 计算。

### 5.6 优化器参数分组

当前训练入口调用：

```python
num_layers = model_without_ddp.get_num_layers()
skip_weight_decay_list = model.no_weight_decay()
```

Wrapper 提供这两个方法后，这部分不需要大范围重写。

AdaBrain 模式需要：

```bash
--layer_decay 1.0
```

这样不会创建 layer-wise decay assigner。

Wrapper 的 `no_weight_decay()` 返回空集合，使 embedding 参与 weight decay。

## 6. 修改 `33Ofull` shell

目标文件：

```text
scripts/33Ofull.finetune_bciiv2a_labrambase_full_finetuen.sh
```

训练参数建议：

```bash
BATCH_SIZE=64
UPDATE_FREQ=1
LR=5e-4
EPOCHS=50
WARMUP_EPOCHS=5
WEIGHT_DECAY=0.05
LAYER_DECAY=1.0
DROP_PATH=0.1
SMOOTHING=0.1
```

命令增加：

```bash
--classifier_mode adabrain_all_token
```

AdaBrain 模型 flags：

```bash
--abs_pos_emb
```

删除：

```bash
--disable_rel_pos_bias
--disable_qkv_bias
```

Full fine-tuning 不应添加：

```bash
--freeze_cnn
```

严格对齐 AdaBrain 时，以下 missing keys 是预期现象：

```text
blocks.*.attn.q_bias
blocks.*.attn.v_bias
fc_norm.weight
fc_norm.bias
head.weight
head.bias
```

Q/V bias 是 AdaBrain 下游模型新增并随机初始化的参数，不代表 checkpoint 没有加载。



## 8. 不需要修改的文件

### `data_processor/bciiv2a.py`

数据 split、重采样、归一化、标签和通道顺序已经对齐。

### `engine_for_finetuning.py`

当前训练和评估入口已经把 `input_chans` 传给模型：

```python
model(samples, input_chans)
```

Wrapper 使用相同 forward 接口即可兼容。

### `Channels_definition.py`

完整 BCI-IV-2a 22 通道定义已经存在。

### `modeling_finetune.py`

当前 `NeuralTransformer.forward_features()` 已支持：

```python
return_all_tokens=True
```

因此使用 wrapper 时无需修改通用 backbone。

## 9. 推荐实施顺序

1. 新增 `modeling_adabrain.py`，实现 constrained head 和 wrapper。
2. 在 `run_class_finetuning.py` 增加 `classifier_mode`，并在 BCI 配置中声明 `num_t=4`。
3. 使用 `dataset_train[0]` 校验 `sample_length == num_t * patch_size`。
4. 在 checkpoint、completion 和 freeze 配置完成后创建 wrapper。
5. 修改 `33Ofull` 为 `adabrain_all_token`，CNN 保持可训练。
6. 做一次模型结构和 checkpoint missing/unexpected keys 审计。
7. 做小批量 forward/backward smoke test。
8. 跑完整 22 通道 baseline，目标接近本地参考的约 59% validation/test BAcc。
9. baseline 锁定后再设计 `33N/33Ah` 的低通道集合、prototype 和 token channel 数。

## 10. 后续通道实验注意事项

使用 all-token head 后，分类 head 输入维度与输出 token 数直接相关：

```text
33O：22 通道 → (22×4+1)×200
33N：低通道 → (低通道数×4+1)×200
33Ah：补全到22通道 → (22×4+1)×200
```

因此 `33N` 的分类 head 参数量会小于 `33O/33Ah`。这会引入“通道信息变化”和“head 参数量变化”两个因素。

在正式实验前需要明确选择：

1. 接受 head 随通道数变化，严格遵循 AdaBrain all-token head 的自然形式；或
2. 固定为 22 通道 token 空间，对缺失通道使用占位或补全，从而固定 head 参数量。

这一选择应在运行 `33N/33Ah` 前锁定并记录。
