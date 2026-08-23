# preexp12 reconstruction step-by-step

本文按从底层到上层的顺序整理 preexp12 需要补齐的内容。

目标不是一次性复制 `labram-preexp-work copy`，而是把每一层依赖先固定清楚，再往上接：

```text
通道定义
-> prototype 生成和检查
-> 命令行参数
-> dataset 输入通道
-> model 状态和直接配置
-> forward_features 补通道
-> pooling / freeze_cnn
-> scripts
-> 最小验证
```

这样做的原因是：补通道最容易错的不是 tensor shape，而是“第 i 行 prototype 到底对应哪个通道”。所以必须先固定通道顺序，再写模型逻辑。

## Step 1. 固定通道定义

改哪里：

```text
LaBraM-origin_preexp12/Channels_definition.py
```

需要包含：

```text
TUEV_13_CHANNELS
TUEV_23_CHANNELS
SEEDV_23_CHANNELS
SEEDV_62_CHANNELS
TUEV23_SEEDV62_EXTRA_CHANNELS
```

为什么：

```text
prototype 生成、补通道映射、检查脚本、训练代码必须使用同一套通道顺序。
如果每个文件自己写一份通道列表，就可能出现 shape 正确但通道顺序错位的问题。
```

注意：

```text
TUEV_13_CHANNELS 必须是 TUEV_23_CHANNELS 的子集。
SEEDV_23_CHANNELS 必须是 SEEDV_62_CHANNELS 的子集。
TUEV23_SEEDV62_EXTRA_CHANNELS 是 TUEV_23 + SEEDV_62 中额外能映射到 standard_1020 的通道。
```

## Step 2. 明确 channel_subset 的含义，放在step5里加代码

改哪里：

```text
run_class_finetuning.py
dataset 读取逻辑
```

`channel_subset` 表示当前输入数据实际使用哪组通道。

建议先只支持这些选择：

```text
dataset=TUEV:
  channel_subset=tuev13
  channel_subset=tuev23

dataset=SEEDV:
  channel_subset=seedv23
  channel_subset=seedv62
```

为什么：

```text
completion_scope 只说明“补到哪里”。
channel_subset 说明“真实输入从哪里来”。
两者不能混在一起。
```

例子：

```text
dataset=TUEV, channel_subset=tuev13, completion_scope=tuev13_with_tuev23
表示真实输入是 TUEV-13，模型内部补到 TUEV-23。

dataset=TUEV, channel_subset=tuev23, completion_scope=none
表示真实输入就是 TUEV-23，不补通道。
```

## Step 4. 先生成可追溯的 prototype

改哪里：

```text
docs/prototypes/
```

建议文件结构：

```text
docs/prototypes/
  00_check_pth.py
  00_check_completion_logic.py
  01_generate_tuev_cnn_patch_prototypes.py
  01_tuev23_cnn_patch_embed_mean.pth
  02_generate_seedv62_cnn_patch_embed_mean.py
  02_seedv62_cnn_patch_embed_mean.pth
  03_build_tuev23_with_seedv62_extra_mean.py
  03_tuev23_with_seedv62_extra_mean.pth
  prototype_record.md
```

prototype `.pth` 至少保存：

```python
{
    "channel_prototypes": Tensor[K, embed_dim],
    "ch_names": List[str],
    "input_chans_index": List[int],
    "channel_input_chans_index": List[int],
    "source_dataset": str,
    "prototype_type": "cnn_patch_embed_mean",
}
```

为什么：

```text
只保存 channel_prototypes: [23, 200] 不够。
必须知道第 0 行、第 1 行分别对应 FP1 还是 FP2。
```

## Step 5. 在 run_class_finetuning.py 增加参数入口

改哪里：

```text
run_class_finetuning.py
```

需要有这些参数：

```python
parser.add_argument("--channel_subset", default="", type=str)

parser.add_argument(
    "--completion_scope",
    default="none",
    choices=[
        "none",
        "tuev13_with_tuev23",
        "seedv23_with_seedv62",
        "tuev23_with_seedv62_extra",
    ],
    type=str,
)

parser.add_argument(
    "--pooling_scope",
    default="low",
    choices=["low", "high"],
    type=str,
)

parser.add_argument("--channel_prototype_path", default="", type=str)

parser.add_argument(
    "--freeze_cnn",
    action="store_true",
    help="Freeze patch_embed/TemporalConv and train transformer/head only",
)
```

为什么：

```text
scripts 会传这些参数。
如果 argparse 没有定义，程序会在训练开始前直接报 unknown argument。
```

这些参数都加到参数部分的最后面

## Step 6. 在 modeling_finetune.py 增加模型状态

改哪里：

```text
modeling_finetune.py
NeuralTransformer.__init__()
```

需要增加：

```python
self.completion_scope = "none"
self.pooling_scope = "low"
self.target_input_chans_index = None
self.real_input_chans_index = None

self.register_buffer(
    "tuev23_channel_prototypes",
    torch.zeros(23, embed_dim),
    persistent=False,
)

self.register_buffer(
    "seedv62_channel_prototypes",
    torch.zeros(62, embed_dim),
    persistent=False,
)

self.register_buffer(
    "tuev23_with_seedv62_extra_channel_prototypes",
    torch.zeros(70, embed_dim),
    persistent=False,
)
```

为什么：

```text
prototype 不能每次 forward 都从磁盘读。
register_buffer 可以让 prototype 跟着模型一起 .to(device)，但不作为可训练参数。
persistent=False 表示它不作为核心 checkpoint 权重保存。
```

默认值建议：

```text
completion_scope 默认 none。
pooling_scope 默认 low。
```

这样 baseline 不补通道时行为最接近原始 LaBraM。


## Step 7. 在 run_class_finetuning.py 加载 prototype 并配置模型

改哪里：

```text
run_class_finetuning.py
```

位置：

```text
model = get_models(args)
加载 finetune checkpoint
配置 completion_scope / pooling_scope / prototype / freeze_cnn
model.to(device)
创建 optimizer
```

推荐顺序：

```python
model.completion_scope = args.completion_scope
model.pooling_scope = args.pooling_scope

if args.completion_scope != "none":
    # 读取 prototype 文件。
    # 例如 tuev13_with_tuev23 使用 docs/prototypes/01_tuev23_cnn_patch_embed_mean.pth。
    ckpt = torch.load(args.channel_prototype_path, map_location="cpu")

    # channel_prototypes 是真正用于补缺失通道的向量。
    # 例如 TUEV-23 prototype 的 shape 应该是 [23, embed_dim]，base 模型里通常是 [23, 200]。
    prototypes = ckpt["channel_prototypes"]

    # target_ch_names 是 prototype 每一行对应的通道名。
    # 例如 target_ch_names[0] = "FP1"，表示 prototypes[0] 是 FP1 的 prototype。
    target_ch_names = ckpt["ch_names"]

    # target_input_chans_index 是 target 通道空间对应的 LaBraM position embedding 索引。
    # 通常包含 cls token，所以长度是 1 + target 通道数。
    # 例如 TUEV-23 时长度应该是 24。
    target_input_chans_index = ckpt["input_chans_index"]
    """  prototype 文件里保存：
      {
          "ch_names": List[str],
          "input_chans_index": List[int], 给模型 pos_embed 用，包含 cls token
          "channel_input_chans_index": List[int], 给 prototype 行和通道对应关系用，不包含 cls token
          "channel_prototypes": Tensor[K, embed_dim],
      }
      input_chans_index = [0] + channel_input_chans_index
      例如 channel_input_chans_index = [
          1, 3, 5, 7, 13, 15, 21, 23, 31, 33,
          9, 11, 17, 19, 25, 27, 29, 30, 6, 14,
          22, 16, 18,
      ]

      对应关系：
      channel_prototypes[i] <-> ch_names[i] <-> channel_input_chans_index[i]
    """

    # real_input_chans_index 是当前真实输入通道对应的 LaBraM position embedding 索引。
    # 它由当前 dataset 返回的 ch_names 计算得到。
    # 例如 TUEV-13 输入时，长度应该是 14，包含 cls token + 13 个真实通道。
    real_input_chans_index = utils.get_input_chans(ch_names)

    """
    def get_input_chans(ch_names):
      input_chans = [0] # for cls token
      for ch_name in ch_names:
          input_chans.append(standard_1020.index(ch_name) + 1)
      return input_chans
    """

    if args.completion_scope == "tuev13_with_tuev23":
        # 把 TUEV-23 prototype 存到模型里。
        # forward_features 会先用它填满 23 个 target 通道，再用真实 TUEV-13 feature 覆盖对应位置。
        model.tuev23_channel_prototypes.copy_(prototypes)

        # target_input_chans_index 告诉模型：补完后的 target 通道用哪些 position embedding。
        model.target_input_chans_index = target_input_chans_index # [0] + channel_input_chans_index

        # real_input_chans_index 告诉模型：真实输入通道在 target 空间里应该覆盖哪些位置。
        model.real_input_chans_index = real_input_chans_index #[0] + 13
    elif args.completion_scope == "seedv23_with_seedv62":
        # 把 SEEDV-62 prototype 存到模型里，用于 SEEDV-23 -> SEEDV-62。
        model.seedv62_channel_prototypes.copy_(prototypes)
        model.target_input_chans_index = target_input_chans_index
        model.real_input_chans_index = real_input_chans_index
    elif args.completion_scope == "tuev23_with_seedv62_extra":
        # 把 70 通道 prototype 存到模型里，用于 TUEV-23 -> TUEV-23 + SEEDV-extra。
        model.tuev23_with_seedv62_extra_channel_prototypes.copy_(prototypes)
        model.target_input_chans_index = target_input_chans_index
        model.real_input_chans_index = real_input_chans_index

if args.freeze_cnn:
    for param in model.patch_embed.parameters():
        param.requires_grad = False
```

为什么：

```text
prototype 加载和模型配置应该发生在 optimizer 创建之前。
freeze_cnn 也必须发生在 optimizer 创建之前，这样 optimizer 不会包含 patch_embed 参数。
```

## Step 8. 在 forward_features 里实现补通道

改哪里：

```text
modeling_finetune.py
NeuralTransformer.forward_features()
```

baseline 逻辑：

```text
completion_scope=none:
  保持原始逻辑。
  x -> patch_embed -> cls/token/pos/time -> transformer -> pooling/head
```

补通道逻辑：

```text
1. 真实输入 x 先过 patch_embed，得到 x_real。
2. 把 x_real reshape 成 [B, real_channels, time_window, embed_dim]。
3. 根据 completion_scope 取对应 prototype。
4. prototype expand 成 x_full: [B, target_channels_num, time_window, embed_dim]。
5. 根据 real_input_chans_index 和 target_input_chans_index 找到真实通道对应的 target index。
6. 用真实 x_real 覆盖 x_full 对应位置。
7. flatten 成 [B, target_channels_num * time_window, embed_dim]。
8. 后面再加 cls token、position embedding、time embedding。
```

中文伪代码：

```text
先对真实输入做 CNN patch_embed
如果不补通道：
    直接使用真实 patch token
否则：
    创建完整 target 通道空间
    每个 target 通道先填对应 prototype
    对于每个真实输入通道：
        找到它在 target 通道空间里的位置
        用真实 patch_embed feature 覆盖 prototype
    把完整 target 通道空间展平成 transformer token
```

关键点：

```text
prototype 只补缺失通道。
真实输入通道的位置必须被真实 patch_embed feature 覆盖。
```
写一下具体代码带注释：

```python
def forward_features(self, x, input_chans=None, return_patch_tokens=False, return_all_tokens=False, **kwargs):
    # x: [B, N, A, T]
    # B = batch size
    # N = 当前真实输入通道数，例如 TUEV-13 时 N=13
    # A = 每个通道切出来的 patch/time window 数
    # T = 每个 patch 的长度，LaBraM 这里通常是 200
    batch_size, n, a, t = x.shape

    # 当前每个通道有多少个 temporal patch。
    # 常见输入是 [B, N, A, 200]，所以 input_time_window = A。
    input_time_window = a if t == self.patch_size else t

    # 原来这里是：
    #   x = self.patch_embed(x)
    #
    # 原来的 x 只包含真实输入通道 token。
    # 现在先命名为 x_real，因为后面可能还要创建补通道后的 x_full。
    # 第一步：只对真实输入通道做 CNN/TemporalConv patch_embed。
    # 输出 x_real: [B, N * input_time_window, embed_dim]
    x_real = self.patch_embed(x)

    if self.completion_scope == "none":
        # completion_scope=none 时，等价于原来的：
        #   x = self.patch_embed(x)
        # 不补通道，保持原始 LaBraM 行为。
        # 后续 token 只包含真实输入通道。
        x = x_real
        # token_input_chans_index 用来选择 position embedding。
        # 不补通道时，它就是 input_chans。
        # input_chans 由当前真实输入的 ch_names 通过 utils.get_input_chans(ch_names) 算出来。
        token_input_chans_index = input_chans
        pool_token_indices = None
        target_channels_num = n #当前 token 对应的通道数
    else:
        # 补通道时，需要先把真实 token reshape 回按通道分组的形式。
        # [B, N * A, C] -> [B, N, A, C]
        x_real = x_real.reshape(batch_size, n, input_time_window, self.embed_dim)

        # 根据 completion_scope 选择对应的 target prototype。
        if self.completion_scope == "tuev13_with_tuev23":
            # prototypes: [23, embed_dim]
            prototypes = self.tuev23_channel_prototypes
        elif self.completion_scope == "seedv23_with_seedv62":
            # prototypes: [62, embed_dim]
            prototypes = self.seedv62_channel_prototypes
        elif self.completion_scope == "tuev23_with_seedv62_extra":
            # prototypes: [70, embed_dim]
            prototypes = self.tuev23_with_seedv62_extra_channel_prototypes
        else:
            raise ValueError(f"Unsupported completion_scope: {self.completion_scope}")

        target_channels_num = prototypes.shape[0]   # 当前 token 对应的通道数

        # 用 prototype 初始化完整 target 通道空间。
        #
        # prototypes 原始形状:
        #   [target_channels_num, embed_dim]
        #
        # expand 后:
        #   [B, target_channels_num, input_time_window, embed_dim]
        #
        # 含义：
        #   每个 target 通道、每个 time patch，先都填这个通道自己的 prototype。
        x_full = prototypes.unsqueeze(0).unsqueeze(2).expand(
            batch_size,
            target_channels_num,
            input_time_window,
            self.embed_dim,
        ).clone()

        # real_input_chans_index 和 target_input_chans_index 都是 LaBraM position embedding 索引。
        # 它们都包含 cls token，所以第 0 个元素是 cls，真正通道从 [1:] 开始。
        #
        # 例如：
        #   target_input_chans_index = [0] + TUEV-23 的 LaBraM pos_embed index
        #   real_input_chans_index   = [0] + TUEV-13 的 LaBraM pos_embed index
        real_channel_pos = list(self.real_input_chans_index[1:])
        target_channel_pos = list(self.target_input_chans_index[1:])

        # 记录真实输入通道在 target tensor 里的通道维下标。
        # 注意：这里保存的是 x_full 的第 1 维下标，例如 TUEV-23 里的 0 到 22。
        # 它不是 LaBraM 128 position embedding 里的 index。
        # pooling_scope=low 时会用它只 pool 真实通道。
        real_channel_indices_in_target_tensor = []

        for real_i, real_pos in enumerate(real_channel_pos):
            # target_i 是该真实通道在 target 空间里的第几个通道。
            # 注意：这不是 LaBraM 128 里的 index，而是 target tensor x_full 的通道维 index。
            target_i = target_channel_pos.index(real_pos)
            real_channel_indices_in_target_tensor.append(target_i)

            # 用真实 patch_embed feature 覆盖 prototype。
            #
            # x_real[:, real_i, :, :]：
            #   第 real_i 个真实输入通道的所有 time patch feature。
            #
            # x_full[:, target_i, :, :]：
            #   target 空间里对应通道的位置。
            x_full[:, target_i, :, :] = x_real[:, real_i, :, :]

        # 补完后，把 [B, target_channels_num, A, C] 展平成 transformer token：
        # [B, target_channels_num * A, C]
        x = x_full.flatten(1, 2)

        # 后面 position embedding 应该使用 target_input_chans_index。
        # 因为此时 token 已经是 target 通道空间，不再是原始真实输入空间。
        token_input_chans_index = self.target_input_chans_index

        # pooling_scope=low：最后只 pool 真实输入通道对应的 token。
        # pooling_scope=high：最后 pool 所有补完后的 target token。
        pool_token_indices = (
            real_channel_indices_in_target_tensor
            if self.pooling_scope == "low"
            else None
        )

    # 原来这里开始就已经进入公共逻辑：
    #   cls_tokens = self.cls_token.expand(batch_size, -1, -1)
    #   x = torch.cat((cls_tokens, x), dim=1)
    #
    # 这两行本身不用变。
    # 变化只在于：这里的 x 可能是原始真实 token，也可能是补通道后的 target token。
    # 加 cls token。
    cls_tokens = self.cls_token.expand(batch_size, -1, -1)
    x = torch.cat((cls_tokens, x), dim=1)

    # 原来这里是：
    #   pos_embed_used = self.pos_embed[:, input_chans] if input_chans is not None else self.pos_embed
    #
    # 现在改成使用 token_input_chans_index：
    #   不补通道时 token_input_chans_index = input_chans
    #     input_chans 由当前真实输入的 ch_names 通过 utils.get_input_chans(ch_names) 算出来。也就是token_input_chans_index = input_chans
    #     所以它对应真实输入通道的 LaBraM position embedding 索引。
    #
    #   补通道时 token_input_chans_index = self.target_input_chans_index
    #     因为此时 x 已经从真实输入通道扩展成 target 通道空间，
    #     position embedding 也必须使用 target 通道空间的索引。
    #
    # 原因是补通道后，x 已经是 target 通道空间，position embedding 也要用 target 通道空间。
    # 加 channel position embedding。
    # token_input_chans_index 包含 cls token，所以可以直接索引 pos_embed。
    if self.pos_embed is not None:
        pos_embed_used = self.pos_embed[:, token_input_chans_index]
        pos_embed = pos_embed_used[:, 1:, :].unsqueeze(2).expand(
            batch_size,
            -1,
            input_time_window,
            -1,
        ).flatten(1, 2)
        pos_embed = torch.cat(
            (pos_embed_used[:, 0:1, :].expand(batch_size, -1, -1), pos_embed),
            dim=1,
        )
        x = x + pos_embed

    # 原来这里是：
    #   nc = n if t == self.patch_size else a   #这里 nc 是通道数。 x.shape = [B, N, A, T]
    #   time_embed = self.time_embed[:, 0:input_time_window, :].unsqueeze(1).expand(
    #       batch_size, nc, -1, -1
    #   ).flatten(1, 2)
    #
    # 现在直接用 target_channels_num：
    #   不补通道时 target_channels_num = n
    #   补通道时 target_channels_num = prototype 的目标通道数
    #
    # 原因是补通道后，token 数量已经变成 target_channels_num * input_time_window。
    # 加 time embedding。
    # 每个通道共享同一套 time embedding。
    if self.time_embed is not None:
        time_embed = self.time_embed[:, 0:input_time_window, :].unsqueeze(1).expand(
            batch_size,
            target_channels_num,
            -1,
            -1,
        ).flatten(1, 2)
        x[:, 1:, :] += time_embed

    # 下面 transformer blocks 和原来一样：
    #   x = self.pos_drop(x)
    #   for blk in self.blocks:
    #       x = blk(x, rel_pos_bias=None)
    #   x = self.norm(x)
    x = self.pos_drop(x)

    for blk in self.blocks:
        x = blk(x, rel_pos_bias=None)

    x = self.norm(x)

    if self.fc_norm is not None:
        if return_all_tokens:
            return self.fc_norm(x)

        patch_tokens = x[:, 1:, :]

        if return_patch_tokens:
            return self.fc_norm(patch_tokens)

        # 原来这里是：
        #   return self.fc_norm(t.mean(1))
        #
        # 原来直接对所有真实输入 patch token 做 mean pooling。
        # 现在如果 pooling_scope=low，需要先从 target token 里取回真实输入通道对应的 token。
        if pool_token_indices is not None:
            # pooling_scope=low:
            # patch_tokens 当前是 [B, target_channels_num * A, C]。
            # 先 reshape 回 [B, target_channels_num, A, C]，
            # 再只取真实输入通道对应的 target index。
            patch_tokens = patch_tokens.reshape(
                batch_size,
                target_channels_num,
                input_time_window,
                self.embed_dim,
            )
            patch_tokens = patch_tokens[:, pool_token_indices, :, :]
            return self.fc_norm(patch_tokens.flatten(1, 2).mean(1))

        # completion_scope=none 或 pooling_scope=high：
        # 直接对当前全部 patch token 做平均。
        return self.fc_norm(patch_tokens.mean(1))

    if return_all_tokens:
        return x
    if return_patch_tokens:
        return x[:, 1:]
    return x[:, 0]
```

注意：

```text
这段是落代码时的参考示例。
真正修改 modeling_finetune.py 时，要结合当前原始 forward_features 的返回分支，
特别是 return_patch_tokens / return_all_tokens，避免破坏原始行为。
```

## Step 9. 参考 preexp10 同步 channel_subset、ch_names 和真实数据

参考哪里：

```text
/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/eeg-test/labram-preexp10/run_class_finetuning.py
/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/eeg-test/labram-preexp10/utils.py
/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/eeg-test/labram-preexp10/docs/preexp2_missing_channels/03_code_changes_and_usage.md
```

preexp10 的核心做法：

```text
channel_subset 不是只作为命令行标签。
它必须同时决定：

1. dataset loader 实际返回哪些通道的数据。
2. get_dataset(args) 返回给 engine 的 ch_names 是哪些通道。
3. engine_for_finetuning.py 用同一份 ch_names 计算 input_chans。
4. model.forward_features() 用 input_chans 给真实输入 token 加 position embedding。
```

为什么要这样：

```text
如果数据实际是 TUEV-13，但 ch_names 仍然是 TUEV-23，
那么 utils.get_input_chans(ch_names) 会给出 23 个通道的位置索引。
模型就会把 13 通道的数据按 23 通道的 position embedding 解释，通道语义会错位。
```

preexp10 的 `run_class_finetuning.py` 结构：

```python
from Channels_definition import (
    TUEV_13_CHANNELS,
    TUEV_23_CHANNELS,
    SEEDV_23_CHANNELS,
    SEEDV_62_CHANNELS,
)

DATASET_CONFIGS = {
    "TUEV": {
        "root": "...",
        "prepare_fn": utils.prepare_TUEV_dataset,
        "ch_names": {
            "tuev13": TUEV_13_CHANNELS,
            "tuev23": TUEV_23_CHANNELS,
        },
        "pass_channel_names": True,
        "nb_classes": 6,
        "metrics": ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted"],
    },
    "SEEDV": {
        "root": "...",
        "prepare_fn": utils.prepare_SEEDV_dataset,
        "ch_names": {
            "seedv23": SEEDV_23_CHANNELS,
            "seedv62": SEEDV_62_CHANNELS,
        },
        "pass_channel_names": True,
        "nb_classes": 5,
        "metrics": ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted"],
    },
}
```

注意：

```text
preexp10 的通道常量来自 utils.py。
preexp12 已经把通道定义迁移到 Channels_definition.py，所以这里不要照抄 utils.TUEV_13_CHANNELS。
训练代码、prototype 生成脚本、检查脚本都应该共用 Channels_definition.py。
```

`get_dataset(args)` 的计划写法：

```python
def get_dataset(args):
    # 原来这里通常是：
    #
    #   if args.dataset == "TUEV":
    #       train_dataset, test_dataset, val_dataset = utils.prepare_TUEV_dataset(root)
    #       ch_names = ["FP1", "FP2", ..., "T2"]  # 固定 TUEV-23
    #
    # 也就是：
    #   1. ch_names 写死成 TUEV-23。
    #   2. prepare_TUEV_dataset 不知道 channel_subset。
    #   3. --channel_subset=tuev13 时，数据和 ch_names 可能不同步。
    #
    # 现在改成 DATASET_CONFIGS：
    #   先根据 args.dataset 找配置，
    #   再根据 args.channel_subset 选择真实输入通道 ch_names。
    cfg = DATASET_CONFIGS[args.dataset]
    """
      DATASET_CONFIGS = {
      "TUEV": {
          "root": "...",
          "prepare_fn": utils.prepare_TUEV_dataset,
          "ch_names": {
              "tuev13": TUEV_13_CHANNELS,
              "tuev23": TUEV_23_CHANNELS,
          },
          "pass_channel_names": True,
          "nb_classes": 6,
          "metrics": ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted"],
      },
    """

    ch_names = cfg["ch_names"]
    if isinstance(ch_names, dict):
        # 原来没有这一层选择。
        # 现在 TUEV 可以在 tuev13 / tuev23 之间选，
        # SEEDV 可以在 seedv23 / seedv62 之间选。
        if args.channel_subset not in ch_names:
            raise ValueError(
                f"Unsupported channel_subset {args.channel_subset} for dataset {args.dataset}"
            )
        ch_names = ch_names[args.channel_subset]

    prepare_fn = cfg["prepare_fn"]
    prepare_kwargs = {}

    if cfg.get("pass_channel_names", False):
        # 原来是：
        #   prepare_fn(cfg["root"])
        #
        # 现在对支持通道裁剪的数据集，显式把 ch_names 传进去。
        # 这样 dataset 返回的 samples.shape[1] 才会和 len(ch_names) 一致。
        train_dataset, test_dataset, val_dataset = prepare_fn(
            cfg["root"],
            channel_names=ch_names,
            **prepare_kwargs,
        )
    else:
        train_dataset, test_dataset, val_dataset = prepare_fn(
            cfg["root"],
            **prepare_kwargs,
        )

    print(f"{args.dataset} channel_subset={args.channel_subset}")
    print(f"{args.dataset} channels ({len(ch_names)}): {ch_names}")

    args.nb_classes = cfg["nb_classes"]
    metrics = cfg["metrics"]
    return train_dataset, test_dataset, val_dataset, ch_names, metrics
```

`utils.py` 里 TUEV loader 的计划写法：

```python
class TUEVLoader(torch.utils.data.Dataset):
    def __init__(self, root, files, sampling_rate=200, channel_indices=None, channel_names=None):
        # 原来 TUEVLoader 大概只保存 root/files/sampling_rate，
        # __getitem__ 直接返回 sample["signal"] 的全部 TUEV-23 通道。
        #
        # 现在新增 channel_names / channel_indices：
        #   channel_names 是 ["FP1", "FP2", ...] 这样的语义通道名。
        #   channel_indices 是这些通道在原始 TUEV-23 数据里的下标。
        self.root = root
        self.files = files
        self.default_rate = 200
        self.sampling_rate = sampling_rate

        if channel_indices is not None and channel_names is not None:
            raise ValueError("Pass either channel_indices or channel_names, not both")

        if channel_names is not None:
            # 原来没有这一步。
            # 现在先检查传入的通道名是否都属于 TUEV-23。
            unknown = [name for name in channel_names if name not in TUEV_23_CHANNELS]
            if unknown:
                raise ValueError(f"Unknown TUEV channel names: {unknown}")

            # 把语义通道名转成原始 TUEV-23 数据里的下标。
            # 例如 channel_names=["FP1", "FP2", "CZ"]，
            # 就会转成这些通道在 TUEV_23_CHANNELS 里的 index。
            channel_indices = [TUEV_23_CHANNELS.index(name) for name in channel_names]

        self.channel_indices = channel_indices

    def __getitem__(self, index):
        sample = pickle.load(open(os.path.join(self.root, self.files[index]), "rb"))
        X = sample["signal"]

        # 原来这里直接使用完整 X，通常是 [23, T]。
        # 现在如果 channel_indices 不为空，就真正裁剪通道。
        # 例如 channel_subset=tuev13 时，X 会从 [23, T] 变成 [13, T]。
        if self.channel_indices is not None:
            X = X[self.channel_indices]

        ...
        return X, y
```

这里最重要的是：

```text
channel_names 是语义名字，例如 ["FP1", "FP2", ..., "CZ"]。
TUEVLoader 内部把 channel_names 转成原始 TUEV-23 顺序里的 channel_indices。
__getitem__ 用 channel_indices 真正裁剪 X。
```

落到 preexp12 的计划：

```text
1. 在 Channels_definition.py 里维护 TUEV_13_CHANNELS / TUEV_23_CHANNELS / SEEDV_23_CHANNELS / SEEDV_62_CHANNELS。
2. 在 run_class_finetuning.py 里从 Channels_definition.py import 这些通道常量。
3. 在 run_class_finetuning.py 里新增 DATASET_CONFIGS。
4. get_dataset(args) 根据 args.channel_subset 选择 ch_names。
5. 对 TUEV/SEEDV 这类支持子集的数据集，把 ch_names 作为 channel_names 传给 prepare_xxx_dataset。
6. 修改 utils.prepare_TUEV_dataset 和 TUEVLoader，让它们支持 channel_names。
7. engine_for_finetuning.py 不需要特殊改：它继续用 ch_names 调 utils.get_input_chans(ch_names)。
8. forward_features 里的 input_chans 就自然表示真实输入通道的位置编码索引。
```

## Step 10. 明确 best epoch 的选择标准

改哪里：

```text
run_class_finetuning.py
训练日志 / checkpoint 保存逻辑
```

当前原始逻辑：

```python
if max_accuracy < val_stats["accuracy"]:
    max_accuracy = val_stats["accuracy"]
    utils.save_model(..., epoch="best", ...)
    max_accuracy_test = test_stats["accuracy"]
```

也就是说：

```text
当前 best checkpoint 是按 validation accuracy 选的。
哪个 epoch 的 val accuracy 最高，就保存哪个 epoch 为 checkpoint-best.pth。
test accuracy 只是记录这个 best-val epoch 对应的 test 表现，不应该用 test 来选 best。
```

建议 preexp12 先保持这个规则：

```text
TUEV:
  best_metric = val cohen_kappa

SEEDV:
  best_metric = val accuracy

TUAB / binary task:
  best_metric = val roc_auc
```

为什么：

```text
best epoch 必须用 validation set 选，不能用 test set 选。
否则 test set 参与模型选择，会造成评估泄漏。
```

建议后续写成显式参数：

```python
parser.add_argument(
    "--best_metric",
    default="accuracy",
    choices=["accuracy", "balanced_accuracy", "f1_weighted", "cohen_kappa", "roc_auc", "pr_auc"],
    type=str,
    # 推荐：
    #   TUAB  -> roc_auc
    #   TUEV  -> cohen_kappa
    #   SEEDV -> accuracy
    #
    # 注意：
    #   这里选的是 validation set 上的 best metric。
    #   test set 只用于报告 best-val epoch 对应的最终表现，不能用来选 best epoch。
)
```

训练时：

```python
current_score = val_stats[args.best_metric]

print(f"Epoch {epoch} val metrics: {val_stats}")
print(f"Epoch {epoch} test metrics: {test_stats}")
print(
    f"Epoch {epoch} metric distribution: "
    f"val={json.dumps(val_stats, sort_keys=True)}, "
    f"test={json.dumps(test_stats, sort_keys=True)}"
)

if current_score > best_score:
    best_score = current_score
    utils.save_model(..., epoch="best", ...)
    best_epoch = epoch
    best_val_stats = val_stats
    best_test_stats = test_stats
```

日志里必须打印：

```text
每个 epoch:
  Best metric name: cohen_kappa / accuracy / roc_auc / ...
  Current val metrics: {accuracy, balanced_accuracy, cohen_kappa, f1_weighted, ...}
  Current test metrics: {accuracy, balanced_accuracy, cohen_kappa, f1_weighted, ...}

更新 best 时:
  Best epoch: ...
  Best metric name: ...
  Best val selected score: ...
  Best val metrics distribution: ...
  Test metrics distribution at best val epoch: ...
```

注意：

```text
如果某个 dataset 的 metrics 里没有 args.best_metric，要尽早 raise ValueError。
例如 TUAB 没有 cohen_kappa，就不能用 --best_metric cohen_kappa。
```

## Step 11. 不要迁移的内容

不要从旧目录复制：

```text
outputs/
__pycache__/
.git/
运行日志
历史 checkpoint
临时实验文件
```

为什么：

```text
这些是运行产物，不是 preexp12 分支应该维护的代码能力。
```
