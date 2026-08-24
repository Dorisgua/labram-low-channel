"""Unified Stage2 downstream classifier for four Transformer input modes.

Modes are ``full``, ``observed_only``, ``prototype`` and ``dynamic``.  The
classifier forward receives only the raw view required by the selected mode;
the dynamic path never constructs a reconstruction target from ``x_full``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from clean_disentangle.modeling import ComponentMode, CompositionMode
from clean_disentangle.prototype import PrototypeProvider


# 四组对照：真实28通道、仅12个观测通道、固定原型补全、Stage1动态补全。
MODES = ("full", "observed_only", "prototype", "dynamic")
DEFAULT_STAGE1 = Path("outputs/missing_prototype_d/missing_prototype_d_seed0_20260818_143337")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def load_backbone(backbone: nn.Module, checkpoint_path: Path) -> None:
    """把不同保存格式的 LaBraM 预训练权重统一加载到 Stage2 backbone。"""
    import utils

    payload = torch.load(checkpoint_path, map_location="cpu")
    source: Any = payload
    if isinstance(payload, dict):
        for key in ("model", "module"):
            if isinstance(payload.get(key), dict):
                source = payload[key]
                break
    if not isinstance(source, dict):
        raise TypeError(f"checkpoint has no state_dict: {checkpoint_path}")
    # 兼容预训练 checkpoint 中以 student. 开头的参数名。
    student = {
        key[len("student.") :]: value
        for key, value in source.items()
        if isinstance(key, str) and key.startswith("student.")
    }
    if student:
        source = student
    expected = backbone.state_dict()
    source = dict(source)
    # 预训练分类头与 ERP-Core 12 分类不匹配时，不加载旧分类头。
    for key in ("head.weight", "head.bias"):
        if key in source and key in expected and tuple(source[key].shape) != tuple(expected[key].shape):
            source.pop(key)
    for key in list(source):
        if "relative_position_index" in key:
            source.pop(key)
    print(f"LABRAM_PRETRAINED_CHECKPOINT={checkpoint_path.resolve()}")
    utils.load_state_dict(backbone, source)


def patch_tokens(backbone: nn.Module, eeg: torch.Tensor, expected_channels: int, *, patch_size: int = 200, num_t: int = 1, trainable: bool = False) -> torch.Tensor:
    """将 EEG 从 [B,C,200] 转为 LaBraM token [B,C,embed_dim]。"""
    # Dataset 通常给出三维 EEG [B,C,200]；LaBraM TemporalConv 需要
    # [B,C,num_t,patch_size]，所以 num_t=1 时在中间补出一个长度为1的维度。
    if eeg.ndim == 3:
        if eeg.shape[-1] != patch_size * num_t:
            raise ValueError(f"unexpected EEG length: {tuple(eeg.shape)}")
        eeg = eeg.reshape(eeg.shape[0], eeg.shape[1], num_t, patch_size)
    if eeg.ndim != 4 or eeg.shape[1] != expected_channels:
        raise ValueError(f"expected [B,{expected_channels},{num_t},{patch_size}], got {tuple(eeg.shape)}")
    # trainable 表示“是否需要经过 patch_embed/TemporalConv 反向传播”：
    # - False：用 no_grad() 生成 token，节省显存，CNN 不会收到梯度。
    # - True：保留计算图，Stage2 分类损失可以回传到 CNN。
    # 参数本身的 requires_grad 还会在 TransformerClassifier.__init__ 里单独设置。
    if trainable and torch.is_grad_enabled():
        tokens = backbone.patch_embed(eeg)  # 交给 LaBraM 的 TemporalConv，并保留梯度。
    else:
        with torch.no_grad():
            tokens = backbone.patch_embed(eeg)  # 只取 token，不为 CNN 建立反向传播图。
    # num_t=1 时，每个 EEG 通道对应一个 embed_dim 维 token。
    expected = (eeg.shape[0], expected_channels * num_t, backbone.embed_dim)
    if tuple(tokens.shape) != expected:
        raise ValueError(f"patch token shape {tuple(tokens.shape)} != {expected}")
    return tokens


def load_prototypes(path: Path, expected_names: tuple[str, ...]) -> tuple[PrototypeProvider, list[int]]:
    """加载固定通道原型，并严格检查28通道名称和顺序。"""
    payload = torch.load(path, map_location="cpu")
    # Prototype 是按通道顺序存储的；顺序错了就会把某通道的原型
    # 填到另一个通道上，所以这里不允许只检查“名称集合相同”。
    names = tuple(str(value).strip().upper() for value in payload["ch_names"])
    if names != expected_names:
        raise ValueError(f"prototype channel order mismatch: {names} != {expected_names}")
    indices = payload.get("input_chans_index")
    if indices is None:
        import utils
        indices = utils.get_input_chans(list(expected_names))
    # input_chans_index 用于从 LaBraM 位置编码表中取出对应位置；
    # 第一个位置留给 CLS，后面才是28个通道。
    if len(indices) != len(expected_names) + 1:
        raise ValueError("prototype input_chans_index must include CLS plus every channel")
    return PrototypeProvider(payload["channel_prototypes"], channel_names=names), [int(v) for v in indices]


def build_stage1_dynamic(
    config: dict[str, Any],
    checkpoint: Path,
    *,
    cnn_checkpoint: Path,
    prototype_checkpoint: Path,
):
    """恢复并冻结 Stage1；它只生成缺失通道 token，不参与 Stage2 更新。"""
    from clean_disentangle.modeling import MISSING_PROTOTYPE_SPEC
    from clean_disentangle.stage1.train_stage1 import (
        build_erpcore_reconstruction_model,
    )

    # dynamic 只接受 Missing + Prototype + D_sub + D_task 的 Stage1 配置。
    expected = ("missing", "prototype", "prototype", "identity", "sum")
    actual = tuple(config.get(key) for key in ("scope", "missing_fill", "output_base", "component_mode", "composition_mode"))
    if actual != expected:
        raise RuntimeError(f"dynamic requires Stage1 C config, got {actual}")
    repo_root = Path(__file__).resolve().parents[2]
    model = build_erpcore_reconstruction_model(
        legacy_root=repo_root,
        cnn_checkpoint=cnn_checkpoint,
        prototype_checkpoint=prototype_checkpoint,
        spec=MISSING_PROTOTYPE_SPEC,
        seed=int(config.get("seed", 0)),
        unfreeze_cnn=False,
    )
    # 先按配置重建结构，再严格加载 Stage1 权重；strict=True 会防止
    # “参数名或 shape 对不上，但仍然带着随机参数继续跑”。
    payload = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(payload["model"], strict=True)
    # Stage2 只能使用 Stage1 的输出，不能反向更新 Stage1。
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()
    return model


class TransformerClassifier(nn.Module):
    """将四种通道输入统一转换为 token，再用 LaBraM 完成12分类。"""
    def __init__(
        self,
        backbone: nn.Module,
        input_mode: str,
        channel_positions: list[int],
        *,
        dynamic_model=None,
        prototype_provider=None,
        last_n_blocks: int = 12,
        train_cnn: bool = False,
    ):
        super().__init__()
        if input_mode not in MODES:
            raise ValueError(f"unknown input mode: {input_mode}")
        self.backbone = backbone
        self.input_mode = input_mode
        self.channel_positions = list(channel_positions)
        self.dynamic_model = dynamic_model
        self.prototype_provider = prototype_provider
        self.last_n_blocks = int(last_n_blocks)
        # self.train_cnn 来自命令行 --train-cnn。它只控制 Stage2 backbone 的
        # patch_embed/TemporalConv，不会解冻 dynamic 模式里的 Stage1。
        self.train_cnn = bool(train_cnn)
        if self.input_mode == "dynamic" and self.train_cnn:
            raise ValueError("dynamic mode uses the frozen Stage1 TemporalConv; TRAIN_CNN must be 0")
        if not 0 <= self.last_n_blocks <= len(backbone.blocks):
            raise ValueError(f"last_n_blocks must be in [0,{len(backbone.blocks)}]")
        # 先全部冻结，再只开放实验指定的模块，便于审计真实训练范围。
        for parameter in self.parameters():
            parameter.requires_grad = False
        # 当 train_cnn=True 时，这里允许 CNN 参数收取梯度；后面调用
        # patch_tokens(..., trainable=self.train_cnn) 则保证前向时确实保留计算图。
        if self.train_cnn:
            for parameter in backbone.patch_embed.parameters():
                parameter.requires_grad = True
        if self.dynamic_model is not None:
            for parameter in self.dynamic_model.parameters():
                parameter.requires_grad = False
        # 仅微调最后 N 个 Transformer block。
        for block in backbone.blocks[-self.last_n_blocks:] if self.last_n_blocks else []:
            for parameter in block.parameters():
                parameter.requires_grad = True
        for parameter in backbone.norm.parameters():
            parameter.requires_grad = True
        if backbone.fc_norm is not None:
            for parameter in backbone.fc_norm.parameters():
                parameter.requires_grad = True
        from modeling_adabrain import LinearWithConstraint

        # 分类头展平所有 token（包括 CLS），不是只读取 CLS。
        self.num_input_tokens = 12 if input_mode == "observed_only" else 28
        self.classifier_input_dim = (self.num_input_tokens + 1) * int(backbone.embed_dim)
        self.task_head = LinearWithConstraint(self.classifier_input_dim, 12, max_norm=1.0, flatten=True)
        for parameter in self.task_head.parameters():
            parameter.requires_grad = True

    def train(self, mode: bool = True):
        """切换训练模式，同时让冻结的 Stage1、CNN 和早期 block 保持 eval。"""
        # model.train() 默认会递归地把所有子模块设为 train。下面手动把
        # 冻结模块切回 eval，避免 Stage1 dropout 或冻结 block 的运行状态改变。
        super().train(mode)
        if self.dynamic_model is not None:
            self.dynamic_model.eval()
            self.dynamic_model.patch_embed.eval()
            self.dynamic_model.stable_core.eval()
        self.backbone.patch_embed.train(bool(mode and self.train_cnn))
        for block in self.backbone.blocks[:-self.last_n_blocks] if self.last_n_blocks else self.backbone.blocks:
            block.eval()
        return self

    def _dynamic_complete(self, x_obs: torch.Tensor) -> torch.Tensor:
        """用冻结 Stage1 将12个观测 token 动态补成28个 token。"""
        # Stage2 只把 Stage1 当成固定的“token 生成器”：整个补全过程不建立梯度图。
        with torch.no_grad():
            corrector = self.dynamic_model
            # h_obs: [B,12,200]；p_miss: [B,16,200]。
            h_obs = corrector.patch_tokens(x_obs, expected_channels=12)
            observed = corrector.observed_token_positions.to(x_obs.device)
            missing = corrector.missing_token_positions.to(x_obs.device)
            provider = corrector.prototype_provider
            p_miss = provider.get_missing(x_obs.shape[0], missing_channel_positions=corrector.missing_channel_positions, num_t=1, device=x_obs.device, dtype=h_obs.dtype)
            # 先用观测 token 和固定 prototype 组成 [B,28,200] 上下文。
            context = h_obs.new_zeros(x_obs.shape[0], 28, corrector.stable_core.embed_dim)
            context = context.index_copy(1, observed, h_obs).index_copy(1, missing, p_miss)
            # StableCore 同时产生 shared/sub/task 分支；这里只从16个缺失位置
            # 取出 D_sub 和 D_task，它们的 shape 都是 [B,16,200]。
            rep = corrector.stable_core.encode_tokens(context)
            components = corrector.build_components(rep, missing, ComponentMode.IDENTITY)
            # 缺失通道预测 = prototype + D_sub + D_task。
            pred = corrector.compose_prediction(
                p_miss,
                components["d_sub"],
                components["d_task"],
                CompositionMode.SUM,
            )
            complete = context.index_copy(1, missing, pred)
            return complete

    def build_transformer_input(self, x: torch.Tensor) -> torch.Tensor:
        """按实验模式构造12或28个 Transformer 输入 token。"""
        if self.input_mode == "full":
            # full：x 是真实28通道 [B,28,200]。
            # trainable=self.train_cnn 的意思是：
            #   self.train_cnn=False -> 冻结 TemporalConv，只取 [B,28,200] token；
            #   self.train_cnn=True  -> 分类 loss 可以回传并更新 TemporalConv。
            return patch_tokens(self.backbone, x, 28, trainable=self.train_cnn)
        if self.input_mode == "observed_only":
            # observed_only：x 只有12个真实观测通道，不构造缺失通道 token。
            # 这里的 trainable 含义与 full 完全一样，只是通道数从28变成12。
            return patch_tokens(self.backbone, x, 12, trainable=self.train_cnn)
        if self.input_mode == "dynamic":
            # dynamic 不调用 Stage2 backbone.patch_embed；它直接使用冻结 Stage1
            # 生成完整28-token 输入，因此该模式禁止 train_cnn=True。
            return self._dynamic_complete(x)
        # prototype 模式：保留12个真实观测 token，用固定原型填入16个缺失位置。
        # prototype 模式中，12个观测 token 仍由 Stage2 TemporalConv 生成；
        # trainable 只决定这个 TemporalConv 是否跟随分类任务更新。
        h_obs = patch_tokens(self.backbone, x, 12, trainable=self.train_cnn)
        observed = torch.as_tensor(self._observed_positions, dtype=torch.long, device=x.device)
        missing = torch.as_tensor(self._missing_positions, dtype=torch.long, device=x.device)
        p_miss = self.prototype_provider.get_missing(x.shape[0], missing_channel_positions=missing, num_t=1, device=x.device, dtype=h_obs.dtype)
        # 先建一个空的28-token 容器，再按固定的 ERP-Core 通道位置写入：
        # observed 位置写真实 token，missing 位置写固定 prototype。
        complete = h_obs.new_zeros(x.shape[0], 28, self.backbone.embed_dim)
        complete = complete.index_copy(1, observed, h_obs)
        if self.input_mode == "prototype":
            return complete.index_copy(1, missing, p_miss)
        raise RuntimeError(f"unhandled input mode: {self.input_mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """完成一次 Stage2 分类前向：原始 EEG -> token -> Transformer -> 12类 logits。

        一般不直接调用 ``forward``；训练代码执行 ``model(source)`` 时，PyTorch
        的 ``nn.Module.__call__`` 会自动进入这里。输入 x 的含义取决于模式：
        ``full`` 接收 [B,28,200] 的 x_full，其余模式接收 [B,12,200] 的 x_obs。
        最终返回 [B,12]，每一列是一个 ERP-Core 类别的未归一化分数（logit）。
        """
        # 这里开始时已经不关心原始 EEG 是怎么补全的：四种模式都先
        # 统一成 [B,N,200] token，区别只是 N=12 或 N=28。
        # build_transformer_input 是“分流器”：
        # full/observed_only 走 Stage2 TemporalConv，prototype 填固定原型，
        # dynamic 则调用冻结 Stage1 生成 P_miss + D_sub + D_task。
        tokens = self.build_transformer_input(x)
        batch_size, num_tokens, _ = tokens.shape
        # observed_only 为13个 token；其余模式为29个 token（均包含 CLS）。
        x = torch.cat((self.backbone.cls_token.expand(batch_size, -1, -1), tokens), dim=1)
        indices = torch.as_tensor(self.channel_positions, dtype=torch.long, device=x.device)
        if self.backbone.pos_embed is not None:
            # 根据实际通道名称选取位置编码，不是简单地把12通道当成前12个位置。
            # indices 中还包含 CLS 的位置，所以 used 为 [1,N+1,200]。
            used = self.backbone.pos_embed[:, indices]
            pos = used[:, 1:, :].unsqueeze(2).expand(batch_size, -1, 1, -1).flatten(1, 2)
            x = x + torch.cat((used[:, :1, :].expand(batch_size, -1, -1), pos), dim=1)
        if self.backbone.time_embed is not None:
            # 当前 num_t=1，所有通道 token 都加上同一个时间块位置编码。
            x[:, 1:] = x[:, 1:] + self.backbone.time_embed[:, :1, :].unsqueeze(1).expand(batch_size, num_tokens, -1, -1).flatten(1, 2)
        x = self.backbone.pos_drop(x)
        # 所有 block 都参与前向；只有最后 last_n_blocks 保存梯度并更新。
        for block in self.backbone.blocks:
            x = block(x, rel_pos_bias=None)
        x = self.backbone.norm(x)
        if self.backbone.fc_norm is not None:
            x = self.backbone.fc_norm(x)
        # task_head 会将 CLS+所有通道 token 展平，输出 [B,12] 分类 logits。
        return self.task_head(x)


def make_model(args: argparse.Namespace, stage1_config: dict[str, Any] | None):
    """组装 Stage2 backbone、原型提供器和可选的冻结 Stage1。"""
    import modeling_finetune  # noqa: F401  # 该导入会向 timm 注册 LaBraM 模型。
    from timm.models import create_model
    from Channels_definition import ERPCORE_12_CHANNELS, ERPCORE_28_CHANNELS

    # 先创建 Stage2 LaBraM 结构，再由 load_backbone() 加载预训练权重。
    # pretrained=False 只是表示 timm 不自动下载权重，不代表最终使用随机初始化。
    backbone = create_model("labram_base_patch200_200", pretrained=False, num_classes=12, drop_rate=0.0, drop_path_rate=0.1, attn_drop_rate=0.0, drop_block_rate=None, use_mean_pooling=True, init_scale=0.001, use_rel_pos_bias=False, use_abs_pos_emb=True, init_values=0.1, qkv_bias=False)
    load_backbone(backbone, args.labram_checkpoint)
    full_names = tuple(ERPCORE_28_CHANNELS)
    # 将12个观测通道映射到固定的28通道顺序中。
    observed_positions = [full_names.index(name) for name in ERPCORE_12_CHANNELS]
    missing_positions = [i for i, name in enumerate(full_names) if name not in ERPCORE_12_CHANNELS]
    prototype_provider = None
    target_indices = None
    # 只有 prototype/dynamic 需要访问固定 prototype bank。
    if args.input_mode in ("prototype", "dynamic"):
        prototype_provider, target_indices = load_prototypes(args.prototype_checkpoint, full_names)
    if args.input_mode == "observed_only":
        import utils
        channel_positions = utils.get_input_chans(list(ERPCORE_12_CHANNELS))
    elif args.input_mode == "full":
        import utils
        channel_positions = utils.get_input_chans(list(ERPCORE_28_CHANNELS))
    else:
        channel_positions = target_indices or []
    dynamic_model = None
    # dynamic 额外携带一套完整但冻结的 Stage1；另外三种模式不加载 Stage1。
    if args.input_mode == "dynamic":
        dynamic_model = build_stage1_dynamic(
            stage1_config,
            args.stage1_checkpoint,
            cnn_checkpoint=args.labram_checkpoint,
            prototype_checkpoint=args.prototype_checkpoint,
        )
    model = TransformerClassifier(
        backbone,
        args.input_mode,
        channel_positions,
        dynamic_model=dynamic_model,
        prototype_provider=prototype_provider,
        last_n_blocks=args.last_n_blocks,
        train_cnn=args.train_cnn,
    )
    model._observed_positions = observed_positions
    model._missing_positions = missing_positions
    return model, full_names, ERPCORE_12_CHANNELS


def validate_dynamic_metadata(args: argparse.Namespace, config: dict[str, Any]) -> None:
    """在加载权重前检查 Stage1 的模式、维度和通道顺序。"""
    expected_spec = {
        "scope": "missing",
        "missing_fill": "prototype",
        "output_base": "prototype",
        "component_mode": "identity",
        "composition_mode": "sum",
    }
    mismatches = [
        f"{key}={config.get(key)!r} (expected {value!r})"
        for key, value in expected_spec.items()
        if config.get(key) != value
    ]
    if mismatches:
        raise RuntimeError("dynamic requires a compatible Missing+Prototype+D Stage1 checkpoint: " + "; ".join(mismatches))
    if int(config.get("full_num_channels", 28)) != 28:
        raise RuntimeError(f"dynamic requires 28 full channels, got {config.get('full_num_channels')!r}")
    if int(config.get("embed_dim", 200)) != 200:
        raise RuntimeError(f"dynamic requires embedding dimension 200, got {config.get('embed_dim')!r}")
    expected_observed = ["FP1", "FP2", "F3", "F4", "F7", "F8", "C3", "C4", "P3", "P4", "O1", "O2"]
    observed = config.get("observed_channels")
    if observed is not None and list(observed) != expected_observed:
        raise RuntimeError(f"dynamic observed channel order mismatch: {observed!r} != {expected_observed!r}")
    print("Stage1 dynamic compatibility: PASS")


def print_resolved_summary(args: argparse.Namespace, model: nn.Module, output_dir: Path) -> None:
    """训练前打印最终生效的模式、冻结范围、超参数和输出目录。"""
    stage1_used = args.input_mode == "dynamic"
    prototype_used = args.input_mode in ("prototype", "dynamic")
    blocks = list(range(max(0, len(model.backbone.blocks) - args.last_n_blocks), len(model.backbone.blocks)))
    print("=" * 60)
    print("Resolved Experiment")
    print("=" * 60)
    print(f"Experiment: {args.exp_name}")
    print(f"Mode: {args.input_mode}")
    print(f"Seed: {args.seed}")
    print(f"Input: {'28 full channels' if args.input_mode == 'full' else '12 observed channels'}")
    print(f"Transformer tokens: {model.num_input_tokens} + CLS")
    print(f"Classifier input dim: {model.classifier_input_dim}")
    print(f"Stage1 checkpoint: {args.stage1_checkpoint.resolve() if stage1_used else 'NOT USED'}")
    print(f"Stage1 StableCore: {'USED, frozen/eval' if stage1_used else 'NOT USED'}")
    print(f"Prototype: {args.prototype_checkpoint.resolve() if prototype_used else 'NOT USED'}")
    print(f"LaBraM pretrained: {args.labram_checkpoint.resolve()}")
    print(f"Transformer depth: {len(model.backbone.blocks)}")
    print(f"Trainable block indices: {blocks}")
    print(f"CNN / patch_embed: {'TRAINABLE' if args.train_cnn else 'FROZEN'}")
    print(f"Epochs / batch size: {args.epochs} / {args.batch_size}")
    print(f"Optimizer / LR / weight decay: AdamW / {args.lr} / {args.weight_decay}")
    print(f"CNN LR: {args.lr * args.cnn_lr_mult if args.train_cnn else 'NOT USED (CNN frozen)'}")
    print(f"Output: {output_dir.resolve()}")
    print("=" * 60)


@torch.no_grad()
def evaluate(model, loader, device):
    """在验证集或测试集上计算四个分类指标。"""
    import utils
    model.eval(); outputs, targets = [], []
    for batch in loader:
        # 只有 full 基线读取真实28通道；其余模式始终只读取12通道 x_obs。
        source = batch["x_full"] if model.input_mode == "full" else batch["x_obs"]
        outputs.append(model(source.to(device)).cpu())
        targets.append(batch["label"].cpu())
    output = torch.cat(outputs).numpy(); target = torch.cat(targets).numpy()
    return {k: float(v) for k, v in utils.get_metrics(output, target, ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted"], False).items()}


def main(args: argparse.Namespace) -> None:
    """完成配置检查、模型构建、训练、评估和 checkpoint 保存。"""
    seed_everything(args.seed)

    # 四种输入模式都必须知道 ERP-Core 数据在哪里。这里先检查参数有没有提供，
    # 但暂时不读取数据；路径是否真的存在会在 dry-run 或正式加载数据时继续检查。
    if args.data_path is None:
        raise ValueError("--data-path is required; keep ERP-Core data outside the repo")

    # 只有 dynamic 模式依赖 Stage1，因此也只有这个模式才读取 stage1_config.json。
    # load_json() 会真的打开文件，所以 dynamic 的配置文件不存在或不是合法 JSON 时，
    # 即使当前是 dry-run，也会在这里提前报错；其他三个模式不会碰 Stage1 配置。
    stage1_config = load_json(args.stage1_config) if args.input_mode == "dynamic" else None
    if args.input_mode == "dynamic":
        # 先检查 Stage1 是否确实是 Missing + Prototype + D 的28通道/200维配置，
        # 避免把结构能勉强加载、但实验含义不一致的 checkpoint 交给 Stage2。
        validate_dynamic_metadata(args, stage1_config)

    # dry-run 的目标是低成本检查“这条命令引用的关键路径是否齐全”。
    # 它不会调用 make_model()、不会加载 EEG 内容、不会占用 GPU，也不会开始训练。
    if args.dry_run:
        # 所有模式都需要 LaBraM 预训练 checkpoint 和 ERP-Core 数据路径。
        # exists() 只证明路径存在，并不验证 checkpoint 内容、数据 shape 或 forward。
        for required in (args.labram_checkpoint, args.data_path):
            if not required.exists():
                raise FileNotFoundError(required)

        # prototype/dynamic 都要用固定通道 prototype；full/observed_only 不需要，
        # 因而后两种模式不会因为 prototype 文件缺失而失败。
        if args.input_mode in ("prototype", "dynamic") and not args.prototype_checkpoint.exists():
            raise FileNotFoundError(args.prototype_checkpoint)

        # dynamic 还必须有 Stage1 权重。Stage1 配置文件没有在这里重复检查，
        # 因为上面的 load_json(args.stage1_config) 已经实际读取并验证过它。
        if args.input_mode == "dynamic" and not args.stage1_checkpoint.exists():
            raise FileNotFoundError(args.stage1_checkpoint)

        # 能走到这里说明当前模式所需的关键路径检查通过。return 很关键：
        # 它保证 dry-run 到此结束，不会继续执行下面的建模、数据加载和训练代码。
        print(f"DRY_RUN_PASS mode={args.input_mode} seed={args.seed} last_n_blocks={args.last_n_blocks}")
        return

    model, full_names, observed_names = make_model(args, stage1_config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 先根据 requires_grad 收集“真正会更新”的参数。
    # CNN 使用较小学习率；其他可训练参数使用主学习率。
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    cnn_params = [p for name, p in trainable if name.startswith("backbone.patch_embed.")]
    other_params = [p for name, p in trainable if not name.startswith("backbone.patch_embed.")]
    groups = [{"params": other_params, "lr": args.lr, "base_lr": args.lr}]
    if cnn_params:
        groups.append({"params": cnn_params, "lr": args.lr * args.cnn_lr_mult, "base_lr": args.lr * args.cnn_lr_mult})
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay, eps=1e-8)
    # 防止 requires_grad=True 的参数漏加或重复加入优化器。
    optimizer_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    trainable_ids = {id(parameter) for _, parameter in trainable}
    if optimizer_ids != trainable_ids:
        raise RuntimeError("optimizer parameters do not exactly match requires_grad=True parameters")
    trainable_names = [name for name, _ in trainable]
    print_resolved_summary(args, model, args.output_dir)
    print(f"Trainable parameter names: {json.dumps(trainable_names)}")
    print(f"Trainable parameter count: {sum(p.numel() for _, p in trainable)}")
    print(f"Frozen parameter count: {sum(p.numel() for p in model.parameters() if not p.requires_grad)}")
    # audit-only 真正读取一个测试 batch 并前向一次，但不训练。
    if args.audit_only:
        from data_processor.erpcore_cslp import prepare_ERPCORE_cslp_dataset
        _, test_dataset, _ = prepare_ERPCORE_cslp_dataset(args.data_path, sampling_rate=200, normalize_method="z_score")
        batch = next(iter(DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)))
        source = batch["x_full"] if args.input_mode == "full" else batch["x_obs"]
        with torch.no_grad(): logits = model(source.to(device))
        print(f"AUDIT_PASS raw={tuple(source.shape)} logits={tuple(logits.shape)}")
        return

    from data_processor.erpcore_cslp import prepare_ERPCORE_cslp_dataset
    # 数据函数的返回顺序是 train、test、val，请勿按常见的 train、val、test 理解。
    train_ds, test_ds, val_ds = prepare_ERPCORE_cslp_dataset(args.data_path, sampling_rate=200, normalize_method="z_score")

    # 训练集 shuffle=True；验证/测试集保持固定顺序。每个 batch 同时含
    # x_full/x_obs，但后面会按 input_mode 只选其中一个送入模型。
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    resolved = {"stage": "stage2", "exp_name": args.exp_name, "stage2_input_mode": args.input_mode, "train_cnn": bool(args.train_cnn), "cnn_lr_mult": args.cnn_lr_mult, "labram_checkpoint": str(args.labram_checkpoint.resolve()), "stage1_checkpoint": str(args.stage1_checkpoint.resolve()) if args.input_mode == "dynamic" else None, "stage1_config": str(args.stage1_config.resolve()) if args.input_mode == "dynamic" else None, "prototype_checkpoint": str(args.prototype_checkpoint.resolve()) if args.input_mode in ("prototype", "dynamic") else None, "prototype_source": "fixed_channel_prototype_bank" if args.input_mode in ("prototype", "dynamic") else None, "observed_channels": list(observed_names), "observed_positions": model._observed_positions, "missing_channels": [name for index, name in enumerate(full_names) if index in model._missing_positions], "missing_positions": model._missing_positions, "num_transformer_input_tokens": model.num_input_tokens, "classifier_input_dim": model.classifier_input_dim, "transformer_total_blocks": len(model.backbone.blocks), "trainable_last_n": args.last_n_blocks, "trainable_block_indices": list(range(max(0, len(model.backbone.blocks) - args.last_n_blocks), len(model.backbone.blocks))), "classifier_protocol": "all_token_linear_with_constraint", "optimizer": "adamw", "lr": args.lr, "cnn_lr": args.lr * args.cnn_lr_mult if args.train_cnn else None, "weight_decay": args.weight_decay, "warmup_epochs": args.warmup_epochs, "epochs": args.epochs, "batch_size": args.batch_size, "seed": args.seed, "full_channel_access": "NONE" if args.input_mode != "full" else "ALLOWED", "checkpoint_selection": ["best-bacc", "best-acc", "last"], "trainable_parameter_names": trainable_names}
    (args.output_dir / "config.json").write_text(json.dumps(resolved, indent=2, sort_keys=True), encoding="utf-8")


    # Stage2 只优化12分类交叉熵；Stage1 的重建/InfoNCE/swap loss 都不在这里使用。
    criterion = nn.CrossEntropyLoss(); best_bacc = -math.inf; best_acc = -math.inf
    fields = ["epoch", "train_loss", "val_accuracy", "val_balanced_accuracy", "val_cohen_kappa", "val_f1_weighted", "test_accuracy", "test_balanced_accuracy", "test_cohen_kappa", "test_f1_weighted"]
    with (args.output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle: csv.DictWriter(handle, fieldnames=fields).writeheader()
    start_epoch = 0
    # resume 同时恢复模型、优化器和起始 epoch。
    # if args.resume:
    #     payload = torch.load(args.resume, map_location="cpu")
    #     resume_config = payload.get("config", {})
    #     for key in ("stage2_input_mode", "classifier_input_dim", "trainable_last_n"):
    #         if resume_config.get(key) != resolved.get(key):
    #             raise RuntimeError(f"resume config mismatch for {key}: {resume_config.get(key)!r} != {resolved.get(key)!r}")
    #     model.load_state_dict(payload["model"], strict=True)
    #     optimizer.load_state_dict(payload["optimizer"])
    #     start_epoch = int(payload.get("epoch", -1)) + 1
    #     print(f"Resumed from {args.resume} at epoch {start_epoch}")
    def save(path, epoch): torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "config": resolved}, path)

    # 每个 epoch：训练 → 验证/测试 → 保存 last 和两个验证集最优 checkpoint。
    for epoch in range(start_epoch, args.epochs):
        model.train(True); losses=[]
        # 前 warmup_epochs 轮线性升高学习率，之后用 cosine 逐渐衰减。
        # scale 会同比例作用于主参数组和 CNN 参数组，保留 cnn_lr_mult 的比例。
        lr = args.lr * (float(epoch+1)/max(args.warmup_epochs,1) if epoch < args.warmup_epochs else 0.5*(1+math.cos(math.pi*(epoch-args.warmup_epochs)/max(args.epochs-args.warmup_epochs,1))))
        scale = lr / args.lr if args.lr else 1.0
        for group in optimizer.param_groups: group["lr"] = group["base_lr"] * scale
        for batch in train_loader:
            # 只有 full 基线读取真实28通道 x_full；其他模式都只把12通道 x_obs
            # 送入模型。prototype/dynamic 所需的16个 token 由模型内部构造。
            optimizer.zero_grad(set_to_none=True); source=batch["x_full"] if args.input_mode == "full" else batch["x_obs"]; logits=model(source.float().to(device)); loss=criterion(logits,batch["label"].to(device)); loss.backward(); optimizer.step(); losses.append(float(loss.detach().item()))
        val=evaluate(model,val_loader,device); test=evaluate(model,test_loader,device); rec={"epoch":epoch,"train_loss":float(np.mean(losses)),**{f"val_{k}":v for k,v in val.items()},**{f"test_{k}":v for k,v in test.items()}}
        with (args.output_dir/"metrics.csv").open("a",newline="",encoding="utf-8") as handle: csv.DictWriter(handle,fieldnames=fields).writerow({k:rec.get(k,"") for k in fields})
        print(json.dumps(rec,sort_keys=True),flush=True); save(args.output_dir/"checkpoint-last.pth",epoch)
        # 两个“最优” checkpoint 的选择标准不同：一个看 Val-BAcc，一个看 Val-Accuracy。
        # Test 指标会记录，但不参与 checkpoint 选择。
        if val["balanced_accuracy"] > best_bacc: best_bacc=val["balanced_accuracy"]; save(args.output_dir/"checkpoint-best-bacc.pth",epoch)
        if val["accuracy"] > best_acc: best_acc=val["accuracy"]; save(args.output_dir/"checkpoint-best-acc.pth",epoch)


def get_args():
    """定义命令行参数；Shell 启动脚本会提供数据路径和实验默认值。"""
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[2]
    stage1 = root / DEFAULT_STAGE1
    parser.add_argument("--input-mode", choices=MODES, default="dynamic")
    parser.add_argument("--exp-name", default="stage2_default")
    parser.add_argument("--stage1-checkpoint", type=Path, default=stage1 / "checkpoints/checkpoint-last.pth")
    parser.add_argument("--stage1-config", type=Path, default=stage1 / "config.json")
    parser.add_argument("--labram-checkpoint", type=Path, default=root / "checkpoints/labram-base.pth")
    parser.add_argument("--prototype-checkpoint", type=Path, default=root / "docs/prototypes/01_erpcore28_cnn_patch_embed_mean.pth")
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=root / "outputs/stage2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--cnn-lr-mult", type=float, default=0.1)
    # 不传 --train-cnn 时默认 False；传入后为 True，Stage2 TemporalConv 才会解冻。
    # dynamic 使用冻结 Stage1 TemporalConv，所以明确禁止开启该选项。
    parser.add_argument("--train-cnn", action="store_true")
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--last-n-blocks", type=int, default=12)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    # parser.add_argument("--resume", type=Path, default=None)
    return parser.parse_args()


if __name__ == "__main__": main(get_args())
