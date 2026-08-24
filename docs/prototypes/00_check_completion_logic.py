import argparse
import runpy
from pathlib import Path
import sys
"""
  python docs/prototypes/00_check_completion_logic.py \
    --pth docs/prototypes/01_tuev23_cnn_patch_embed_mean.pth \
    --scope tuev13_with_tuev23
    
    
  目标通道列表 = TUEV-23 的 23 个通道
  真实输入通道列表 = TUEV-13 的 13 个通道

  先创建一个完整的 23 通道容器 x_full

  第一步：
    对于 TUEV-23 里的每一个通道：
      x_full 里这个通道的位置先填入对应通道的 prototype

  第二步：
    对于真实输入 TUEV-13 里的每一个通道：
      找到这个通道在 TUEV-23 目标通道列表里的位置
      用这个真实通道的 patch_embed feature 覆盖 x_full 里的对应位置

  最后：
    TUEV-13 真实存在的通道 = 真实 feature
    TUEV-13 没有但 TUEV-23 有的通道 = prototype feature
"""
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import utils  # noqa: E402


def load_channels():
    channels_path = REPO_ROOT / "Channels_definition.py"
    namespace = runpy.run_path(
        str(channels_path),
        init_globals={"standard_1020": utils.standard_1020},
    )
    return namespace


def get_default_real_channels(scope, channels):
    if scope == "tuev13_with_tuev23":
        return list(channels["TUEV_13_CHANNELS"])
    if scope == "bciiv2a13_with_bciiv2a22":
        return list(channels["BCIIV2A_13_CHANNELS"])
    if scope == "physionet23_with_physionet64":
        return list(channels["PHYSIONET_23_CHANNELS"])
    if scope == "physionet32_with_physionet64":
        return list(channels["PHYSIONET_32_CHANNELS"])
    if scope == "seedv23_with_seedv62":
        return list(channels["SEEDV_23_CHANNELS"])
    raise ValueError(
        f"No default real channels for scope={scope}. "
        "Pass --real_channels explicitly."
    )


def check_completion_mapping(channel_prototypes, target_ch_names, real_ch_names):
    if not torch.is_tensor(channel_prototypes):
        raise TypeError("channel_prototypes must be a torch.Tensor")
    if channel_prototypes.ndim != 2:
        raise ValueError(f"channel_prototypes must be [K, D], got {tuple(channel_prototypes.shape)}")
    if channel_prototypes.shape[0] != len(target_ch_names):
        raise ValueError(
            "prototype rows must match target_ch_names; "
            f"got {channel_prototypes.shape[0]} rows and {len(target_ch_names)} channel names"
        )

    missing = [ch for ch in real_ch_names if ch not in target_ch_names]
    if missing:
        raise ValueError(f"real channels are not in target_ch_names: {missing}")

    real_channel_index = torch.tensor(
        [target_ch_names.index(ch) for ch in real_ch_names],
        dtype=torch.long,
    )

    batch_size = 2
    time_patches = 5
    embed_dim = channel_prototypes.shape[1]
    num_real_channels = len(real_ch_names)

    # Use synthetic real features with channel-specific sentinel values.
    # This makes wrong channel placement easy to detect.
    x_real = torch.zeros(batch_size, num_real_channels, time_patches, embed_dim)
    for i in range(num_real_channels):
        x_real[:, i, :, :] = 1000 + i

    x_full = channel_prototypes.view(1, len(target_ch_names), 1, embed_dim)
    x_full = x_full.expand(batch_size, len(target_ch_names), time_patches, embed_dim).clone()
    x_full[:, real_channel_index, :, :] = x_real

    for i, ch in enumerate(real_ch_names):
        target_i = target_ch_names.index(ch)
        if not torch.equal(x_full[:, target_i, :, :], x_real[:, i, :, :]):
            raise AssertionError(
                f"real channel {ch} was not copied to target index {target_i} correctly"
            )

    real_set = set(real_ch_names)
    for target_i, ch in enumerate(target_ch_names):
        if ch in real_set:
            continue
        expected = channel_prototypes[target_i].view(1, 1, embed_dim).expand(
            batch_size, time_patches, embed_dim
        )
        if not torch.equal(x_full[:, target_i, :, :], expected):
            raise AssertionError(
                f"missing channel {ch} at target index {target_i} did not preserve prototype"
            )

    return real_channel_index.tolist(), x_full


def main():
    parser = argparse.ArgumentParser("Check channel completion mapping logic")
    parser.add_argument(
        "--pth",
        default="docs/prototypes/01_tuev23_cnn_patch_embed_mean.pth",
        type=str,
        help="Prototype .pth file containing channel_prototypes and ch_names",
    )
    parser.add_argument(
        "--scope",
        default="tuev13_with_tuev23",
        choices=[
            "tuev13_with_tuev23",
            "bciiv2a13_with_bciiv2a22",
            "physionet23_with_physionet64",
            "physionet32_with_physionet64",
            "seedv23_with_seedv62",
        ],
        help="Completion scope to check",
    )
    parser.add_argument(
        "--real_channels",
        nargs="+",
        default=None,
        help="Optional real input channel names. Default comes from Channels_definition.py for the scope.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit printed channel mappings. Default prints all.",
    )
    args = parser.parse_args()

    ckpt = torch.load(args.pth, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise TypeError(f"Expected dict checkpoint, got {type(ckpt).__name__}")
    if "channel_prototypes" not in ckpt:
        raise KeyError("checkpoint missing key: channel_prototypes")
    if "ch_names" not in ckpt:
        raise KeyError("checkpoint missing key: ch_names")

    channel_prototypes = ckpt["channel_prototypes"]
    target_ch_names = list(ckpt["ch_names"])

    channels = load_channels()
    real_ch_names = list(args.real_channels) if args.real_channels else get_default_real_channels(args.scope, channels)

    real_channel_index, _ = check_completion_mapping(
        channel_prototypes=channel_prototypes,
        target_ch_names=target_ch_names,
        real_ch_names=real_ch_names,
    )

    print(f"prototype file: {args.pth}")
    print(f"scope: {args.scope}")
    print(f"channel_prototypes.shape: {tuple(channel_prototypes.shape)}")
    print(f"num target channels: {len(target_ch_names)}")
    print(f"num real channels: {len(real_ch_names)}")
    print()
    print("real channel -> target index mapping:")
    n = len(real_ch_names) if args.limit is None else min(len(real_ch_names), args.limit)
    for i in range(n):
        print(f"  real[{i}] {real_ch_names[i]} -> target[{real_channel_index[i]}]")
    if args.limit is not None and len(real_ch_names) > args.limit:
        print(f"  ... ({len(real_ch_names) - args.limit} more)")

    missing_target_channels = [ch for ch in target_ch_names if ch not in set(real_ch_names)]
    print()
    print(f"missing target channels kept as prototype ({len(missing_target_channels)}):")
    n_missing = len(missing_target_channels) if args.limit is None else min(len(missing_target_channels), args.limit)
    for i in range(n_missing):
        ch = missing_target_channels[i]
        print(f"  {ch} at target[{target_ch_names.index(ch)}]")
    if args.limit is not None and len(missing_target_channels) > args.limit:
        print(f"  ... ({len(missing_target_channels) - args.limit} more)")

    print()
    print("completion mapping check passed")


if __name__ == "__main__":
    main()
