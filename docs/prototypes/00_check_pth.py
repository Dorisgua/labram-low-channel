import argparse
from pathlib import Path

import torch
# python docs/prototypes/00_check_pth.py docs/prototypes/01_tuev23_cnn_patch_embed_mean.pth

def _shape(value):
    if torch.is_tensor(value):
        return tuple(value.shape)
    if isinstance(value, (list, tuple)):
        return f"len={len(value)}"
    if isinstance(value, dict):
        return f"dict_keys={list(value.keys())}"
    return type(value).__name__


def _print_sequence(name, value, limit=None):
    if value is None:
        print(f"{name}: <missing>")
        return
    if not isinstance(value, (list, tuple)):
        print(f"{name}: {value}")
        return

    print(f"{name} ({len(value)}):")
    seq = value if limit is None else value[:limit]
    for i, item in enumerate(seq):
        print(f"  [{i}] {item}")
    if limit is not None and len(value) > limit:
        print(f"  ... ({len(value) - limit} more)")


def main():
    parser = argparse.ArgumentParser("Inspect prototype .pth metadata")
    parser.add_argument("pth", type=str, help="Path to prototype .pth file")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of channels/items printed for long lists. Default prints all.",
    )
    args = parser.parse_args()

    pth_path = Path(args.pth)
    ckpt = torch.load(pth_path, map_location="cpu")

    if not isinstance(ckpt, dict):
        print(f"file: {pth_path}")
        print(f"object type: {type(ckpt).__name__}")
        if torch.is_tensor(ckpt):
            print(f"tensor shape: {tuple(ckpt.shape)}")
        return

    print(f"file: {pth_path}")
    print("keys:")
    for key in sorted(ckpt.keys()):
        print(f"  {key}: {_shape(ckpt[key])}")

    channel_prototypes = ckpt.get("channel_prototypes")
    ch_names = ckpt.get("ch_names")
    input_chans_index = ckpt.get("input_chans_index", ckpt.get("input_chans"))
    channel_input_chans_index = ckpt.get(
        "channel_input_chans_index",
        ckpt.get("channel_input_chans"),
    )
    target_ch_names = ckpt.get("target_ch_names")
    target_input_chans_index = ckpt.get(
        "target_input_chans_index",
        ckpt.get("target_input_chans"),
    )

    print()
    if torch.is_tensor(channel_prototypes):
        print(f"channel_prototypes.shape: {tuple(channel_prototypes.shape)}")
    else:
        print("channel_prototypes: <missing or not tensor>")

    print(f"source_dataset: {ckpt.get('source_dataset', '<missing>')}")
    print(f"prototype_type: {ckpt.get('prototype_type', '<missing>')}")
    if "completion_scope" in ckpt:
        print(f"optional completion_scope: {ckpt['completion_scope']}")

    print()
    _print_sequence("ch_names", ch_names, args.limit)
    print()
    _print_sequence("input_chans_index", input_chans_index, args.limit)
    print()
    _print_sequence("channel_input_chans_index", channel_input_chans_index, args.limit)

    if target_ch_names is not None or target_input_chans_index is not None:
        print()
        _print_sequence("target_ch_names", target_ch_names, args.limit)
        print()
        _print_sequence("target_input_chans_index", target_input_chans_index, args.limit)

    print()
    if torch.is_tensor(channel_prototypes) and isinstance(ch_names, (list, tuple)):
        ok = channel_prototypes.shape[0] == len(ch_names)
        print(f"check rows == len(ch_names): {channel_prototypes.shape[0]} == {len(ch_names)} -> {ok}")
    if torch.is_tensor(channel_prototypes) and isinstance(channel_input_chans_index, (list, tuple)):
        ok = channel_prototypes.shape[0] == len(channel_input_chans_index)
        print(
            "check rows == len(channel_input_chans_index): "
            f"{channel_prototypes.shape[0]} == {len(channel_input_chans_index)} -> {ok}"
        )
    if isinstance(ch_names, (list, tuple)) and isinstance(channel_input_chans_index, (list, tuple)):
        ok = len(ch_names) == len(channel_input_chans_index)
        print(
            "check len(ch_names) == len(channel_input_chans_index): "
            f"{len(ch_names)} == {len(channel_input_chans_index)} -> {ok}"
        )

    if isinstance(ch_names, (list, tuple)) and isinstance(channel_input_chans_index, (list, tuple)):
        print()
        print("prototype row mapping:")
        n = len(ch_names) if args.limit is None else min(len(ch_names), args.limit)
        for i in range(n):
            print(
                f"  channel_prototypes[{i}] <-> {ch_names[i]} "
                f"<-> channel_input_chans_index[{i}]={channel_input_chans_index[i]}"
            )
        if args.limit is not None and len(ch_names) > args.limit:
            print(f"  ... ({len(ch_names) - args.limit} more)")


if __name__ == "__main__":
    main()
