"""Generate BCI-IV-2a 22-channel prototypes from training-set CNN features."""

import argparse
import sys
from pathlib import Path

import torch
from einops import rearrange
from timm.models import create_model


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import modeling_finetune  # noqa: F401,E402
import utils  # noqa: E402
from Channels_definition import BCIIV2A_22_CHANNELS  # noqa: E402
from data_processor.bciiv2a import (  # noqa: E402
    prepare_BCIIV2A_multisession_dataset,
)


BCIIV2A_ROOT = (
    "/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/"
    "global_user/7461-chenxinhe/eeg-test/AdaBrain-PreExp34-35-repro/"
    "AdaBrain-Bench-main_film/preprocessing/BCI-IV-2A/multi_subject_json"
)


def get_args():
    parser = argparse.ArgumentParser(
        "Generate BCI-IV-2a 22-channel CNN patch_embed prototypes"
    )
    parser.add_argument("--data_root", default=BCIIV2A_ROOT, type=str)
    parser.add_argument(
        "--finetune",
        default="./checkpoints/labram-base.pth",
        type=str,
        help="LaBraM checkpoint used to initialize patch_embed",
    )
    parser.add_argument(
        "--output",
        default="docs/prototypes/01_bciiv2a22_cnn_patch_embed_mean.pth",
        type=str,
    )
    parser.add_argument("--model", default="labram_base_patch200_200", type=str)
    parser.add_argument("--sampling_rate", default=200, type=int)
    parser.add_argument("--norm_method", default="z_score", type=str)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    return parser.parse_args()


def _load_checkpoint_state(path):
    checkpoint = torch.load(path, map_location="cpu")
    if "model" in checkpoint:
        state = checkpoint["model"]
    elif "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
    else:
        state = checkpoint

    state = {
        key: value
        for key, value in state.items()
        if not (
            key.startswith("loss")
            or key.startswith("teacher")
            or key.startswith("scaling")
        )
    }
    if not any(key.startswith("patch_embed.") for key in state):
        state = {
            key[8:]: value
            for key, value in state.items()
            if key.startswith("student.")
        }
    return state


def build_labram(args):
    state = _load_checkpoint_state(args.finetune)
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=4,
        drop_path_rate=0.1,
        use_mean_pooling=True,
        init_scale=0.001,
        use_abs_pos_emb=True,
        use_rel_pos_bias=True,
        init_values=0.1,
        qkv_bias=True,
    )
    state.pop("head.weight", None)
    state.pop("head.bias", None)

    missing_keys, unexpected_keys = model.load_state_dict(state, strict=False)
    print(f"Loaded LaBraM checkpoint from {args.finetune}")
    print(f"Missing keys: {missing_keys}")
    print(f"Unexpected keys: {unexpected_keys}")
    return model.eval()


@torch.no_grad()
def main():
    args = get_args()
    device = torch.device(args.device)

    train_dataset, _, _ = prepare_BCIIV2A_multisession_dataset(
        args.data_root,
        sampling_rate=args.sampling_rate,
        normalize_method=args.norm_method,
        channel_names=BCIIV2A_22_CHANNELS,
    )
    data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    model = build_labram(args).to(device)
    ch_names = list(BCIIV2A_22_CHANNELS)
    input_chans_index = utils.get_input_chans(ch_names)
    channel_input_chans_index = input_chans_index[1:]

    channel_sum = torch.zeros(
        len(ch_names), model.embed_dim, device=device, dtype=torch.float64
    )
    channel_count = torch.zeros(
        len(ch_names), device=device, dtype=torch.float64
    )

    num_patches = None
    for step, (samples, _) in enumerate(data_loader):
        if samples.shape[1] != len(ch_names):
            raise ValueError(
                f"Expected {len(ch_names)} channels, got {samples.shape[1]}"
            )
        samples = samples.float().to(device, non_blocking=True)
        if samples.shape[-1] % 200 != 0:
            raise ValueError(
                f"Expected sample length divisible by 200, got {samples.shape[-1]}"
            )

        samples = rearrange(samples, "B N (A T) -> B N A T", T=200)
        if num_patches is None:
            num_patches = samples.shape[2]

        features = model.patch_embed(samples)
        features = features.reshape(
            samples.shape[0], samples.shape[1], samples.shape[2], -1
        )
        channel_sum += features.double().sum(dim=(0, 2))
        channel_count += features.shape[0] * features.shape[2]

        if step % 10 == 0:
            print(f"Processed {step + 1}/{len(data_loader)} batches", flush=True)

    channel_prototypes = (channel_sum / channel_count[:, None]).float().cpu()
    num_tokens = len(train_dataset) * len(ch_names) * num_patches

    if channel_prototypes.shape != (len(ch_names), model.embed_dim):
        raise ValueError(
            f"Unexpected prototype shape: {tuple(channel_prototypes.shape)}"
        )

    payload = {
        "channel_prototypes": channel_prototypes,
        "ch_names": ch_names,
        "input_chans_index": input_chans_index,
        "channel_input_chans_index": channel_input_chans_index,
        "source_dataset": "BCI-IV-2a train split",
        "prototype_type": "cnn_patch_embed_mean",
        "num_samples": len(train_dataset),
        "num_channels": len(ch_names),
        "num_patches": num_patches,
        "num_tokens": num_tokens,
        "embed_dim": channel_prototypes.shape[-1],
        "feature_source": "labram.patch_embed TemporalConv output",
        "source_checkpoint": args.finetune,
        "sample_scale": "1.0 after loader z_score",
        "sampling_rate": args.sampling_rate,
        "normalize_method": args.norm_method,
        "patch_size": 200,
        "note": (
            "BCI-IV-2a 22-channel train-set prototypes from CNN patch "
            "embedding only; no validation/test samples and no Transformer blocks."
        ),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)

    print(f"Saved BCI-IV-2a prototypes to {output_path}", flush=True)
    print(f"prototype shape: {tuple(channel_prototypes.shape)}", flush=True)
    print(f"num_patches: {num_patches}", flush=True)
    print(f"channels: {ch_names}", flush=True)
    print(f"input_chans_index: {input_chans_index}", flush=True)


if __name__ == "__main__":
    main()
