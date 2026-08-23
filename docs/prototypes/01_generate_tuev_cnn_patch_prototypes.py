import argparse
import runpy
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


CHANNELS = runpy.run_path(
    str(REPO_ROOT / "Channels_definition.py"),
    init_globals={"standard_1020": utils.standard_1020},
)
TUEV_23_CHANNELS = list(CHANNELS["TUEV_23_CHANNELS"])

# TUEV_ROOT = "/inspire/hdd/project/sais-medical/public/share_medical/EEG/TUEZ/v2.0.1/processed_labram/processed"
TUEV_ROOT = "/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/TUEZ/v2.0.1/processed_labram/processed"

def get_args():
    parser = argparse.ArgumentParser(
        "Generate TUEV 23-channel CNN patch_embed prototypes"
    )
    parser.add_argument("--tuev_root", default=TUEV_ROOT, type=str)
    parser.add_argument(
        "--finetune",
        default="./checkpoints/labram-base.pth",
        type=str,
        help="LaBraM checkpoint used to initialize patch_embed",
    )
    parser.add_argument(
        "--output",
        default="docs/prototypes/01_tuev23_cnn_patch_embed_mean.pth",
        type=str,
    )
    parser.add_argument("--model", default="labram_base_patch200_200", type=str)
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

    return {
        k: v for k, v in state.items()
        if not (k.startswith("loss") or k.startswith("teacher") or k.startswith("scaling"))
    }


def build_labram(args):
    state = _load_checkpoint_state(args.finetune)
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=6,
        drop_path_rate=0.1,
        use_mean_pooling=True,
        init_scale=0.001,
        use_abs_pos_emb=True,
        use_rel_pos_bias=False,
        init_values=0.1,
        qkv_bias=False,
    )

    if not any(k.startswith("patch_embed.") for k in state):
        state = {
            k[8:]: v for k, v in state.items()
            if k.startswith("student.")
        }
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

    train_dataset, _, _ = utils.prepare_TUEV_dataset(args.tuev_root)
    data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    model = build_labram(args).to(device)
    ch_names = TUEV_23_CHANNELS
    input_chans_index = utils.get_input_chans(ch_names)
    channel_input_chans_index = input_chans_index[1:]

    channel_sum = torch.zeros(
        len(ch_names), model.embed_dim,
        device=device, dtype=torch.float64,
    )
    channel_count = torch.zeros(len(ch_names), device=device, dtype=torch.float64)

    num_patches = None
    for step, (samples, _) in enumerate(data_loader):
        if samples.shape[1] != len(ch_names):
            raise ValueError(
                "Expected TUEV samples to have channels matching TUEV_23_CHANNELS; "
                f"got samples.shape[1]={samples.shape[1]} and len(ch_names)={len(ch_names)}. "
                "If this dataset is not already in TUEV-23 order, add explicit channel selection/mapping before prototype generation."
            )

        # TUEV finetuning scales raw input by /100 before tokenization.
        samples = samples.float().to(device, non_blocking=True) / 100
        if samples.shape[-1] % 200 != 0:
            raise ValueError(f"Expected sample length divisible by 200, got {samples.shape[-1]}")

        samples = rearrange(samples, "B N (A T) -> B N A T", T=200)
        if num_patches is None:
            num_patches = samples.shape[2]

        # Only use the CNN/TemporalConv patch embedding. Do not add position/time
        # embeddings and do not run Transformer blocks.
        features = model.patch_embed(samples)
        features = features.reshape(samples.shape[0], samples.shape[1], samples.shape[2], -1)

        channel_sum += features.double().sum(dim=(0, 2))
        channel_count += features.shape[0] * features.shape[2]

        if step % 50 == 0:
            print(f"Processed {step + 1}/{len(data_loader)} batches", flush=True)

    channel_prototypes = (channel_sum / channel_count[:, None]).float().cpu()
    num_tokens = len(train_dataset) * len(ch_names) * num_patches

    if channel_prototypes.shape[0] != len(ch_names):
        raise ValueError(
            f"prototype rows ({channel_prototypes.shape[0]}) must match ch_names ({len(ch_names)})"
        )
    if channel_prototypes.shape[0] != len(channel_input_chans_index):
        raise ValueError(
            "prototype rows must match channel_input_chans_index; "
            f"got {channel_prototypes.shape[0]} and {len(channel_input_chans_index)}"
        )

    payload = {
        "channel_prototypes": channel_prototypes,
        "ch_names": ch_names,
        # Full LaBraM input_chans_index includes the leading cls-token index 0.
        "input_chans_index": input_chans_index,
        # Per-channel position indices align one-to-one with ch_names/prototype rows.
        "channel_input_chans_index": channel_input_chans_index,
        "source_dataset": "TUEV",
        "prototype_type": "cnn_patch_embed_mean",
        "num_samples": len(train_dataset),
        "num_channels": len(ch_names),
        "num_patches": num_patches,
        "num_tokens": num_tokens,
        "embed_dim": channel_prototypes.shape[-1],
        "feature_source": "labram.patch_embed TemporalConv output",
        "source_checkpoint": args.finetune,
        "sample_scale": "/100",
        "patch_size": 200,
        "note": "TUEV 23-channel train-set prototypes from CNN patch embedding only; no Transformer blocks.",
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)

    print(f"Saved TUEV 23-channel CNN patch_embed prototypes to {output_path}", flush=True)
    print(f"prototype shape: {tuple(channel_prototypes.shape)}", flush=True)
    print(f"num_patches: {num_patches}", flush=True)
    print(f"channels: {ch_names}", flush=True)
    print(f"input_chans_index: {input_chans_index}", flush=True)
    print(f"channel_input_chans_index: {channel_input_chans_index}", flush=True)
    print("source_dataset: TUEV", flush=True)
    print("prototype_type: cnn_patch_embed_mean", flush=True)


if __name__ == "__main__":
    main()
