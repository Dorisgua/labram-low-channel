"""Temporary one-batch audit for the unified Stage2 input modes."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from clean_disentangle.stage2.train_stage2 import (
    MODES,
    load_json,
    make_model,
    patch_tokens,
)


def max_diff(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left - right).abs().max().item())


def args_for_mode(base: argparse.Namespace, mode: str) -> argparse.Namespace:
    values = vars(base).copy()
    values["input_mode"] = mode
    return argparse.Namespace(**values)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[2]
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--labram-checkpoint", type=Path, default=root / "checkpoints/labram-base.pth")
    parser.add_argument("--prototype-checkpoint", type=Path, default=root / "docs/prototypes/01_erpcore28_cnn_patch_embed_mean.pth")
    stage1_dir = root / "outputs/missing_prototype_d/missing_prototype_d_seed0_20260818_143337"
    parser.add_argument("--stage1-checkpoint", type=Path, default=stage1_dir / "checkpoints/checkpoint-last.pth")
    parser.add_argument("--stage1-config", type=Path, default=stage1_dir / "config.json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--last-n-blocks", type=int, default=12)
    ns = parser.parse_args()

    from data_processor.erpcore_cslp import prepare_ERPCORE_cslp_dataset

    _, test_dataset, _ = prepare_ERPCORE_cslp_dataset(ns.data_path, sampling_rate=200, normalize_method="z_score")
    batch = next(iter(DataLoader(test_dataset, batch_size=ns.batch_size, shuffle=False, num_workers=0)))
    print(f"TEST_BATCH x_full={tuple(batch['x_full'].shape)} x_obs={tuple(batch['x_obs'].shape)}")
    print(f"TEST_DATASET n={len(test_dataset)}")

    for mode in MODES:
        model_args = args_for_mode(ns, mode)
        stage1_config = load_json(ns.stage1_config) if mode == "dynamic" else None
        model, full_names, observed_names = make_model(model_args, stage1_config)
        model.to(ns.device).eval()
        source = batch["x_full"] if mode == "full" else batch["x_obs"]
        source = source.to(ns.device).float()
        transformer_input = model.build_transformer_input(source)
        logits = model(source)
        print(f"MODE={mode}")
        print(f"  raw_input={tuple(source.shape)}")
        print(f"  transformer_input={tuple(transformer_input.shape)}")
        print(f"  num_input_tokens={model.num_input_tokens} classifier_input_dim={model.classifier_input_dim}")
        print(f"  logits={tuple(logits.shape)}")
        print(f"  channel_positions={model.channel_positions}")

        if mode in ("prototype", "dynamic"):
            h_obs = patch_tokens(model.backbone, source, 12)
            observed = torch.as_tensor(model._observed_positions, device=source.device)
            observed_diff = max_diff(transformer_input.index_select(1, observed), h_obs)
            print(f"  observed_max_abs_diff={observed_diff:.9g}")
            if mode == "prototype":
                p_miss = model.prototype_provider.get_missing(
                    source.shape[0], missing_channel_positions=model._missing_positions,
                    num_t=1, device=source.device, dtype=h_obs.dtype,
                )
                missing = torch.as_tensor(model._missing_positions, device=source.device)
                print(f"  prototype_missing_max_abs_diff={max_diff(transformer_input.index_select(1, missing), p_miss):.9g}")
            else:
                corrector = model.dynamic_model
                missing = corrector.missing_token_positions.to(source.device)
                p_miss = corrector.prototype_provider.get_missing(
                    source.shape[0], missing_channel_positions=corrector.missing_channel_positions,
                    num_t=1, device=source.device, dtype=h_obs.dtype,
                )
                context = h_obs.new_zeros(source.shape[0], 28, corrector.stable_core.embed_dim)
                context = context.index_copy(1, observed, h_obs).index_copy(1, missing, p_miss)
                rep = corrector.stable_core.encode_tokens(context)
                components = corrector.build_components(rep, missing, __import__("clean_disentangle.modeling", fromlist=["ComponentMode"]).ComponentMode.IDENTITY)
                expected_missing = p_miss + components["d_sub"] + components["d_task"]
                print(f"  dynamic_missing_max_abs_diff={max_diff(transformer_input.index_select(1, missing), expected_missing):.9g}")
        if mode == "observed_only":
            print(f"  observed_channel_names={list(observed_names)}")
            print(f"  observed_original_channel_positions={model.channel_positions[1:]}")

    print("AUDIT_PASS")


if __name__ == "__main__":
    main()
