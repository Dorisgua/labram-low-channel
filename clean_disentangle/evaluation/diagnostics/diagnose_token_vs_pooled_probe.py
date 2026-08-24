#!/usr/bin/env python3
"""Generic Stage1 diagnostic: token-level versus mean-pooled probes."""

from __future__ import annotations

import csv
import gc
import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from clean_disentangle.engine import move_batch
from clean_disentangle.evaluation.probe_latents import (
    _fit_and_score,
    _load_checkpoint_config,
    _required,
    _resolve_device,
    _spec_from_config,
    _undersample_task_training_fold,
    _xgb_parameters,
)
from clean_disentangle.modeling import ComponentMode
# from clean_disentangle.modeling import MissingFillMode
# from clean_disentangle.modeling import OutputBaseMode
# from clean_disentangle.modeling import ReconstructionScope
from clean_disentangle.stage1.train_stage1 import build_erpcore_reconstruction_model


CV_SEED = 42
XGB_SEED = 42
UNDERSAMPLE_SEED = 42
BATCH_SIZE = 256
NUM_WORKERS = 4


PROBES = (
    {
        "representation": "D_sub(flat)",
        "feature": "d_sub_flat",
        "target": "Subject",
        "cv": "subject_cv",
    },
    {
        "representation": "z_sub",
        "feature": "z_sub",
        "target": "Subject",
        "cv": "subject_cv",
    },
    {
        "representation": "D_sub(flat)",
        "feature": "d_sub_flat",
        "target": "Task",
        "cv": "subject_cv",
    },
    {
        "representation": "z_sub",
        "feature": "z_sub",
        "target": "Task",
        "cv": "subject_cv",
    },
    {
        "representation": "D_task(flat)",
        "feature": "d_task_flat",
        "target": "Task",
        "cv": "task_cv",
    },
    {
        "representation": "z_task",
        "feature": "z_task",
        "target": "Task",
        "cv": "task_cv",
    },
    {
        "representation": "D_task(flat)",
        "feature": "d_task_flat",
        "target": "Subject",
        "cv": "task_cv",
    },
    {
        "representation": "z_task",
        "feature": "z_task",
        "target": "Subject",
        "cv": "task_cv",
    },
)


class DifferenceAccumulator:
    def __init__(self) -> None:
        self.maximum = 0.0
        self.absolute_sum = 0.0
        self.count = 0

    def update(self, left: torch.Tensor, right: torch.Tensor) -> None:
        difference = (left - right).detach().abs().float()
        self.maximum = max(self.maximum, float(difference.max().item()))
        self.absolute_sum += float(difference.double().sum().item())
        self.count += difference.numel()

    def result(self) -> dict[str, float]:
        return {
            "max_abs_diff": self.maximum,
            "mean_abs_diff": self.absolute_sum / self.count,
        }


def extract_representations(checkpoint_path: Path, *, device_name: str, batch_size: int, num_workers: int) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise TypeError("checkpoint must be a mapping")
    config = _load_checkpoint_config(checkpoint, checkpoint_path)
    spec = _spec_from_config(config)
    if spec.component_mode is not ComponentMode.IDENTITY:
        raise RuntimeError(f"token-vs-pooled diagnostic requires IDENTITY components, got {spec.component_mode}")

    model = build_erpcore_reconstruction_model(
        legacy_root=Path(_required(config, "legacy_root")),
        cnn_checkpoint=Path(_required(config, "cnn_checkpoint")),
        prototype_checkpoint=Path(config["prototype_checkpoint"]) if config.get("prototype_checkpoint") else None,
        spec=spec,
        seed=int(_required(config, "seed")),
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    device = _resolve_device(device_name)
    model.to(device)
    model.eval()

    from data_processor.erpcore_cslp import prepare_ERPCORE_cslp_dataset

    _, test_dataset, _ = prepare_ERPCORE_cslp_dataset(
        _required(config, "data_path"),
        sampling_rate=int(_required(config, "sampling_rate")),
        normalize_method=str(_required(config, "norm_method")),
    )
    loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    arrays: dict[str, list[np.ndarray]] = {
        "d_sub": [],
        "z_sub": [],
        "d_task": [],
        "z_task": [],
        "subjects": [],
        "tasks": [],
    }
    differences = {
        "d_sub_vs_sub_tokens": DifferenceAccumulator(),
        "d_task_vs_task_tokens": DifferenceAccumulator(),
        "z_sub_vs_mean_sub_tokens": DifferenceAccumulator(),
        "z_task_vs_mean_task_tokens": DifferenceAccumulator(),
        "z_sub_vs_mean_d_sub": DifferenceAccumulator(),
        "z_task_vs_mean_d_task": DifferenceAccumulator(),
    }
    with torch.no_grad():
        for batch_number, batch in enumerate(loader, start=1):
            moved = move_batch(
                batch,
                device,
                input_scale=float(_required(config, "input_scale")),
            )
            output = model.forward_reconstruction(moved, spec)
            # In FULL mode d_* and branch tokens both contain all positions.
            # In MISSING mode d_* is already selected at target_positions
            # (16 missing tokens), while sub_tokens/task_tokens retain the
            # complete 28-position context. Compare like-for-like here.
            target_positions = output["target_positions"]
            selected_sub_tokens = output["sub_tokens"].index_select(1, target_positions)
            selected_task_tokens = output["task_tokens"].index_select(1, target_positions)
            differences["d_sub_vs_sub_tokens"].update(
                output["d_sub"], selected_sub_tokens
            )
            differences["d_task_vs_task_tokens"].update(
                output["d_task"], selected_task_tokens
            )
            differences["z_sub_vs_mean_sub_tokens"].update(
                output["z_sub"], output["sub_tokens"].mean(dim=1)
            )
            differences["z_task_vs_mean_task_tokens"].update(
                output["z_task"], output["task_tokens"].mean(dim=1)
            )
            differences["z_sub_vs_mean_d_sub"].update(
                output["z_sub"], output["d_sub"].mean(dim=1)
            )
            differences["z_task_vs_mean_d_task"].update(
                output["z_task"], output["d_task"].mean(dim=1)
            )
            for name in ("d_sub", "z_sub", "d_task", "z_task"):
                arrays[name].append(output[name].detach().float().cpu().numpy())
            arrays["subjects"].append(batch["subject"].numpy())
            arrays["tasks"].append(batch["task"].numpy())
            if batch_number == 1 or batch_number % 20 == 0:
                print(f"extraction batch {batch_number}/{len(loader)}", flush=True)

    result = {name: np.concatenate(parts, axis=0) for name, parts in arrays.items()}
    result["subjects"] = result["subjects"].astype(np.int64, copy=False)
    result["tasks"] = result["tasks"].astype(np.int64, copy=False)
    if not np.array_equal(result["subjects"], np.asarray(test_dataset.subjects)):
        raise RuntimeError("subject order mismatch")
    if not np.array_equal(result["tasks"], np.asarray(test_dataset.labels)):
        raise RuntimeError("task order mismatch")
    if not all(np.isfinite(result[name]).all() for name in ("d_sub", "z_sub", "d_task", "z_task")):
        raise RuntimeError("non-finite extracted representation")

    metadata = {
        "checkpoint": str(checkpoint_path.resolve()),
        "spec": {
            "scope": spec.scope.value,
            "missing_fill": spec.missing_fill.value,
            "output_base": spec.output_base.value,
            "component_mode": spec.component_mode.value,
            "composition_mode": spec.composition_mode.value,
        },
        "shapes": {name: list(result[name].shape) for name in ("d_sub", "z_sub", "d_task", "z_task")},
        "differences": {name: accumulator.result() for name, accumulator in differences.items()},
        "n_samples": len(test_dataset),
        "n_subjects": len(np.unique(result["subjects"])),
        "n_tasks": len(np.unique(result["tasks"])),
    }
    del model, checkpoint
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result, metadata


def evaluate(
    definition: Mapping[str, str],
    *,
    features: np.ndarray,
    targets: np.ndarray,
    tasks: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    xgb_params: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    fold_rows = []
    for fold, (train_indices, eval_indices) in enumerate(splits, start=1):
        fit_indices = np.asarray(train_indices, dtype=np.int64)
        train_before = len(fit_indices)
        if definition["target"] == "Task":
            fit_indices = _undersample_task_training_fold(
                fit_indices,
                tasks,
                seed=UNDERSAMPLE_SEED,
                fold=fold,
            )
        accuracy, balanced_accuracy = _fit_and_score(
            features,
            targets,
            fit_indices,
            np.asarray(eval_indices, dtype=np.int64),
            xgb_params=xgb_params,
        )
        row = {
            "fold": fold,
            "train_size_before_balance": train_before,
            "train_size_after_balance": len(fit_indices),
            "eval_size": len(eval_indices),
            "accuracy": accuracy,
            "balanced_accuracy": balanced_accuracy,
        }
        fold_rows.append(row)
        print(
            f"{definition['representation']:<13} -> {definition['target']:<7} "
            f"fold={fold} train={train_before}->{len(fit_indices)} "
            f"eval={len(eval_indices)} acc={accuracy:.9f} "
            f"bacc={balanced_accuracy:.9f}",
            flush=True,
        )
        gc.collect()

    accuracies = np.asarray([row["accuracy"] for row in fold_rows])
    balanced = np.asarray([row["balanced_accuracy"] for row in fold_rows])
    summary = {
        "representation": definition["representation"],
        "target": definition["target"],
        "cv_type": definition["cv"],
        "dim": features.shape[1],
        "accuracy_mean": float(accuracies.mean()),
        "accuracy_std": float(accuracies.std(ddof=0)),
        "balanced_accuracy_mean": float(balanced.mean()),
        "balanced_accuracy_std": float(balanced.std(ddof=0)),
    }
    return summary, fold_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[3]
    default_run = root / "outputs/full_d_only/full_d_only_seed0_20260818_131233"
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--cv-seed", type=int, default=42)
    parser.add_argument("--xgb-seed", type=int, default=42)
    parser.add_argument("--undersample-seed", type=int, default=42)
    args = parser.parse_args()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    global CV_SEED, XGB_SEED, UNDERSAMPLE_SEED
    CV_SEED, XGB_SEED, UNDERSAMPLE_SEED = args.cv_seed, args.xgb_seed, args.undersample_seed
    arrays, metadata = extract_representations(
        checkpoint_path,
        device_name=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    d_sub_flat = arrays["d_sub"].reshape(len(arrays["d_sub"]), -1)
    d_task_flat = arrays["d_task"].reshape(len(arrays["d_task"]), -1)
    features = {
        "d_sub_flat": d_sub_flat,
        "z_sub": arrays["z_sub"],
        "d_task_flat": d_task_flat,
        "z_task": arrays["z_task"],
    }
    targets = {"Subject": arrays["subjects"], "Task": arrays["tasks"]}
    n_samples = len(arrays["subjects"])
    subject_splits = list(
        StratifiedKFold(n_splits=5, shuffle=True, random_state=CV_SEED).split(
            np.zeros(n_samples), arrays["subjects"]
        )
    )
    task_splits = list(
        StratifiedKFold(n_splits=5, shuffle=True, random_state=CV_SEED).split(
            np.zeros(n_samples), arrays["tasks"]
        )
    )
    splits = {"subject_cv": subject_splits, "task_cv": task_splits}
    xgb_params = _xgb_parameters(xgb_seed=XGB_SEED, n_jobs=-1)

    summaries = []
    fold_results: dict[str, list[dict[str, Any]]] = {}
    for definition in PROBES:
        print(
            f"\n=== {definition['representation']} -> {definition['target']} "
            f"({definition['cv']}) ===",
            flush=True,
        )
        summary, folds = evaluate(
            definition,
            features=features[definition["feature"]],
            targets=targets[definition["target"]],
            tasks=arrays["tasks"],
            splits=splits[definition["cv"]],
            xgb_params=xgb_params,
        )
        summaries.append(summary)
        fold_results[f"{definition['representation']} -> {definition['target']}"] = folds

    csv_path = output_dir / "token_vs_pooled_probe.csv"
    fieldnames = (
        "representation", "target", "cv_type", "dim", "accuracy_mean",
        "accuracy_std", "balanced_accuracy_mean", "balanced_accuracy_std",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)

    by_key = {(row["representation"], row["target"]): row for row in summaries}
    metadata.update(
        {
            "protocol": {
                "cv_seed": CV_SEED,
                "xgb_seed": XGB_SEED,
                "undersample_seed": UNDERSAMPLE_SEED,
                "xgb_params": xgb_params,
                "feature_preprocessing": "none",
                "task_undersampling": "training fold only",
                "evaluation_undersampling": "none",
                "std_ddof": 0,
            },
            "fold_results": fold_results,
            "task_pooling_bacc_drop_pp": 100.0 * (
                by_key[("D_task(flat)", "Task")]["balanced_accuracy_mean"]
                - by_key[("z_task", "Task")]["balanced_accuracy_mean"]
            ),
            "subject_pooling_bacc_drop_pp": 100.0 * (
                by_key[("D_sub(flat)", "Subject")]["balanced_accuracy_mean"]
                - by_key[("z_sub", "Subject")]["balanced_accuracy_mean"]
            ),
        }
    )
    json_path = output_dir / "token_vs_pooled_probe.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print("\nRepresentation        Target      Dim       Acc mean±std       BAcc mean±std")
    for row in summaries:
        print(
            f"{row['representation']:<21} {row['target']:<10} {row['dim']:<9} "
            f"{row['accuracy_mean']:.6f}±{row['accuracy_std']:.6f}   "
            f"{row['balanced_accuracy_mean']:.6f}±{row['balanced_accuracy_std']:.6f}"
        )
    print(f"\nTask pooling BAcc drop: {metadata['task_pooling_bacc_drop_pp']:.6f} pp")
    print(f"Subject pooling BAcc drop: {metadata['subject_pooling_bacc_drop_pp']:.6f} pp")
    print(f"Saved: {csv_path}")
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
