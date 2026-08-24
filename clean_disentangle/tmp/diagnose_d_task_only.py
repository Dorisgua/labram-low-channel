#!/usr/bin/env python3
"""One-off diagnostic: only retain and probe D_task(flat) -> Task."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from pathlib import Path

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
    _undersample_task_training_fold,
    _xgb_parameters,
)
from clean_disentangle.modeling import (
    ComponentMode,
    CompositionMode,
    MissingFillMode,
    OutputBaseMode,
    ReconstructionScope,
    ReconstructionSpec,
)
from clean_disentangle.run import build_erpcore_reconstruction_model


REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = REPO_ROOT / "outputs/full_d_only/full_d_only_seed0_20260818_131233"
CHECKPOINT = RUN_DIR / "checkpoints/checkpoint-last.pth"
OUTPUT_DIR = RUN_DIR / "evaluation/diagnostics"


def main() -> None:
    checkpoint = torch.load(CHECKPOINT, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise TypeError("checkpoint must be a mapping")
    config = _load_checkpoint_config(checkpoint, CHECKPOINT)
    spec = ReconstructionSpec(
        scope=ReconstructionScope(_required(config, "scope")),
        missing_fill=MissingFillMode(_required(config, "missing_fill")),
        output_base=OutputBaseMode(_required(config, "output_base")),
        component_mode=ComponentMode(_required(config, "component_mode")),
        composition_mode=CompositionMode(_required(config, "composition_mode")),
    )
    if spec.scope is not ReconstructionScope.FULL or spec.output_base is not OutputBaseMode.NONE:
        raise RuntimeError(f"expected Full + D-only config, got {spec}")

    model = build_erpcore_reconstruction_model(
        legacy_root=Path(_required(config, "legacy_root")),
        cnn_checkpoint=Path(_required(config, "cnn_checkpoint")),
        prototype_checkpoint=None,
        spec=spec,
        seed=int(_required(config, "seed")),
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    device = _resolve_device("cuda")
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
        batch_size=256,
        shuffle=False,
        num_workers=4,
        pin_memory=device.type == "cuda",
    )

    d_task_parts: list[np.ndarray] = []
    task_parts: list[np.ndarray] = []
    with torch.no_grad():
        for batch_number, batch in enumerate(loader, start=1):
            moved = move_batch(
                batch,
                device,
                input_scale=float(_required(config, "input_scale")),
            )
            output = model.forward_reconstruction(moved, spec)
            d_task_parts.append(output["d_task"].detach().float().cpu().numpy())
            task_parts.append(batch["task"].numpy())
            if batch_number == 1 or batch_number % 20 == 0:
                print(f"extraction batch {batch_number}/{len(loader)}", flush=True)

    d_task = np.concatenate(d_task_parts, axis=0)
    tasks = np.concatenate(task_parts).astype(np.int64, copy=False)
    if not np.array_equal(tasks, np.asarray(test_dataset.labels)):
        raise RuntimeError("task order mismatch")
    if not np.isfinite(d_task).all():
        raise RuntimeError("D_task contains NaN or Inf")

    features = d_task.reshape(len(d_task), -1)
    splits = list(
        StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(
            np.zeros(len(tasks)), tasks
        )
    )
    rows = []
    params = _xgb_parameters(xgb_seed=42, n_jobs=-1)
    for fold, (train_indices, eval_indices) in enumerate(splits, start=1):
        fit_indices = _undersample_task_training_fold(
            train_indices, tasks, seed=42, fold=fold
        )
        accuracy, balanced_accuracy = _fit_and_score(
            features,
            tasks,
            fit_indices,
            eval_indices,
            xgb_params=params,
        )
        row = {
            "fold": fold,
            "train_size_before_balance": len(train_indices),
            "train_size_after_balance": len(fit_indices),
            "eval_size": len(eval_indices),
            "accuracy": accuracy,
            "balanced_accuracy": balanced_accuracy,
        }
        rows.append(row)
        print(
            f"D_task(flat) -> Task fold={fold} "
            f"acc={accuracy:.9f} bacc={balanced_accuracy:.9f}",
            flush=True,
        )

    accuracies = np.asarray([row["accuracy"] for row in rows])
    balanced = np.asarray([row["balanced_accuracy"] for row in rows])
    summary = {
        "representation": "D_task(flat)",
        "target": "Task",
        "dim": int(features.shape[1]),
        "accuracy_mean": float(accuracies.mean()),
        "accuracy_std": float(accuracies.std(ddof=0)),
        "balanced_accuracy_mean": float(balanced.mean()),
        "balanced_accuracy_std": float(balanced.std(ddof=0)),
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / "d_task_flat_to_task_only.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "summary": summary,
                "folds": rows,
                "n_samples": len(tasks),
                "n_tasks": len(np.unique(tasks)),
                "feature_shape": list(features.shape),
                "protocol": {
                    "cv_seed": 42,
                    "xgb_seed": 42,
                    "undersample_seed": 42,
                    "task_undersampling": "training fold only",
                    "feature_preprocessing": "none",
                },
            },
            handle,
            indent=2,
        )
    with (OUTPUT_DIR / "d_task_flat_to_task_only.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fields = tuple(rows[0])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
