#!/usr/bin/env python3
"""Quantify subject/task information in clean Stage1 z_sub and z_task latents."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
import xgboost as xgb
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from clean_disentangle.engine import move_batch
from clean_disentangle.modeling import (
    ComponentMode,
    CompositionMode,
    MissingFillMode,
    OutputBaseMode,
    ReconstructionScope,
    ReconstructionSpec,
)
from clean_disentangle.run import build_erpcore_reconstruction_model


PROBE_DEFINITIONS = (
    {
        "metric": "S.acc",
        "internal_name": "subject_from_zsub",
        "latent": "z_sub",
        "target": "subject",
        "cv_type": "subject_cv",
        "undersample_task": False,
        "display": "z_sub -> Subject",
    },
    {
        "metric": "T⊢S.acc",
        "internal_name": "task_from_zsub",
        "latent": "z_sub",
        "target": "task",
        "cv_type": "subject_cv",
        "undersample_task": True,
        "display": "z_sub -> Task",
    },
    {
        "metric": "T.acc",
        "internal_name": "task_from_ztask",
        "latent": "z_task",
        "target": "task",
        "cv_type": "task_cv",
        "undersample_task": True,
        "display": "z_task -> Task",
    },
    {
        "metric": "S⊢T.acc",
        "internal_name": "subject_from_ztask",
        "latent": "z_task",
        "target": "subject",
        "cv_type": "task_cv",
        "undersample_task": False,
        "display": "z_task -> Subject",
    },
)

RESULT_FIELDS = (
    "metric",
    "internal_name",
    "latent",
    "target",
    "cv_type",
    "fold",
    "train_size_before_balance",
    "train_size_after_balance",
    "eval_size",
    "accuracy",
    "balanced_accuracy",
)

SUMMARY_FIELDS = (
    "metric",
    "internal_name",
    "latent",
    "target",
    "cv_type",
    "accuracy_mean",
    "accuracy_std",
    "balanced_accuracy_mean",
    "balanced_accuracy_std",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--cv-seed", type=int, default=42)
    parser.add_argument("--xgb-seed", type=int, default=42)
    parser.add_argument("--undersample-seed", type=int, default=42)
    parser.add_argument("--xgb-n-jobs", type=int, default=-1)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def _load_checkpoint_config(
    checkpoint: Mapping[str, Any],
    checkpoint_path: Path,
) -> dict[str, Any]:
    embedded = checkpoint.get("config")
    if isinstance(embedded, Mapping):
        return dict(embedded)

    config_path = checkpoint_path.parent.parent / "config.json"
    if not config_path.is_file():
        raise KeyError(
            "checkpoint has no embedded config and adjacent config.json is missing: "
            f"{config_path}"
        )
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise TypeError(f"run config must be a JSON object: {config_path}")
    return config


def _required(config: Mapping[str, Any], key: str) -> Any:
    if key not in config or config[key] is None:
        raise KeyError(f"checkpoint config is missing required field {key!r}")
    return config[key]


def _spec_from_config(config: Mapping[str, Any]) -> ReconstructionSpec:
    return ReconstructionSpec(
        scope=ReconstructionScope(_required(config, "scope")),
        missing_fill=MissingFillMode(_required(config, "missing_fill")),
        output_base=OutputBaseMode(_required(config, "output_base")),
        component_mode=ComponentMode(_required(config, "component_mode")),
        composition_mode=CompositionMode(_required(config, "composition_mode")),
    )


def _resolve_device(requested: str) -> torch.device:
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA is unavailable; falling back to CPU", flush=True)
        return torch.device("cpu")
    return device


def _extract_test_latents(
    *,
    checkpoint: Mapping[str, Any],
    config: Mapping[str, Any],
    spec: ReconstructionSpec,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    prototype_value = config.get("prototype_checkpoint")
    prototype_checkpoint = (
        Path(prototype_value).expanduser().resolve()
        if prototype_value is not None
        else None
    )
    model = build_erpcore_reconstruction_model(
        legacy_root=Path(_required(config, "legacy_root")),
        cnn_checkpoint=Path(_required(config, "cnn_checkpoint")),
        prototype_checkpoint=prototype_checkpoint,
        spec=spec,
        seed=int(_required(config, "seed")),
    )
    state_dict = checkpoint.get("model")
    if not isinstance(state_dict, Mapping):
        raise KeyError("checkpoint has no model state_dict")
    model.load_state_dict(state_dict, strict=True)
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

    z_sub_parts: list[np.ndarray] = []
    z_task_parts: list[np.ndarray] = []
    subject_parts: list[np.ndarray] = []
    task_parts: list[np.ndarray] = []
    with torch.no_grad():
        for batch_index, batch in enumerate(loader, start=1):
            moved_batch = move_batch(
                batch,
                device,
                input_scale=float(_required(config, "input_scale")),
            )
            output = model.forward_reconstruction(moved_batch, spec)
            z_sub_parts.append(output["z_sub"].detach().float().cpu().numpy())
            z_task_parts.append(output["z_task"].detach().float().cpu().numpy())
            subject_parts.append(batch["subject"].numpy())
            task_parts.append(batch["task"].numpy())
            if batch_index == 1 or batch_index % 20 == 0:
                print(
                    f"latent extraction batch {batch_index}/{len(loader)}",
                    flush=True,
                )

    z_sub = np.concatenate(z_sub_parts, axis=0)
    z_task = np.concatenate(z_task_parts, axis=0)
    subjects = np.concatenate(subject_parts).astype(np.int64, copy=False)
    tasks = np.concatenate(task_parts).astype(np.int64, copy=False)
    sample_indices = np.arange(len(test_dataset), dtype=np.int64)
    sample_ids = np.asarray(test_dataset.indices, dtype=np.int64)

    expected_subjects = np.asarray(test_dataset.subjects, dtype=np.int64)
    expected_tasks = np.asarray(test_dataset.labels, dtype=np.int64)
    if not np.array_equal(subjects, expected_subjects):
        raise RuntimeError("extracted subject order differs from ERP-Core test order")
    if not np.array_equal(tasks, expected_tasks):
        raise RuntimeError("extracted task order differs from ERP-Core test order")
    expected_rows = len(test_dataset)
    if z_sub.shape[0] != expected_rows or z_task.shape[0] != expected_rows:
        raise RuntimeError(
            "latent/sample count mismatch: "
            f"z_sub={z_sub.shape}, z_task={z_task.shape}, expected={expected_rows}"
        )
    if z_sub.ndim != 2 or z_task.ndim != 2:
        raise RuntimeError(
            f"probe features must be raw [N,D] latents, got {z_sub.shape}/{z_task.shape}"
        )
    if not np.isfinite(z_sub).all() or not np.isfinite(z_task).all():
        raise RuntimeError("z_sub or z_task contains NaN or Inf")
    return z_sub, z_task, subjects, tasks, sample_indices, sample_ids


def _make_fold_assignments(
    splits: list[tuple[np.ndarray, np.ndarray]],
    n_samples: int,
) -> np.ndarray:
    assignments = np.zeros(n_samples, dtype=np.int64)
    for fold, (_, eval_indices) in enumerate(splits, start=1):
        if np.any(assignments[eval_indices] != 0):
            raise RuntimeError("CV evaluation folds overlap")
        assignments[eval_indices] = fold
    if np.any(assignments == 0):
        raise RuntimeError("CV folds do not cover every test sample")
    return assignments


def _undersample_task_training_fold(
    train_indices: np.ndarray,
    tasks: np.ndarray,
    *,
    seed: int,
    fold: int,
) -> np.ndarray:
    train_indices = np.asarray(train_indices, dtype=np.int64)
    train_tasks = tasks[train_indices]
    task_classes, counts = np.unique(train_tasks, return_counts=True)
    min_class_count = int(counts.min())
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(fold)]))
    selected = []
    for task_class in task_classes:
        class_indices = train_indices[train_tasks == task_class]
        selected.append(
            rng.choice(class_indices, size=min_class_count, replace=False)
        )
    balanced = np.concatenate(selected).astype(np.int64, copy=False)
    if not np.all(np.isin(balanced, train_indices)):
        raise RuntimeError("task undersampling selected an index outside training fold")
    return balanced


def _xgb_parameters(*, xgb_seed: int, n_jobs: int) -> dict[str, Any]:
    """Match the classifier settings in the local CSLP-AE author implementation."""

    return {
        "n_estimators": 300,
        "max_bin": 100,
        "learning_rate": 0.3,
        "grow_policy": "depthwise",
        "objective": "multi:softmax",
        "tree_method": "hist",
        "n_jobs": int(n_jobs),
        "random_state": int(xgb_seed),
    }


def _fit_and_score(
    features: np.ndarray,
    targets: np.ndarray,
    train_indices: np.ndarray,
    eval_indices: np.ndarray,
    *,
    xgb_params: Mapping[str, Any],
) -> tuple[float, float]:
    train_classes = np.unique(targets[train_indices])
    eval_classes = np.unique(targets[eval_indices])
    missing_train_classes = sorted(set(eval_classes) - set(train_classes))
    if missing_train_classes:
        raise RuntimeError(
            f"evaluation contains classes absent from training: {missing_train_classes}"
        )
    encoded_train = np.searchsorted(train_classes, targets[train_indices])
    classifier = xgb.XGBClassifier(**dict(xgb_params))
    classifier.fit(features[train_indices], encoded_train)
    encoded_prediction = np.asarray(
        classifier.predict(features[eval_indices]),
        dtype=np.int64,
    ).reshape(-1)
    if np.any(encoded_prediction < 0) or np.any(
        encoded_prediction >= len(train_classes)
    ):
        raise RuntimeError("XGBoost predicted an invalid encoded class")
    prediction = train_classes[encoded_prediction]
    truth = targets[eval_indices]
    return (
        float(accuracy_score(truth, prediction)),
        float(balanced_accuracy_score(truth, prediction)),
    )


def _evaluate_probe(
    definition: Mapping[str, Any],
    *,
    features: np.ndarray,
    targets: np.ndarray,
    tasks: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    undersample_seed: int,
    xgb_params: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fold, (train_indices, eval_indices) in enumerate(splits, start=1):
        fit_indices = np.asarray(train_indices, dtype=np.int64)
        train_size_before = len(fit_indices)
        if bool(definition["undersample_task"]):
            fit_indices = _undersample_task_training_fold(
                fit_indices,
                tasks,
                seed=undersample_seed,
                fold=fold,
            )
        train_size_after = len(fit_indices)
        accuracy, balanced_accuracy = _fit_and_score(
            features,
            targets,
            fit_indices,
            np.asarray(eval_indices, dtype=np.int64),
            xgb_params=xgb_params,
        )
        row = {
            "metric": definition["metric"],
            "internal_name": definition["internal_name"],
            "latent": definition["latent"],
            "target": definition["target"],
            "cv_type": definition["cv_type"],
            "fold": fold,
            "train_size_before_balance": train_size_before,
            "train_size_after_balance": train_size_after,
            "eval_size": len(eval_indices),
            "accuracy": accuracy,
            "balanced_accuracy": balanced_accuracy,
        }
        rows.append(row)
        print(
            f"{definition['metric']:<8} fold={fold} "
            f"train={train_size_before}->{train_size_after} "
            f"eval={len(eval_indices)} accuracy={accuracy:.9f} "
            f"balanced_accuracy={balanced_accuracy:.9f}",
            flush=True,
        )
    return rows


def _summarize_results(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for definition in PROBE_DEFINITIONS:
        selected = [
            row
            for row in rows
            if row["internal_name"] == definition["internal_name"]
        ]
        if len(selected) != 5:
            raise RuntimeError(
                f"expected five rows for {definition['internal_name']}, got {len(selected)}"
            )
        accuracies = np.asarray([row["accuracy"] for row in selected], dtype=float)
        balanced = np.asarray(
            [row["balanced_accuracy"] for row in selected],
            dtype=float,
        )
        summary.append(
            {
                "metric": definition["metric"],
                "internal_name": definition["internal_name"],
                "latent": definition["latent"],
                "target": definition["target"],
                "cv_type": definition["cv_type"],
                "accuracy_mean": float(accuracies.mean()),
                "accuracy_std": float(accuracies.std(ddof=0)),
                "balanced_accuracy_mean": float(balanced.mean()),
                "balanced_accuracy_std": float(balanced.std(ddof=0)),
            }
        )
    return summary


def _write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: tuple[str, ...],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _print_summary(
    summary: list[dict[str, Any]],
    *,
    n_subjects: int,
    n_tasks: int,
) -> None:
    print("\nMetric     Actual prediction             Acc mean±std                  BAcc mean±std")
    for definition, row in zip(PROBE_DEFINITIONS, summary):
        accuracy_mean = float(row["accuracy_mean"])
        accuracy_std = float(row["accuracy_std"])
        balanced_mean = float(row["balanced_accuracy_mean"])
        balanced_std = float(row["balanced_accuracy_std"])
        print(
            f"{definition['metric']:<10} {definition['display']:<29} "
            f"{accuracy_mean:.6f}±{accuracy_std:.6f} "
            f"({100 * accuracy_mean:.2f}%±{100 * accuracy_std:.2f}%)    "
            f"{balanced_mean:.6f}±{balanced_std:.6f} "
            f"({100 * balanced_mean:.2f}%±{100 * balanced_std:.2f}%)"
        )
    print("\nDesired disentanglement direction:")
    print("S.acc      ↑")
    print("T⊢S.acc    ↓")
    print("T.acc      ↑")
    print("S⊢T.acc    ↓")
    print(f"\nSubject classes: {n_subjects}")
    print(f"Subject uniform chance: {1.0 / n_subjects:.9f} ({100.0 / n_subjects:.2f}%)")
    print(f"Task classes: {n_tasks}")
    print(f"Task uniform chance: {1.0 / n_tasks:.9f} ({100.0 / n_tasks:.2f}%)")


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError(f"batch_size must be positive, got {args.batch_size}")
    if args.num_workers < 0:
        raise ValueError(f"num_workers must be non-negative, got {args.num_workers}")
    if args.xgb_n_jobs == 0:
        raise ValueError("xgb_n_jobs cannot be zero")

    checkpoint_path = args.checkpoint.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else checkpoint_path.parent.parent / "evaluation" / "probe"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"checkpoint must contain a mapping: {checkpoint_path}")
    config = _load_checkpoint_config(checkpoint, checkpoint_path)
    if _required(config, "dataset") != "erpcore":
        raise ValueError(f"only ERP-Core is supported, got {config['dataset']!r}")
    spec = _spec_from_config(config)
    device = _resolve_device(args.device)

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Run name: {config.get('run_name', '')}")
    print(
        "ReconstructionSpec: "
        f"scope={spec.scope.value}, missing_fill={spec.missing_fill.value}, "
        f"output_base={spec.output_base.value}, "
        f"component_mode={spec.component_mode.value}, "
        f"composition_mode={spec.composition_mode.value}"
    )
    print("Split: ERP-Core test; extraction shuffle=False")
    print("Probe features: raw high-dimensional z_sub and z_task (no scaling/projection)")

    (
        z_sub,
        z_task,
        subjects,
        tasks,
        sample_indices,
        sample_ids,
    ) = _extract_test_latents(
        checkpoint=checkpoint,
        config=config,
        spec=spec,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    del checkpoint
    if device.type == "cuda":
        torch.cuda.empty_cache()

    n_samples = len(subjects)
    n_subjects = len(np.unique(subjects))
    n_tasks = len(np.unique(tasks))
    if n_subjects < 2 or n_tasks < 2:
        raise RuntimeError(
            f"probe requires multiple classes, got subjects={n_subjects}, tasks={n_tasks}"
        )
    print(
        f"Extracted n_samples={n_samples}, n_subjects={n_subjects}, "
        f"n_tasks={n_tasks}, z_sub={z_sub.shape}, z_task={z_task.shape}"
    )

    subject_cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=args.cv_seed,
    )
    task_cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=args.cv_seed,
    )
    subject_splits = list(subject_cv.split(np.zeros(n_samples), subjects))
    task_splits = list(task_cv.split(np.zeros(n_samples), tasks))
    subject_cv_fold = _make_fold_assignments(subject_splits, n_samples)
    task_cv_fold = _make_fold_assignments(task_splits, n_samples)

    latent_path = output_dir / "probe_latents.npz"
    np.savez_compressed(
        latent_path,
        z_sub=z_sub,
        z_task=z_task,
        subjects=subjects,
        tasks=tasks,
        sample_indices=sample_indices,
        sample_ids=sample_ids,
        subject_cv_fold=subject_cv_fold,
        task_cv_fold=task_cv_fold,
    )
    print(f"Saved raw probe latents: {latent_path}")

    features_by_name = {"z_sub": z_sub, "z_task": z_task}
    targets_by_name = {"subject": subjects, "task": tasks}
    splits_by_name = {"subject_cv": subject_splits, "task_cv": task_splits}
    xgb_params = _xgb_parameters(
        xgb_seed=args.xgb_seed,
        n_jobs=args.xgb_n_jobs,
    )
    print(f"XGBoost version: {xgb.__version__}")
    print(f"XGBoost parameters: {json.dumps(xgb_params, sort_keys=True)}")

    result_rows: list[dict[str, Any]] = []
    for definition in PROBE_DEFINITIONS:
        print(
            f"\n=== {definition['metric']}: {definition['display']} "
            f"using {definition['cv_type']} ===",
            flush=True,
        )
        result_rows.extend(
            _evaluate_probe(
                definition,
                features=features_by_name[str(definition["latent"])],
                targets=targets_by_name[str(definition["target"])],
                tasks=tasks,
                splits=splits_by_name[str(definition["cv_type"])],
                undersample_seed=args.undersample_seed,
                xgb_params=xgb_params,
            )
        )

    summary_rows = _summarize_results(result_rows)
    results_path = output_dir / "probe_results.csv"
    summary_path = output_dir / "probe_summary.csv"
    _write_csv(results_path, result_rows, RESULT_FIELDS)
    _write_csv(summary_path, summary_rows, SUMMARY_FIELDS)

    probe_config = {
        "checkpoint": str(checkpoint_path),
        "run_name": str(config.get("run_name", "")),
        "training_seed": int(_required(config, "seed")),
        "spec": {
            "scope": spec.scope.value,
            "missing_fill": spec.missing_fill.value,
            "output_base": spec.output_base.value,
            "component_mode": spec.component_mode.value,
            "composition_mode": spec.composition_mode.value,
        },
        "dataset": "erpcore",
        "split": "test",
        "n_samples": n_samples,
        "n_subjects": n_subjects,
        "n_tasks": n_tasks,
        "cv": {
            "type": "StratifiedKFold",
            "n_splits": 5,
            "shuffle": True,
            "cv_seed": args.cv_seed,
        },
        "xgboost_version": xgb.__version__,
        "xgboost_parameters": xgb_params,
        "xgboost_parameter_source": "local CSLP-AE/utils.py fit_clf_fn",
        "undersample_seed": args.undersample_seed,
        "task_undersampling": "training fold only; minimum task class count",
        "subject_balancing": "none",
        "evaluation_fold_balancing": "none",
        "feature_preprocessing": "none; raw z_sub/z_task",
        "std_ddof": 0,
    }
    config_path = output_dir / "probe_config.json"
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(probe_config, handle, indent=2, sort_keys=True)
        handle.write("\n")

    _print_summary(summary_rows, n_subjects=n_subjects, n_tasks=n_tasks)
    print(f"\nSaved fold results: {results_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved probe config: {config_path}")


if __name__ == "__main__":
    main()
