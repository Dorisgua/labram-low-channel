#!/usr/bin/env python3
"""One-off C probe using real observed tokens plus missing D tokens.

This is deliberately separate from D_sub(flat)/D_task(flat).  For C it builds
two 28-position representations:

  observed_plus_d_sub  = real H_obs at observed positions + D_sub_miss
  observed_plus_d_task = real H_obs at observed positions + D_task_miss

Both are flattened to 28*200=5600 features and evaluated with the same
CSLP-style XGBoost protocol as the regular token-vs-pooled diagnostic.
"""

from __future__ import annotations

import argparse
import csv
import json
# from collections.abc import Mapping
from pathlib import Path
# from typing import Any

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from clean_disentangle.engine import move_batch
from clean_disentangle.evaluation.probe_latents import (
    _load_checkpoint_config,
    _required,
    _resolve_device,
    _spec_from_config,
    _undersample_task_training_fold,
    _xgb_parameters,
)
from clean_disentangle.modeling import ReconstructionScope, MissingFillMode, OutputBaseMode, ComponentMode
from clean_disentangle.stage1.train_stage1 import build_erpcore_reconstruction_model
from clean_disentangle.evaluation.diagnostics.diagnose_token_vs_pooled_probe import (
    _fit_and_score,
)


def _extract(checkpoint_path: Path, device_name: str, batch_size: int, num_workers: int):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = _load_checkpoint_config(checkpoint, checkpoint_path)
    spec = _spec_from_config(config)
    expected = (ReconstructionScope.MISSING, MissingFillMode.PROTOTYPE, OutputBaseMode.PROTOTYPE, ComponentMode.IDENTITY)
    actual = (spec.scope, spec.missing_fill, spec.output_base, spec.component_mode)
    if actual != expected:
        raise RuntimeError(f"C diagnostic requires scope/fill/base/component={expected}, got {actual}")

    model = build_erpcore_reconstruction_model(
        legacy_root=Path(_required(config, "legacy_root")),
        cnn_checkpoint=Path(_required(config, "cnn_checkpoint")),
        prototype_checkpoint=Path(_required(config, "prototype_checkpoint")),
        spec=spec,
        seed=int(_required(config, "seed")),
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    device = _resolve_device(device_name)
    model.to(device).eval()

    from data_processor.erpcore_cslp import prepare_ERPCORE_cslp_dataset

    _, test_dataset, _ = prepare_ERPCORE_cslp_dataset(
        _required(config, "data_path"),
        sampling_rate=int(_required(config, "sampling_rate")),
        normalize_method=str(_required(config, "norm_method")),
    )
    loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=device.type == "cuda")

    sub_parts, task_parts, subjects, tasks = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            moved = move_batch(batch, device, input_scale=float(_required(config, "input_scale")))
            output = model.forward_reconstruction(moved, spec)
            # input_tokens contains real H_obs at observed positions and P_miss
            # at missing positions. Replace only missing positions with D.
            positions = output["target_positions"]
            observed_plus_sub = output["input_tokens"].index_copy(1, positions, output["d_sub"])
            observed_plus_task = output["input_tokens"].index_copy(1, positions, output["d_task"])
            sub_parts.append(observed_plus_sub.cpu().numpy())
            task_parts.append(observed_plus_task.cpu().numpy())
            subjects.extend(batch["subject"].cpu().numpy().tolist())
            tasks.extend(batch["task"].cpu().numpy().tolist())

    return {
        "observed_plus_d_sub": np.concatenate(sub_parts).reshape(len(subjects), -1),
        "observed_plus_d_task": np.concatenate(task_parts).reshape(len(subjects), -1),
        "subjects": np.asarray(subjects, dtype=np.int64),
        "tasks": np.asarray(tasks, dtype=np.int64),
    }, config


def _run_probe(name: str, features: np.ndarray, target_name: str, target: np.ndarray, splits, xgb_params):
    rows = []
    for fold, (train_idx, eval_idx) in enumerate(splits, 1):
        fit_idx = np.asarray(train_idx, dtype=np.int64)
        before = len(fit_idx)
        if target_name == "Task":
            fit_idx = _undersample_task_training_fold(fit_idx, target, seed=42, fold=fold)
        acc, bacc = _fit_and_score(features, target, fit_idx, np.asarray(eval_idx), xgb_params=xgb_params)
        rows.append({"representation": name, "target": target_name, "fold": fold, "train_size_before_balance": before, "train_size_after_balance": len(fit_idx), "eval_size": len(eval_idx), "accuracy": acc, "balanced_accuracy": bacc})
        print(f"{name} -> {target_name} fold={fold} acc={acc:.9f} bacc={bacc:.9f}", flush=True)
    accs = np.asarray([r["accuracy"] for r in rows]); baccs = np.asarray([r["balanced_accuracy"] for r in rows])
    return {"representation": name, "target": target_name, "dim": int(features.shape[1]), "accuracy_mean": float(accs.mean()), "accuracy_std": float(accs.std()), "balanced_accuracy_mean": float(baccs.mean()), "balanced_accuracy_std": float(baccs.std())}, rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    arrays, config = _extract(args.checkpoint.resolve(), args.device, args.batch_size, args.num_workers)
    n = len(arrays["subjects"])
    subject_cv = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(n), arrays["subjects"]))
    task_cv = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(n), arrays["tasks"]))
    xgb_params = _xgb_parameters(xgb_seed=42, n_jobs=-1)
    definitions = [
        ("observed_plus_D_sub(flat)", arrays["observed_plus_d_sub"], "Subject", arrays["subjects"], subject_cv),
        ("observed_plus_D_sub(flat)", arrays["observed_plus_d_sub"], "Task", arrays["tasks"], subject_cv),
        ("observed_plus_D_task(flat)", arrays["observed_plus_d_task"], "Task", arrays["tasks"], task_cv),
        ("observed_plus_D_task(flat)", arrays["observed_plus_d_task"], "Subject", arrays["subjects"], task_cv),
    ]
    summaries, fold_rows = [], []
    for name, features, target_name, target, splits in definitions:
        summary, rows = _run_probe(name, features, target_name, target, splits, xgb_params)
        summaries.append(summary); fold_rows.extend(rows)
    with (args.output_dir / "c_observed_plus_d_probe.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fold_rows[0])); writer.writeheader(); writer.writerows(fold_rows)
    (args.output_dir / "c_observed_plus_d_summary.json").write_text(json.dumps({"checkpoint": str(args.checkpoint.resolve()), "feature_dim": 5600, "summaries": summaries}, indent=2) + "\n")
    for s in summaries:
        print(f"{s['representation']} -> {s['target']}: dim={s['dim']} acc={s['accuracy_mean']:.6f}±{s['accuracy_std']:.6f} bacc={s['balanced_accuracy_mean']:.6f}±{s['balanced_accuracy_std']:.6f}")


if __name__ == "__main__":
    main()
