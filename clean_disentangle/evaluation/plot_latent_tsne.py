#!/usr/bin/env python3
"""Plot clean Stage1 ERP-Core z_sub/z_task latents from one checkpoint."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Subset

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


TASK_NAMES = {
    0: "ERN/Incorrect",
    1: "ERN/Correct",
    2: "LRP/Contralateral",
    3: "LRP/Ipsilateral",
    4: "MMN/Deviants",
    5: "MMN/Standards",
    6: "N2pc/Contralateral",
    7: "N2pc/Ipsilateral",
    8: "N400/Unrelated",
    9: "N400/Related",
    10: "P3/Rare",
    11: "P3/Frequent",
}

TASK_COLORS = {
    0: "#1479D1",
    1: "#73B7F2",
    2: "#F07818",
    3: "#FFB15C",
    4: "#159447",
    5: "#78C86A",
    6: "#D62828",
    7: "#F18181",
    8: "#7441A8",
    9: "#B28BD0",
    10: "#8C564B",
    11: "#D59B8B",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=1500)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--tsne-seed", type=int, default=1968125571)
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


def _experiment_name(config: Mapping[str, Any], spec: ReconstructionSpec) -> str:
    key = (spec.scope.value, spec.missing_fill.value, spec.output_base.value)
    known = {
        ("full", "not_applicable", "none"): "Full D Only",
        ("full", "not_applicable", "prototype"): "Full Prototype + D",
        ("missing", "prototype", "prototype"): "Missing Prototype + D",
    }
    return known.get(key, str(config.get("run_name") or "Clean Stage1"))


def _resolve_device(requested: str) -> torch.device:
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA is unavailable; falling back to CPU", flush=True)
        return torch.device("cpu")
    return device


def _selected_test_indices(
    test_size: int,
    max_samples: int,
    sample_seed: int,
) -> np.ndarray:
    if max_samples < 1:
        raise ValueError(f"max_samples must be positive, got {max_samples}")
    count = min(int(max_samples), int(test_size))
    rng = np.random.default_rng(int(sample_seed))
    selected = rng.choice(np.arange(test_size), count, replace=False)
    return np.sort(selected.astype(np.int64, copy=False))


def _fit_tsne(
    latent: np.ndarray,
    *,
    max_iter: int,
    tsne_seed: int,
) -> np.ndarray:
    if latent.shape[0] <= 30:
        raise ValueError(
            f"t-SNE perplexity=30 requires more than 30 samples, got {latent.shape[0]}"
        )
    return TSNE(
        n_components=2,
        perplexity=30,
        init="pca",
        learning_rate="auto",
        max_iter=int(max_iter),
        random_state=int(tsne_seed),
        n_jobs=-1,
        verbose=1,
    ).fit_transform(latent)


def _plot_latents(
    *,
    z_sub_tsne: np.ndarray,
    z_task_tsne: np.ndarray,
    subjects: np.ndarray,
    tasks: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    unique_subjects = sorted(int(value) for value in np.unique(subjects))
    unique_tasks = sorted(int(value) for value in np.unique(tasks))
    unknown_tasks = sorted(set(unique_tasks) - set(TASK_NAMES))
    if unknown_tasks:
        raise ValueError(f"unknown ERP task ids: {unknown_tasks}")

    subject_colors = plt.get_cmap("nipy_spectral")(
        np.linspace(0, 1, len(unique_subjects))
    )
    subject_map = {
        value: subject_colors[index] for index, value in enumerate(unique_subjects)
    }
    task_map = {value: TASK_COLORS[value] for value in unique_tasks}

    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    panels = (
        (z_sub_tsne, subjects, subject_map, "z_sub, colored by subject"),
        (z_sub_tsne, tasks, task_map, "z_sub, colored by task"),
        (z_task_tsne, tasks, task_map, "z_task, colored by task"),
        (z_task_tsne, subjects, subject_map, "z_task, colored by subject"),
    )
    for axis, (coordinates, values, color_map, panel_title) in zip(
        axes.flat,
        panels,
    ):
        axis.scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            c=[color_map[int(value)] for value in values],
            s=8,
            alpha=0.75,
            linewidths=0,
        )
        axis.set_title(panel_title)
        axis.set_xticks([])
        axis.set_yticks([])

    subject_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=6,
            color=subject_map[value],
            label=f"Subject {value}",
        )
        for value in unique_subjects
    ]
    task_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=6,
            color=task_map[value],
            label=TASK_NAMES[value],
        )
        for value in unique_tasks
    ]
    fig.legend(
        handles=subject_handles,
        title="Subject",
        loc="upper right",
        bbox_to_anchor=(0.995, 0.78),
        frameon=False,
    )
    fig.legend(
        handles=task_handles,
        title="ERP task",
        loc="lower right",
        bbox_to_anchor=(0.995, 0.38),
        frameon=False,
    )
    fig.suptitle(title, fontsize=16)
    fig.subplots_adjust(
        left=0.03,
        right=0.84,
        bottom=0.04,
        top=0.92,
        wspace=0.08,
        hspace=0.22,
    )
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError(f"batch_size must be positive, got {args.batch_size}")
    if args.num_workers < 0:
        raise ValueError(f"num_workers must be non-negative, got {args.num_workers}")
    if args.max_iter < 250:
        raise ValueError(f"max_iter must be at least 250, got {args.max_iter}")

    checkpoint_path = args.checkpoint.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else checkpoint_path.parent.parent / "evaluation" / "tsne"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"checkpoint must contain a mapping: {checkpoint_path}")
    state_dict = checkpoint.get("model")
    if not isinstance(state_dict, Mapping):
        raise KeyError(f"checkpoint has no model state_dict: {checkpoint_path}")
    config = _load_checkpoint_config(checkpoint, checkpoint_path)
    if _required(config, "dataset") != "erpcore":
        raise ValueError(f"only ERP-Core is supported, got {config['dataset']!r}")
    spec = _spec_from_config(config)

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
    model.load_state_dict(state_dict, strict=True)
    device = _resolve_device(args.device)
    model.to(device)
    model.eval()

    from data_processor.erpcore_cslp import prepare_ERPCORE_cslp_dataset

    _, test_dataset, _ = prepare_ERPCORE_cslp_dataset(
        _required(config, "data_path"),
        sampling_rate=int(_required(config, "sampling_rate")),
        normalize_method=str(_required(config, "norm_method")),
    )
    sample_indices = _selected_test_indices(
        len(test_dataset),
        args.max_samples,
        args.sample_seed,
    )
    expected_subjects = np.asarray(test_dataset.subjects, dtype=np.int64)[sample_indices]
    expected_tasks = np.asarray(test_dataset.labels, dtype=np.int64)[sample_indices]
    loader = DataLoader(
        Subset(test_dataset, sample_indices.tolist()),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    z_sub_batches: list[np.ndarray] = []
    z_task_batches: list[np.ndarray] = []
    subject_batches: list[np.ndarray] = []
    task_batches: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            moved_batch = move_batch(
                batch,
                device,
                input_scale=float(_required(config, "input_scale")),
            )
            output = model.forward_reconstruction(moved_batch, spec)
            z_sub_batches.append(output["z_sub"].detach().cpu().numpy())
            z_task_batches.append(output["z_task"].detach().cpu().numpy())
            subject_batches.append(batch["subject"].numpy())
            task_batches.append(batch["task"].numpy())

    z_sub = np.concatenate(z_sub_batches, axis=0)
    z_task = np.concatenate(z_task_batches, axis=0)
    subjects = np.concatenate(subject_batches).astype(np.int64, copy=False)
    tasks = np.concatenate(task_batches).astype(np.int64, copy=False)
    if not np.array_equal(subjects, expected_subjects):
        raise RuntimeError("DataLoader subject order differs from selected test indices")
    if not np.array_equal(tasks, expected_tasks):
        raise RuntimeError("DataLoader task order differs from selected test indices")
    if z_sub.shape[0] != len(sample_indices) or z_task.shape[0] != len(sample_indices):
        raise RuntimeError(
            "latent/sample count mismatch: "
            f"z_sub={z_sub.shape}, z_task={z_task.shape}, "
            f"samples={len(sample_indices)}"
        )
    if not np.isfinite(z_sub).all() or not np.isfinite(z_task).all():
        raise RuntimeError("z_sub or z_task contains NaN or Inf")

    print(f"z_sub: {z_sub.shape} -> t-SNE", flush=True)
    z_sub_tsne = _fit_tsne(
        z_sub,
        max_iter=args.max_iter,
        tsne_seed=args.tsne_seed,
    )
    print(f"z_task: {z_task.shape} -> t-SNE", flush=True)
    z_task_tsne = _fit_tsne(
        z_task,
        max_iter=args.max_iter,
        tsne_seed=args.tsne_seed,
    )

    experiment_name = _experiment_name(config, spec)
    title = f"{experiment_name} | seed={config['seed']}"
    png_path = output_dir / "z_latent_tsne.png"
    npz_path = output_dir / "z_latent_tsne.npz"
    json_path = output_dir / "tsne_config.json"
    _plot_latents(
        z_sub_tsne=z_sub_tsne,
        z_task_tsne=z_task_tsne,
        subjects=subjects,
        tasks=tasks,
        title=title,
        output_path=png_path,
    )
    np.savez_compressed(
        npz_path,
        z_sub=z_sub,
        z_task=z_task,
        z_sub_tsne=z_sub_tsne,
        z_task_tsne=z_task_tsne,
        subjects=subjects,
        tasks=tasks,
        sample_indices=sample_indices,
        checkpoint_path=np.asarray(str(checkpoint_path)),
        run_name=np.asarray(str(config.get("run_name", ""))),
        sample_seed=np.asarray(args.sample_seed, dtype=np.int64),
        tsne_seed=np.asarray(args.tsne_seed, dtype=np.int64),
    )
    tsne_config = {
        "checkpoint": str(checkpoint_path),
        "run_name": str(config.get("run_name", "")),
        "experiment_name": experiment_name,
        "training_seed": int(config["seed"]),
        "spec": {
            "scope": spec.scope.value,
            "missing_fill": spec.missing_fill.value,
            "output_base": spec.output_base.value,
            "component_mode": spec.component_mode.value,
            "composition_mode": spec.composition_mode.value,
        },
        "dataset": "erpcore",
        "split": "test",
        "test_size": len(test_dataset),
        "selected_samples": len(sample_indices),
        "sample_seed": args.sample_seed,
        "tsne": {
            "n_components": 2,
            "perplexity": 30,
            "init": "pca",
            "learning_rate": "auto",
            "max_iter": args.max_iter,
            "random_state": args.tsne_seed,
            "n_jobs": -1,
        },
        "device": str(device),
        "z_sub_shape": list(z_sub.shape),
        "z_task_shape": list(z_task.shape),
    }
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(tsne_config, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"subjects: {len(np.unique(subjects))}; tasks: {len(np.unique(tasks))}")
    print(f"saved: {png_path}")
    print(f"saved: {npz_path}")
    print(f"saved: {json_path}")


if __name__ == "__main__":
    main()
