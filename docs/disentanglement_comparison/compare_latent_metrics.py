#!/usr/bin/env python3
"""Plot TUEV Stage1 correction latents on validation or all test samples."""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Dataset

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from Channels_definition import TUEV_13_CHANNELS, TUEV_23_CHANNELS
from run_dynamic_stage1 import get_models
import utils


class SelectedTUEV(Dataset):
    def __init__(self, root: Path, max_samples: int, max_subjects: int, seed: int, split: str):
        self.split = split
        directory = root / ("processed_test" if split == "test" else "processed_eval")
        files = sorted(directory.glob("*.pkl"))
        if not files:
            raise FileNotFoundError(f"No pkl files in {directory}")
        if split == "val":
            counts = Counter(p.name.split("_", 1)[0] for p in files)
            chosen = {name for name, _ in counts.most_common(max_subjects)}
            files = [p for p in files if p.name.split("_", 1)[0] in chosen]
            rng = np.random.default_rng(seed)
            indices = np.sort(rng.choice(len(files), min(max_samples, len(files)), replace=False))
            files = [files[i] for i in indices]
        self.files = files
        self.subject_names = ([p.name.split("_", 1)[0] for p in files] if split == "val" else [])
        self.observed_indices = [TUEV_23_CHANNELS.index(name) for name in TUEV_13_CHANNELS]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        with self.files[index].open("rb") as handle:
            sample = pickle.load(handle)
        full = torch.as_tensor(np.asarray(sample["signal"]), dtype=torch.float32)
        if full.shape != (23, 1000):
            raise ValueError(f"Unexpected shape {tuple(full.shape)}: {self.files[index]}")
        label = int(np.asarray(sample["label"]).reshape(-1)[0]) - 1
        if label not in range(6):
            raise ValueError(f"Invalid label {label}: {self.files[index]}")
        return full[self.observed_indices], full, label


def embed(features: np.ndarray, seed: int) -> np.ndarray:
    features = features.reshape(len(features), -1)
    components = min(50, len(features) - 1, features.shape[1])
    reduced = (PCA(n_components=components, svd_solver="randomized", random_state=seed).fit_transform(features)
               if features.shape[1] > 50 else features)
    return TSNE(n_components=2, perplexity=min(30, (len(features) - 1) / 3),
                init="pca", learning_rate="auto", random_state=seed).fit_transform(reduced)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=3000)
    parser.add_argument("--max-subjects", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    cli = parser.parse_args()
    if cli.split == "val" and cli.max_samples < 5:
        parser.error("--max-samples must be at least 5")

    checkpoint = torch.load(cli.checkpoint, map_location="cpu", weights_only=False)
    args = checkpoint["args"]
    if args.dataset != "TUEV" or args.completion_scope != "tuev13_with_tuev23":
        raise ValueError("Expected a TUEV 13-to-23 Dynamic Stage1 checkpoint")
    root = cli.data_path or Path(args.data_path)
    dataset = SelectedTUEV(root, cli.max_samples, cli.max_subjects, cli.seed, cli.split)
    device = torch.device(cli.device)
    model = get_models(args)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.completion_scope = args.completion_scope
    model.pooling_scope = args.pooling_scope
    model.real_input_chans_index = [int(x) for x in utils.get_input_chans(TUEV_13_CHANNELS)]
    model.target_input_chans_index = [int(x) for x in utils.get_input_chans(TUEV_23_CHANNELS)]
    model.to(device).eval()
    loader = DataLoader(dataset, batch_size=cli.batch_size, shuffle=False,
                        num_workers=cli.num_workers, pin_memory=device.type == "cuda")
    subs, tasks, labels = [], [], []
    scale = float(getattr(args, "input_scale", 0.01))
    projection = None
    if cli.split == "test":
        generator = torch.Generator().manual_seed(cli.seed)
        projection = torch.randn(10 * 5 * 200, 50, generator=generator).to(device) / np.sqrt(50)
    with torch.inference_mode():
        for batch_index, (observed, _, task) in enumerate(loader):
            observed = observed.to(device, non_blocking=True) * scale
            observed = observed.reshape(observed.shape[0], observed.shape[1], 5, 200)
            output = model._encode_dynamic_tokens(model._patch_tokens(observed))
            sub = output["d_sub"].float().flatten(1)
            task_latent = output["d_task"].float().flatten(1)
            if projection is not None:
                sub = sub @ projection
                task_latent = task_latent @ projection
            subs.append(sub.cpu().numpy())
            tasks.append(task_latent.cpu().numpy())
            labels.append(task.numpy())
            if cli.split == "test" and (batch_index + 1) % 50 == 0:
                print(f"Extracted {min((batch_index + 1) * cli.batch_size, len(dataset))}/{len(dataset)}", flush=True)
    d_sub = np.concatenate(subs)
    d_task = np.concatenate(tasks)
    task_labels = np.concatenate(labels)
    subject_names = sorted(set(dataset.subject_names))
    subject_to_id = {name: i for i, name in enumerate(subject_names)}
    subject_labels = np.array([subject_to_id[name] for name in dataset.subject_names])
    print(f"Extracted {len(dataset)} {cli.split} samples, {len(subject_names)} verified subjects, 6 tasks", flush=True)
    sub_xy = embed(d_sub, cli.seed)
    print("d_sub t-SNE complete", flush=True)
    task_xy = embed(d_task, cli.seed)
    print("d_task t-SNE complete", flush=True)

    fig, axes = plt.subplots(1, 2 if cli.split == "test" else 4,
                            figsize=(12 if cli.split == "test" else 20, 5), constrained_layout=True)
    subject_colors = plt.get_cmap("tab20", max(1, len(subject_names)))
    task_colors = plt.get_cmap("tab10", 6)
    panels = ((sub_xy, task_labels, task_colors, "d_sub by task"),
              (task_xy, task_labels, task_colors, "d_task by task")) if cli.split == "test" else (
              (sub_xy, subject_labels, subject_colors, "d_sub by subject"),
              (sub_xy, task_labels, task_colors, "d_sub by task"),
              (task_xy, task_labels, task_colors, "d_task by task"),
              (task_xy, subject_labels, subject_colors, "d_task by subject"))
    for ax, (xy, values, cmap, title) in zip(axes, panels):
        ax.scatter(xy[:, 0], xy[:, 1], c=values, cmap=cmap, vmin=-0.5,
                   vmax=(len(subject_names) if "subject" in title else 6) - 0.5,
                   s=1.5 if cli.split == "test" else 3, alpha=0.65,
                   linewidths=0, rasterized=True)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    task_handles = [Line2D([0], [0], marker="o", linestyle="", color=task_colors(i),
                           label=f"Class {i + 1}") for i in range(6)]
    axes[1 if cli.split == "test" else 2].legend(
        handles=task_handles, loc="upper left", bbox_to_anchor=(1.01, 1),
        fontsize=8, frameon=False)
    fig.suptitle(f"TUEV {cli.split} | Stage1 checkpoint-best | "
                 f"13 real + 10 missing channels | n={len(dataset)}", fontsize=12)
    cli.output_dir.mkdir(parents=True, exist_ok=True)
    figure = cli.output_dir / ("test_all_task_d_tsne.png" if cli.split == "test" else "comparison_d_tsne.png")
    fig.savefig(figure, dpi=200)
    plt.close(fig)
    metadata = {"checkpoint": str(cli.checkpoint.resolve()), "data_path": str(root.resolve()),
                "split": "processed_test" if cli.split == "test" else "processed_eval",
                "samples": len(dataset), "subjects": len(subject_names),
                "subject_names": subject_names,
                "task_counts": np.bincount(task_labels, minlength=6).tolist(), "seed": cli.seed,
                "latent_shape": list(d_sub.shape[1:]), "input_scale": scale,
                "projection": "Gaussian random projection 10000->50" if cli.split == "test" else "PCA 10000->50",
                "note": "Independent t-SNE fits for d_sub and d_task; compare label structure, not cross-panel distances."}
    (cli.output_dir / ("test_all_task_d_tsne.json" if cli.split == "test" else "comparison_d_tsne.json")).write_text(
        json.dumps(metadata, indent=2) + "\n")
    print(f"Saved {figure}", flush=True)


if __name__ == "__main__":
    main()
