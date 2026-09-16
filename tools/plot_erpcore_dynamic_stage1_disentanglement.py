#!/usr/bin/env python3
"""Plot subject/task separation for an ERP-Core Dynamic Stage1 checkpoint."""

import argparse
import csv
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


TASK_NAMES = {
    0: "ERN/Incorrect", 1: "ERN/Correct",
    2: "LRP/Contralateral", 3: "LRP/Ipsilateral",
    4: "MMN/Deviants", 5: "MMN/Standards",
    6: "N2pc/Contralateral", 7: "N2pc/Ipsilateral",
    8: "N400/Unrelated", 9: "N400/Related",
    10: "P3/Rare", 11: "P3/Frequent",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "outputs/erpcore/erp_core_D_stage1/checkpoint-best.pth",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=14485)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--d-pca-dim", type=int, default=50)
    args = parser.parse_args()
    if args.max_samples < 32 or args.max_iter < 250:
        parser.error("max-samples must be >=32 and max-iter must be >=250")
    if args.output_dir is None:
        args.output_dir = args.checkpoint.parent / "disentanglement_best"
    return args


def main():
    args = parse_args()
    args.checkpoint = args.checkpoint.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = str(args.threads)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import numpy as np
    import torch
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from torch.utils.data import DataLoader, Subset

    os.chdir(ROOT)
    import utils
    from Channels_definition import ERPCORE_12_CHANNELS
    from data_processor.erpcore import prepare_ERPCORE_pt_dataset
    from run_dynamic_stage1 import get_models, _validate_completion_prototype

    torch.set_num_threads(args.threads)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    train_args = checkpoint["args"]
    original_args = vars(train_args).copy()
    train, test, val = prepare_ERPCORE_pt_dataset(
        train_args.data_path,
        sampling_rate=train_args.sampling_rate,
        normalize_method=train_args.norm_method,
        channel_names=ERPCORE_12_CHANNELS,
    )
    del train, val

    prototype_ckpt = torch.load(
        train_args.channel_prototype_path, map_location="cpu", weights_only=False
    )
    prototypes = prototype_ckpt["channel_prototypes"]
    target_names = prototype_ckpt["ch_names"]
    target_indices = [int(v) for v in prototype_ckpt["input_chans_index"]]
    real_indices = [int(v) for v in utils.get_input_chans(ERPCORE_12_CHANNELS)]
    _validate_completion_prototype(
        args=train_args,
        ch_names=ERPCORE_12_CHANNELS,
        target_ch_names=target_names,
        target_input_chans_index=target_indices,
        prototypes=prototypes,
    )

    model = get_models(train_args)
    model.completion_scope = train_args.completion_scope
    model.pooling_scope = train_args.pooling_scope
    model.erpcore28_channel_prototypes.copy_(prototypes)
    model.target_input_chans_index = target_indices
    model.real_input_chans_index = real_indices
    model.load_state_dict(checkpoint["model"], strict=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(device).eval().requires_grad_(False)

    rng = np.random.default_rng(42)
    sample_count = min(args.max_samples, len(test))
    selected = np.sort(rng.choice(len(test), sample_count, replace=False))
    subjects = np.asarray(
        [int(test.subject_values[int(test.indices[i])]) for i in selected]
    )
    tasks = np.asarray(test.labels)[selected]
    loader = DataLoader(
        Subset(test, selected.tolist()), batch_size=args.batch_size,
        num_workers=args.num_workers, shuffle=False, pin_memory=device.type == "cuda",
    )

    chunks = {key: [] for key in ("z_sub", "z_task", "d_sub", "d_task")}
    with torch.no_grad():
        done = 0
        for batch in loader:
            x_obs = batch[2].float().to(device) * float(train_args.input_scale)
            x_obs = x_obs.reshape(x_obs.shape[0], x_obs.shape[1], 1, 200)
            outputs = model._encode_dynamic_tokens(model._patch_tokens(x_obs))
            for key in chunks:
                value = outputs[key]
                if key.startswith("d_"):
                    value = value.flatten(1)
                chunks[key].append(value.cpu().numpy())
            done += x_obs.shape[0]
            print(f"Extracted {done}/{sample_count}", flush=True)

    features = {key: np.concatenate(value, axis=0) for key, value in chunks.items()}
    for key, value in features.items():
        if not np.isfinite(value).all():
            raise ValueError(f"Non-finite features: {key}")
        np.save(args.output_dir / f"{key}.npy", value)
    with (args.output_dir / "sample_alignment.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["test_local_index", "payload_index", "subject", "task"])
        writer.writerows(zip(selected, np.asarray(test.indices)[selected], subjects, tasks))

    metadata = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": hashlib.file_digest(
            args.checkpoint.open("rb"), "sha256"
        ).hexdigest(),
        "checkpoint_epoch_field": checkpoint.get("epoch"),
        "checkpoint_args": original_args,
        "samples": sample_count,
        "split": "test",
        "sample_seed": 42,
        "tsne_seed": 1968125571,
        "features": {},
        "note": (
            "Dynamic Stage1 accepts 12 observed channels and predicts 16 missing "
            "positions; unlike the three-stage reference, it has no matched full28 "
            "Corrector forward. z is pre-tanh missing-token mean; d is flattened "
            "0.02*tanh correction."
        ),
    }
    coordinates = {}
    for key, value in features.items():
        transformed = value
        info = {"shape": list(value.shape), "pca_dim": None}
        if key.startswith("d_") and args.d_pca_dim:
            dim = min(args.d_pca_dim, sample_count - 1, value.shape[1])
            pca = PCA(n_components=dim, svd_solver="randomized", random_state=42)
            transformed = pca.fit_transform(value)
            info["pca_dim"] = dim
            info["pca_explained_variance"] = float(
                pca.explained_variance_ratio_.sum()
            )
        kwargs = dict(
            n_components=2, perplexity=30, init="pca", learning_rate="auto",
            random_state=1968125571, n_jobs=args.threads, verbose=1,
        )
        iteration_key = "max_iter" if "max_iter" in inspect.signature(TSNE).parameters else "n_iter"
        kwargs[iteration_key] = args.max_iter
        estimator = TSNE(**kwargs)
        print(f"{key}: {value.shape} -> t-SNE", flush=True)
        coordinates[key] = estimator.fit_transform(transformed)
        np.save(args.output_dir / f"{key}_tsne.npy", coordinates[key])
        info["kl_divergence"] = float(estimator.kl_divergence_)
        metadata["features"][key] = info
        (args.output_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, default=str)
        )

    subject_ids = sorted(np.unique(subjects))
    subject_colors = dict(zip(
        subject_ids, plt.get_cmap("nipy_spectral")(np.linspace(0, 1, len(subject_ids)))
    ))
    task_colors = dict(zip(
        sorted(np.unique(tasks)), plt.get_cmap("tab20")(np.linspace(0, 1, 12))
    ))
    subject_handles = [
        Line2D([], [], marker="o", linestyle="", color=subject_colors[v], label=f"Subject {v}")
        for v in subject_ids
    ]
    task_handles = [
        Line2D([], [], marker="o", linestyle="", color=task_colors[v], label=TASK_NAMES[v])
        for v in sorted(np.unique(tasks))
    ]
    fig, axes = plt.subplots(2, 4, figsize=(24, 10))
    for row, kind in enumerate(("z", "d")):
        for axis, (branch, label_name) in zip(
            axes[row], (("sub", "subject"), ("sub", "task"),
                        ("task", "task"), ("task", "subject"))
        ):
            key = f"{kind}_{branch}"
            labels = subjects if label_name == "subject" else tasks
            colors = subject_colors if label_name == "subject" else task_colors
            xy = coordinates[key]
            axis.scatter(
                xy[:, 0], xy[:, 1], c=[colors[int(v)] for v in labels],
                s=3, alpha=0.65, linewidths=0, rasterized=True,
            )
            axis.set_title(f"{key} ({features[key].shape[1]}D), by {label_name}")
            axis.set_xticks([])
            axis.set_yticks([])
        axes[row, 0].set_ylabel(
            "pre-tanh z (mean over missing positions)" if kind == "z"
            else f"actual d = {train_args.correction_scale} * tanh(raw)"
        )
    fig.suptitle(
        "ERP-Core Dynamic Stage1 checkpoint-best | 12 real -> 16 missing | "
        f"Test n={sample_count}"
    )
    fig.legend(handles=subject_handles, title="Subject", loc="upper right", bbox_to_anchor=(0.998, 0.88), fontsize=8)
    fig.legend(handles=task_handles, title="ERP task", loc="lower right", bbox_to_anchor=(0.998, 0.08), fontsize=8)
    fig.subplots_adjust(left=0.04, right=0.84, bottom=0.06, top=0.90, wspace=0.10, hspace=0.22)
    fig.savefig(args.output_dir / "comparison_z_d_tsne.png", dpi=180)
    fig.savefig(args.output_dir / "comparison_z_d_tsne.pdf", dpi=180)
    plt.close(fig)
    print(f"Done: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
