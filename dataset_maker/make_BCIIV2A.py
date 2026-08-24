"""Create BCI Competition IV 2a multi-session JSON manifests.

The source directory is expected to contain AdaBrain-style per-trial pickle
files:

    processed_data/A01/1_1.pkl ... 1_576.pkl
    ...
    processed_data/A09/9_1.pkl ... 9_576.pkl

Each pickle stores ``{"X": array[22, 1000], "Y": class_id}`` at 250 Hz.
The output ``train.json``, ``val.json`` and ``test.json`` match
``data_processor/bciiv2a.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np


DEFAULT_INPUT_ROOT = Path(
    "/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/"
    "global_user/7461-chenxinhe/BCI-IV-2A/processed_data"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/"
    "global_user/7461-chenxinhe/BCI-IV-2A/multi_subject_json"
)

CHANNELS = [
    "Fz", "FC3", "FC1", "FCZ", "FC2", "FC4", "C5", "C3", "C1", "CZ",
    "C2", "C4", "C6", "CP3", "CP1", "CPZ", "CP2", "CP4", "P1", "PZ",
    "P2", "POZ",
]
SPLITS = {
    "train": range(1, 289),
    "val": range(289, 433),
    "test": range(433, 577),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create BCI-IV-2a multi-session train/val/test JSON manifests."
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_trial(path: Path) -> tuple[np.ndarray, int]:
    with path.open("rb") as handle:
        sample = pickle.load(handle)
    if not isinstance(sample, dict) or "X" not in sample or "Y" not in sample:
        raise ValueError(f"Invalid BCI-IV-2a pickle schema: {path}")
    x = np.asarray(sample["X"])
    if x.shape != (len(CHANNELS), 1000):
        raise ValueError(f"Invalid BCI-IV-2a X shape in {path}: {x.shape}")
    if not np.isfinite(x).all():
        raise ValueError(f"NaN or Inf in BCI-IV-2a trial: {path}")
    label = int(np.asarray(sample["Y"]).reshape(()))
    if label not in {0, 1, 2, 3}:
        raise ValueError(f"Invalid BCI-IV-2a label in {path}: {label}")
    return x, label


def calculate_train_stats(records: list[dict]) -> dict:
    max_value = -float("inf")
    min_value = float("inf")
    channel_means = np.zeros(len(CHANNELS), dtype=np.float64)
    channel_stds = np.zeros(len(CHANNELS), dtype=np.float64)
    for record in records:
        x, _ = load_trial(Path(record["file"]))
        max_value = max(max_value, float(np.max(x)))
        min_value = min(min_value, float(np.min(x)))
        channel_means += np.mean(x, axis=-1)
        channel_stds += np.std(x, axis=-1)
    channel_means /= len(records)
    channel_stds /= len(records)
    if not np.isfinite(channel_means).all() or not np.isfinite(channel_stds).all():
        raise ValueError("Invalid BCI-IV-2a normalization statistics")
    return {
        "sampling_rate": 250,
        "ch_names": CHANNELS,
        "min": min_value,
        "max": max_value,
        "mean": channel_means.tolist(),
        "std": channel_stds.tolist(),
    }


def build_records(input_root: Path) -> dict[str, list[dict]]:
    output = {name: [] for name in SPLITS}
    for subject_index in range(9):
        subject_id = subject_index + 1
        subject_name = f"A{subject_id:02d}"
        subject_dir = input_root / subject_name
        if not subject_dir.is_dir():
            raise FileNotFoundError(f"Missing BCI-IV-2a subject directory: {subject_dir}")
        for trial in range(1, 577):
            path = subject_dir / f"{subject_id}_{trial}.pkl"
            if not path.is_file():
                raise FileNotFoundError(f"Missing BCI-IV-2a trial: {path}")
            _, label = load_trial(path)
            record = {
                "subject_id": subject_index,
                "subject_name": subject_name,
                "file": str(path),
                "label": label,
            }
            for split, trial_range in SPLITS.items():
                if trial in trial_range:
                    output[split].append(record)
                    break
    expected_counts = {"train": 2592, "val": 1296, "test": 1296}
    for split, expected in expected_counts.items():
        if len(output[split]) != expected:
            raise ValueError(
                f"Unexpected BCI-IV-2a {split} count: {len(output[split])}, expected {expected}"
            )
    return output


def atomic_json_dump(payload: dict, destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    if not input_root.is_dir():
        raise FileNotFoundError(f"BCI-IV-2a input root not found: {input_root}")
    records = build_records(input_root)
    dataset_info = calculate_train_stats(records["train"])
    payloads = {
        split: {"dataset_info": dataset_info, "subject_data": split_records}
        for split, split_records in records.items()
    }
    print(
        "BCI-IV-2a audit: "
        + ", ".join(f"{split}={len(value)}" for split, value in records.items())
        + f", output={output_root}"
    )
    if args.dry_run:
        print("Dry run completed successfully; no output was written.")
        return
    output_root.mkdir(parents=True, exist_ok=True)
    for split, payload in payloads.items():
        destination = output_root / f"{split}.json"
        if destination.exists() and not args.overwrite:
            raise FileExistsError(f"Output exists: {destination}; pass --overwrite")
        atomic_json_dump(payload, destination)
    print(f"Completed BCI-IV-2a manifests: {output_root}")


if __name__ == "__main__":
    main()
