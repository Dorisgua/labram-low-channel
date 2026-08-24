"""SEED emotion loaders for the AdaBrain cross-subject protocol."""

from __future__ import annotations

import pickle
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from Channels_definition import SEED_62_CHANNELS


EXPECTED_SUBJECT_SAMPLES = 10137
EXPECTED_SPLIT_SAMPLES = {
    "train": 97308,
    "val": 24336,
    "test": 30411,
}
TRAIN_SUBJECTS = tuple(range(1, 13))
TEST_SUBJECTS = tuple(range(13, 16))

# Original label.mat: positive/neutral/negative are 1/0/-1 and preprocessing
# maps them to class ids 2/1/0 respectively.
TRIAL_LABELS = (2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0)
FILENAME_PATTERN = re.compile(
    r"^S(?P<subject>\d+)_(?P<session>[1-3])_"
    r"(?P<trial>1[0-5]|[1-9])_(?P<window>\d+)\.pkl$"
)


def _parse_sample_path(path: Path, expected_subject: int) -> tuple[int, int, int, int]:
    match = FILENAME_PATTERN.fullmatch(path.name)
    if match is None:
        raise ValueError(f"Unexpected SEED sample filename: {path}")
    values = tuple(int(match.group(name)) for name in ("subject", "session", "trial", "window"))
    if values[0] != expected_subject:
        raise ValueError(
            f"SEED sample subject mismatch: folder={expected_subject}, file={path.name}"
        )
    return values


def _discover_subject_files(root: Path) -> dict[int, list[Path]]:
    if not root.is_dir():
        raise FileNotFoundError(f"SEED processed-data directory not found: {root}")

    unexpected_dirs = sorted(
        path.name
        for path in root.iterdir()
        if path.is_dir() and (not path.name.isdigit() or not 1 <= int(path.name) <= 15)
    )
    if unexpected_dirs:
        raise ValueError(f"Unexpected directories below SEED root: {unexpected_dirs}")

    subject_files: dict[int, list[Path]] = {}
    incomplete: dict[int, int] = {}
    for subject in range(1, 16):
        subject_dir = root / str(subject)
        if not subject_dir.is_dir():
            incomplete[subject] = 0
            continue
        parsed = []
        for path in subject_dir.iterdir():
            if path.suffix != ".pkl":
                continue
            key = _parse_sample_path(path, subject)
            parsed.append((key, path))
        parsed.sort(key=lambda item: item[0])
        files = [path for _, path in parsed]
        subject_files[subject] = files
        if len(files) != EXPECTED_SUBJECT_SAMPLES:
            incomplete[subject] = len(files)

    if incomplete:
        details = ", ".join(
            f"S{subject}={count}" for subject, count in sorted(incomplete.items())
        )
        raise ValueError(
            "Incomplete SEED processed dataset: expected 10137 pickle samples "
            f"for each of 15 subjects, found {details}. Complete/copy the "
            "preprocessing output before training."
        )
    return subject_files


def _label_for_path(path: Path) -> int:
    match = FILENAME_PATTERN.fullmatch(path.name)
    if match is None:
        raise ValueError(f"Unexpected SEED sample filename: {path}")
    return TRIAL_LABELS[int(match.group("trial")) - 1]


def _make_cross_subject_records(
    subject_files: dict[int, list[Path]],
) -> dict[str, list[tuple[Path, int]]]:
    """Reproduce AdaBrain's seed-42 stratified cross-subject split."""

    rng = random.Random(42)
    train_records: list[tuple[Path, int]] = []
    val_records: list[tuple[Path, int]] = []

    for subject in TRAIN_SUBJECTS:
        # Preserve first-seen label insertion order to match the existing
        # cross_json_process.py manifests exactly.
        by_label: dict[int, list[Path]] = {}
        for path in subject_files[subject]:
            label = _label_for_path(path)
            by_label.setdefault(label, []).append(path)
        for label, paths in by_label.items():
            rng.shuffle(paths)
            split_index = int(len(paths) * 0.8)
            train_records.extend((path, label) for path in paths[:split_index])
            val_records.extend((path, label) for path in paths[split_index:])

    test_records = [
        (path, _label_for_path(path))
        for subject in TEST_SUBJECTS
        for path in subject_files[subject]
    ]
    records = {
        "train": train_records,
        "val": val_records,
        "test": test_records,
    }
    for split, expected in EXPECTED_SPLIT_SAMPLES.items():
        if len(records[split]) != expected:
            raise ValueError(
                f"Unexpected SEED {split} size: got {len(records[split])}, "
                f"expected {expected}"
            )
    return records


class SEEDCrossSubjectLoader(Dataset):
    """Load one deterministic SEED cross-subject split from pickle windows."""

    def __init__(self, records, split: str, channel_names=None):
        self.records = list(records)
        self.split = split
        self.manifest_channel_names = list(SEED_62_CHANNELS)
        if channel_names is None:
            self.channel_names = list(self.manifest_channel_names)
        else:
            self.channel_names = [str(name).upper() for name in channel_names]
            if len(set(self.channel_names)) != len(self.channel_names):
                raise ValueError(f"Duplicate SEED channel names: {self.channel_names}")
            unknown = [
                name for name in self.channel_names
                if name not in self.manifest_channel_names
            ]
            if unknown:
                raise ValueError(f"Unknown SEED channel names: {unknown}")
        self.channel_indices = np.asarray(
            [self.manifest_channel_names.index(name) for name in self.channel_names],
            dtype=np.int64,
        )
        self.label_counts = Counter(label for _, label in self.records)

    def __len__(self):
        return len(self.records)

    def get_ch_names(self):
        return list(self.channel_names)

    def __getitem__(self, index):
        path, expected_label = self.records[index]
        with path.open("rb") as handle:
            sample = pickle.load(handle)
        if not isinstance(sample, dict) or "X" not in sample or "Y" not in sample:
            raise ValueError(f"Invalid SEED pickle schema: {path}")

        x = np.asarray(sample["X"])
        if x.shape != (len(self.manifest_channel_names), 200):
            raise ValueError(
                f"Unexpected SEED sample shape in {path}: got {x.shape}, "
                f"expected ({len(self.manifest_channel_names)}, 200)"
            )
        label = int(sample["Y"])
        if label != expected_label:
            raise ValueError(
                f"SEED label mismatch in {path}: pickle={label}, "
                f"filename/trial={expected_label}"
            )
        x = np.ascontiguousarray(x[self.channel_indices], dtype=np.float32)
        if not np.isfinite(x).all():
            raise ValueError(f"SEED sample contains NaN or Inf: {path}")
        return torch.from_numpy(x), label


def prepare_SEED_cross_subject_dataset(root, channel_names=None):
    """Build train/test/validation datasets using subjects 1-12/13-15."""

    root = Path(root)
    subject_files = _discover_subject_files(root)
    records = _make_cross_subject_records(subject_files)
    datasets = {
        split: SEEDCrossSubjectLoader(
            records[split], split=split, channel_names=channel_names
        )
        for split in ("train", "val", "test")
    }

    expected_channels = datasets["train"].get_ch_names()
    if any(
        dataset.get_ch_names() != expected_channels
        for dataset in datasets.values()
    ):
        raise ValueError("SEED channel order differs across train/val/test")

    print(
        "SEED cross-subject audit: "
        + ", ".join(
            f"{split}={len(dataset)} labels={dict(sorted(dataset.label_counts.items()))}"
            for split, dataset in datasets.items()
        )
    )
    # Keep this repository's established return order.
    return datasets["train"], datasets["test"], datasets["val"]
