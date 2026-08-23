"""Zuo2025 4-second cross-subject dataset reader."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from Channels_definition import ZUO2025_30_CHANNELS


EXPECTED_SUBJECTS = tuple(range(1, 31))
EXPECTED_TOTAL_TRIALS = 14034
EXPECTED_RATE = 200
EXPECTED_SAMPLES = 800
TRAIN_SUBJECTS = tuple(range(1, 25))
VAL_SUBJECTS = (25, 26, 27)
TEST_SUBJECTS = (28, 29, 30)


def _safe_manifest_path(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"Manifest path escapes dataset root: {value}") from exc
    return path


def _load_manifest(root: Path) -> tuple[dict, dict[int, dict]]:
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Zuo2025 manifest not found: {manifest_path}. Run "
            "dataset_maker/make_Zuo2025.py first."
        )
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    expected_values = {
        "dataset": "Zuo2025",
        "schema_version": 1,
        "target_sampling_rate": EXPECTED_RATE,
        "sample_shape": [len(ZUO2025_30_CHANNELS), EXPECTED_SAMPLES],
        "dtype": "float32",
        "unit": "microvolt",
        "selected_seconds_relative_to_cue": [0, 4],
    }
    for key, expected in expected_values.items():
        if manifest.get(key) != expected:
            raise ValueError(
                f"Invalid Zuo2025 manifest field {key}: "
                f"got {manifest.get(key)!r}, expected {expected!r}"
            )
    channels = [str(value).upper() for value in manifest.get("channels", [])]
    if channels != ZUO2025_30_CHANNELS:
        raise ValueError(f"Unexpected Zuo2025 channel order: {channels}")
    records = manifest.get("subject_records")
    if not isinstance(records, list) or not records:
        raise ValueError("Zuo2025 manifest has no subject_records")
    by_subject: dict[int, dict] = {}
    for record in records:
        subject = int(record["subject_id"])
        if subject in by_subject:
            raise ValueError(f"Duplicate Zuo2025 subject in manifest: {subject}")
        by_subject[subject] = record
    return manifest, by_subject


def _validate_splits(
    available: set[int],
    train_subjects: tuple[int, ...],
    val_subjects: tuple[int, ...],
    test_subjects: tuple[int, ...],
) -> None:
    split_sets = [set(train_subjects), set(val_subjects), set(test_subjects)]
    names = ["train", "val", "test"]
    for name, values in zip(names, split_sets):
        if not values:
            raise ValueError(f"Zuo2025 {name} subject split is empty")
        missing = values - available
        if missing:
            raise ValueError(f"Zuo2025 {name} subjects missing: {sorted(missing)}")
    if split_sets[0] & split_sets[1] or split_sets[0] & split_sets[2] or split_sets[1] & split_sets[2]:
        raise ValueError("Zuo2025 train/val/test subject splits overlap")


def _training_statistics(
    by_subject: dict[int, dict], train_subjects: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray]:
    channel_count = len(ZUO2025_30_CHANNELS)
    total_sum = np.zeros(channel_count, dtype=np.float64)
    total_squared_sum = np.zeros(channel_count, dtype=np.float64)
    total_points = 0
    for subject in train_subjects:
        record = by_subject[subject]
        channel_sum = np.asarray(record["channel_sum"], dtype=np.float64)
        channel_squared_sum = np.asarray(
            record["channel_squared_sum"], dtype=np.float64
        )
        if channel_sum.shape != (channel_count,) or channel_squared_sum.shape != (
            channel_count,
        ):
            raise ValueError(f"Invalid normalization statistics for subject {subject}")
        total_sum += channel_sum
        total_squared_sum += channel_squared_sum
        total_points += int(record["sample_points_per_channel"])
    mean = total_sum / total_points
    variance = total_squared_sum / total_points - np.square(mean)
    variance = np.maximum(variance, 0.0)
    std = np.sqrt(variance)
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("Invalid Zuo2025 training normalization statistics")
    return mean, std


def _subject_records(
    root: Path, by_subject: dict[int, dict], subjects: tuple[int, ...]
) -> list[tuple[Path, int, int, int]]:
    output: list[tuple[Path, int, int, int]] = []
    for subject in subjects:
        record = by_subject[subject]
        eeg_path = _safe_manifest_path(root, record["eeg_file"])
        label_path = _safe_manifest_path(root, record["label_file"])
        if not eeg_path.is_file() or not label_path.is_file():
            raise FileNotFoundError(
                f"Missing Zuo2025 arrays for subject {subject}: {eeg_path}, {label_path}"
            )
        eeg = np.load(eeg_path, mmap_mode="r", allow_pickle=False)
        labels = np.load(label_path, mmap_mode="r", allow_pickle=False)
        trials = int(record["trials"])
        expected_shape = (trials, len(ZUO2025_30_CHANNELS), EXPECTED_SAMPLES)
        if eeg.shape != expected_shape or eeg.dtype != np.float32:
            raise ValueError(f"Invalid Zuo2025 EEG array {eeg_path}: {eeg.shape}, {eeg.dtype}")
        if labels.shape != (trials,) or labels.dtype != np.int64:
            raise ValueError(
                f"Invalid Zuo2025 label array {label_path}: {labels.shape}, {labels.dtype}"
            )
        label_counts = Counter(int(value) for value in np.asarray(labels))
        expected_counts = {
            int(key): int(value) for key, value in record["label_counts"].items()
        }
        if label_counts != Counter(expected_counts):
            raise ValueError(f"Label counts disagree with manifest for subject {subject}")
        if set(label_counts) - {0, 1}:
            raise ValueError(f"Unknown Zuo2025 class ids for subject {subject}")
        output.extend(
            (eeg_path, trial_index, int(label), subject)
            for trial_index, label in enumerate(labels)
        )
    return output


class Zuo2025CrossSubjectLoader(Dataset):
    """Lazily memory-map one subject-disjoint Zuo2025 split."""

    def __init__(
        self,
        records,
        split: str,
        mean: np.ndarray,
        std: np.ndarray,
        normalize_method: str = "z_score",
        channel_names=None,
    ):
        self.records = list(records)
        self.split = split
        self.normalize_method = normalize_method
        self.manifest_channel_names = list(ZUO2025_30_CHANNELS)
        self.channel_names = (
            list(self.manifest_channel_names)
            if channel_names is None
            else [str(name).upper() for name in channel_names]
        )
        if len(set(self.channel_names)) != len(self.channel_names):
            raise ValueError(f"Duplicate Zuo2025 channel names: {self.channel_names}")
        unknown = [
            name for name in self.channel_names if name not in self.manifest_channel_names
        ]
        if unknown:
            raise ValueError(f"Unknown Zuo2025 channel names: {unknown}")
        self.channel_indices = np.asarray(
            [self.manifest_channel_names.index(name) for name in self.channel_names],
            dtype=np.int64,
        )
        self.mean = np.asarray(mean, dtype=np.float64)[self.channel_indices]
        self.std = np.asarray(std, dtype=np.float64)[self.channel_indices]
        self.label_counts = Counter(label for _, _, label, _ in self.records)
        self.subjects = tuple(sorted({subject for _, _, _, subject in self.records}))
        self._arrays: dict[Path, np.ndarray] = {}

    def __len__(self):
        return len(self.records)

    def get_ch_names(self):
        return list(self.channel_names)

    def _array(self, path: Path) -> np.ndarray:
        if path not in self._arrays:
            self._arrays[path] = np.load(path, mmap_mode="r", allow_pickle=False)
        return self._arrays[path]

    def _normalize(self, eeg: np.ndarray) -> np.ndarray:
        if self.normalize_method == "z_score":
            return (eeg - self.mean[:, None]) / (self.std[:, None] + 1e-8)
        if self.normalize_method == "0.1mv":
            # Source derivatives are in microvolts: 100 uV equals one 0.1 mV unit.
            return eeg / 100.0
        if self.normalize_method == "95":
            scale = np.quantile(np.abs(eeg), 0.95, axis=-1, keepdims=True)
            return eeg / (scale + 1e-8)
        raise ValueError(f"Unsupported Zuo2025 normalization: {self.normalize_method}")

    def __getitem__(self, index):
        eeg_path, trial_index, label, _ = self.records[index]
        eeg = self._array(eeg_path)[trial_index, self.channel_indices]
        eeg = self._normalize(np.asarray(eeg))
        eeg = np.ascontiguousarray(eeg, dtype=np.float32)
        if eeg.shape != (len(self.channel_names), EXPECTED_SAMPLES):
            raise ValueError(f"Unexpected Zuo2025 sample shape: {eeg.shape}")
        if not np.isfinite(eeg).all():
            raise ValueError(f"NaN or Inf in normalized Zuo2025 sample: {eeg_path}")
        return torch.from_numpy(eeg), label


def prepare_Zuo2025_cross_subject_dataset(
    root,
    sampling_rate=200,
    normalize_method="z_score",
    channel_names=None,
    train_subjects=TRAIN_SUBJECTS,
    val_subjects=VAL_SUBJECTS,
    test_subjects=TEST_SUBJECTS,
):
    """Build a configurable subject-disjoint 24/3/3 split.

    The default subject ids are a deterministic initial protocol, not a claim
    that the Zuo2025 paper defines this exact benchmark.  Pass explicit subject
    tuples when reproducing another published split.
    """

    if int(sampling_rate) != EXPECTED_RATE:
        raise ValueError(
            f"Zuo2025 files are fixed at {EXPECTED_RATE} Hz after preprocessing; "
            "rerun the maker for another rate instead of resampling in the loader"
        )
    root = Path(root).resolve()
    manifest, by_subject = _load_manifest(root)
    available = set(by_subject)
    if available == set(EXPECTED_SUBJECTS):
        if int(manifest["total_trials"]) != EXPECTED_TOTAL_TRIALS:
            raise ValueError(
                f"Expected {EXPECTED_TOTAL_TRIALS} Zuo2025 trials, "
                f"found {manifest['total_trials']}"
            )
    train_subjects = tuple(int(value) for value in train_subjects)
    val_subjects = tuple(int(value) for value in val_subjects)
    test_subjects = tuple(int(value) for value in test_subjects)
    _validate_splits(available, train_subjects, val_subjects, test_subjects)
    mean, std = _training_statistics(by_subject, train_subjects)
    split_subjects = {
        "train": train_subjects,
        "val": val_subjects,
        "test": test_subjects,
    }
    datasets = {
        split: Zuo2025CrossSubjectLoader(
            _subject_records(root, by_subject, subjects),
            split=split,
            mean=mean,
            std=std,
            normalize_method=normalize_method,
            channel_names=channel_names,
        )
        for split, subjects in split_subjects.items()
    }
    expected_channels = datasets["train"].get_ch_names()
    if any(dataset.get_ch_names() != expected_channels for dataset in datasets.values()):
        raise ValueError("Zuo2025 channel order differs across train/val/test")
    print(
        "Zuo2025 cross-subject audit: "
        + ", ".join(
            f"{split}=subjects{dataset.subjects}/trials{len(dataset)}/"
            f"labels{dict(sorted(dataset.label_counts.items()))}"
            for split, dataset in datasets.items()
        )
        + f", sampling_rate={EXPECTED_RATE}, normalization={normalize_method}"
    )
    return datasets["train"], datasets["test"], datasets["val"]
