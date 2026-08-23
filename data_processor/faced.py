"""Reader for FACED 32-channel emotion windows."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from Channels_definition import FACED_32_CHANNELS


EXPECTED_SUBJECTS = tuple(range(123))
EXPECTED_RATE = 200
EXPECTED_SAMPLES = 2000
TRAIN_SUBJECTS = tuple(range(80))
VAL_SUBJECTS = tuple(range(80, 100))
TEST_SUBJECTS = tuple(range(100, 123))


def _safe_manifest_path(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"FACED manifest path escapes dataset root: {value}") from exc
    return path


def _load_manifest(root: Path) -> tuple[dict, dict[int, dict]]:
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"FACED manifest not found: {manifest_path}. Run dataset_maker/make_FACED.py first."
        )
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    expected_values = {
        "dataset": "FACED",
        "schema_version": 1,
        "target_sampling_rate": EXPECTED_RATE,
        "sample_shape": [len(FACED_32_CHANNELS), EXPECTED_SAMPLES],
        "dtype": "float32",
        "label_mode": "emotion9",
    }
    for key, expected in expected_values.items():
        if manifest.get(key) != expected:
            raise ValueError(
                f"Invalid FACED manifest field {key}: got {manifest.get(key)!r}, "
                f"expected {expected!r}"
            )
    channels = [str(value).strip().upper() for value in manifest.get("channels", [])]
    if channels != FACED_32_CHANNELS:
        raise ValueError(f"Unexpected FACED channel order: {channels}")
    records = manifest.get("subject_records")
    if not isinstance(records, list) or not records:
        raise ValueError("FACED manifest has no subject_records")
    by_subject: dict[int, dict] = {}
    for record in records:
        subject = int(record["subject_id"])
        if subject in by_subject:
            raise ValueError(f"Duplicate FACED subject in manifest: {subject}")
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
            raise ValueError(f"FACED {name} subject split is empty")
        missing = values - available
        if missing:
            raise ValueError(f"FACED {name} subjects missing: {sorted(missing)}")
    if split_sets[0] & split_sets[1] or split_sets[0] & split_sets[2] or split_sets[1] & split_sets[2]:
        raise ValueError("FACED train/val/test subject splits overlap")


def _training_statistics(
    by_subject: dict[int, dict], train_subjects: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray]:
    channel_count = len(FACED_32_CHANNELS)
    total_sum = np.zeros(channel_count, dtype=np.float64)
    total_squared_sum = np.zeros(channel_count, dtype=np.float64)
    total_points = 0
    for subject in train_subjects:
        record = by_subject[subject]
        channel_sum = np.asarray(record["channel_sum"], dtype=np.float64)
        channel_squared_sum = np.asarray(record["channel_squared_sum"], dtype=np.float64)
        if channel_sum.shape != (channel_count,) or channel_squared_sum.shape != (channel_count,):
            raise ValueError(f"Invalid FACED normalization statistics for subject {subject}")
        total_sum += channel_sum
        total_squared_sum += channel_squared_sum
        total_points += int(record["sample_points_per_channel"])
    mean = total_sum / total_points
    variance = total_squared_sum / total_points - np.square(mean)
    std = np.sqrt(np.maximum(variance, 0.0))
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("Invalid FACED training normalization statistics")
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
                f"Missing FACED arrays for subject {subject}: {eeg_path}, {label_path}"
            )
        eeg = np.load(eeg_path, mmap_mode="r", allow_pickle=False)
        labels = np.load(label_path, mmap_mode="r", allow_pickle=False)
        trials = int(record["trials"])
        expected_shape = (trials, len(FACED_32_CHANNELS), EXPECTED_SAMPLES)
        if eeg.shape != expected_shape or eeg.dtype != np.float32:
            raise ValueError(f"Invalid FACED EEG array {eeg_path}: {eeg.shape}, {eeg.dtype}")
        if labels.shape != (trials,) or labels.dtype != np.int64:
            raise ValueError(f"Invalid FACED labels {label_path}: {labels.shape}, {labels.dtype}")
        label_counts = Counter(int(value) for value in np.asarray(labels))
        expected_counts = {
            int(key): int(value) for key, value in record["label_counts"].items()
        }
        if label_counts != Counter(expected_counts):
            raise ValueError(f"FACED label counts disagree with manifest for subject {subject}")
        if set(label_counts) - set(range(9)):
            raise ValueError(f"Unknown FACED class ids for subject {subject}")
        output.extend(
            (eeg_path, trial_index, int(label), subject)
            for trial_index, label in enumerate(labels)
        )
    return output


class FACEDCrossSubjectLoader(Dataset):
    """Lazy memory-mapped FACED split with train-only normalization."""

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
        self.manifest_channel_names = list(FACED_32_CHANNELS)
        self.channel_names = (
            list(self.manifest_channel_names)
            if channel_names is None
            else [str(name).strip().upper() for name in channel_names]
        )
        if len(set(self.channel_names)) != len(self.channel_names):
            raise ValueError(f"Duplicate FACED channel names: {self.channel_names}")
        unknown = [
            name for name in self.channel_names if name not in self.manifest_channel_names
        ]
        if unknown:
            raise ValueError(f"Unknown FACED channel names: {unknown}")
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
            return eeg / 100.0
        if self.normalize_method == "95":
            scale = np.quantile(np.abs(eeg), 0.95, axis=-1, keepdims=True)
            return eeg / (scale + 1e-8)
        raise ValueError(f"Unsupported FACED normalization: {self.normalize_method}")

    def __getitem__(self, index):
        eeg_path, trial_index, label, _ = self.records[index]
        eeg = self._array(eeg_path)[trial_index, self.channel_indices]
        eeg = self._normalize(np.asarray(eeg))
        eeg = np.ascontiguousarray(eeg, dtype=np.float32)
        if eeg.shape != (len(self.channel_names), EXPECTED_SAMPLES):
            raise ValueError(f"Unexpected FACED sample shape: {eeg.shape}")
        if not np.isfinite(eeg).all():
            raise ValueError(f"NaN or Inf in normalized FACED sample: {eeg_path}")
        return torch.from_numpy(eeg), label


def prepare_FACED_cross_subject_dataset(
    root,
    sampling_rate=200,
    normalize_method="z_score",
    channel_names=None,
    train_subjects=TRAIN_SUBJECTS,
    val_subjects=VAL_SUBJECTS,
    test_subjects=TEST_SUBJECTS,
):
    """Build CBraMod's subject-disjoint 80/20/23 FACED split."""

    if int(sampling_rate) != EXPECTED_RATE:
        raise ValueError(
            f"FACED files are fixed at {EXPECTED_RATE} Hz after preprocessing; "
            "rerun the maker for another rate instead of resampling in the loader"
        )
    root = Path(root).resolve()
    _, by_subject = _load_manifest(root)
    available = set(by_subject)
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
        split: FACEDCrossSubjectLoader(
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
        raise ValueError("FACED channel order differs across train/val/test")
    print(
        "FACED cross-subject audit: "
        + ", ".join(
            f"{split}=subjects{dataset.subjects}/trials{len(dataset)}/"
            f"labels{dict(sorted(dataset.label_counts.items()))}"
            for split, dataset in datasets.items()
        )
        + f", sampling_rate={EXPECTED_RATE}, normalization={normalize_method}"
    )
    return datasets["train"], datasets["test"], datasets["val"]
