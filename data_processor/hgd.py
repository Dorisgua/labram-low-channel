"""Reader for the preprocessed Schirrmeister2017 High Gamma Dataset."""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from Channels_definition import HGD_78_CHANNELS


EXPECTED_SUBJECTS = tuple(range(1, 15))
EXPECTED_RATE = 200
EXPECTED_SAMPLES = 800
EXPECTED_CLASSES = {0, 1, 2, 3}
DEFAULT_VALID_FRACTION = 0.2
DEFAULT_SPLIT_SEED = 42


def _safe_path(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"HGD manifest path escapes dataset root: {value}") from exc
    return path


def _load_manifest(root: Path) -> tuple[dict, dict[int, dict]]:
    path = root / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"HGD manifest not found: {path}. Run dataset_maker/make_HGD.py first."
        )
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    expected = {
        "dataset": "Schirrmeister2017-HGD",
        "schema_version": 1,
        "target_sampling_rate": EXPECTED_RATE,
        "selected_seconds_relative_to_annotation": [0, 4],
        "sample_shape": [len(HGD_78_CHANNELS), EXPECTED_SAMPLES],
        "dtype": "float32",
        "unit": "microvolt",
    }
    for key, expected_value in expected.items():
        if manifest.get(key) != expected_value:
            raise ValueError(
                f"Invalid HGD manifest field {key}: got {manifest.get(key)!r}, "
                f"expected {expected_value!r}"
            )
    if manifest.get("selected_channels") != HGD_78_CHANNELS:
        raise ValueError("HGD manifest channel order differs from HGD_78_CHANNELS")
    records = manifest.get("subject_records")
    if not isinstance(records, list) or not records:
        raise ValueError("HGD manifest has no subject_records")
    by_subject = {}
    for record in records:
        subject = int(record["subject_id"])
        if subject in by_subject:
            raise ValueError(f"Duplicate HGD subject in manifest: {subject}")
        if set(record.get("splits", {})) != {"train", "test"}:
            raise ValueError(f"HGD subject {subject} lacks official train/test splits")
        by_subject[subject] = record
    return manifest, by_subject


def _load_subject_split(
    root: Path, record: dict, split: str
) -> tuple[Path, np.ndarray]:
    split_record = record["splits"][split]
    eeg_path = _safe_path(root, split_record["eeg_file"])
    label_path = _safe_path(root, split_record["label_file"])
    if not eeg_path.is_file() or not label_path.is_file():
        raise FileNotFoundError(f"Missing HGD arrays: {eeg_path}, {label_path}")
    eeg = np.load(eeg_path, mmap_mode="r", allow_pickle=False)
    labels = np.load(label_path, mmap_mode="r", allow_pickle=False)
    expected_trials = int(split_record["kept_trials"])
    if eeg.shape != (expected_trials, len(HGD_78_CHANNELS), EXPECTED_SAMPLES):
        raise ValueError(f"Invalid HGD EEG shape: {eeg_path} {eeg.shape}")
    if eeg.dtype != np.float32:
        raise ValueError(f"Invalid HGD EEG dtype: {eeg_path} {eeg.dtype}")
    if labels.shape != (expected_trials,) or labels.dtype != np.int64:
        raise ValueError(f"Invalid HGD labels: {label_path} {labels.shape} {labels.dtype}")
    counts = Counter(int(value) for value in np.asarray(labels))
    expected_counts = Counter(
        {int(key): int(value) for key, value in split_record["label_counts"].items()}
    )
    if counts != expected_counts or set(counts) - EXPECTED_CLASSES:
        raise ValueError(f"HGD labels disagree with manifest: {label_path}")
    return eeg_path, labels


def _make_records(
    root: Path,
    by_subject: dict[int, dict],
    valid_fraction: float,
    split_seed: int,
) -> dict[str, list[tuple[Path, int, int, int, str]]]:
    if not 0.0 < valid_fraction < 1.0:
        raise ValueError(f"valid_fraction must be within (0, 1), got {valid_fraction}")
    rng = random.Random(split_seed)
    output = {"train": [], "val": [], "test": []}
    for subject in sorted(by_subject):
        official_train_path, train_labels = _load_subject_split(
            root, by_subject[subject], "train"
        )
        by_label: dict[int, list[int]] = defaultdict(list)
        for trial_index, label in enumerate(train_labels):
            by_label[int(label)].append(trial_index)
        if set(by_label) != EXPECTED_CLASSES:
            raise ValueError(f"HGD subject {subject} official train lacks a class")
        for label in sorted(by_label):
            indices = by_label[label]
            rng.shuffle(indices)
            train_count = int(len(indices) * (1.0 - valid_fraction))
            if train_count <= 0 or train_count >= len(indices):
                raise ValueError(
                    f"Cannot split HGD subject {subject} class {label}: {len(indices)}"
                )
            output["train"].extend(
                (official_train_path, index, label, subject, "train")
                for index in indices[:train_count]
            )
            output["val"].extend(
                (official_train_path, index, label, subject, "train")
                for index in indices[train_count:]
            )

        official_test_path, test_labels = _load_subject_split(
            root, by_subject[subject], "test"
        )
        output["test"].extend(
            (official_test_path, index, int(label), subject, "test")
            for index, label in enumerate(test_labels)
        )
    train_keys = {(path, index) for path, index, *_ in output["train"]}
    val_keys = {(path, index) for path, index, *_ in output["val"]}
    test_keys = {(path, index) for path, index, *_ in output["test"]}
    if train_keys & val_keys or train_keys & test_keys or val_keys & test_keys:
        raise ValueError("HGD train/validation/test trial records overlap")
    return output


def _training_statistics(
    records: list[tuple[Path, int, int, int, str]],
) -> tuple[np.ndarray, np.ndarray]:
    by_path: dict[Path, list[int]] = defaultdict(list)
    for path, trial_index, *_ in records:
        by_path[path].append(trial_index)
    channel_sum = np.zeros(len(HGD_78_CHANNELS), dtype=np.float64)
    channel_squared_sum = np.zeros(len(HGD_78_CHANNELS), dtype=np.float64)
    sample_points = 0
    for path, indices in by_path.items():
        eeg = np.load(path, mmap_mode="r", allow_pickle=False)
        for start in range(0, len(indices), 64):
            batch_indices = indices[start : start + 64]
            batch = np.asarray(eeg[batch_indices], dtype=np.float64)
            channel_sum += np.sum(batch, axis=(0, 2), dtype=np.float64)
            channel_squared_sum += np.sum(
                np.square(batch), axis=(0, 2), dtype=np.float64
            )
            sample_points += batch.shape[0] * batch.shape[2]
    mean = channel_sum / sample_points
    variance = channel_squared_sum / sample_points - np.square(mean)
    std = np.sqrt(np.maximum(variance, 0.0))
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("Invalid HGD training normalization statistics")
    return mean, std


class HGDDataset(Dataset):
    """Lazy memory-mapped HGD split with train-only normalization."""

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
        self.manifest_channel_names = list(HGD_78_CHANNELS)
        self.channel_names = (
            list(self.manifest_channel_names)
            if channel_names is None
            else [str(name) for name in channel_names]
        )
        if len(set(self.channel_names)) != len(self.channel_names):
            raise ValueError(f"Duplicate HGD channel names: {self.channel_names}")
        unknown = [
            name for name in self.channel_names if name not in self.manifest_channel_names
        ]
        if unknown:
            raise ValueError(f"Unknown HGD channel names: {unknown}")
        self.channel_indices = np.asarray(
            [self.manifest_channel_names.index(name) for name in self.channel_names],
            dtype=np.int64,
        )
        self.mean = np.asarray(mean, dtype=np.float64)[self.channel_indices]
        self.std = np.asarray(std, dtype=np.float64)[self.channel_indices]
        self.label_counts = Counter(label for _, _, label, _, _ in self.records)
        self.subjects = tuple(sorted({subject for _, _, _, subject, _ in self.records}))
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
        raise ValueError(f"Unsupported HGD normalization: {self.normalize_method}")

    def __getitem__(self, index):
        path, trial_index, label, _, _ = self.records[index]
        eeg = self._array(path)[trial_index, self.channel_indices]
        eeg = self._normalize(np.asarray(eeg))
        eeg = np.ascontiguousarray(eeg, dtype=np.float32)
        if eeg.shape != (len(self.channel_names), EXPECTED_SAMPLES):
            raise ValueError(f"Unexpected HGD sample shape: {eeg.shape}")
        if not np.isfinite(eeg).all():
            raise ValueError(f"NaN or Inf in normalized HGD sample: {path}")
        return torch.from_numpy(eeg), label


def prepare_HGD_official_dataset(
    root,
    sampling_rate=200,
    normalize_method="z_score",
    channel_names=None,
    valid_fraction=DEFAULT_VALID_FRACTION,
    split_seed=DEFAULT_SPLIT_SEED,
):
    """Pool subjects while preserving official recording-level test files.

    Validation is a deterministic class-stratified fraction of each subject's
    official train file.  This is not a cross-subject protocol: all 14 subjects
    occur in train, validation and official test.
    """

    if int(sampling_rate) != EXPECTED_RATE:
        raise ValueError(
            f"HGD arrays are fixed at {EXPECTED_RATE} Hz; rerun preprocessing "
            "for another sampling rate"
        )
    root = Path(root).resolve()
    _, by_subject = _load_manifest(root)
    if set(by_subject) != set(EXPECTED_SUBJECTS):
        raise ValueError(
            f"Incomplete HGD subject set: got {sorted(by_subject)}, "
            f"expected {list(EXPECTED_SUBJECTS)}"
        )
    records = _make_records(root, by_subject, valid_fraction, int(split_seed))
    mean, std = _training_statistics(records["train"])
    datasets = {
        split: HGDDataset(
            records[split],
            split=split,
            mean=mean,
            std=std,
            normalize_method=normalize_method,
            channel_names=channel_names,
        )
        for split in ("train", "val", "test")
    }
    expected_channels = datasets["train"].get_ch_names()
    if any(dataset.get_ch_names() != expected_channels for dataset in datasets.values()):
        raise ValueError("HGD channel order differs across train/val/test")
    print(
        "HGD official-split audit: "
        + ", ".join(
            f"{split}=trials{len(dataset)}/labels{dict(sorted(dataset.label_counts.items()))}"
            for split, dataset in datasets.items()
        )
        + f", subjects=1-14 in every split, validation_fraction={valid_fraction}, "
        f"split_seed={split_seed}, sampling_rate={EXPECTED_RATE}, "
        f"normalization={normalize_method}"
    )
    return datasets["train"], datasets["test"], datasets["val"]
