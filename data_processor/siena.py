"""AdaBrain-compatible cross-subject reader for Siena seizure detection."""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from Channels_definition import SIENA_29_CHANNELS


TRAIN_VALID_PATIENTS = (
    "PN00", "PN01", "PN03", "PN05", "PN06", "PN07", "PN09",
    "PN10", "PN11", "PN12", "PN13", "PN14",
)
TEST_PATIENTS = ("PN16", "PN17")
ALL_PATIENTS = TRAIN_VALID_PATIENTS + TEST_PATIENTS
EXPECTED_RATE = 200
EXPECTED_SAMPLES = 2000
EXPECTED_SPLIT_COUNTS = {
    "train": {"samples": 38128, "labels": {0: 37652, 1: 476}},
    "val": {"samples": 9542, "labels": {0: 9419, 1: 123}},
    "test": {"samples": 3679, "labels": {0: 3594, 1: 85}},
}
VALID_FRACTION = 0.2
SPLIT_SEED = 42


def _safe_path(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"Siena manifest path escapes dataset root: {value}") from exc
    return path


def _load_manifest(root: Path) -> tuple[dict, dict[str, dict]]:
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Siena manifest not found: {manifest_path}. Run "
            "dataset_maker/make_Siena.py first."
        )
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    expected = {
        "dataset": "Siena",
        "schema_version": 1,
        "target_sampling_rate": EXPECTED_RATE,
        "window_seconds": 10,
        "sample_shape": [len(SIENA_29_CHANNELS), EXPECTED_SAMPLES],
        "dtype": "float32",
        "unit": "microvolt",
        "channels": SIENA_29_CHANNELS,
        "label_names": {"0": "non_seizure", "1": "seizure"},
    }
    for key, expected_value in expected.items():
        if manifest.get(key) != expected_value:
            raise ValueError(
                f"Invalid Siena manifest field {key}: got {manifest.get(key)!r}, "
                f"expected {expected_value!r}"
            )
    records = manifest.get("subject_records")
    if not isinstance(records, list) or not records:
        raise ValueError("Siena manifest has no subject_records")
    by_patient = {}
    for record in records:
        patient = str(record["patient_id"])
        if patient in by_patient:
            raise ValueError(f"Duplicate Siena patient in manifest: {patient}")
        by_patient[patient] = record
    if set(by_patient) != set(ALL_PATIENTS):
        raise ValueError(
            f"Incomplete Siena patients: got {sorted(by_patient)}, "
            f"expected {list(ALL_PATIENTS)}"
        )
    return manifest, by_patient


def _load_patient_arrays(
    root: Path, record: dict
) -> tuple[Path, Path, np.ndarray, np.ndarray]:
    eeg_path = _safe_path(root, record["eeg_file"])
    label_path = _safe_path(root, record["label_file"])
    stats_path = _safe_path(root, record["stats_file"])
    for path in (eeg_path, label_path, stats_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing Siena array: {path}")
    eeg = np.load(eeg_path, mmap_mode="r", allow_pickle=False)
    labels = np.load(label_path, mmap_mode="r", allow_pickle=False)
    stats = np.load(stats_path, mmap_mode="r", allow_pickle=False)
    samples = int(record["samples"])
    if eeg.shape != (samples, len(SIENA_29_CHANNELS), EXPECTED_SAMPLES):
        raise ValueError(f"Invalid Siena EEG shape: {eeg_path} {eeg.shape}")
    if eeg.dtype != np.float32:
        raise ValueError(f"Invalid Siena EEG dtype: {eeg_path} {eeg.dtype}")
    if labels.shape != (samples,) or labels.dtype != np.int64:
        raise ValueError(f"Invalid Siena labels: {label_path} {labels.shape} {labels.dtype}")
    if (
        stats.shape not in (
            (samples, len(SIENA_29_CHANNELS), 2),
            (samples, len(SIENA_29_CHANNELS), 4),
        )
        or stats.dtype != np.float64
    ):
        raise ValueError(f"Invalid Siena statistics: {stats_path} {stats.shape} {stats.dtype}")
    counts = Counter(int(value) for value in np.asarray(labels))
    expected_counts = Counter(
        {int(key): int(value) for key, value in record["label_counts"].items()}
    )
    if counts != expected_counts or set(counts) - {0, 1}:
        raise ValueError(f"Siena labels disagree with manifest for {record['patient_id']}")
    return eeg_path, stats_path, labels, stats


def _make_records(
    root: Path, by_patient: dict[str, dict]
) -> dict[str, list[tuple[Path, Path, int, int, str]]]:
    rng = random.Random(SPLIT_SEED)
    output = {"train": [], "val": [], "test": []}
    for patient in TRAIN_VALID_PATIENTS:
        eeg_path, stats_path, labels, _ = _load_patient_arrays(
            root, by_patient[patient]
        )
        # Preserve AdaBrain's first-seen class insertion order and one global
        # seed-42 RNG stream across patients.
        by_label: dict[int, list[int]] = {}
        for trial_index, value in enumerate(labels):
            label = int(value)
            by_label.setdefault(label, []).append(trial_index)
        if set(by_label) != {0, 1}:
            raise ValueError(f"Siena training patient {patient} lacks a class")
        for label, indices in by_label.items():
            rng.shuffle(indices)
            train_count = int(len(indices) * (1.0 - VALID_FRACTION))
            output["train"].extend(
                (eeg_path, stats_path, index, label, patient)
                for index in indices[:train_count]
            )
            output["val"].extend(
                (eeg_path, stats_path, index, label, patient)
                for index in indices[train_count:]
            )
    for patient in TEST_PATIENTS:
        eeg_path, stats_path, labels, _ = _load_patient_arrays(
            root, by_patient[patient]
        )
        output["test"].extend(
            (eeg_path, stats_path, index, int(label), patient)
            for index, label in enumerate(labels)
        )
    for split, expected in EXPECTED_SPLIT_COUNTS.items():
        counts = Counter(record[3] for record in output[split])
        if len(output[split]) != expected["samples"] or counts != Counter(
            expected["labels"]
        ):
            raise ValueError(
                f"Siena {split} differs from AdaBrain benchmark: "
                f"samples={len(output[split])}, labels={dict(counts)}, expected={expected}"
            )
    train_patients = {record[4] for record in output["train"]}
    val_patients = {record[4] for record in output["val"]}
    test_patients = {record[4] for record in output["test"]}
    if train_patients & test_patients or val_patients & test_patients:
        raise ValueError("Siena test patients overlap train/validation patients")
    return output


def _training_statistics(
    records: list[tuple[Path, Path, int, int, str]],
) -> tuple[np.ndarray, np.ndarray]:
    by_stats_path: dict[Path, list[int]] = defaultdict(list)
    for _, stats_path, trial_index, _, _ in records:
        by_stats_path[stats_path].append(trial_index)

    first_stats_path = next(iter(by_stats_path))
    first_stats = np.load(first_stats_path, mmap_mode="r", allow_pickle=False)
    if first_stats.shape[-1] == 4:
        mean_sum = np.zeros(len(SIENA_29_CHANNELS), dtype=np.float64)
        std_sum = np.zeros(len(SIENA_29_CHANNELS), dtype=np.float64)
        total_windows = 0
        for stats_path, indices in by_stats_path.items():
            stats = np.load(stats_path, mmap_mode="r", allow_pickle=False)
            if stats.shape[-1] != 4:
                raise ValueError("Mixed Siena statistics schemas are not supported")
            for start in range(0, len(indices), 1024):
                selected = np.asarray(stats[indices[start : start + 1024]])
                mean_sum += np.sum(selected[:, :, 2], axis=0, dtype=np.float64)
                std_sum += np.sum(selected[:, :, 3], axis=0, dtype=np.float64)
            total_windows += len(indices)
        mean = mean_sum / total_windows
        std = std_sum / total_windows
        if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
            raise ValueError("Invalid Siena train-only AdaBrain normalization statistics")
        return mean, std

    channel_sum = np.zeros(len(SIENA_29_CHANNELS), dtype=np.float64)
    channel_squared_sum = np.zeros(len(SIENA_29_CHANNELS), dtype=np.float64)
    total_points = 0
    for stats_path, indices in by_stats_path.items():
        stats = np.load(stats_path, mmap_mode="r", allow_pickle=False)
        for start in range(0, len(indices), 1024):
            selected = np.asarray(stats[indices[start : start + 1024]])
            channel_sum += np.sum(selected[:, :, 0], axis=0, dtype=np.float64)
            channel_squared_sum += np.sum(
                selected[:, :, 1], axis=0, dtype=np.float64
            )
        total_points += len(indices) * EXPECTED_SAMPLES
    mean = channel_sum / total_points
    variance = channel_squared_sum / total_points - np.square(mean)
    std = np.sqrt(np.maximum(variance, 0.0))
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("Invalid Siena train-only normalization statistics")
    return mean, std


class SienaCrossSubjectDataset(Dataset):
    """Lazy Siena split using benchmark patient isolation and train statistics."""

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
        self.manifest_channel_names = list(SIENA_29_CHANNELS)
        self.channel_names = (
            list(self.manifest_channel_names)
            if channel_names is None
            else [str(name).upper() for name in channel_names]
        )
        if len(set(self.channel_names)) != len(self.channel_names):
            raise ValueError(f"Duplicate Siena channel names: {self.channel_names}")
        unknown = [
            name for name in self.channel_names if name not in self.manifest_channel_names
        ]
        if unknown:
            raise ValueError(f"Unknown Siena channel names: {unknown}")
        self.channel_indices = np.asarray(
            [self.manifest_channel_names.index(name) for name in self.channel_names],
            dtype=np.int64,
        )
        self.mean = np.asarray(mean, dtype=np.float64)[self.channel_indices]
        self.std = np.asarray(std, dtype=np.float64)[self.channel_indices]
        self.label_counts = Counter(record[3] for record in self.records)
        self.patients = tuple(sorted({record[4] for record in self.records}))
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
        raise ValueError(f"Unsupported Siena normalization: {self.normalize_method}")

    def __getitem__(self, index):
        eeg_path, _, trial_index, label, _ = self.records[index]
        eeg = self._array(eeg_path)[trial_index, self.channel_indices]
        eeg = self._normalize(np.asarray(eeg))
        eeg = np.ascontiguousarray(eeg, dtype=np.float32)
        if eeg.shape != (len(self.channel_names), EXPECTED_SAMPLES):
            raise ValueError(f"Unexpected Siena sample shape: {eeg.shape}")
        if not np.isfinite(eeg).all():
            raise ValueError(f"NaN or Inf in normalized Siena sample: {eeg_path}")
        return torch.from_numpy(eeg), label


def prepare_Siena_cross_subject_dataset(
    root,
    sampling_rate=200,
    normalize_method="z_score",
    channel_names=None,
):
    """Reproduce AdaBrain's PN00--14 train/val and PN16--17 test split."""

    if int(sampling_rate) != EXPECTED_RATE:
        raise ValueError(
            f"Siena arrays are fixed at {EXPECTED_RATE} Hz; rerun preprocessing "
            "for another sampling rate"
        )
    root = Path(root).resolve()
    _, by_patient = _load_manifest(root)
    records = _make_records(root, by_patient)
    mean, std = _training_statistics(records["train"])
    datasets = {
        split: SienaCrossSubjectDataset(
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
        raise ValueError("Siena channel order differs across train/val/test")
    print(
        "Siena cross-subject audit: "
        + ", ".join(
            f"{split}=patients{dataset.patients}/samples{len(dataset)}/"
            f"labels{dict(sorted(dataset.label_counts.items()))}"
            for split, dataset in datasets.items()
        )
        + f", sampling_rate={EXPECTED_RATE}, window=10s, "
        f"normalization={normalize_method}"
    )
    return datasets["train"], datasets["test"], datasets["val"]
