"""Reader for the processed FATIG pickle dataset with EEG-Deformer LOSO splits."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from Channels_definition import FATIG_30_CHANNELS


EXPECTED_RATE = 200
WINDOW_SECONDS = 3
EXPECTED_SAMPLES = EXPECTED_RATE * WINDOW_SECONDS


def _safe_path(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"Fatig manifest path escapes dataset root: {value}") from exc
    return path


def _load_manifest(root: Path) -> tuple[dict, dict[str, dict]]:
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Fatig manifest not found: {manifest_path}. Run "
            "dataset_maker/make_Fatig.py first."
        )
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    expected = {
        "dataset": "Fatig",
        "schema_version": 2,
        "target_sampling_rate": EXPECTED_RATE,
        "window_seconds": WINDOW_SECONDS,
        "sample_shape": [len(FATIG_30_CHANNELS), EXPECTED_SAMPLES],
        "dtype": "float32",
        "channels": FATIG_30_CHANNELS,
        "label_names": {"0": "not_fatigued", "1": "fatigued"},
    }
    for key, expected_value in expected.items():
        if manifest.get(key) != expected_value:
            raise ValueError(
                f"Invalid Fatig manifest field {key}: got {manifest.get(key)!r}, "
                f"expected {expected_value!r}"
            )
    records = manifest.get("subject_records")
    if not isinstance(records, list) or not records:
        raise ValueError("Fatig manifest has no subject_records")
    by_subject = {}
    for record in records:
        subject = str(record["subject_id"])
        if subject in by_subject:
            raise ValueError(f"Duplicate Fatig subject in manifest: {subject}")
        by_subject[subject] = record
    return manifest, by_subject


def _load_subject_arrays(
    root: Path, record: dict
) -> tuple[Path, Path, np.ndarray, np.ndarray]:
    eeg_path = _safe_path(root, record["eeg_file"])
    label_path = _safe_path(root, record["label_file"])
    stats_path = _safe_path(root, record["stats_file"])
    for path in (eeg_path, label_path, stats_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing Fatig array: {path}")
    eeg = np.load(eeg_path, mmap_mode="r", allow_pickle=False)
    labels = np.load(label_path, mmap_mode="r", allow_pickle=False)
    stats = np.load(stats_path, mmap_mode="r", allow_pickle=False)
    samples = int(record["samples"])
    if eeg.shape != (samples, len(FATIG_30_CHANNELS), EXPECTED_SAMPLES):
        raise ValueError(f"Invalid Fatig EEG shape: {eeg_path} {eeg.shape}")
    if eeg.dtype != np.float32:
        raise ValueError(f"Invalid Fatig EEG dtype: {eeg_path} {eeg.dtype}")
    if labels.shape != (samples,) or labels.dtype != np.int64:
        raise ValueError(f"Invalid Fatig labels: {label_path} {labels.shape} {labels.dtype}")
    if stats.shape != (samples, len(FATIG_30_CHANNELS), 2) or stats.dtype != np.float64:
        raise ValueError(f"Invalid Fatig statistics: {stats_path} {stats.shape} {stats.dtype}")
    counts = Counter(int(value) for value in np.asarray(labels))
    expected_counts = Counter(
        {int(key): int(value) for key, value in record["label_counts"].items()}
    )
    if counts != expected_counts or set(counts) - {0, 1}:
        raise ValueError(f"Fatig labels disagree with manifest for {record['subject_id']}")
    return eeg_path, stats_path, labels, stats


def _split_subjects(manifest: dict, available_subjects: set[str]) -> dict[str, tuple[str, ...]]:
    split = manifest.get("subject_split")
    if not isinstance(split, dict):
        raise ValueError("Fatig manifest has no subject_split")
    output = {}
    for name in ("train", "val", "test"):
        subjects = tuple(str(value) for value in split.get(name, []))
        if not subjects:
            raise ValueError(f"Fatig {name} split is empty")
        missing = set(subjects) - available_subjects
        if missing:
            raise ValueError(f"Fatig {name} split subjects missing: {sorted(missing)}")
        output[name] = subjects
    if set(output["train"]) & set(output["test"]):
        raise ValueError("Fatig train and test subjects overlap")
    if set(output["train"]) & set(output["val"]):
        raise ValueError("Fatig train and val subjects overlap")
    if set(output["val"]) & set(output["test"]):
        raise ValueError("Fatig val and test subjects overlap")
    return output


def _select_loso_fold(
    manifest: dict, available_subjects: set[str], loso_fold: int | None
) -> dict | None:
    folds = manifest.get("loso_folds")
    if folds is None:
        return None
    if not isinstance(folds, list) or not folds:
        raise ValueError("Fatig manifest has invalid loso_folds")
    if loso_fold is None:
        loso_fold = int(manifest.get("default_loso_fold", 0))
    selected = None
    for fold in folds:
        if int(fold.get("fold_index", -1)) == int(loso_fold):
            selected = fold
            break
    if selected is None:
        raise ValueError(
            f"Fatig LOSO fold {loso_fold} not found; "
            f"available={[fold.get('fold_index') for fold in folds]}"
        )
    test_subject = str(selected["test_subject"])
    train_subjects = tuple(str(value) for value in selected.get("train_subjects", []))
    if test_subject not in available_subjects:
        raise ValueError(f"Fatig LOSO test subject missing: {test_subject}")
    missing = set(train_subjects) - available_subjects
    if missing:
        raise ValueError(f"Fatig LOSO train subjects missing: {sorted(missing)}")
    if test_subject in set(train_subjects):
        raise ValueError("Fatig LOSO test subject overlaps train subjects")
    return selected


def _make_records_from_subject_split(root: Path, by_subject: dict[str, dict], split_subjects):
    output = {"train": [], "val": [], "test": []}
    for split, subjects in split_subjects.items():
        for subject in subjects:
            eeg_path, stats_path, labels, _ = _load_subject_arrays(root, by_subject[subject])
            output[split].extend(
                (eeg_path, stats_path, index, int(label), subject)
                for index, label in enumerate(labels)
            )
    return output


def _validate_loso_indices(indices: list[int], samples: int, subject: str, split: str) -> tuple[int, ...]:
    values = tuple(int(value) for value in indices)
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate Fatig LOSO {split} indices for subject {subject}")
    invalid = [value for value in values if value < 0 or value >= samples]
    if invalid:
        raise ValueError(
            f"Invalid Fatig LOSO {split} indices for subject {subject}: {invalid[:5]}"
        )
    return values


def _make_records_from_loso_fold(root: Path, by_subject: dict[str, dict], fold: dict):
    output = {"train": [], "val": [], "test": []}
    test_subject = str(fold["test_subject"])
    train_subjects = tuple(str(value) for value in fold.get("train_subjects", []))
    train_indices = fold.get("train_indices")
    val_indices = fold.get("val_indices")
    if not isinstance(train_indices, dict) or not isinstance(val_indices, dict):
        raise ValueError("Fatig LOSO fold has no train_indices/val_indices")

    for split, split_indices in (("train", train_indices), ("val", val_indices)):
        for subject in train_subjects:
            eeg_path, stats_path, labels, _ = _load_subject_arrays(root, by_subject[subject])
            selected = _validate_loso_indices(
                split_indices.get(subject, []), len(labels), subject, split
            )
            output[split].extend(
                (eeg_path, stats_path, index, int(labels[index]), subject)
                for index in selected
            )

    eeg_path, stats_path, labels, _ = _load_subject_arrays(root, by_subject[test_subject])
    output["test"].extend(
        (eeg_path, stats_path, index, int(label), test_subject)
        for index, label in enumerate(labels)
    )

    for subject in train_subjects:
        train_set = set(
            _validate_loso_indices(
                train_indices.get(subject, []),
                int(by_subject[subject]["samples"]),
                subject,
                "train",
            )
        )
        val_set = set(
            _validate_loso_indices(
                val_indices.get(subject, []),
                int(by_subject[subject]["samples"]),
                subject,
                "val",
            )
        )
        if train_set & val_set:
            raise ValueError(f"Fatig LOSO train/val indices overlap for subject {subject}")
    return output


def _training_statistics(records) -> tuple[np.ndarray, np.ndarray]:
    by_stats_path: dict[Path, list[int]] = defaultdict(list)
    for _, stats_path, trial_index, _, _ in records:
        by_stats_path[stats_path].append(trial_index)
    channel_sum = np.zeros(len(FATIG_30_CHANNELS), dtype=np.float64)
    channel_squared_sum = np.zeros(len(FATIG_30_CHANNELS), dtype=np.float64)
    total_points = 0
    for stats_path, indices in by_stats_path.items():
        stats = np.load(stats_path, mmap_mode="r", allow_pickle=False)
        for start in range(0, len(indices), 1024):
            selected = np.asarray(stats[indices[start : start + 1024]])
            channel_sum += np.sum(selected[:, :, 0], axis=0, dtype=np.float64)
            channel_squared_sum += np.sum(selected[:, :, 1], axis=0, dtype=np.float64)
        total_points += len(indices) * EXPECTED_SAMPLES
    mean = channel_sum / total_points
    variance = channel_squared_sum / total_points - np.square(mean)
    std = np.sqrt(np.maximum(variance, 0.0))
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("Invalid Fatig train-only normalization statistics")
    return mean, std


class FatigRestTaskDataset(Dataset):
    """Lazy Fatig split using subject-disjoint windows and train statistics."""

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
        self.manifest_channel_names = list(FATIG_30_CHANNELS)
        self.channel_names = (
            list(self.manifest_channel_names)
            if channel_names is None
            else [str(name).upper() for name in channel_names]
        )
        if len(set(self.channel_names)) != len(self.channel_names):
            raise ValueError(f"Duplicate Fatig channel names: {self.channel_names}")
        unknown = [
            name for name in self.channel_names
            if name not in self.manifest_channel_names
        ]
        if unknown:
            raise ValueError(f"Unknown Fatig channel names: {unknown}")
        self.channel_indices = np.asarray(
            [self.manifest_channel_names.index(name) for name in self.channel_names],
            dtype=np.int64,
        )
        self.mean = np.asarray(mean, dtype=np.float64)[self.channel_indices]
        self.std = np.asarray(std, dtype=np.float64)[self.channel_indices]
        self.label_counts = Counter(record[3] for record in self.records)
        self.subjects = tuple(sorted({record[4] for record in self.records}, key=int))
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
        raise ValueError(f"Unsupported Fatig normalization: {self.normalize_method}")

    def __getitem__(self, index):
        eeg_path, _, trial_index, label, _ = self.records[index]
        eeg = self._array(eeg_path)[trial_index, self.channel_indices]
        eeg = self._normalize(np.asarray(eeg))
        eeg = np.ascontiguousarray(eeg, dtype=np.float32)
        if eeg.shape != (len(self.channel_names), EXPECTED_SAMPLES):
            raise ValueError(f"Unexpected Fatig sample shape: {eeg.shape}")
        if not np.isfinite(eeg).all():
            raise ValueError(f"NaN or Inf in normalized Fatig sample: {eeg_path}")
        return torch.from_numpy(eeg), label


def prepare_Fatig_rest_task_dataset(
    root,
    sampling_rate=200,
    normalize_method="z_score",
    channel_names=None,
    loso_fold=None,
):
    """Build the Fatig rest-vs-task split from preprocessed arrays."""

    if int(sampling_rate) != EXPECTED_RATE:
        raise ValueError(
            f"Fatig arrays are fixed at {EXPECTED_RATE} Hz; rerun preprocessing "
            "for another sampling rate"
        )
    root = Path(root).resolve()
    manifest, by_subject = _load_manifest(root)
    fold = _select_loso_fold(manifest, set(by_subject), loso_fold)
    if fold is None:
        split_subjects = _split_subjects(manifest, set(by_subject))
        records = _make_records_from_subject_split(root, by_subject, split_subjects)
        split_description = "subject_split"
    else:
        records = _make_records_from_loso_fold(root, by_subject, fold)
        split_description = (
            f"LOSO fold={fold['fold_index']} test_subject={fold['test_subject']}"
        )
    mean, std = _training_statistics(records["train"])
    datasets = {
        split: FatigRestTaskDataset(
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
        raise ValueError("Fatig channel order differs across train/val/test")
    print(
        "Fatig rest-task audit: "
        + ", ".join(
            f"{split}=subjects{dataset.subjects}/samples{len(dataset)}/"
            f"labels{dict(sorted(dataset.label_counts.items()))}"
            for split, dataset in datasets.items()
        )
        + f", sampling_rate={EXPECTED_RATE}, window={WINDOW_SECONDS}s, "
        f"normalization={normalize_method}, split={split_description}"
    )
    return datasets["train"], datasets["test"], datasets["val"]
