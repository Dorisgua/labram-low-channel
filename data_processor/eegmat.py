"""EEGMAT loader matching AdaBrain-Bench's cross-subject protocol."""

from __future__ import annotations

import pickle
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from scipy.signal import resample
from torch.utils.data import Dataset

from Channels_definition import EEGMAT_19_CHANNELS


DEFAULT_SAMPLING_RATE = 500
EXPECTED_SAMPLE_SHAPE = (19, 2000)
EXPECTED_SUBJECT_SAMPLES = 30
EXPECTED_SPLIT_SAMPLES = {
    "train": 768,
    "val": 192,
    "test": 120,
}
TRAIN_SUBJECTS = tuple(range(32))
TEST_SUBJECTS = tuple(range(32, 36))
FILENAME_PATTERN = re.compile(
    r"^Subject(?P<subject>\d{2})_(?P<session>[12])_(?P<segment>\d+)\.pkl$"
)


def _parse_sample_path(path: Path, expected_subject: int) -> tuple[int, int, int]:
    match = FILENAME_PATTERN.fullmatch(path.name)
    if match is None:
        raise ValueError(f"Unexpected EEGMAT sample filename: {path}")
    values = tuple(
        int(match.group(name)) for name in ("subject", "session", "segment")
    )
    if values[0] != expected_subject:
        raise ValueError(
            f"EEGMAT subject mismatch: folder={expected_subject}, file={path.name}"
        )
    if not 1 <= values[2] <= 15:
        raise ValueError(f"Unexpected EEGMAT segment index in {path}")
    return values


def _discover_subject_files(root: Path) -> dict[int, list[Path]]:
    if not root.is_dir():
        raise FileNotFoundError(f"EEGMAT processed-data directory not found: {root}")

    subject_files: dict[int, list[Path]] = {}
    incomplete: dict[int, int] = {}
    for subject in range(36):
        subject_dir = root / f"Subject{subject:02d}"
        if not subject_dir.is_dir():
            incomplete[subject] = 0
            continue
        parsed = []
        for path in subject_dir.iterdir():
            if path.suffix != ".pkl":
                continue
            parsed.append((_parse_sample_path(path, subject), path))
        parsed.sort(key=lambda item: item[0])
        files = [path for _, path in parsed]
        subject_files[subject] = files
        if len(files) != EXPECTED_SUBJECT_SAMPLES:
            incomplete[subject] = len(files)

    if incomplete:
        details = ", ".join(
            f"Subject{subject:02d}={count}"
            for subject, count in sorted(incomplete.items())
        )
        raise ValueError(
            "Incomplete EEGMAT dataset: expected 30 pickle samples for each "
            f"of 36 subjects, found {details}"
        )
    return subject_files


def _expected_label(path: Path) -> int:
    match = FILENAME_PATTERN.fullmatch(path.name)
    if match is None:
        raise ValueError(f"Unexpected EEGMAT sample filename: {path}")
    return int(match.group("session")) - 1


def _make_cross_subject_records(
    subject_files: dict[int, list[Path]],
) -> dict[str, list[tuple[Path, int]]]:
    """Reproduce the seed-42, class-balanced AdaBrain-Bench split."""

    rng = random.Random(42)
    train_records: list[tuple[Path, int]] = []
    val_records: list[tuple[Path, int]] = []
    for subject in TRAIN_SUBJECTS:
        by_label: dict[int, list[Path]] = {}
        for path in subject_files[subject]:
            label = _expected_label(path)
            by_label.setdefault(label, []).append(path)
        for label, paths in by_label.items():
            rng.shuffle(paths)
            split_index = int(len(paths) * 0.8)
            train_records.extend((path, label) for path in paths[:split_index])
            val_records.extend((path, label) for path in paths[split_index:])

    test_records = [
        (path, _expected_label(path))
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
                f"Unexpected EEGMAT {split} size: got {len(records[split])}, "
                f"expected {expected}"
            )
    return records


def _read_validated_sample(path: Path, expected_label: int) -> np.ndarray:
    with path.open("rb") as handle:
        sample = pickle.load(handle)
    if not isinstance(sample, dict) or "X" not in sample or "Y" not in sample:
        raise ValueError(f"Invalid EEGMAT pickle schema: {path}")
    x = np.asarray(sample["X"])
    if x.shape != EXPECTED_SAMPLE_SHAPE:
        raise ValueError(
            f"Unexpected EEGMAT sample shape in {path}: got {x.shape}, "
            f"expected {EXPECTED_SAMPLE_SHAPE}"
        )
    label = int(sample["Y"])
    if label != expected_label:
        raise ValueError(
            f"EEGMAT label mismatch in {path}: pickle={label}, "
            f"filename/session={expected_label}"
        )
    if not np.isfinite(x).all():
        raise ValueError(f"EEGMAT sample contains NaN or Inf: {path}")
    return x


def _compute_training_statistics(
    records: list[tuple[Path, int]],
) -> tuple[np.ndarray, np.ndarray]:
    """Match AdaBrain-Bench: average per-window channel means and stds."""

    mean_sum = np.zeros(len(EEGMAT_19_CHANNELS), dtype=np.float64)
    std_sum = np.zeros(len(EEGMAT_19_CHANNELS), dtype=np.float64)
    for path, label in records:
        x = _read_validated_sample(path, label)
        mean_sum += x.mean(axis=1)
        std_sum += x.std(axis=1)
    mean = mean_sum / len(records)
    std = std_sum / len(records)
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("Invalid EEGMAT training normalization statistics")
    return mean, std


class EEGMATCrossSubjectLoader(Dataset):
    """Load one deterministic EEGMAT cross-subject split."""

    def __init__(
        self,
        records,
        split: str,
        mean: np.ndarray,
        std: np.ndarray,
        sampling_rate: int = 200,
        normalize_method: str = "z_score",
        channel_names=None,
    ):
        if sampling_rate <= 0:
            raise ValueError(f"sampling_rate must be positive, got {sampling_rate}")
        self.records = list(records)
        self.split = split
        self.default_rate = DEFAULT_SAMPLING_RATE
        self.sampling_rate = int(sampling_rate)
        self.normalize_method = normalize_method
        self.manifest_channel_names = list(EEGMAT_19_CHANNELS)
        if channel_names is None:
            self.channel_names = list(self.manifest_channel_names)
        else:
            self.channel_names = [str(name).upper() for name in channel_names]
        if len(set(self.channel_names)) != len(self.channel_names):
            raise ValueError(f"Duplicate EEGMAT channel names: {self.channel_names}")
        unknown = [
            name for name in self.channel_names
            if name not in self.manifest_channel_names
        ]
        if unknown:
            raise ValueError(f"Unknown EEGMAT channel names: {unknown}")
        self.channel_indices = np.asarray(
            [self.manifest_channel_names.index(name) for name in self.channel_names],
            dtype=np.int64,
        )
        self.mean = np.asarray(mean, dtype=np.float64)[self.channel_indices]
        self.std = np.asarray(std, dtype=np.float64)[self.channel_indices]
        self.label_counts = Counter(label for _, label in self.records)

    def __len__(self):
        return len(self.records)

    def get_ch_names(self):
        return list(self.channel_names)

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        if self.normalize_method == "z_score":
            return (x - self.mean[:, None]) / (self.std[:, None] + 1e-8)
        if self.normalize_method == "0.1mv":
            # Pickles are in volts. Convert to units of 0.1 mV.
            return x * 1e4
        if self.normalize_method == "95":
            scale = np.quantile(np.abs(x), 0.95, axis=-1, keepdims=True)
            return x / (scale + 1e-8)
        raise ValueError(f"Unsupported EEGMAT normalization: {self.normalize_method}")

    def __getitem__(self, index):
        path, expected_label = self.records[index]
        x = _read_validated_sample(path, expected_label)
        x = x[self.channel_indices]
        if self.sampling_rate != self.default_rate:
            sample_count = int(x.shape[-1] * self.sampling_rate / self.default_rate)
            x = resample(x, sample_count, axis=-1)
        x = self._normalize(x)
        x = np.ascontiguousarray(x, dtype=np.float32)
        if not np.isfinite(x).all():
            raise ValueError(f"Normalized EEGMAT sample contains NaN or Inf: {path}")
        return torch.from_numpy(x), expected_label


def prepare_EEGMAT_cross_subject_dataset(
    root, sampling_rate=200, normalize_method="z_score", channel_names=None
):
    """Build the AdaBrain-Bench EEGMAT train/validation/test split."""

    root = Path(root)
    subject_files = _discover_subject_files(root)
    records = _make_cross_subject_records(subject_files)
    mean, std = _compute_training_statistics(records["train"])
    datasets = {
        split: EEGMATCrossSubjectLoader(
            records[split],
            split=split,
            mean=mean,
            std=std,
            sampling_rate=sampling_rate,
            normalize_method=normalize_method,
            channel_names=channel_names,
        )
        for split in ("train", "val", "test")
    }

    expected_channels = datasets["train"].get_ch_names()
    if any(
        dataset.get_ch_names() != expected_channels
        for dataset in datasets.values()
    ):
        raise ValueError("EEGMAT channel order differs across train/val/test")

    print(
        "EEGMAT cross-subject audit: "
        + ", ".join(
            f"{split}={len(dataset)} labels={dict(sorted(dataset.label_counts.items()))}"
            for split, dataset in datasets.items()
        )
        + f", sampling_rate={sampling_rate}, normalization={normalize_method}"
    )
    return datasets["train"], datasets["test"], datasets["val"]
