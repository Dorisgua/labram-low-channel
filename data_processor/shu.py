"""SHU motor-imagery loader matching AdaBrain-Bench cross-subject manifests."""

from __future__ import annotations

import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.signal import resample
from torch.utils.data import Dataset

from Channels_definition import SHU_13_CHANNELS, SHU_32_CHANNELS


EXPECTED_CROSS_SUBJECT_SPLITS = {
    "train": {"samples": 8465, "subjects": list(range(22))},
    "val": {"samples": 2138, "subjects": list(range(22))},
    "test": {"samples": 1385, "subjects": [22, 23, 24]},
}


class SHUCrossSubjectLoader(Dataset):
    """Load one AdaBrain SHU cross-subject JSON split."""

    def __init__(
        self,
        json_path,
        sampling_rate=200,
        normalize_method="z_score",
        factor=100,
        channel_names=None,
    ):
        self.json_path = Path(json_path)
        if not self.json_path.is_file():
            raise FileNotFoundError(f"SHU split manifest not found: {self.json_path}")

        with self.json_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if "dataset_info" not in payload or "subject_data" not in payload:
            raise ValueError(
                f"Invalid SHU manifest {self.json_path}: expected dataset_info and subject_data"
            )

        info = payload["dataset_info"]
        self.files = payload["subject_data"]
        self.default_rate = int(info["sampling_rate"])
        self.sampling_rate = int(sampling_rate)
        self.normalize_method = normalize_method
        self.factor = factor
        self.manifest_channel_names = [str(name).upper() for name in info["ch_names"]]

        full_mean_value = np.asarray(info["mean"], dtype=np.float64)[:, None]
        full_std_value = np.asarray(info["std"], dtype=np.float64)[:, None]
        num_manifest_channels = len(self.manifest_channel_names)
        if full_mean_value.shape != (num_manifest_channels, 1):
            raise ValueError(f"Mean/channel mismatch in {self.json_path}")
        if full_std_value.shape != (num_manifest_channels, 1):
            raise ValueError(f"Std/channel mismatch in {self.json_path}")
        if not self.files:
            raise ValueError(f"SHU split is empty: {self.json_path}")

        if channel_names is None:
            self.channel_names = list(self.manifest_channel_names)
        else:
            self.channel_names = [str(name).upper() for name in channel_names]
            if len(set(self.channel_names)) != len(self.channel_names):
                raise ValueError(f"Duplicate SHU channel names requested: {self.channel_names}")
            unknown = [
                name for name in self.channel_names
                if name not in self.manifest_channel_names
            ]
            if unknown:
                raise ValueError(f"Unknown SHU channel names: {unknown}")

        self.channel_indices = [
            self.manifest_channel_names.index(name) for name in self.channel_names
        ]
        self.mean_value = full_mean_value[self.channel_indices]
        self.std_value = full_std_value[self.channel_indices]
        self.full_mean_value = full_mean_value
        self.full_std_value = full_std_value
        if self.manifest_channel_names != SHU_32_CHANNELS:
            raise ValueError("SHU manifest channel order differs from SHU_32_CHANNELS")
        self.observed_indices_in_full = [
            self.manifest_channel_names.index(name) for name in SHU_13_CHANNELS
        ]
        self.subject_indices = defaultdict(list)
        self.task_indices = defaultdict(list)
        for index, record in enumerate(self.files):
            self.subject_indices[int(record["subject_id"])].append(index)
            self.task_indices[int(record["label"])].append(index)

    def __len__(self):
        return len(self.files)

    def get_ch_names(self):
        return list(self.channel_names)

    def _normalize(self, x):
        if self.normalize_method in {"none", ""}:
            return x
        if self.normalize_method == "z_score":
            return (x - self.full_mean_value) / (self.full_std_value + 1e-8)
        if self.normalize_method == "0.1mv":
            return x / self.factor
        if self.normalize_method == "95":
            scale = np.quantile(np.abs(x), q=0.95, method="linear", axis=-1, keepdims=True)
            return x / (scale + 1e-8)
        raise ValueError(f"Unsupported SHU normalization: {self.normalize_method}")

    def __getitem__(self, index):
        record = self.files[index]
        file_path = record["file"]
        with open(file_path, "rb") as handle:
            sample = pickle.load(handle)

        x = np.asarray(sample["X"])
        if x.ndim != 2:
            raise ValueError(f"SHU sample must be [channels, time], got {x.shape}: {file_path}")
        if x.shape[0] != len(self.manifest_channel_names):
            raise ValueError(
                f"Channel mismatch in {file_path}: got {x.shape[0]}, "
                f"expected {len(self.manifest_channel_names)}"
            )
        if self.sampling_rate != self.default_rate:
            sample_count = int(x.shape[-1] * self.sampling_rate / self.default_rate)
            x = resample(x, sample_count, axis=-1)
        x = self._normalize(x)

        label = int(float(sample["Y"]))
        if self.sampling_rate == 200 and x.shape != (len(SHU_32_CHANNELS), 800):
            raise ValueError(f"Unexpected SHU sample shape: {x.shape}: {file_path}")
        if not np.isfinite(x).all():
            raise ValueError(f"SHU sample contains NaN or Inf: {file_path}")
        if label != int(record["label"]):
            raise ValueError(f"SHU label mismatch: {file_path}")
        x_full = torch.as_tensor(np.ascontiguousarray(x), dtype=torch.float32)
        x_input = torch.as_tensor(np.ascontiguousarray(x[self.channel_indices]), dtype=torch.float32)
        x_obs = torch.as_tensor(np.ascontiguousarray(x[self.observed_indices_in_full]), dtype=torch.float32)
        subject = int(record["subject_id"])
        return x_input, label, x_obs, x_full, subject, label


def prepare_SHU_cross_subject_dataset(
    root, sampling_rate=200, normalize_method="z_score", channel_names=None,
):
    """Build AdaBrain's SHU cross-subject split.

    AdaBrain uses subjects 1-22 for train/validation and subjects 23-25 for test.
    The JSON manifests carry the train-set channel statistics used for z-score.
    """

    root = Path(root)
    datasets = {
        split: SHUCrossSubjectLoader(
            root / f"{split}.json",
            sampling_rate=sampling_rate,
            normalize_method=normalize_method,
            channel_names=channel_names,
        )
        for split in ("train", "val", "test")
    }

    train_dataset = datasets["train"]
    expected_channels = train_dataset.get_ch_names()
    for split, dataset in datasets.items():
        if dataset.get_ch_names() != expected_channels:
            raise ValueError(f"SHU cross-subject {split} channel order differs from train")
        if not np.array_equal(dataset.mean_value, train_dataset.mean_value):
            raise ValueError(f"SHU cross-subject {split} mean statistics differ from train")
        if not np.array_equal(dataset.std_value, train_dataset.std_value):
            raise ValueError(f"SHU cross-subject {split} std statistics differ from train")

    split_files = {
        split: {record["file"] for record in dataset.files}
        for split, dataset in datasets.items()
    }
    if split_files["train"] & split_files["val"]:
        raise ValueError("SHU cross-subject train and val manifests overlap")
    if split_files["train"] & split_files["test"]:
        raise ValueError("SHU cross-subject train and test manifests overlap")
    if split_files["val"] & split_files["test"]:
        raise ValueError("SHU cross-subject val and test manifests overlap")

    audit = {}
    split_subjects = {}
    for split, dataset in datasets.items():
        expected = EXPECTED_CROSS_SUBJECT_SPLITS[split]
        subject_counts = Counter(int(record["subject_id"]) for record in dataset.files)
        subjects = sorted(subject_counts)
        labels = {int(record["label"]) for record in dataset.files}

        if len(dataset) != expected["samples"]:
            raise ValueError(
                f"Unexpected SHU cross-subject {split} size: "
                f"got {len(dataset)}, expected {expected['samples']}"
            )
        if subjects != expected["subjects"]:
            raise ValueError(
                f"Unexpected SHU cross-subject {split} subjects: "
                f"{subjects}, expected {expected['subjects']}"
            )
        if labels != {0, 1}:
            raise ValueError(f"Unexpected SHU cross-subject {split} labels: {sorted(labels)}")
        audit[split] = (len(dataset), dict(subject_counts))
        split_subjects[split] = set(subjects)

    if split_subjects["test"] & split_subjects["train"]:
        raise ValueError("SHU cross-subject train and test subjects overlap")
    if split_subjects["test"] & split_subjects["val"]:
        raise ValueError("SHU cross-subject val and test subjects overlap")

    print(
        "SHU cross-subject audit: "
        + ", ".join(
            f"{split}={samples} subjects={sorted(counts)}"
            for split, (samples, counts) in audit.items()
        )
    )
    return datasets["train"], datasets["test"], datasets["val"]
