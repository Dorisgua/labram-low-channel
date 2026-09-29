"""TUEV pickle reader with ERP-Core-compatible six-field samples.

The source pickles are the existing processed_train/processed_eval/
processed_test files. Signals keep the same units as utils.TUEVLoader;
the training engine applies its input scale afterwards.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import pickle
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from Channels_definition import TUEV_13_CHANNELS, TUEV_23_CHANNELS


def _subject_key(filename: str, split: str) -> str:
    if split not in {"train", "val"}:
        raise ValueError("TUEV test filenames do not contain verified subject IDs")
    # Same rule used to make the existing TUEV subject-disjoint split.
    return filename.split("_", 1)[0]


class TUEVPklLoader(Dataset):
    def __init__(self, root, split: str, channel_names=None):
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unknown TUEV split: {split}")
        self.root = Path(root)
        self.split = split
        self.files = sorted(path.name for path in self.root.glob("*.pkl"))
        if not self.files:
            raise ValueError(f"No TUEV pickle files in {self.root}")

        self.channel_names = list(TUEV_13_CHANNELS if channel_names is None else channel_names)
        if self.channel_names not in (TUEV_13_CHANNELS, TUEV_23_CHANNELS):
            raise ValueError("TUEV channel_names must be the exact 13- or 23-channel layout")
        self.full_channel_names = list(TUEV_23_CHANNELS)
        self.observed_channel_names = list(TUEV_13_CHANNELS)
        self.observed_indices_in_full = [
            self.full_channel_names.index(name) for name in self.observed_channel_names
        ]
        self.use_full_input = self.channel_names == self.full_channel_names

        self.labels = []
        self.subject_names = []
        self.subject_values = []
        self.subject_indices = defaultdict(list)
        self.task_indices = defaultdict(list)
        self.preload = os.environ.get("PRELOAD_DATA", "0") == "1"
        self.signals = torch.empty((len(self.files), len(TUEV_23_CHANNELS), 1000), dtype=torch.float32) if self.preload else None
        for index, filename in enumerate(self.files):
            with (self.root / filename).open("rb") as handle:
                sample = pickle.load(handle)
            raw_label = float(np.asarray(sample["label"]).reshape(-1)[0])
            if not raw_label.is_integer():
                raise ValueError(f"Noninteger TUEV label in {filename}: {raw_label}")
            label = int(raw_label) - 1
            if label not in range(6):
                raise ValueError(f"Invalid TUEV label in {filename}: {label}")
            if self.signals is not None:
                signal = torch.as_tensor(np.asarray(sample["signal"]), dtype=torch.float32)
                if signal.shape != (len(TUEV_23_CHANNELS), 1000) or not torch.isfinite(signal).all():
                    raise ValueError(f"Invalid TUEV signal in {filename}: {tuple(signal.shape)}")
                self.signals[index].copy_(signal)
            self.labels.append(label)
            self.subject_names.append(_subject_key(filename, split) if split != "test" else None)
            self.task_indices[label].append(index)

        subject_ids = {
            name: index for index, name in enumerate(sorted(set(self.subject_names)))
        } if split != "test" else {}
        for index, name in enumerate(self.subject_names):
            subject = subject_ids[name] if split != "test" else -1
            self.subject_values.append(subject)
            if split != "test":
                self.subject_indices[subject].append(index)
        self.subjects = tuple(sorted(subject_ids))
        self.label_counts = Counter(self.labels)
        self.has_subject_ids = split != "test"

    def __len__(self):
        return len(self.files)

    def get_ch_names(self):
        return list(self.channel_names)

    def __getitem__(self, index):
        filename = self.files[index]
        if self.signals is None:
            with (self.root / filename).open("rb") as handle:
                sample = pickle.load(handle)
            x_full = torch.as_tensor(np.asarray(sample["signal"]), dtype=torch.float32)
        else:
            x_full = self.signals[index]
        if x_full.shape != (len(TUEV_23_CHANNELS), 1000):
            raise ValueError(f"Unexpected TUEV signal shape in {filename}: {tuple(x_full.shape)}")
        if not torch.isfinite(x_full).all():
            raise ValueError(f"Nonfinite TUEV signal in {filename}")
        x_obs = x_full[self.observed_indices_in_full].contiguous()
        x = x_full if self.use_full_input else x_obs
        label = self.labels[index]
        return (
            x.contiguous(), label, x_obs, x_full.contiguous(),
            self.subject_values[index], label,
        )


def prepare_TUEV_dynamic_dataset(root, channel_names=None):
    """Return train, test, val in the order expected by current run scripts."""
    root = Path(root)
    datasets = {
        split: TUEVPklLoader(root / directory, split, channel_names)
        for split, directory in (
            ("train", "processed_train"),
            ("val", "processed_eval"),
            ("test", "processed_test"),
        )
    }
    overlap = set(datasets["train"].subjects) & set(datasets["val"].subjects)
    if overlap:
        raise ValueError(f"TUEV train/val subject overlap: {sorted(overlap)[:5]}")
    print(
        "TUEV audit: "
        + ", ".join(
            f"{split}={len(dataset)} samples/{len(dataset.subjects)} groups/"
            f"labels={dict(dataset.label_counts)}"
            for split, dataset in datasets.items()
        )
    )
    print("TUEV test has no verified subject IDs; subject-pair evaluation must be skipped")
    return datasets["train"], datasets["test"], datasets["val"]
