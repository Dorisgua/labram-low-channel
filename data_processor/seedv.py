"""SEED-V loader for the preprocessed one-second LaBraM windows."""

import pickle
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from Channels_definition import SEEDV_62_CHANNELS


EXPECTED_SPLITS = {
    "train": {"directory": "processed_train", "samples": 33693},
    "val": {"directory": "processed_eval", "samples": 41931},
    "test": {"directory": "processed_test", "samples": 39377},
}
EXPECTED_SAMPLE_SHAPE = (62, 200)
EXPECTED_LABELS = {0, 1, 2, 3, 4}
EXPECTED_SUBJECTS = set(range(1, 17))
EXPECTED_SESSIONS = 47

FILE_NAME_PATTERN = re.compile(
    r"^sub(?P<subject>\d+)_sess(?P<session>\d+)_trial(?P<trial>\d+)_"
    r"win(?P<window>\d+)_(?P<label_name>[a-z]+)\.pkl$"
)


class SEEDVMotorEmotionDataset(Dataset):
    """Load one trial-based SEED-V split from preprocessed pickle windows."""

    def __init__(self, root, split, channel_names=None):
        if split not in EXPECTED_SPLITS:
            raise ValueError(f"Unsupported SEED-V split: {split}")

        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"SEED-V root not found: {self.root}")

        expected = EXPECTED_SPLITS[split]
        self.split = split
        self.split_root = self.root / expected["directory"]
        if not self.split_root.is_dir():
            raise FileNotFoundError(
                f"SEED-V {split} directory not found: {self.split_root}"
            )

        self.files = sorted(self.split_root.glob("*.pkl"))
        if len(self.files) != expected["samples"]:
            raise ValueError(
                f"Unexpected SEED-V {split} size: got {len(self.files)}, "
                f"expected {expected['samples']}"
            )

        self.manifest_channel_names = list(SEEDV_62_CHANNELS)
        if channel_names is None:
            self.channel_names = list(self.manifest_channel_names)
        else:
            self.channel_names = [str(name).upper() for name in channel_names]
            if len(set(self.channel_names)) != len(self.channel_names):
                raise ValueError(
                    f"Duplicate SEED-V channel names requested: {self.channel_names}"
                )
            unknown = [
                name
                for name in self.channel_names
                if name not in self.manifest_channel_names
            ]
            if unknown:
                raise ValueError(f"Unknown SEED-V channel names: {unknown}")

        self.channel_indices = np.asarray(
            [self.manifest_channel_names.index(name) for name in self.channel_names],
            dtype=np.int64,
        )

        subjects = set()
        sessions = set()
        label_name_counts = Counter()
        for path in self.files:
            match = FILE_NAME_PATTERN.match(path.name)
            if match is None:
                raise ValueError(f"Unexpected SEED-V file name: {path.name}")
            subject = int(match.group("subject"))
            session = int(match.group("session"))
            subjects.add(subject)
            sessions.add((subject, session))
            label_name_counts[match.group("label_name")] += 1

        if subjects != EXPECTED_SUBJECTS:
            raise ValueError(
                f"Unexpected SEED-V {split} subjects: got {sorted(subjects)}, "
                f"expected {sorted(EXPECTED_SUBJECTS)}"
            )
        if len(sessions) != EXPECTED_SESSIONS:
            raise ValueError(
                f"Unexpected SEED-V {split} subject-session count: "
                f"got {len(sessions)}, expected {EXPECTED_SESSIONS}"
            )
        self.subjects = subjects
        self.sessions = sessions
        self.label_name_counts = label_name_counts

        self._validate_record(self.files[0], record_index=0)

    def __len__(self):
        return len(self.files)

    def get_ch_names(self):
        return list(self.channel_names)

    def _load_record(self, path):
        with path.open("rb") as handle:
            record = pickle.load(handle)
        if not isinstance(record, dict):
            raise TypeError(
                f"Expected a dict in {path}, got {type(record).__name__}"
            )
        return record

    def _validate_record(self, path, record_index):
        record = self._load_record(path)
        required = {"signal", "label", "subject", "ch_names", "sampling_rate"}
        missing = required - set(record)
        if missing:
            raise ValueError(
                f"SEED-V sample {record_index} is missing keys: {sorted(missing)}"
            )

        signal = np.asarray(record["signal"], dtype=np.float32)
        if signal.shape != EXPECTED_SAMPLE_SHAPE:
            raise ValueError(
                f"SEED-V sample {record_index} has shape {signal.shape}, "
                f"expected {EXPECTED_SAMPLE_SHAPE}"
            )
        if not np.isfinite(signal).all():
            raise ValueError(f"SEED-V sample {record_index} contains NaN or Inf")

        record_ch_names = [str(name).upper() for name in record["ch_names"]]
        if record_ch_names != self.manifest_channel_names:
            raise ValueError(
                f"SEED-V sample {record_index} channel order differs from "
                "SEEDV_62_CHANNELS"
            )
        if int(record["sampling_rate"]) != 200:
            raise ValueError(
                f"SEED-V sample {record_index} sampling rate is "
                f"{record['sampling_rate']}, expected 200"
            )

        labels = np.asarray(record["label"]).reshape(-1)
        if labels.size != 1:
            raise ValueError(
                f"SEED-V sample {record_index} has invalid label shape: "
                f"{np.asarray(record['label']).shape}"
            )
        label = int(labels[0])
        if label not in EXPECTED_LABELS:
            raise ValueError(
                f"SEED-V sample {record_index} has invalid label {label}"
            )
        return signal, label

    def __getitem__(self, index):
        path = self.files[int(index)]
        signal, label = self._validate_record(path, record_index=int(index))
        signal = np.ascontiguousarray(signal[self.channel_indices])
        return torch.from_numpy(signal), label


def prepare_SEEDV_dataset(root, channel_names=None):
    """Build the established trial-based SEED-V train/test/validation splits."""

    datasets = {
        split: SEEDVMotorEmotionDataset(
            root=root,
            split=split,
            channel_names=channel_names,
        )
        for split in ("train", "val", "test")
    }

    train_channels = datasets["train"].get_ch_names()
    for split, dataset in datasets.items():
        if dataset.get_ch_names() != train_channels:
            raise ValueError(f"SEED-V {split} channel order differs from train")

    print(
        "SEED-V audit: "
        + ", ".join(
            f"{split}={len(dataset)} subjects={len(dataset.subjects)} "
            f"sessions={len(dataset.sessions)} "
            f"labels={dict(sorted(dataset.label_name_counts.items()))}"
            for split, dataset in datasets.items()
        )
    )
    # Keep the repository's established return order.
    return datasets["train"], datasets["test"], datasets["val"]
