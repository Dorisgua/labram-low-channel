"""PhysioNet EEGMMIDB motor-imagery loader for EEG-FM-Bench Arrow data."""

from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from datasets import Dataset as ArrowDataset
    from datasets import concatenate_datasets
except ImportError as exc:  # pragma: no cover - depends on the runtime environment
    ArrowDataset = None
    concatenate_datasets = None
    _DATASETS_IMPORT_ERROR = exc
else:
    _DATASETS_IMPORT_ERROR = None

from Channels_definition import PHYSIONET_64_CHANNELS


EXPECTED_SPLITS = {
    "train": {"arrow_name": "train", "samples": 6210, "subjects": range(1, 70)},
    "val": {"arrow_name": "validation", "samples": 1734, "subjects": range(70, 89)},
    "test": {"arrow_name": "test", "samples": 1893, "subjects": range(89, 110)},
}

EXPECTED_SAMPLE_SHAPE = (64, 800)
EXPECTED_LABELS = {0, 1, 2, 3}
# These are EEG-FM-Bench electrode-set indices.  They verify the Arrow row
# order only; LaBraM position indices are rebuilt from channel names in the
# training entrypoint because the two projects use different index spaces.
EXPECTED_EEGFMBENCH_CHS = (
    0, 1, 2, 4, 6, 8, 10, 12, 16, 17, 18, 19, 20, 21, 22, 23, 24,
    28, 29, 30, 31, 32, 33, 34, 35, 36, 39, 40, 41, 42, 43, 44, 45,
    46, 47, 48, 49, 52, 53, 54, 55, 56, 57, 58, 59, 60, 63, 64, 65,
    66, 67, 68, 69, 70, 71, 74, 76, 78, 80, 82, 84, 85, 86, 88,
)


class PhysioNetMotorImageryDataset(Dataset):
    """Memory-map one EEG-FM-Bench PhysioMI Arrow split."""

    def __init__(self, root, split, channel_names=None):
        if ArrowDataset is None:
            raise ImportError(
                "PhysioNet Arrow loading requires the 'datasets' package and "
                "its dependencies"
            ) from _DATASETS_IMPORT_ERROR
        if split not in EXPECTED_SPLITS:
            raise ValueError(f"Unsupported PhysioNet split: {split}")

        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"PhysioNet Arrow directory not found: {self.root}")

        expected = EXPECTED_SPLITS[split]
        arrow_name = expected["arrow_name"]
        shard_paths = sorted(self.root.glob(f"motor_mv_img-{arrow_name}-*.arrow"))
        if not shard_paths:
            raise FileNotFoundError(
                f"No PhysioNet {split} Arrow shards found in {self.root}"
            )

        shards = [ArrowDataset.from_file(str(path)) for path in shard_paths]
        self.arrow_dataset = (
            shards[0] if len(shards) == 1 else concatenate_datasets(shards)
        )
        self.split = split
        self.shard_paths = shard_paths
        self.manifest_channel_names = list(PHYSIONET_64_CHANNELS)

        required_columns = {"data", "chs", "subject", "label"}
        missing_columns = required_columns - set(self.arrow_dataset.column_names)
        if missing_columns:
            raise ValueError(
                f"PhysioNet {split} Arrow data is missing columns: "
                f"{sorted(missing_columns)}"
            )

        if channel_names is None:
            self.channel_names = list(self.manifest_channel_names)
        else:
            self.channel_names = [str(name).upper() for name in channel_names]
            if len(set(self.channel_names)) != len(self.channel_names):
                raise ValueError(
                    f"Duplicate PhysioNet channel names requested: {self.channel_names}"
                )
            unknown = [
                name for name in self.channel_names
                if name not in self.manifest_channel_names
            ]
            if unknown:
                raise ValueError(f"Unknown PhysioNet channel names: {unknown}")

        self.channel_indices = np.asarray(
            [self.manifest_channel_names.index(name) for name in self.channel_names],
            dtype=np.int64,
        )

        if len(self.arrow_dataset) != expected["samples"]:
            raise ValueError(
                f"Unexpected PhysioNet {split} size: got {len(self.arrow_dataset)}, "
                f"expected {expected['samples']}"
            )

        subjects = {int(subject) for subject in self.arrow_dataset["subject"]}
        expected_subjects = set(expected["subjects"])
        if subjects != expected_subjects:
            raise ValueError(
                f"Unexpected PhysioNet {split} subjects: got {sorted(subjects)}, "
                f"expected {sorted(expected_subjects)}"
            )
        self.subjects = subjects

        self.label_counts = Counter(
            int(label) for label in self.arrow_dataset["label"]
        )
        if set(self.label_counts) != EXPECTED_LABELS:
            raise ValueError(
                f"Unexpected PhysioNet {split} labels: "
                f"{sorted(self.label_counts)}"
            )

        first_record = self.arrow_dataset[0]
        first_x = np.asarray(first_record["data"], dtype=np.float32)
        if first_x.shape != EXPECTED_SAMPLE_SHAPE:
            raise ValueError(
                f"Unexpected PhysioNet {split} sample shape: got {first_x.shape}, "
                f"expected {EXPECTED_SAMPLE_SHAPE}"
            )
        first_chs = np.asarray(first_record["chs"])
        if first_chs.shape != (EXPECTED_SAMPLE_SHAPE[0],):
            raise ValueError(
                f"Unexpected PhysioNet {split} channel-index shape: "
                f"got {first_chs.shape}, expected {(EXPECTED_SAMPLE_SHAPE[0],)}"
            )
        if tuple(int(value) for value in first_chs) != EXPECTED_EEGFMBENCH_CHS:
            raise ValueError(
                f"Unexpected PhysioNet {split} EEG-FM-Bench channel indices; "
                "the Arrow row order is not the expected 64-channel montage"
            )

    def __len__(self):
        return len(self.arrow_dataset)

    def get_ch_names(self):
        return list(self.channel_names)

    def __getitem__(self, index):
        record = self.arrow_dataset[int(index)]
        x = np.asarray(record["data"], dtype=np.float32)
        if x.shape != EXPECTED_SAMPLE_SHAPE:
            raise ValueError(
                f"PhysioNet sample {index} has shape {x.shape}, "
                f"expected {EXPECTED_SAMPLE_SHAPE}"
            )
        x = np.ascontiguousarray(x[self.channel_indices])
        if not np.isfinite(x).all():
            raise ValueError(f"PhysioNet sample {index} contains NaN or Inf")

        label = int(record["label"])
        if label not in EXPECTED_LABELS:
            raise ValueError(f"PhysioNet sample {index} has invalid label {label}")
        return torch.from_numpy(x), label


def prepare_PhysioNet_motor_imagery_dataset(root, channel_names=None):
    """Build subject-independent train/test/validation PhysioMI datasets."""

    datasets = {
        split: PhysioNetMotorImageryDataset(
            root=root,
            split=split,
            channel_names=channel_names,
        )
        for split in ("train", "val", "test")
    }

    train_channels = datasets["train"].get_ch_names()
    for split, dataset in datasets.items():
        if dataset.get_ch_names() != train_channels:
            raise ValueError(
                f"PhysioNet {split} channel order differs from train"
            )

    if datasets["train"].subjects & datasets["val"].subjects:
        raise ValueError("PhysioNet train and val subjects overlap")
    if datasets["train"].subjects & datasets["test"].subjects:
        raise ValueError("PhysioNet train and test subjects overlap")
    if datasets["val"].subjects & datasets["test"].subjects:
        raise ValueError("PhysioNet val and test subjects overlap")

    print(
        "PhysioNet motor-imagery audit: "
        + ", ".join(
            f"{split}={len(dataset)} subjects="
            f"[{min(dataset.subjects)}-{max(dataset.subjects)}] "
            f"labels={dict(sorted(dataset.label_counts.items()))}"
            for split, dataset in datasets.items()
        )
    )
    # Keep the repository's established return order.
    return datasets["train"], datasets["test"], datasets["val"]
