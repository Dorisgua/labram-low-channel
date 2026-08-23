"""BCI Competition IV 2a loaders for the AdaBrain multi-session protocol."""

import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from scipy.signal import resample
from torch.utils.data import Dataset


EXPECTED_MULTISESSION_SPLITS = {
    "train": {"samples": 2592, "trial_range": (1, 288), "per_subject": 288},
    "val": {"samples": 1296, "trial_range": (289, 432), "per_subject": 144},
    "test": {"samples": 1296, "trial_range": (433, 576), "per_subject": 144},
}


class BCIIV2AMultiSessionLoader(Dataset):
    """Load one AdaBrain BCI-IV-2a multi-session JSON split."""

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
            raise FileNotFoundError(f"BCI-IV-2a split manifest not found: {self.json_path}")

        with self.json_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if "dataset_info" not in payload or "subject_data" not in payload:
            raise ValueError(
                f"Invalid BCI-IV-2a manifest {self.json_path}: "
                "expected dataset_info and subject_data"
            )

        info = payload["dataset_info"]
        self.files = payload["subject_data"]
        self.default_rate = int(info["sampling_rate"])
        self.sampling_rate = int(sampling_rate)
        self.manifest_channel_names = [
            str(name).upper() for name in info["ch_names"]
        ]
        full_mean_value = np.asarray(info["mean"], dtype=np.float64)[:, None]
        full_std_value = np.asarray(info["std"], dtype=np.float64)[:, None]
        self.normalize_method = normalize_method
        self.factor = factor

        num_manifest_channels = len(self.manifest_channel_names)
        if full_mean_value.shape != (num_manifest_channels, 1):
            raise ValueError(f"Mean/channel mismatch in {self.json_path}")
        if full_std_value.shape != (num_manifest_channels, 1):
            raise ValueError(f"Std/channel mismatch in {self.json_path}")
        if not self.files:
            raise ValueError(f"BCI-IV-2a split is empty: {self.json_path}")

        if channel_names is None:
            self.channel_names = list(self.manifest_channel_names)
        else:
            self.channel_names = [str(name).upper() for name in channel_names]
            if len(set(self.channel_names)) != len(self.channel_names):
                raise ValueError(
                    f"Duplicate BCI-IV-2a channel names requested: {self.channel_names}"
                )
            unknown = [
                name
                for name in self.channel_names
                if name not in self.manifest_channel_names
            ]
            if unknown:
                raise ValueError(f"Unknown BCI-IV-2a channel names: {unknown}")

        self.channel_indices = [
            self.manifest_channel_names.index(name) for name in self.channel_names
        ]
        self.mean_value = full_mean_value[self.channel_indices]
        self.std_value = full_std_value[self.channel_indices]

    def __len__(self):
        return len(self.files)

    def get_ch_names(self):
        return self.channel_names

    def _normalize(self, X):
        if self.normalize_method == "z_score":
            return (X - self.mean_value) / (self.std_value + 1e-8)
        if self.normalize_method == "0.1mv":
            return X / self.factor
        if self.normalize_method == "95":
            scale = np.quantile(
                np.abs(X), q=0.95, method="linear", axis=-1, keepdims=True
            )
            return X / (scale + 1e-8)
        raise ValueError(f"Unsupported BCI-IV-2a normalization: {self.normalize_method}")

    def __getitem__(self, index):
        record = self.files[index]
        file_path = record["file"]
        with open(file_path, "rb") as handle:
            sample = pickle.load(handle)

        X = np.asarray(sample["X"])
        if X.ndim < 2:
            X = np.expand_dims(X, axis=0)
        if X.shape[0] != len(self.manifest_channel_names):
            raise ValueError(
                f"Channel mismatch in {file_path}: got {X.shape[0]}, "
                f"expected {len(self.manifest_channel_names)}"
            )
        X = X[self.channel_indices]
        if self.sampling_rate != self.default_rate:
            sample_count = int(X.shape[-1] * self.sampling_rate / self.default_rate)
            X = resample(X, sample_count, axis=-1)
        X = self._normalize(X)

        # Match AdaBrain: the pkl label is authoritative; JSON has a duplicate.
        Y = int(float(sample["Y"]))
        return torch.as_tensor(X, dtype=torch.float32), Y


def prepare_BCIIV2A_multisession_dataset(
    root, sampling_rate=200, normalize_method="z_score", channel_names=None
):
    """Build AdaBrain's train-session/validation/test-session split."""

    root = Path(root)
    datasets = {
        split: BCIIV2AMultiSessionLoader(
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
            raise ValueError(f"BCI-IV-2a {split} channel order differs from train")
        if not np.array_equal(dataset.mean_value, train_dataset.mean_value):
            raise ValueError(f"BCI-IV-2a {split} mean statistics differ from train")
        if not np.array_equal(dataset.std_value, train_dataset.std_value):
            raise ValueError(f"BCI-IV-2a {split} std statistics differ from train")

    split_files = {
        split: {record["file"] for record in dataset.files}
        for split, dataset in datasets.items()
    }
    if split_files["train"] & split_files["val"]:
        raise ValueError("BCI-IV-2a train and val manifests overlap")
    if split_files["train"] & split_files["test"]:
        raise ValueError("BCI-IV-2a train and test manifests overlap")
    if split_files["val"] & split_files["test"]:
        raise ValueError("BCI-IV-2a val and test manifests overlap")

    audit = {}
    for split, dataset in datasets.items():
        expected = EXPECTED_MULTISESSION_SPLITS[split]
        subject_counts = Counter(int(record["subject_id"]) for record in dataset.files)
        subjects = sorted(subject_counts)
        try:
            trial_numbers = [
                int(Path(record["file"]).stem.split("_")[-1])
                for record in dataset.files
            ]
        except ValueError as exc:
            raise ValueError(
                f"Cannot parse BCI-IV-2a trial number in {split} manifest"
            ) from exc
        labels = {int(record["label"]) for record in dataset.files}

        if len(dataset) != expected["samples"]:
            raise ValueError(
                f"Unexpected BCI-IV-2a {split} size: got {len(dataset)}, "
                f"expected {expected['samples']}"
            )
        if subjects != list(range(9)):
            raise ValueError(f"Unexpected BCI-IV-2a {split} subjects: {subjects}")
        if set(subject_counts.values()) != {expected["per_subject"]}:
            raise ValueError(
                f"Unexpected BCI-IV-2a {split} samples per subject: "
                f"{dict(subject_counts)}"
            )
        trial_range = (min(trial_numbers), max(trial_numbers))
        if trial_range != expected["trial_range"]:
            raise ValueError(
                f"Unexpected BCI-IV-2a {split} trial range: got {trial_range}, "
                f"expected {expected['trial_range']}"
            )
        if labels != {0, 1, 2, 3}:
            raise ValueError(f"Unexpected BCI-IV-2a {split} labels: {sorted(labels)}")
        audit[split] = (len(dataset), trial_range, expected["per_subject"])

    print(
        "BCI-IV-2a multisession audit: "
        + ", ".join(
            f"{split}={samples} trials[{trials[0]}-{trials[1]}] "
            f"per_subject={per_subject}"
            for split, (samples, trials, per_subject) in audit.items()
        )
    )
    return datasets["train"], datasets["test"], datasets["val"]
