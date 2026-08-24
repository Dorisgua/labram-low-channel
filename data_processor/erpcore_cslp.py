"""ERP Core loader for Dynamic Prototype + CSLP losses."""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from Channels_definition import ERPCORE_12_CHANNELS, ERPCORE_28_CHANNELS, ERPCORE_30_CHANNELS
from data_processor.erpcore import (
    SOURCE_SAMPLES,
    SOURCE_SAMPLING_RATE,
    TASK_REMAP,
    TEST_SUBJECTS,
    TRAIN_SUBJECTS,
    VAL_SUBJECTS,
    _resolve_pt_path,
    _split_indices,
    _training_statistics,
)


class ERPCORECSLPDataset(Dataset):
    """Return full/observed/missing ERP Core views plus subject and task ids."""

    def __init__(
        self,
        payload: dict,
        indices: np.ndarray,
        split: str,
        mean: np.ndarray,
        std: np.ndarray,
        normalize_method: str,
        target_samples: int,
    ):
        self.data = payload["data"]
        self.tasks = payload["tasks"].cpu().numpy()
        self.subject_values = payload["subjects"].cpu().numpy()
        self.indices = np.asarray(indices, dtype=np.int64)
        self.split = str(split)
        self.normalize_method = normalize_method
        self.full_channel_names = list(ERPCORE_28_CHANNELS)
        self.obs_channel_names = list(ERPCORE_12_CHANNELS)
        self.miss_channel_names = [
            ch for ch in self.full_channel_names if ch not in self.obs_channel_names
        ]
        self.manifest_channel_names = list(ERPCORE_30_CHANNELS)
        self.full_channel_indices = np.asarray(
            [self.manifest_channel_names.index(name) for name in self.full_channel_names],
            dtype=np.int64,
        )
        self.obs_indices_in_full = [
            self.full_channel_names.index(name) for name in self.obs_channel_names
        ]
        self.miss_indices_in_full = [
            self.full_channel_names.index(name) for name in self.miss_channel_names
        ]
        self.mean = torch.as_tensor(mean, dtype=torch.float32)
        self.std = torch.as_tensor(std, dtype=torch.float32)
        self.target_samples = int(target_samples)

        self.labels = np.asarray(
            [TASK_REMAP[int(self.tasks[idx])] for idx in self.indices],
            dtype=np.int64,
        )
        self.subjects = np.asarray(
            [int(self.subject_values[idx]) for idx in self.indices],
            dtype=np.int64,
        )
        self.subject_indices = defaultdict(list)
        self.task_indices = defaultdict(list)
        for local_idx, (subject, label) in enumerate(zip(self.subjects, self.labels)):
            self.subject_indices[int(subject)].append(local_idx)
            self.task_indices[int(label)].append(local_idx)

    def __len__(self):
        return len(self.indices)

    def get_ch_names(self):
        return list(self.full_channel_names)

    def get_obs_ch_names(self):
        return list(self.obs_channel_names)

    def get_miss_ch_names(self):
        return list(self.miss_channel_names)

    def _normalize(self, eeg: torch.Tensor) -> torch.Tensor:
        if self.normalize_method == "z_score":
            return (eeg - self.mean[:, None]) / (self.std[:, None] + 1e-8)
        if self.normalize_method == "0.1mv":
            return eeg / 100.0
        if self.normalize_method == "95":
            scale = torch.quantile(torch.abs(eeg), 0.95, dim=-1, keepdim=True)
            return eeg / (scale + 1e-8)
        raise ValueError(f"Unsupported ERP CORE normalization: {self.normalize_method}")

    def __getitem__(self, index):
        global_index = int(self.indices[index])
        eeg = self.data[global_index, self.full_channel_indices, :].float()
        eeg = self._normalize(eeg)
        if eeg.shape[-1] != self.target_samples:
            eeg = F.interpolate(
                eeg.unsqueeze(0),
                size=self.target_samples,
                mode="linear",
                align_corners=False,
            ).squeeze(0)
        if eeg.shape != (len(self.full_channel_names), self.target_samples):
            raise ValueError(f"Unexpected ERP CORE sample shape: {tuple(eeg.shape)}")
        if not torch.isfinite(eeg).all():
            raise ValueError(f"NaN or Inf in normalized ERP CORE sample {global_index}")

        label = int(self.labels[index])
        subject = int(self.subjects[index])
        return {
            "x_full": eeg.contiguous(),
            "x_obs": eeg[self.obs_indices_in_full].contiguous(),
            "x_miss": eeg[self.miss_indices_in_full].contiguous(),
            "label": torch.tensor(label, dtype=torch.long),
            "subject": torch.tensor(subject, dtype=torch.long),
            "task": torch.tensor(label, dtype=torch.long),
        }

    def sample_pair_batch(self, property_name, batch_size):
        if property_name == "subject":
            index_by_value = self.subject_indices
        elif property_name == "task":
            index_by_value = self.task_indices
        else:
            raise ValueError(f"Unsupported pair property: {property_name}")

        values = list(index_by_value.keys())
        left_indices = []
        right_indices = []
        for _ in range(batch_size):
            value = random.choice(values)
            candidates = index_by_value[value]
            if len(candidates) >= 2:
                i, j = random.sample(candidates, 2)
            else:
                i = j = candidates[0]
            left_indices.append(i)
            right_indices.append(j)
        return _collate([self[i] for i in left_indices]), _collate([self[i] for i in right_indices])

    def sample_group_pair_batch(self, property_name, num_groups, samples_per_group):
        if property_name == "subject":
            index_by_value = self.subject_indices
        elif property_name == "task":
            index_by_value = self.task_indices
        else:
            raise ValueError(f"Unsupported pair property: {property_name}")

        values = list(index_by_value.keys())
        if not values:
            raise ValueError(f"No values available for property: {property_name}")
        replace_groups = int(num_groups) > len(values)
        group_values = (
            random.choices(values, k=int(num_groups))
            if replace_groups
            else random.sample(values, int(num_groups))
        )

        left_indices = []
        right_indices = []
        for value in group_values:
            candidates = index_by_value[value]
            replace_samples = len(candidates) < 2 * int(samples_per_group)
            if replace_samples:
                left = random.choices(candidates, k=int(samples_per_group))
                right = random.choices(candidates, k=int(samples_per_group))
            else:
                picked = random.sample(candidates, 2 * int(samples_per_group))
                left = picked[: int(samples_per_group)]
                right = picked[int(samples_per_group) :]
            left_indices.extend(left)
            right_indices.extend(right)

        return (
            _collate([self[i] for i in left_indices]),
            _collate([self[i] for i in right_indices]),
            len(group_values),
            int(samples_per_group),
        )

    def sample_cslpae_pair_batch(self, property_name, batch_size):
        if property_name == "subject":
            index_by_value = self.subject_indices
        elif property_name == "task":
            index_by_value = self.task_indices
        else:
            raise ValueError(f"Unsupported pair property: {property_name}")

        values = sorted(index_by_value.keys())
        if not values:
            raise ValueError(f"No values available for property: {property_name}")

        samples_per_repeat = len(values)
        repeats = max(int(batch_size) // (2 * samples_per_repeat), 1)
        left_indices = []
        right_indices = []
        for _ in range(repeats):
            for value in values:
                candidates = index_by_value[value]
                if len(candidates) >= 2:
                    left, right = random.sample(candidates, 2)
                else:
                    left = right = candidates[0]
                left_indices.append(left)
                right_indices.append(right)

        return (
            _collate([self[i] for i in left_indices]),
            _collate([self[i] for i in right_indices]),
            repeats,
            samples_per_repeat,
        )


def _collate(items):
    return {
        key: torch.stack([item[key] for item in items])
        for key in items[0].keys()
    }


def prepare_ERPCORE_cslp_dataset(
    root,
    sampling_rate=200,
    normalize_method="z_score",
    train_subjects=TRAIN_SUBJECTS,
    val_subjects=VAL_SUBJECTS,
    test_subjects=TEST_SUBJECTS,
):
    """Build the CSLP-AE subject-disjoint ERP Core split with 28->12 views."""

    pt_path = _resolve_pt_path(root)
    payload = torch.load(pt_path, map_location="cpu")
    required = {"data", "subjects", "tasks"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"ERP CORE PT missing keys: {sorted(missing)}")
    if tuple(payload["data"].shape[1:]) != (len(ERPCORE_30_CHANNELS), SOURCE_SAMPLES):
        raise ValueError(f"Unexpected ERP CORE data shape: {tuple(payload['data'].shape)}")

    target_samples = int(sampling_rate)
    if target_samples not in {200, SOURCE_SAMPLES}:
        raise ValueError(
            f"ERP CORE simple_data.pt is {SOURCE_SAMPLING_RATE} Hz with {SOURCE_SAMPLES} samples; "
            "this loader currently supports sampling_rate=200 or 256"
        )

    subjects = payload["subjects"].cpu().numpy()
    tasks = payload["tasks"].cpu().numpy()
    split_subjects = {
        "train": tuple(int(value) for value in train_subjects),
        "val": tuple(int(value) for value in val_subjects),
        "test": tuple(int(value) for value in test_subjects),
    }
    indices = {
        split: _split_indices(subjects, tasks, values)
        for split, values in split_subjects.items()
    }
    if any(len(values) == 0 for values in indices.values()):
        split_lengths = {split: len(values) for split, values in indices.items()}
        raise ValueError(f"Empty ERP CORE split: {split_lengths}")

    full_channel_indices = np.asarray(
        [ERPCORE_30_CHANNELS.index(name) for name in ERPCORE_28_CHANNELS],
        dtype=np.int64,
    )
    mean, std = _training_statistics(payload["data"], indices["train"], full_channel_indices)
    datasets = {
        split: ERPCORECSLPDataset(
            payload,
            split_indices,
            split=split,
            mean=mean,
            std=std,
            normalize_method=normalize_method,
            target_samples=target_samples,
        )
        for split, split_indices in indices.items()
    }
    print(
        "ERP CORE CSLP audit: "
        f"pt_path={pt_path}, source_rate={SOURCE_SAMPLING_RATE}, "
        f"target_samples={target_samples}, full_channels={len(ERPCORE_28_CHANNELS)}, "
        f"observed_channels={len(ERPCORE_12_CHANNELS)}, "
        f"missing_channels={len(datasets['train'].get_miss_ch_names())}, "
        f"train={len(datasets['train'])}, val={len(datasets['val'])}, test={len(datasets['test'])}"
    )
    print(f"Observed channels: {datasets['train'].get_obs_ch_names()}")
    print(f"Missing channels: {datasets['train'].get_miss_ch_names()}")
    return datasets["train"], datasets["test"], datasets["val"]
