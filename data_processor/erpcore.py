"""ERP CORE simple_data.pt reader for LaBraM fine-tuning."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

"""from Channels_definition import ERPCORE_30_CHANNELS"""

from Channels_definition import (
    ERPCORE_12_CHANNELS,
    ERPCORE_28_CHANNELS,
    ERPCORE_30_CHANNELS,
)


SOURCE_SAMPLING_RATE = 256
SOURCE_SAMPLES = 256

TRAIN_SUBJECTS = (1, 2, 3, 6, 8, 9, 10, 11, 12, 13, 16, 17, 18, 19, 21, 24, 25, 28, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40)
VAL_SUBJECTS = (4, 7, 27, 33)
TEST_SUBJECTS = (5, 14, 15, 20, 22, 23, 26, 29)

# Exclude N170/Faces and N170/Cars to match the CSLP-AE 12-class setting.
TASK_REMAP = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
    7: 7,
    10: 8,
    11: 9,
    12: 10,
    13: 11,
}


def _resolve_pt_path(root: str | Path) -> Path:
    root = Path(root).expanduser().resolve()
    if root.is_file():
        return root
    path = root / "simple_data.pt"
    if path.is_file():
        return path
    raise FileNotFoundError(f"ERP CORE simple_data.pt not found at {root} or {path}")


def _split_indices(
    subjects: np.ndarray,
    tasks: np.ndarray,
    split_subjects: tuple[int, ...],
) -> np.ndarray:
    valid_tasks = np.asarray(tuple(TASK_REMAP), dtype=np.int64)
    mask = np.isin(subjects, split_subjects) & np.isin(tasks, valid_tasks)
    return np.flatnonzero(mask).astype(np.int64)


def _training_statistics(
    data: torch.Tensor,
    train_indices: np.ndarray,
    channel_indices: np.ndarray,
    chunk_size: int = 1024,
) -> tuple[np.ndarray, np.ndarray]:
    total_sum = torch.zeros(len(channel_indices), dtype=torch.float64)
    total_squared_sum = torch.zeros(len(channel_indices), dtype=torch.float64)
    total_points = 0
    channel_index_tensor = torch.as_tensor(channel_indices, dtype=torch.long)
    for start in range(0, len(train_indices), chunk_size):
        batch_indices = torch.as_tensor(train_indices[start:start + chunk_size], dtype=torch.long)
        chunk = data.index_select(0, batch_indices).index_select(1, channel_index_tensor)
        total_sum += chunk.sum(dim=(0, 2))
        total_squared_sum += torch.square(chunk).sum(dim=(0, 2))
        total_points += int(chunk.shape[0] * chunk.shape[2])
    mean = total_sum / total_points
    variance = total_squared_sum / total_points - torch.square(mean)
    std = torch.sqrt(torch.clamp(variance, min=0.0))
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or torch.any(std <= 0):
        raise ValueError("Invalid ERP CORE training normalization statistics")
    return mean.numpy(), std.numpy()


class ERPCOREPtLoader(Dataset):
    """In-memory ERP CORE split with train-subject normalization."""

    def __init__(
        self,
        payload: dict,
        indices: np.ndarray,
        split: str,
        mean: np.ndarray,
        std: np.ndarray,
        normalize_method: str,
        channel_names: list[str],
        target_samples: int,
    ):
        self.data = payload["data"]
        self.tasks = payload["tasks"].cpu().numpy()
        self.subject_values = payload["subjects"].cpu().numpy()
        self.indices = np.asarray(indices, dtype=np.int64)
        self.split = split
        self.normalize_method = normalize_method
        self.manifest_channel_names = list(ERPCORE_30_CHANNELS)
        self.channel_names = [str(name).strip().upper() for name in channel_names]
        # 旧逻辑只保存当前实验输入导联在原始 30 通道中的索引：
        # self.channel_indices = np.asarray(
        #     [self.manifest_channel_names.index(name) for name in self.channel_names],
        #     dtype=np.int64,
        # )
        # 固定 12 导联观测空间和 28 导联目标空间，并分别计算它们在
        # 原始 30 通道及目标 28 通道中的位置，保证后续抽取顺序一致。
        self.observed_channel_names = list(ERPCORE_12_CHANNELS)
        self.full_channel_names = list(ERPCORE_28_CHANNELS)
        self.full_channel_indices = np.asarray(
            [self.manifest_channel_names.index(name) for name in self.full_channel_names],
            dtype=np.int64,
        )
        self.observed_indices_in_full = np.asarray(
            [self.full_channel_names.index(name) for name in self.observed_channel_names],
            dtype=np.int64,
        )
        self.use_full_input = self.channel_names == self.full_channel_names
        self.mean = torch.as_tensor(mean, dtype=torch.float32)
        self.std = torch.as_tensor(std, dtype=torch.float32)
        self.target_samples = int(target_samples)
        self.labels = np.asarray([TASK_REMAP[int(self.tasks[idx])] for idx in self.indices], dtype=np.int64)
        self.label_counts = Counter(int(value) for value in self.labels)
        self.subjects = tuple(sorted({int(self.subject_values[idx]) for idx in self.indices}))
        
        self.subject_indices = defaultdict(list) # 该 subject 对应的样本位置
        self.task_indices = defaultdict(list)
        for local_index, (global_index, task) in enumerate(zip(self.indices, self.labels)):
            subject = int(self.subject_values[int(global_index)]) # 根据全局索引查出该样本属于哪个 subject。
            self.subject_indices[subject].append(local_index)  # e.g.subject_indices[3] = [0, 5, 9, 12]
            self.task_indices[int(task)].append(local_index)

    def __len__(self):
        return len(self.indices)

    def get_ch_names(self):
        return list(self.channel_names)

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
        # 旧逻辑只读取当前实验需要的导联，并只返回 (eeg, label)：
        # eeg = self.data[global_index, self.channel_indices, :].float()
        # eeg = self._normalize(eeg)
        # if eeg.shape[-1] != self.target_samples:
        #     eeg = F.interpolate(
        #         eeg.unsqueeze(0),
        #         size=self.target_samples,
        #         mode="linear",
        #         align_corners=False,
        #     ).squeeze(0)
        # if eeg.shape != (len(self.channel_names), self.target_samples):
        #     raise ValueError(f"Unexpected ERP CORE sample shape: {tuple(eeg.shape)}")
        # if not torch.isfinite(eeg).all():
        #     raise ValueError(f"NaN or Inf in normalized ERP CORE sample {global_index}")
        # return eeg.contiguous(), int(self.labels[index])

        # 从同一个原始样本先生成归一化、重采样后的 28 导联 x_full，
        # 再按固定索引抽取 12 导联 x_obs，避免两种输入的预处理不一致。
        x_full = self.data[global_index, self.full_channel_indices, :].float()
        x_full = self._normalize(x_full)
        if x_full.shape[-1] != self.target_samples:
            x_full = F.interpolate(
                x_full.unsqueeze(0),
                size=self.target_samples,
                mode="linear",
                align_corners=False,
            ).squeeze(0)
        if x_full.shape != (len(self.full_channel_names), self.target_samples):
            raise ValueError(f"Unexpected ERP CORE x_full shape: {tuple(x_full.shape)}")
        if not torch.isfinite(x_full).all():
            raise ValueError(f"NaN or Inf in normalized ERP CORE sample {global_index}")

        x_obs = x_full[self.observed_indices_in_full]
        if x_obs.shape != (len(self.observed_channel_names), self.target_samples):
            raise ValueError(f"Unexpected ERP CORE x_obs shape: {tuple(x_obs.shape)}")

        # O 将 x_full 作为主输入，其余实验将 x_obs 作为主输入；所有实验
        # 统一返回 (x, label, x_obs, x_full, subject, task)。
        x = x_full if self.use_full_input else x_obs
        label = int(self.labels[index])
        subject = int(self.subject_values[global_index])
        task = label
        return (
            x.contiguous(),
            label,
            x_obs.contiguous(),
            x_full.contiguous(),
            subject,
            task,
        )
        """  (
            x,        # [B, 12, 200]
            label,    # [B]
            x_obs,    # [B, 12, 200]
            x_full,   # [B, 28, 200]
            subject,  # [B]
            task,     # [B]
        )"""


def prepare_ERPCORE_pt_dataset(
    root,
    sampling_rate=200,
    normalize_method="z_score",
    channel_names=None,
    train_subjects=TRAIN_SUBJECTS,
    val_subjects=VAL_SUBJECTS,
    test_subjects=TEST_SUBJECTS,
):
    """Build the CSLP-AE subject-disjoint ERP CORE split from simple_data.pt."""

    pt_path = _resolve_pt_path(root)
    payload = torch.load(pt_path, map_location="cpu")
    required = {"data", "subjects", "tasks"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"ERP CORE PT missing keys: {sorted(missing)}")
    if tuple(payload["data"].shape[1:]) != (len(ERPCORE_30_CHANNELS), SOURCE_SAMPLES):
        raise ValueError(f"Unexpected ERP CORE data shape: {tuple(payload['data'].shape)}")

    # 旧逻辑默认读取原始 30 通道：
    # channel_names = list(ERPCORE_30_CHANNELS if channel_names is None else channel_names)
    # 新逻辑默认使用去掉 HEOG/VEOG 后的 28 导联目标空间。
    channel_names = list(ERPCORE_28_CHANNELS if channel_names is None else channel_names)
    channel_names = [str(name).strip().upper() for name in channel_names]
    unknown = [name for name in channel_names if name not in ERPCORE_30_CHANNELS]
    if unknown:
        raise ValueError(f"Unknown ERP CORE channel names: {unknown}")
    if len(set(channel_names)) != len(channel_names):
        raise ValueError(f"Duplicate ERP CORE channel names: {channel_names}")
    # 旧逻辑根据当前 channel_names 生成索引：
    # channel_indices = np.asarray(
    #     [ERPCORE_30_CHANNELS.index(name) for name in channel_names],
    #     dtype=np.int64,
    # )
    # 只接受实验协议中的 12/28 导联布局，并生成目标 28 导联在
    # 原始 30 通道中的索引，用于统一计算 x_full 的训练集统计量。
    supported_channel_layouts = {
        tuple(ERPCORE_12_CHANNELS),
        tuple(ERPCORE_28_CHANNELS),
    }
    if tuple(channel_names) not in supported_channel_layouts:
        raise ValueError(
            "ERP CORE channel_names must equal ERPCORE_12_CHANNELS or "
            "ERPCORE_28_CHANNELS"
        )
    full_channel_indices = np.asarray([ERPCORE_30_CHANNELS.index(name) for name in ERPCORE_28_CHANNELS], dtype=np.int64)

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

    # 旧逻辑只为当前实验输入导联计算统计量：
    # mean, std = _training_statistics(
    #     payload["data"], indices["train"], channel_indices
    # )
    # 只用训练 split 为目标 28 导联计算 mean/std，val/test 共用该统计量。
    mean, std = _training_statistics(payload["data"], indices["train"], full_channel_indices)
    datasets = {
        split: ERPCOREPtLoader(
            payload,
            split_indices,
            split=split,
            mean=mean,
            std=std,
            normalize_method=normalize_method,
            channel_names=channel_names,
            target_samples=target_samples,
        )
        for split, split_indices in indices.items()
    }
    print(
        "ERP CORE PT audit: "
        f"pt_path={pt_path}, source_rate={SOURCE_SAMPLING_RATE}, "
        f"target_samples={target_samples}, channels={len(channel_names)}, "
        f"train_subjects={datasets['train'].subjects}, "
        f"val_subjects={datasets['val'].subjects}, "
        f"test_subjects={datasets['test'].subjects}, "
        f"train={len(datasets['train'])}, val={len(datasets['val'])}, "
        f"test={len(datasets['test'])}, "
        f"train_labels={dict(datasets['train'].label_counts)}, "
        f"val_labels={dict(datasets['val'].label_counts)}, "
        f"test_labels={dict(datasets['test'].label_counts)}"
    )
    return datasets["train"], datasets["test"], datasets["val"]
