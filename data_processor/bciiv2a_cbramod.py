"""BCI-IV-2A subject-disjoint split used by the CBraMod protocol."""

from collections import Counter
from pathlib import Path

import numpy as np

from data_processor.bciiv2a import BCIIV2AMultiSessionLoader


EXPECTED_SUBJECTS = {"train": set(range(5)), "val": {5, 6}, "test": {7, 8}}


def prepare_BCIIV2A_cbramod_dataset(
    root, sampling_rate=200, normalize_method="z_score", channel_names=None,
    preload=True,
):
    """Return train, test, val with the same six-field sample format as BCI D."""
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
    train = datasets["train"]
    paths = {}
    for split, dataset in datasets.items():
        expected_subjects = EXPECTED_SUBJECTS[split]
        counts = Counter(int(record["subject_id"]) for record in dataset.files)
        if set(counts) != expected_subjects or set(counts.values()) != {576}:
            raise ValueError(f"Unexpected CBraMod {split} subjects/trials: {dict(counts)}")
        if set(int(record["label"]) for record in dataset.files) != set(range(4)):
            raise ValueError(f"Unexpected CBraMod {split} labels")
        if dataset.get_ch_names() != train.get_ch_names():
            raise ValueError(f"CBraMod {split} channel order differs from train")
        if not np.array_equal(dataset.full_mean_value, train.full_mean_value) or not np.array_equal(
            dataset.full_std_value, train.full_std_value
        ):
            raise ValueError(f"CBraMod {split} normalization differs from train")
        paths[split] = {record["file"] for record in dataset.files}
        print(f"BCI-IV-2A CBraMod {split}: {len(dataset)} trials, subjects {sorted(counts)}")
    if paths["train"] & paths["val"] or paths["train"] & paths["test"] or paths["val"] & paths["test"]:
        raise ValueError("CBraMod split manifests overlap")
    if preload:
        for dataset in datasets.values():
            dataset.preload_into_memory()
    return datasets["train"], datasets["test"], datasets["val"]
