"""Validate PhysioNet EEGMMIDB EEG-FM-Bench Arrow shards.

The current ``data_processor/physionet.py`` consumes EEG-FM-Bench Arrow files
directly. This maker verifies the expected shard directory and writes a small
manifest for auditability; it does not rewrite the large Arrow shards.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys

import numpy as np

try:
    from datasets import Dataset as ArrowDataset
    from datasets import concatenate_datasets
except ImportError as exc:  # pragma: no cover
    ArrowDataset = None
    concatenate_datasets = None
    DATASETS_IMPORT_ERROR = exc
else:
    DATASETS_IMPORT_ERROR = None

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Channels_definition import PHYSIONET_64_CHANNELS


DEFAULT_ARROW_ROOT = Path(
    "/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/"
    "global_user/7461-chenxinhe/physionet/physionet.org/files/eegmmidb/"
    "processed_eegfmbench/processed/fs_200/motor_mv_img/finetune/1.0.0"
)
EXPECTED_SPLITS = {
    "train": {"arrow_name": "train", "samples": 6210, "subjects": range(1, 70)},
    "val": {"arrow_name": "validation", "samples": 1734, "subjects": range(70, 89)},
    "test": {"arrow_name": "test", "samples": 1893, "subjects": range(89, 110)},
}
EXPECTED_SAMPLE_SHAPE = (64, 800)
EXPECTED_LABELS = {0, 1, 2, 3}
CLASS_NAMES = {
    "0": "left_fist",
    "1": "right_fist",
    "2": "both_fists",
    "3": "both_feet",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate PhysioNet EEGMMIDB motor-imagery Arrow shards."
    )
    parser.add_argument("--arrow-root", type=Path, default=DEFAULT_ARROW_ROOT)
    parser.add_argument("--manifest-name", default="labram_manifest.json")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--strict-arrow",
        action="store_true",
        help="load Arrow shards and validate rows; requires datasets and dependencies",
    )
    return parser.parse_args()


def load_arrow_split(root: Path, split: str):
    if ArrowDataset is None:
        raise ImportError(
            "PhysioNet Arrow validation requires the 'datasets' package"
        ) from DATASETS_IMPORT_ERROR
    expected = EXPECTED_SPLITS[split]
    paths = sorted(root.glob(f"motor_mv_img-{expected['arrow_name']}-*.arrow"))
    if not paths:
        raise FileNotFoundError(f"No PhysioNet {split} Arrow shards in {root}")
    shards = [ArrowDataset.from_file(str(path)) for path in paths]
    dataset = shards[0] if len(shards) == 1 else concatenate_datasets(shards)
    return paths, dataset


def audit_split(root: Path, split: str, strict_arrow: bool) -> dict:
    expected = EXPECTED_SPLITS[split]
    paths = sorted(root.glob(f"motor_mv_img-{expected['arrow_name']}-*.arrow"))
    if not paths:
        raise FileNotFoundError(f"No PhysioNet {split} Arrow shards in {root}")
    if not strict_arrow and ArrowDataset is None:
        return {
            "split": split,
            "arrow_files": [path.name for path in paths],
            "samples": expected["samples"],
            "subjects": [min(expected["subjects"]), max(expected["subjects"])],
            "label_counts": "not_validated_without_datasets_dependency",
            "validation": "file_presence_only",
            "dependency_error": repr(DATASETS_IMPORT_ERROR),
        }
    _, dataset = load_arrow_split(root, split)
    expected = EXPECTED_SPLITS[split]
    if len(dataset) != expected["samples"]:
        raise ValueError(
            f"Unexpected PhysioNet {split} size: got {len(dataset)}, "
            f"expected {expected['samples']}"
        )
    required = {"data", "chs", "subject", "label"}
    missing = required - set(dataset.column_names)
    if missing:
        raise ValueError(f"PhysioNet {split} missing columns: {sorted(missing)}")
    subjects = {int(value) for value in dataset["subject"]}
    expected_subjects = set(expected["subjects"])
    if subjects != expected_subjects:
        raise ValueError(
            f"Unexpected PhysioNet {split} subjects: got {sorted(subjects)}, "
            f"expected {sorted(expected_subjects)}"
        )
    label_counts = Counter(int(value) for value in dataset["label"])
    if set(label_counts) != EXPECTED_LABELS:
        raise ValueError(f"Unexpected PhysioNet {split} labels: {sorted(label_counts)}")
    first = np.asarray(dataset[0]["data"], dtype=np.float32)
    if first.shape != EXPECTED_SAMPLE_SHAPE:
        raise ValueError(f"Unexpected PhysioNet sample shape: {first.shape}")
    return {
        "split": split,
        "arrow_files": [path.name for path in paths],
        "samples": len(dataset),
        "subjects": [min(subjects), max(subjects)],
        "label_counts": {str(key): int(label_counts[key]) for key in sorted(label_counts)},
        "validation": "arrow_rows",
    }


def atomic_json_dump(payload: dict, destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    args = parse_args()
    root = args.arrow_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"PhysioNet Arrow root not found: {root}")
    split_records = [
        audit_split(root, split, strict_arrow=args.strict_arrow)
        for split in ("train", "val", "test")
    ]
    manifest = {
        "dataset": "PhysioNet EEGMMIDB",
        "schema_version": 1,
        "source_format": "EEG-FM-Bench Arrow",
        "sample_shape": list(EXPECTED_SAMPLE_SHAPE),
        "sampling_rate": 200,
        "channels": PHYSIONET_64_CHANNELS,
        "class_names": CLASS_NAMES,
        "split_protocol": "subject-independent EEG-FM-Bench split",
        "splits": split_records,
    }
    print(
        "PhysioNet audit: "
        + ", ".join(f"{item['split']}={item['samples']}" for item in split_records)
        + f", root={root}"
    )
    if args.dry_run:
        print("Dry run completed successfully; no output was written.")
        return
    destination = root / args.manifest_name
    if destination.exists() and not args.overwrite:
        raise FileExistsError(f"Manifest exists: {destination}; pass --overwrite")
    atomic_json_dump(manifest, destination)
    print(f"Completed PhysioNet manifest: {destination}")


if __name__ == "__main__":
    main()
