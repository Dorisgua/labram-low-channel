"""Validate EEGMAT pickles and write an auditable manifest.

The current ``data_processor/eegmat.py`` reads the processed pickle directory
directly. This maker keeps that format intact and verifies the exact
AdaBrain-Bench protocol expected by the reader:

    Subject00 ... Subject35
    SubjectXX_1_1.pkl ... SubjectXX_1_15.pkl
    SubjectXX_2_1.pkl ... SubjectXX_2_15.pkl

Each pickle stores ``{"X": array[19, 2000], "Y": class_id}``.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
import pickle
from pathlib import Path

import numpy as np


DEFAULT_INPUT_ROOT = Path(
    "/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/"
    "global_user/7461-chenxinhe/EEGMAT"
)
EXPECTED_SUBJECTS = tuple(range(36))
EXPECTED_SAMPLE_SHAPE = (19, 2000)
TRAIN_SUBJECTS = tuple(range(32))
TEST_SUBJECTS = tuple(range(32, 36))
CLASS_NAMES = {"0": "session_1", "1": "session_2"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate EEGMAT processed pickles.")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--manifest-name", default="manifest.json")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_sample(path: Path, expected_label: int) -> np.ndarray:
    with path.open("rb") as handle:
        sample = pickle.load(handle)
    if not isinstance(sample, dict) or "X" not in sample or "Y" not in sample:
        raise ValueError(f"Invalid EEGMAT pickle schema: {path}")
    x = np.asarray(sample["X"])
    if x.shape != EXPECTED_SAMPLE_SHAPE:
        raise ValueError(f"Unexpected EEGMAT shape in {path}: {x.shape}")
    label = int(sample["Y"])
    if label != expected_label:
        raise ValueError(
            f"EEGMAT label mismatch in {path}: got {label}, expected {expected_label}"
        )
    if not np.isfinite(x).all():
        raise ValueError(f"NaN or Inf in EEGMAT sample: {path}")
    return x


def audit_subject(root: Path, subject: int) -> dict:
    subject_name = f"Subject{subject:02d}"
    subject_dir = root / subject_name
    if not subject_dir.is_dir():
        raise FileNotFoundError(f"Missing EEGMAT subject directory: {subject_dir}")
    label_counts: Counter[int] = Counter()
    files = []
    for session in (1, 2):
        expected_label = session - 1
        for segment in range(1, 16):
            path = subject_dir / f"{subject_name}_{session}_{segment}.pkl"
            if not path.is_file():
                raise FileNotFoundError(f"Missing EEGMAT sample: {path}")
            load_sample(path, expected_label)
            label_counts[expected_label] += 1
            files.append(str(path.relative_to(root)))
    return {
        "subject_id": subject,
        "subject_name": subject_name,
        "samples": len(files),
        "label_counts": {str(key): int(label_counts[key]) for key in sorted(label_counts)},
        "files": files,
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
    root = args.input_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"EEGMAT root not found: {root}")
    subject_records = [audit_subject(root, subject) for subject in EXPECTED_SUBJECTS]
    label_counts: Counter[int] = Counter()
    for record in subject_records:
        label_counts.update({int(k): int(v) for k, v in record["label_counts"].items()})
    manifest = {
        "dataset": "EEGMAT",
        "schema_version": 1,
        "sample_shape": list(EXPECTED_SAMPLE_SHAPE),
        "dtype": "source",
        "label_names": CLASS_NAMES,
        "subjects": list(EXPECTED_SUBJECTS),
        "train_subjects": list(TRAIN_SUBJECTS),
        "test_subjects": list(TEST_SUBJECTS),
        "validation_split": "seed-42 class-balanced 80/20 within train subjects, applied in data_processor/eegmat.py",
        "total_samples": sum(record["samples"] for record in subject_records),
        "label_counts": {str(key): int(label_counts[key]) for key in sorted(label_counts)},
        "subject_records": subject_records,
    }
    destination = root / args.manifest_name
    print(
        f"EEGMAT audit: subjects={len(subject_records)}, "
        f"samples={manifest['total_samples']}, labels={manifest['label_counts']}"
    )
    if args.dry_run:
        print("Dry run completed successfully; no output was written.")
        return
    if destination.exists() and not args.overwrite:
        raise FileExistsError(f"Manifest exists: {destination}; pass --overwrite")
    atomic_json_dump(manifest, destination)
    print(f"Completed EEGMAT manifest: {destination}")


if __name__ == "__main__":
    main()
