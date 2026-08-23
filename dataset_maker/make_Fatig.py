"""Convert data_eeg_FATIG_FTG pickle subjects into LaBraM-ready arrays.

The shared FATIG package is already windowed: each ``sub*.pkl`` contains a
``data`` list of ``(1, 30, 384)`` float32 arrays and a matching ``label`` list.
This maker validates those 128 Hz windows, resamples them to 200 Hz, removes
the singleton axis, and writes one EEG/label/statistics array per subject plus
an EEG-Deformer-style LOSO manifest consumed by ``data_processor/fatig.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

sys.path.append(str(Path(__file__).resolve().parents[1]))

from Channels_definition import FATIG_30_CHANNELS


DEFAULT_INPUT_ROOT = Path(
    "/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/"
    "global_user/7461-chenxinhe/fatig/data_eeg_FATIG_FTG"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/"
    "global_user/7461-chenxinhe/fatig/processed_data_3s_200hz"
)

SOURCE_RATE = 128
TARGET_RATE = 200
WINDOW_SECONDS = 3
SOURCE_SAMPLES = SOURCE_RATE * WINDOW_SECONDS
TARGET_SAMPLES = TARGET_RATE * WINDOW_SECONDS
SUBJECT_PATTERN = re.compile(r"^sub(?P<subject>\d+)\.pkl$")
LABEL_NAMES = {"0": "not_fatigued", "1": "fatigued"}
EEG_DEFORMER_FATIG_SUBJECTS = (0, 4, 21, 30, 34, 40, 41, 42, 43, 44, 52)
DEFAULT_VAL_RATE = 0.2
DEFAULT_SPLIT_SEED = 2023


def parse_subjects(value: str) -> tuple[int, ...]:
    subjects: set[int] = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if item.startswith("sub"):
            item = item[3:]
        subjects.add(int(item))
    if not subjects:
        raise argparse.ArgumentTypeError("Subject list is empty")
    return tuple(sorted(subjects))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert FATIG pickle windows to 3-second, 200 Hz arrays."
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--subjects",
        type=parse_subjects,
        default=EEG_DEFORMER_FATIG_SUBJECTS,
        help=(
            "subject ids to process, e.g. 0,4,21; default matches "
            "EEG-Deformer FATIG LOSO subjects"
        ),
    )
    parser.add_argument(
        "--val-rate",
        type=float,
        default=DEFAULT_VAL_RATE,
        help="sample-level validation fraction inside each LOSO training pool",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=DEFAULT_SPLIT_SEED,
        help="random seed for each LOSO sample-level train/val split",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def temporary_path(destination: Path) -> Path:
    return destination.with_name(f".{destination.name}.tmp-{os.getpid()}")


def atomic_json_dump(payload: dict, destination: Path) -> None:
    temporary = temporary_path(destination)
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_numpy_save(array: np.ndarray, destination: Path) -> None:
    temporary = temporary_path(destination)
    try:
        with temporary.open("wb") as handle:
            np.save(handle, array, allow_pickle=False)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def subject_id_from_path(path: Path) -> int:
    match = SUBJECT_PATTERN.match(path.name)
    if match is None:
        raise ValueError(f"Unexpected FATIG subject filename: {path.name}")
    return int(match.group("subject"))


def source_path(input_root: Path, subject_id: int) -> Path:
    return input_root / f"sub{subject_id}.pkl"


def load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def discover_subjects(input_root: Path, requested: tuple[int, ...] | None) -> tuple[int, ...]:
    if requested is not None:
        subjects = tuple(sorted(requested))
    else:
        subjects = tuple(
            sorted(subject_id_from_path(path) for path in input_root.glob("sub*.pkl"))
        )
    missing = [subject for subject in subjects if not source_path(input_root, subject).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing FATIG subject pickle(s): {missing}")
    if len(subjects) < 3:
        raise ValueError("FATIG preprocessing needs at least 3 subjects")
    return subjects


def load_channel_info(input_root: Path) -> list[str]:
    info_path = input_root / "dataset_info.pkl"
    if not info_path.is_file():
        raise FileNotFoundError(f"Missing FATIG dataset_info.pkl: {info_path}")
    info = load_pickle(info_path)
    channels = [str(name).upper() for name in info.get("original channel", [])]
    if channels != FATIG_30_CHANNELS:
        raise ValueError(
            f"Unexpected FATIG channel order: got {channels}, expected {FATIG_30_CHANNELS}"
        )
    return channels


def make_subject_arrays(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    payload = load_pickle(path)
    if not isinstance(payload, dict) or set(payload) < {"data", "label"}:
        raise ValueError(f"Invalid FATIG pickle payload: {path}")
    data_items = payload["data"]
    label_items = payload["label"]
    if len(data_items) != len(label_items) or not data_items:
        raise ValueError(f"Invalid FATIG sample/label count in {path}")

    samples = len(data_items)
    source_eeg = np.empty(
        (samples, len(FATIG_30_CHANNELS), SOURCE_SAMPLES), dtype=np.float32
    )
    labels = np.empty((samples,), dtype=np.int64)
    for index, (sample, label) in enumerate(zip(data_items, label_items)):
        sample = np.asarray(sample)
        if sample.shape != (1, len(FATIG_30_CHANNELS), SOURCE_SAMPLES):
            raise ValueError(f"Invalid FATIG sample shape in {path}: {sample.shape}")
        label = np.asarray(label).reshape(-1)
        if label.shape != (1,) or int(label[0]) not in (0, 1):
            raise ValueError(f"Invalid FATIG label in {path}: {label}")
        source_eeg[index] = sample[0].astype(np.float32, copy=False)
        labels[index] = int(label[0])

    if not np.isfinite(source_eeg).all():
        raise ValueError(f"NaN or Inf in FATIG data: {path}")
    eeg = resample_poly(source_eeg, up=25, down=16, axis=-1)
    eeg = np.asarray(eeg, dtype=np.float32)
    if eeg.shape != (samples, len(FATIG_30_CHANNELS), TARGET_SAMPLES):
        raise ValueError(f"Unexpected FATIG resampled shape in {path}: {eeg.shape}")
    if not np.isfinite(eeg).all():
        raise ValueError(f"NaN or Inf in resampled FATIG data: {path}")
    stats = np.empty((samples, len(FATIG_30_CHANNELS), 2), dtype=np.float64)
    eeg64 = eeg.astype(np.float64, copy=False)
    stats[:, :, 0] = np.sum(eeg64, axis=-1)
    stats[:, :, 1] = np.sum(np.square(eeg64), axis=-1)
    return eeg, labels, stats


def validate_split_args(val_rate: float) -> None:
    if not 0.0 < val_rate < 1.0:
        raise ValueError(f"FATIG val-rate must be between 0 and 1, got {val_rate}")


def make_loso_fold_splits(
    subjects: tuple[int, ...],
    sample_counts: dict[int, int],
    val_rate: float,
    split_seed: int,
) -> list[dict]:
    folds = []
    for fold_index, test_subject in enumerate(subjects):
        train_subjects = [subject for subject in subjects if subject != test_subject]
        pooled = [
            (subject, sample_index)
            for subject in train_subjects
            for sample_index in range(sample_counts[subject])
        ]
        shuffled = list(range(len(pooled)))
        random.Random(split_seed).shuffle(shuffled)
        split_at = int(len(shuffled) * (1.0 - val_rate))
        train_positions = shuffled[:split_at]
        val_positions = shuffled[split_at:]
        split_indices = {
            "train": {str(subject): [] for subject in train_subjects},
            "val": {str(subject): [] for subject in train_subjects},
        }
        for name, positions in (("train", train_positions), ("val", val_positions)):
            for position in positions:
                subject, sample_index = pooled[int(position)]
                split_indices[name][str(subject)].append(int(sample_index))
        for name in ("train", "val"):
            for subject in train_subjects:
                split_indices[name][str(subject)].sort()
        folds.append(
            {
                "fold_index": fold_index,
                "test_subject": test_subject,
                "train_subjects": train_subjects,
                "val_rate": val_rate,
                "split_seed": split_seed,
                "split_unit": "sample_after_subject_pooling",
                "train_indices": split_indices["train"],
                "val_indices": split_indices["val"],
                "train_samples": int(len(train_positions)),
                "val_samples": int(len(val_positions)),
                "test_samples": int(sample_counts[test_subject]),
            }
        )
    return folds


def preprocess(args: argparse.Namespace) -> None:
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    subjects_dir = output_root / "subjects"
    config_path = output_root / "preprocess_config.json"
    manifest_path = output_root / "manifest.json"
    if not input_root.is_dir():
        raise FileNotFoundError(f"FATIG input root not found: {input_root}")
    validate_split_args(args.val_rate)

    channels = load_channel_info(input_root)
    subjects = discover_subjects(input_root, args.subjects)
    if tuple(subjects) != EEG_DEFORMER_FATIG_SUBJECTS:
        print(
            "Warning: FATIG subjects differ from EEG-Deformer default "
            f"{list(EEG_DEFORMER_FATIG_SUBJECTS)}; using requested/discovered order.",
            flush=True,
        )
    print(
        f"FATIG source audit: subjects={list(subjects)}, "
        f"LOSO folds={len(subjects)}, val_rate={args.val_rate}, split_seed={args.split_seed}",
        flush=True,
    )
    if args.dry_run:
        print("Dry run completed successfully; no output was written.")
        return

    run_config = {
        "dataset": "Fatig",
        "schema_version": 2,
        "source_format": "data_eeg_FATIG_FTG_pickle",
        "source_root": str(input_root),
        "source_sampling_rate": SOURCE_RATE,
        "target_sampling_rate": TARGET_RATE,
        "window_seconds": WINDOW_SECONDS,
        "sample_shape": [len(channels), TARGET_SAMPLES],
        "dtype": "float32",
        "unit": "source",
        "channels": channels,
        "label_names": LABEL_NAMES,
        "resampling": "scipy.signal.resample_poly(up=25, down=16)",
        "subjects": list(subjects),
        "split_protocol": "EEG-Deformer FATIG LOSO",
        "validation_split": "sample-level random split after pooling non-test subjects",
        "val_rate": args.val_rate,
        "split_seed": args.split_seed,
    }
    if config_path.is_file():
        with config_path.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing != run_config and not (args.overwrite or args.resume):
            raise ValueError(
                f"Existing FATIG preprocessing config differs: {config_path}. "
                "Use a new --output-root, --resume, or --overwrite."
            )
    elif subjects_dir.exists() and any(subjects_dir.glob("*.npy")):
        raise FileNotFoundError(
            f"Found FATIG arrays without {config_path}; use a new --output-root"
        )

    output_root.mkdir(parents=True, exist_ok=True)
    subjects_dir.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(run_config, config_path)

    subject_records = []
    global_counts: Counter[int] = Counter()
    for position, subject_id in enumerate(subjects, start=1):
        source = source_path(input_root, subject_id)
        stem = f"sub{subject_id}"
        eeg_path = subjects_dir / f"{stem}_eeg.npy"
        label_path = subjects_dir / f"{stem}_labels.npy"
        stats_path = subjects_dir / f"{stem}_stats.npy"
        outputs = (eeg_path, label_path, stats_path)
        outputs_exist = all(path.is_file() for path in outputs)
        any_output = any(path.exists() for path in outputs)
        if outputs_exist and args.resume:
            print(f"[{position:02d}/{len(subjects)}] {stem}: validating existing")
        elif any_output and not args.overwrite and not args.resume:
            raise FileExistsError(
                f"FATIG output exists for {stem}; use --resume or --overwrite"
            )
        else:
            print(f"[{position:02d}/{len(subjects)}] {stem}: processing", flush=True)
            eeg, labels, stats = make_subject_arrays(source)
            atomic_numpy_save(eeg, eeg_path)
            atomic_numpy_save(labels, label_path)
            atomic_numpy_save(stats, stats_path)

        eeg = np.load(eeg_path, mmap_mode="r", allow_pickle=False)
        labels = np.load(label_path, mmap_mode="r", allow_pickle=False)
        stats = np.load(stats_path, mmap_mode="r", allow_pickle=False)
        if eeg.shape[1:] != (len(channels), TARGET_SAMPLES) or eeg.dtype != np.float32:
            raise ValueError(f"Invalid FATIG EEG output: {eeg_path} {eeg.shape} {eeg.dtype}")
        if labels.shape != (eeg.shape[0],) or labels.dtype != np.int64:
            raise ValueError(f"Invalid FATIG label output: {label_path}")
        if stats.shape != (eeg.shape[0], len(channels), 2) or stats.dtype != np.float64:
            raise ValueError(f"Invalid FATIG stats output: {stats_path}")
        label_counts = Counter(int(value) for value in np.asarray(labels))
        if set(label_counts) != {0, 1}:
            raise ValueError(f"FATIG subject lacks one class: {stem} {label_counts}")
        global_counts.update(label_counts)
        subject_records.append(
            {
                "subject_id": subject_id,
                "eeg_file": str(eeg_path.relative_to(output_root)),
                "label_file": str(label_path.relative_to(output_root)),
                "stats_file": str(stats_path.relative_to(output_root)),
                "source_file": str(source),
                "samples": int(eeg.shape[0]),
                "label_counts": {
                    str(key): int(label_counts[key]) for key in sorted(label_counts)
                },
            }
        )

    sample_counts = {
        int(record["subject_id"]): int(record["samples"]) for record in subject_records
    }
    loso_folds = make_loso_fold_splits(
        subjects=subjects,
        sample_counts=sample_counts,
        val_rate=args.val_rate,
        split_seed=args.split_seed,
    )
    manifest = {
        **run_config,
        "processed_root": str(output_root),
        "total_samples": sum(record["samples"] for record in subject_records),
        "label_counts": {
            str(key): int(global_counts[key]) for key in sorted(global_counts)
        },
        "loso_folds": loso_folds,
        "default_loso_fold": 0,
        "subject_records": subject_records,
    }
    atomic_json_dump(manifest, manifest_path)
    print(
        f"Completed: subjects={len(subject_records)}, samples={manifest['total_samples']}, "
        f"labels={manifest['label_counts']}, manifest={manifest_path}"
    )


def main() -> None:
    preprocess(parse_args())


if __name__ == "__main__":
    main()
