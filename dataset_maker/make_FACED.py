"""Convert FACED processed pickles into LaBraM-ready 10-second windows.

The public folder already contains FACED ``Processed_data/subXXX.pkl`` files.
Each pickle is an array shaped ``[28, 32, 7500]``: 28 video trials, 32
channels, 30 seconds at 250 Hz.  This maker follows CBraMod's FACED protocol:
resample each trial to 200 Hz and split it into three non-overlapping
10-second samples.

Labels are CBraMod's 9-class emotion labels for the 28 videos.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Channels_definition import FACED_32_CHANNELS


DEFAULT_INPUT_ROOT = Path(
    "/inspire/alluxio/project/sais-medical/public/share_medical/EEG/"
    "faced/srcfiles/FACED/Processed_data"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/"
    "global_user/7461-chenxinhe/FACED/processed_data_10s_200hz"
)

EXPECTED_SUBJECTS = tuple(range(123))
TRAIN_SUBJECTS = tuple(range(80))
VAL_SUBJECTS = tuple(range(80, 100))
TEST_SUBJECTS = tuple(range(100, 123))
SOURCE_RATE = 250
TARGET_RATE = 200
WINDOW_SECONDS = 10
SOURCE_SHAPE = (28, 32, 7500)
TARGET_SAMPLES = TARGET_RATE * WINDOW_SECONDS
WINDOWS_PER_TRIAL = 3
TRIAL_EMOTION_LABELS = np.asarray(
    [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4,
     5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8],
    dtype=np.int64,
)
CLASS_NAMES = {
    "0": "Anger",
    "1": "Disgust",
    "2": "Fear",
    "3": "Sadness",
    "4": "Neutral",
    "5": "Amusement",
    "6": "Inspiration",
    "7": "Joy",
    "8": "Tenderness",
}


def parse_subjects(value: str) -> tuple[int, ...]:
    if value.lower() == "all":
        return EXPECTED_SUBJECTS
    subjects = tuple(sorted({int(item) for item in value.split(",") if item.strip()}))
    invalid = [subject for subject in subjects if subject not in EXPECTED_SUBJECTS]
    if invalid:
        raise ValueError(f"Subjects must be within 0--122; invalid={invalid}")
    return subjects


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build 32-channel, ten-second, 200 Hz FACED arrays."
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--subjects", type=parse_subjects, default=EXPECTED_SUBJECTS)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def source_path(root: Path, subject: int) -> Path:
    return root / f"sub{subject:03d}.pkl"


def output_paths(root: Path, subject: int) -> tuple[Path, Path]:
    subjects_dir = root / "subjects"
    return subjects_dir / f"sub{subject:03d}_eeg.npy", subjects_dir / f"sub{subject:03d}_labels.npy"


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


def atomic_numpy_save(array: np.ndarray, destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}.npy")
    try:
        np.save(temporary, array, allow_pickle=False)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_subject(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        data = pickle.load(handle)
    data = np.asarray(data)
    if data.shape != SOURCE_SHAPE:
        raise ValueError(f"Unexpected FACED source shape: {path} {data.shape}")
    if not np.isfinite(data).all():
        raise ValueError(f"NaN or Inf in FACED source: {path}")
    return data


def make_windows(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    resampled = resample_poly(data, up=4, down=5, axis=-1)
    expected_trial_samples = SOURCE_SHAPE[-1] * TARGET_RATE // SOURCE_RATE
    if resampled.shape != (SOURCE_SHAPE[0], SOURCE_SHAPE[1], expected_trial_samples):
        raise ValueError(f"Unexpected FACED resampled shape: {resampled.shape}")
    kept = resampled[:, :, : WINDOWS_PER_TRIAL * TARGET_SAMPLES]
    windows = kept.reshape(
        SOURCE_SHAPE[0],
        SOURCE_SHAPE[1],
        WINDOWS_PER_TRIAL,
        TARGET_SAMPLES,
    )
    windows = np.transpose(windows, (0, 2, 1, 3)).reshape(
        SOURCE_SHAPE[0] * WINDOWS_PER_TRIAL,
        SOURCE_SHAPE[1],
        TARGET_SAMPLES,
    )
    labels = np.repeat(TRIAL_EMOTION_LABELS, WINDOWS_PER_TRIAL)
    return np.ascontiguousarray(windows, dtype=np.float32), labels


def validate_saved_arrays(eeg_path: Path, label_path: Path) -> tuple[np.ndarray, np.ndarray]:
    eeg = np.load(eeg_path, mmap_mode="r", allow_pickle=False)
    labels = np.load(label_path, mmap_mode="r", allow_pickle=False)
    expected_shape = (
        SOURCE_SHAPE[0] * WINDOWS_PER_TRIAL,
        len(FACED_32_CHANNELS),
        TARGET_SAMPLES,
    )
    if eeg.shape != expected_shape or eeg.dtype != np.float32:
        raise ValueError(f"Invalid FACED EEG array: {eeg_path} {eeg.shape} {eeg.dtype}")
    if labels.shape != (expected_shape[0],) or labels.dtype != np.int64:
        raise ValueError(f"Invalid FACED labels: {label_path} {labels.shape} {labels.dtype}")
    if set(np.asarray(labels).tolist()) - set(range(len(CLASS_NAMES))):
        raise ValueError(f"Unknown FACED class ids in {label_path}")
    return eeg, labels


def subject_statistics(eeg: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    data = np.asarray(eeg, dtype=np.float64)
    channel_sum = data.sum(axis=(0, 2))
    channel_squared_sum = np.square(data).sum(axis=(0, 2))
    sample_points = data.shape[0] * data.shape[2]
    return channel_sum, channel_squared_sum, int(sample_points)


def make_subject_record(
    output_root: Path,
    subject: int,
    eeg_path: Path,
    label_path: Path,
    labels: np.ndarray,
    channel_sum: np.ndarray,
    channel_squared_sum: np.ndarray,
    sample_points: int,
) -> dict:
    label_counts = Counter(int(value) for value in np.asarray(labels))
    return {
        "subject_id": subject,
        "subject_name": f"sub{subject:03d}",
        "eeg_file": str(eeg_path.relative_to(output_root)),
        "label_file": str(label_path.relative_to(output_root)),
        "trials": int(labels.shape[0]),
        "label_counts": {str(key): int(label_counts[key]) for key in sorted(label_counts)},
        "channel_sum": channel_sum.tolist(),
        "channel_squared_sum": channel_squared_sum.tolist(),
        "sample_points_per_channel": sample_points,
    }


def process_subject(input_root: Path, output_root: Path, subject: int, overwrite: bool, resume: bool) -> dict:
    src = source_path(input_root, subject)
    if not src.is_file():
        raise FileNotFoundError(f"FACED source not found: {src}")
    eeg_path, label_path = output_paths(output_root, subject)
    if eeg_path.is_file() and label_path.is_file() and resume:
        eeg, labels = validate_saved_arrays(eeg_path, label_path)
        channel_sum, channel_squared_sum, sample_points = subject_statistics(eeg)
        return make_subject_record(
            output_root, subject, eeg_path, label_path, labels,
            channel_sum, channel_squared_sum, sample_points,
        )
    if (eeg_path.exists() or label_path.exists()) and not overwrite:
        raise FileExistsError(f"Output exists for sub{subject:03d}; use --resume or --overwrite")

    windows, labels = make_windows(load_subject(src))
    atomic_numpy_save(windows, eeg_path)
    atomic_numpy_save(labels, label_path)
    eeg, labels = validate_saved_arrays(eeg_path, label_path)
    channel_sum, channel_squared_sum, sample_points = subject_statistics(eeg)
    return make_subject_record(
        output_root, subject, eeg_path, label_path, labels,
        channel_sum, channel_squared_sum, sample_points,
    )


def preprocess(args: argparse.Namespace) -> None:
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    subjects = tuple(args.subjects)
    if not input_root.is_dir():
        raise FileNotFoundError(f"FACED input root not found: {input_root}")
    missing = [subject for subject in subjects if not source_path(input_root, subject).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing FACED subjects: {missing}")
    print(f"Validated FACED sources for subjects={subjects}", flush=True)
    if args.dry_run:
        return

    (output_root / "subjects").mkdir(parents=True, exist_ok=True)
    run_config = {
        "dataset": "FACED",
        "schema_version": 1,
        "source_root": str(input_root),
        "subjects": list(subjects),
        "source_sampling_rate": SOURCE_RATE,
        "target_sampling_rate": TARGET_RATE,
        "window_seconds": WINDOW_SECONDS,
        "windows_per_trial": WINDOWS_PER_TRIAL,
        "sample_shape": [len(FACED_32_CHANNELS), TARGET_SAMPLES],
        "dtype": "float32",
        "unit": "source_processed",
        "channels": list(FACED_32_CHANNELS),
        "label_mode": "emotion9",
        "class_names": CLASS_NAMES,
        "trial_emotion_labels": TRIAL_EMOTION_LABELS.tolist(),
        "resampling": "scipy.signal.resample_poly(up=4, down=5)",
        "default_split": {
            "train_subjects": list(TRAIN_SUBJECTS),
            "val_subjects": list(VAL_SUBJECTS),
            "test_subjects": list(TEST_SUBJECTS),
        },
    }
    atomic_json_dump(run_config, output_root / "preprocess_config.json")

    records = []
    global_label_counts: Counter[int] = Counter()
    for position, subject in enumerate(subjects, start=1):
        print(f"[{position:03d}/{len(subjects)}] sub{subject:03d}: processing", flush=True)
        record = process_subject(
            input_root,
            output_root,
            subject,
            overwrite=args.overwrite,
            resume=args.resume,
        )
        records.append(record)
        global_label_counts.update({int(k): int(v) for k, v in record["label_counts"].items()})
        print(
            f"sub{subject:03d}: windows={record['trials']}, labels={record['label_counts']}",
            flush=True,
        )

    manifest = {
        **run_config,
        "processed_root": str(output_root),
        "total_trials": int(sum(record["trials"] for record in records)),
        "label_counts": {
            str(key): int(global_label_counts[key]) for key in sorted(global_label_counts)
        },
        "subject_records": records,
    }
    atomic_json_dump(manifest, output_root / "manifest.json")
    print(f"Completed FACED preprocessing: {output_root / 'manifest.json'}", flush=True)


def main() -> None:
    preprocess(parse_args())


if __name__ == "__main__":
    main()
