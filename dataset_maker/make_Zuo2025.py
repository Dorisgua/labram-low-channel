"""Build a LaBraM-ready 4-second version of the Zuo2025 dataset.

The public derivative files already contain artifact-rejected, average-
referenced, 2--45 Hz band-passed and 49--51 Hz notch-filtered EEG.  Their
10-second epochs cover -2 to +8 seconds relative to the motor-imagery cue.
This script keeps cue-relative 0--4 seconds (source samples 1000:3000 at
500 Hz), resamples it to 200 Hz with a polyphase anti-aliasing filter, and
writes one NumPy array per subject::

    subjects/sub01_eeg.npy     # [trials, 30, 800], float32, microvolts
    subjects/sub01_labels.npy  # [trials], int64, left=0/right=1
    manifest.json

The per-subject sufficient statistics in ``manifest.json`` let the loader
derive normalization statistics from training subjects only.

Examples::

    python dataset_maker/make_Zuo2025.py --dry-run
    python dataset_maker/make_Zuo2025.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
from scipy.signal import resample_poly


DEFAULT_INPUT_DIR = Path(
    "/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/"
    "global_user/7461-chenxinhe/Zuo2025/derivatives"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/"
    "global_user/7461-chenxinhe/Zuo2025/processed_data_4s_200hz"
)

EXPECTED_SUBJECTS = tuple(range(1, 31))
SOURCE_RATE = 500
TARGET_RATE = 200
SOURCE_TRIAL_SECONDS = 10
SOURCE_CUE_OFFSET_SECONDS = 2
WINDOW_SECONDS = 4
CROP_START = SOURCE_CUE_OFFSET_SECONDS * SOURCE_RATE
CROP_STOP = (SOURCE_CUE_OFFSET_SECONDS + WINDOW_SECONDS) * SOURCE_RATE
TARGET_SAMPLES = WINDOW_SECONDS * TARGET_RATE
EXPECTED_CHANNELS = 30
EXPECTED_TOTAL_TRIALS = 14034
SOURCE_LABEL_TO_CLASS_ID = {1: 0, 2: 1}
CLASS_NAMES = {0: "left_leg_motor_imagery", 1: "right_leg_motor_imagery"}
CHANNEL_NAMES = [
    "FP1", "FP2", "FZ", "F3", "F4", "F7", "F8", "FCZ",
    "FC3", "FC4", "FT7", "FT8", "CZ", "C3", "C4", "T3",
    "T4", "CPZ", "CP3", "CP4", "TP7", "TP8", "PZ", "P3",
    "P4", "T5", "T6", "OZ", "O1", "O2",
]
SOURCE_PATTERN = re.compile(r"^sub(?P<subject>\d{2})_task-MI_eeg-extraction\.mat$")


def parse_subjects(value: str) -> tuple[int, ...]:
    """Parse comma-separated ids and inclusive ranges, e.g. ``1-3,8``."""

    subjects: set[int] = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start_text, stop_text = item.split("-", 1)
            start, stop = int(start_text), int(stop_text)
            if start > stop:
                raise argparse.ArgumentTypeError(f"Invalid subject range: {item}")
            subjects.update(range(start, stop + 1))
        else:
            subjects.add(int(item))
    invalid = sorted(subjects - set(EXPECTED_SUBJECTS))
    if not subjects or invalid:
        raise argparse.ArgumentTypeError(
            f"Subjects must be within 1--30; invalid={invalid}"
        )
    return tuple(sorted(subjects))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crop Zuo2025 post-cue EEG to 4 s and resample to 200 Hz."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--subjects",
        type=parse_subjects,
        default=EXPECTED_SUBJECTS,
        help="subject ids/ranges to process (default: 1-30; useful for smoke tests)",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def source_path(input_dir: Path, subject: int) -> Path:
    return input_dir / f"sub{subject:02d}_task-MI_eeg-extraction.mat"


def inspect_source(path: Path, subject: int) -> tuple[int, Counter[int]]:
    if not path.is_file():
        raise FileNotFoundError(f"Zuo2025 subject {subject:02d} not found: {path}")
    match = SOURCE_PATTERN.fullmatch(path.name)
    if match is None or int(match.group("subject")) != subject:
        raise ValueError(f"Unexpected Zuo2025 derivative filename: {path.name}")
    with h5py.File(path, "r") as handle:
        if set(handle.keys()) != {"EEG_data", "label"}:
            raise ValueError(f"Unexpected HDF5 variables in {path}: {list(handle.keys())}")
        eeg = handle["EEG_data"]
        labels = np.asarray(handle["label"]).reshape(-1)
        if eeg.ndim != 3 or eeg.shape[1:] != (
            EXPECTED_CHANNELS,
            SOURCE_TRIAL_SECONDS * SOURCE_RATE,
        ):
            raise ValueError(f"Unexpected EEG_data shape in {path}: {eeg.shape}")
        if eeg.dtype != np.float32:
            raise ValueError(f"Expected float32 EEG_data in {path}, got {eeg.dtype}")
        if labels.size != eeg.shape[0]:
            raise ValueError(f"Trial/label mismatch in {path}: {eeg.shape[0]} vs {labels.size}")
        rounded = labels.astype(np.int64)
        if not np.array_equal(labels, rounded):
            raise ValueError(f"Non-integer labels in {path}")
        unknown = set(rounded.tolist()) - set(SOURCE_LABEL_TO_CLASS_ID)
        if unknown:
            raise ValueError(f"Unknown labels in {path}: {sorted(unknown)}")
    return int(labels.size), Counter(int(value) for value in rounded)


def atomic_numpy_save(array: np.ndarray, destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            np.save(handle, array, allow_pickle=False)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


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


def load_and_transform(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as handle:
        source = np.asarray(handle["EEG_data"][:, :, CROP_START:CROP_STOP])
        source_labels = np.asarray(handle["label"]).reshape(-1).astype(np.int64)
    if not np.isfinite(source).all():
        raise ValueError(f"NaN or Inf in cropped source data: {path}")

    # 500 -> 200 Hz reduces exactly by 2/5; resample_poly applies its own
    # anti-aliasing FIR filter before decimation.
    eeg = resample_poly(source, up=2, down=5, axis=-1)
    eeg = np.ascontiguousarray(eeg, dtype=np.float32)
    if eeg.shape != (source.shape[0], EXPECTED_CHANNELS, TARGET_SAMPLES):
        raise ValueError(f"Unexpected transformed shape for {path}: {eeg.shape}")
    if not np.isfinite(eeg).all():
        raise ValueError(f"NaN or Inf after resampling: {path}")
    labels = np.asarray(
        [SOURCE_LABEL_TO_CLASS_ID[int(value)] for value in source_labels],
        dtype=np.int64,
    )
    return eeg, labels


def validate_saved_arrays(
    eeg_path: Path, label_path: Path, expected_trials: int
) -> tuple[np.ndarray, np.ndarray]:
    eeg = np.load(eeg_path, mmap_mode="r", allow_pickle=False)
    labels = np.load(label_path, mmap_mode="r", allow_pickle=False)
    expected_shape = (expected_trials, EXPECTED_CHANNELS, TARGET_SAMPLES)
    if eeg.shape != expected_shape or eeg.dtype != np.float32:
        raise ValueError(f"Invalid saved EEG array {eeg_path}: {eeg.shape}, {eeg.dtype}")
    if labels.shape != (expected_trials,) or labels.dtype != np.int64:
        raise ValueError(
            f"Invalid saved label array {label_path}: {labels.shape}, {labels.dtype}"
        )
    if not np.isfinite(eeg).all():
        raise ValueError(f"NaN or Inf in saved EEG array: {eeg_path}")
    return eeg, labels


def subject_statistics(eeg: np.ndarray) -> tuple[list[float], list[float], int]:
    channel_sum = np.sum(eeg, axis=(0, 2), dtype=np.float64)
    channel_squared_sum = np.sum(
        np.square(eeg, dtype=np.float64), axis=(0, 2), dtype=np.float64
    )
    count = int(eeg.shape[0] * eeg.shape[2])
    return channel_sum.tolist(), channel_squared_sum.tolist(), count


def preprocess(args: argparse.Namespace) -> None:
    input_dir = args.input_dir.resolve()
    output_root = args.output_root.resolve()
    subjects_dir = output_root / "subjects"
    config_path = output_root / "preprocess_config.json"
    manifest_path = output_root / "manifest.json"
    subjects = tuple(args.subjects)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Zuo2025 derivative directory not found: {input_dir}")

    source_audit: dict[int, tuple[int, Counter[int]]] = {}
    for subject in subjects:
        source_audit[subject] = inspect_source(source_path(input_dir, subject), subject)
    total_trials = sum(item[0] for item in source_audit.values())
    if subjects == EXPECTED_SUBJECTS and total_trials != EXPECTED_TOTAL_TRIALS:
        raise ValueError(
            f"Expected {EXPECTED_TOTAL_TRIALS} total trials, found {total_trials}"
        )
    print(
        f"Validated subjects={len(subjects)}, trials={total_trials}, "
        f"source_shape=[trial,30,5000], crop={CROP_START}:{CROP_STOP}, "
        f"target_shape=[trial,30,{TARGET_SAMPLES}]",
        flush=True,
    )
    if args.dry_run:
        print("Dry run completed successfully; no output was written.")
        return

    run_config = {
        "dataset": "Zuo2025",
        "schema_version": 1,
        "source_dir": str(input_dir),
        "subjects": list(subjects),
        "source_sampling_rate": SOURCE_RATE,
        "target_sampling_rate": TARGET_RATE,
        "source_epoch_seconds_relative_to_cue": [-2, 8],
        "selected_seconds_relative_to_cue": [0, 4],
        "source_crop_python_slice": [CROP_START, CROP_STOP],
        "sample_shape": [EXPECTED_CHANNELS, TARGET_SAMPLES],
        "dtype": "float32",
        "unit": "microvolt",
        "channels": CHANNEL_NAMES,
        "source_to_class_id": {
            str(key): value for key, value in SOURCE_LABEL_TO_CLASS_ID.items()
        },
        "resampling": "scipy.signal.resample_poly(up=2, down=5)",
    }
    if config_path.is_file():
        with config_path.open("r", encoding="utf-8") as handle:
            old_config = json.load(handle)
        if old_config != run_config:
            raise ValueError(
                f"Existing preprocessing config differs: {config_path}. "
                "Use a new --output-root."
            )
    elif subjects_dir.exists() and any(subjects_dir.glob("*.npy")):
        raise FileNotFoundError(
            f"Found output arrays without {config_path}; use a new --output-root"
        )

    output_root.mkdir(parents=True, exist_ok=True)
    subjects_dir.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(run_config, config_path)
    manifest_subjects = []
    global_label_counts: Counter[int] = Counter()

    for position, subject in enumerate(subjects, start=1):
        source = source_path(input_dir, subject)
        trials = source_audit[subject][0]
        eeg_path = subjects_dir / f"sub{subject:02d}_eeg.npy"
        label_path = subjects_dir / f"sub{subject:02d}_labels.npy"
        already_complete = eeg_path.is_file() and label_path.is_file()
        if already_complete and args.resume:
            print(f"[{position:02d}/{len(subjects)}] sub{subject:02d}: validating existing")
        elif (eeg_path.exists() or label_path.exists()) and not args.overwrite:
            raise FileExistsError(
                f"Output exists for sub{subject:02d}; use --resume or --overwrite"
            )
        else:
            print(f"[{position:02d}/{len(subjects)}] sub{subject:02d}: processing", flush=True)
            eeg, labels = load_and_transform(source)
            atomic_numpy_save(eeg, eeg_path)
            atomic_numpy_save(labels, label_path)

        eeg, labels = validate_saved_arrays(eeg_path, label_path, trials)
        label_counts = Counter(int(value) for value in np.asarray(labels))
        if set(label_counts) != set(CLASS_NAMES):
            raise ValueError(f"Incomplete label set for sub{subject:02d}: {label_counts}")
        channel_sum, channel_squared_sum, sample_points = subject_statistics(eeg)
        global_label_counts.update(label_counts)
        manifest_subjects.append(
            {
                "subject_id": subject,
                "source_file": str(source),
                "eeg_file": str(eeg_path.relative_to(output_root)),
                "label_file": str(label_path.relative_to(output_root)),
                "trials": trials,
                "label_counts": {
                    str(key): int(label_counts[key]) for key in sorted(label_counts)
                },
                "channel_sum": channel_sum,
                "channel_squared_sum": channel_squared_sum,
                "sample_points_per_channel": sample_points,
            }
        )

    manifest = {
        **run_config,
        "processed_root": str(output_root),
        "class_names": {str(key): value for key, value in CLASS_NAMES.items()},
        "total_trials": total_trials,
        "label_counts": {
            str(key): int(global_label_counts[key]) for key in sorted(global_label_counts)
        },
        "source_derivative_preprocessing": [
            "2-45 Hz band-pass",
            "49-51 Hz notch",
            "average rereference",
            "artifact rejection by the dataset authors",
        ],
        "subject_records": manifest_subjects,
    }
    atomic_json_dump(manifest, manifest_path)
    print(
        f"Completed: trials={total_trials}, labels={manifest['label_counts']}, "
        f"manifest={manifest_path}"
    )


def main() -> None:
    preprocess(parse_args())


if __name__ == "__main__":
    main()
