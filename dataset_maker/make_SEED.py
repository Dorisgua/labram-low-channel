"""Preprocess the SEED emotion dataset into LaBraM-ready pickle samples.

The official ``Preprocessed_EEG`` MAT files are already sampled at 200 Hz.
This script therefore filters them without resampling, cuts non-overlapping
one-second windows, converts the arrays to float32, and writes samples with
the following schema::

    {"X": np.ndarray(shape=(62, 200), dtype=np.float32), "Y": int}

Labels are mapped as negative/neutral/positive -> 0/1/2.  Output files are
grouped by the original subject id and named as follows::

    processed_data/<subject>/S<subject>_<session>_<trial>_<window>.pkl

By default, the final complete window of every trial is omitted to reproduce
AdaBrain's published preprocessing script and its 152,055-sample dataset.
Pass ``--keep-last-full-window`` only when AdaBrain sample-count compatibility
is not required.

Example (the defaults point to the shared raw data and the requested SSD
destination)::

    python dataset_maker/make_SEED.py --dry-run
    python dataset_maker/make_SEED.py
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import mne
import numpy as np
import scipy.io as sio


DEFAULT_INPUT_DIR = Path(
    "/inspire/hdd/project/sais-medical/public/share_medical/EEG/SEED/"
    "raw_data/Preprocessed_EEG"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/"
    "global_user/7461-chenxinhe/SEED"
)

SAMPLING_RATE = 200
EXPECTED_SUBJECTS = 15
EXPECTED_SESSIONS_PER_SUBJECT = 3
EXPECTED_TRIALS_PER_SESSION = 15
LABEL_MAP = {-1: 0, 0: 1, 1: 2}
LABEL_NAMES = {0: "negative", 1: "neutral", 2: "positive"}

SEED_62_CHANNELS = [
    "FP1", "FPZ", "FP2", "AF3", "AF4", "F7", "F5", "F3",
    "F1", "FZ", "F2", "F4", "F6", "F8", "FT7", "FC5",
    "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8", "T7",
    "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8",
    "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6",
    "TP8", "P7", "P5", "P3", "P1", "PZ", "P2", "P4",
    "P6", "P8", "PO7", "PO5", "PO3", "POZ", "PO4", "PO6",
    "PO8", "CB1", "O1", "OZ", "O2", "CB2",
]

SESSION_PATTERN = re.compile(r"^(?P<subject>\d+)_(?P<date>\d{8})\.mat$")
TRIAL_PATTERN = re.compile(r"eeg(?P<trial>\d+)$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert SEED 200 Hz MAT sessions to LaBraM pickle windows."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="directory containing the 45 session MAT files and label.mat",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="destination root; samples are written below processed_data/",
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=1.0,
        help="non-overlapping window length; 1 second gives 62x200 samples",
    )
    parser.add_argument("--l-freq", type=float, default=0.1)
    parser.add_argument("--h-freq", type=float, default=75.0)
    parser.add_argument("--notch-freq", type=float, default=50.0)
    parser.add_argument(
        "--keep-last-full-window",
        action="store_true",
        help=(
            "retain every complete window; default omits the final complete "
            "window of each trial to match AdaBrain's 152,055 samples"
        ),
    )
    output_mode = parser.add_mutually_exclusive_group()
    output_mode.add_argument(
        "--resume",
        action="store_true",
        help="skip sample files that already exist (for an interrupted run)",
    )
    output_mode.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite sample files that already exist",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate source structure and report expected output without writing",
    )
    return parser.parse_args()


def session_sort_key(path: Path) -> tuple[int, str]:
    match = SESSION_PATTERN.fullmatch(path.name)
    if match is None:
        raise ValueError(f"Unexpected SEED session filename: {path.name}")
    return int(match.group("subject")), match.group("date")


def discover_sessions(input_dir: Path) -> list[tuple[int, int, Path]]:
    session_paths = [
        path for path in input_dir.glob("*.mat") if path.name != "label.mat"
    ]
    session_paths.sort(key=session_sort_key)

    by_subject: dict[int, list[Path]] = defaultdict(list)
    for path in session_paths:
        subject_id, _ = session_sort_key(path)
        by_subject[subject_id].append(path)

    expected_subject_ids = set(range(1, EXPECTED_SUBJECTS + 1))
    if set(by_subject) != expected_subject_ids:
        raise ValueError(
            "SEED subject ids are incomplete: "
            f"got {sorted(by_subject)}, expected {sorted(expected_subject_ids)}"
        )

    sessions = []
    for subject_id in sorted(by_subject):
        subject_paths = sorted(by_subject[subject_id], key=session_sort_key)
        if len(subject_paths) != EXPECTED_SESSIONS_PER_SUBJECT:
            raise ValueError(
                f"Subject {subject_id} has {len(subject_paths)} sessions; "
                f"expected {EXPECTED_SESSIONS_PER_SUBJECT}"
            )
        for session_id, path in enumerate(subject_paths, start=1):
            sessions.append((subject_id, session_id, path))
    return sessions


def load_labels(input_dir: Path) -> list[int]:
    label_path = input_dir / "label.mat"
    if not label_path.is_file():
        raise FileNotFoundError(f"SEED label file not found: {label_path}")
    payload = sio.loadmat(label_path)
    if "label" not in payload:
        raise ValueError(f"Variable 'label' not found in {label_path}")
    source_labels = np.asarray(payload["label"]).reshape(-1)
    if source_labels.size != EXPECTED_TRIALS_PER_SESSION:
        raise ValueError(
            f"Expected {EXPECTED_TRIALS_PER_SESSION} labels, "
            f"found {source_labels.size}"
        )
    unknown = set(int(value) for value in source_labels) - set(LABEL_MAP)
    if unknown:
        raise ValueError(f"Unexpected SEED source labels: {sorted(unknown)}")
    return [LABEL_MAP[int(value)] for value in source_labels]


def ordered_trial_keys(payload: dict) -> list[tuple[int, str]]:
    trials = []
    for key in payload:
        match = TRIAL_PATTERN.search(key)
        if match is not None:
            trials.append((int(match.group("trial")), key))
    trials.sort()
    expected = list(range(1, EXPECTED_TRIALS_PER_SESSION + 1))
    if [trial_id for trial_id, _ in trials] != expected:
        raise ValueError(
            f"Unexpected trial variables: {[key for _, key in trials]}"
        )
    return trials


def validate_mat_headers(
    sessions: Iterable[tuple[int, int, Path]],
    window_samples: int,
    drop_last_full_window: bool,
) -> tuple[int, int]:
    total_time_points = 0
    expected_windows = 0
    for subject_id, session_id, path in sessions:
        variables = sio.whosmat(path)
        trial_shapes = []
        for name, shape, _ in variables:
            match = TRIAL_PATTERN.search(name)
            if match is not None:
                trial_shapes.append((int(match.group("trial")), shape))
        trial_shapes.sort()
        expected_trial_ids = list(range(1, EXPECTED_TRIALS_PER_SESSION + 1))
        if [trial_id for trial_id, _ in trial_shapes] != expected_trial_ids:
            raise ValueError(f"Invalid trial variables in {path}")
        for trial_id, shape in trial_shapes:
            if len(shape) != 2 or shape[0] != len(SEED_62_CHANNELS):
                raise ValueError(
                    f"Invalid shape in subject={subject_id} session={session_id} "
                    f"trial={trial_id}: {shape}"
                )
            total_time_points += shape[1]
            trial_windows = shape[1] // window_samples
            if drop_last_full_window:
                trial_windows = max(0, trial_windows - 1)
            expected_windows += trial_windows
    return total_time_points, expected_windows


def filter_trial(
    data: np.ndarray, l_freq: float, h_freq: float, notch_freq: float
) -> np.ndarray:
    if data.ndim != 2 or data.shape[0] != len(SEED_62_CHANNELS):
        raise ValueError(f"Unexpected SEED trial shape: {data.shape}")
    if not np.isfinite(data).all():
        raise ValueError("SEED trial contains NaN or Inf before filtering")

    filtered = mne.filter.filter_data(
        np.asarray(data, dtype=np.float64),
        sfreq=SAMPLING_RATE,
        l_freq=l_freq,
        h_freq=h_freq,
        verbose="ERROR",
    )
    filtered = mne.filter.notch_filter(
        filtered,
        Fs=SAMPLING_RATE,
        freqs=np.asarray([notch_freq]),
        verbose="ERROR",
    )
    if not np.isfinite(filtered).all():
        raise ValueError("SEED trial contains NaN or Inf after filtering")
    return filtered


def atomic_pickle_dump(payload: dict, destination: Path) -> None:
    temporary = destination.with_name(
        f".{destination.name}.tmp-{os.getpid()}"
    )
    try:
        with temporary.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_json_dump(payload: dict, destination: Path) -> None:
    temporary = destination.with_name(
        f".{destination.name}.tmp-{os.getpid()}"
    )
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def ensure_output_is_safe(
    processed_dir: Path,
    config_path: Path,
    config: dict,
    resume: bool,
    overwrite: bool,
) -> None:
    if not processed_dir.exists():
        return
    try:
        first_existing = next(processed_dir.glob("*/*.pkl"))
    except StopIteration:
        return
    if not resume and not overwrite:
        raise FileExistsError(
            f"Output already contains samples, for example {first_existing}. "
            "Use --resume to skip them or --overwrite to replace them."
        )
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Output contains samples but has no {config_path.name}; refusing to "
            "mix data from an unknown preprocessing configuration. Use a new "
            "--output-root."
        )
    with config_path.open("r", encoding="utf-8") as handle:
        existing_config = json.load(handle)
    if existing_config != config:
        raise ValueError(
            "Existing samples were generated with different preprocessing "
            f"parameters ({config_path}). Use a new --output-root."
        )


def preprocess(args: argparse.Namespace) -> None:
    input_dir = args.input_dir.resolve()
    output_root = args.output_root.resolve()
    processed_dir = output_root / "processed_data"
    config_path = output_root / "preprocess_config.json"

    if not input_dir.is_dir():
        raise FileNotFoundError(f"SEED input directory not found: {input_dir}")
    window_samples_float = args.window_seconds * SAMPLING_RATE
    window_samples = int(round(window_samples_float))
    drop_last_full_window = not args.keep_last_full_window
    if window_samples <= 0 or not np.isclose(window_samples, window_samples_float):
        raise ValueError(
            "--window-seconds must produce a positive integer number of samples"
        )
    if not (0 <= args.l_freq < args.h_freq < SAMPLING_RATE / 2):
        raise ValueError("Filter cutoffs must satisfy 0 <= low < high < 100 Hz")
    if not (0 < args.notch_freq < SAMPLING_RATE / 2):
        raise ValueError("Notch frequency must be between 0 and 100 Hz")

    sessions = discover_sessions(input_dir)
    labels = load_labels(input_dir)
    total_raw_points, expected_windows = validate_mat_headers(
        sessions, window_samples, drop_last_full_window
    )
    print(
        f"Validated {len(sessions)} sessions from {EXPECTED_SUBJECTS} subjects; "
        f"sampling_rate={SAMPLING_RATE} Hz, channels={len(SEED_62_CHANNELS)}, "
        f"window={window_samples} samples ({args.window_seconds:g} s), "
        f"adabrain_drop_last={drop_last_full_window}."
    )
    if args.dry_run:
        print(
            "Dry run completed successfully. "
            f"Expected windows={expected_windows}; "
            f"total per-channel time points={total_raw_points}."
        )
        return

    run_config = {
        "dataset": "SEED",
        "source_dir": str(input_dir),
        "sampling_rate": SAMPLING_RATE,
        "window_seconds": args.window_seconds,
        "window_samples": window_samples,
        "sample_shape": [len(SEED_62_CHANNELS), window_samples],
        "dtype": "float32",
        "drop_last_full_window": drop_last_full_window,
        "channels": SEED_62_CHANNELS,
        "filter": {
            "l_freq": args.l_freq,
            "h_freq": args.h_freq,
            "notch_freq": args.notch_freq,
        },
        "source_to_class_id": {str(key): value for key, value in LABEL_MAP.items()},
    }
    ensure_output_is_safe(
        processed_dir,
        config_path,
        run_config,
        args.resume,
        args.overwrite,
    )
    processed_dir.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(run_config, config_path)

    total_written = 0
    total_skipped = 0
    label_counts: Counter[int] = Counter()
    session_summaries = []

    for session_number, (subject_id, session_id, mat_path) in enumerate(
        sessions, start=1
    ):
        print(
            f"[{session_number:02d}/{len(sessions)}] subject={subject_id:02d} "
            f"session={session_id} file={mat_path.name}",
            flush=True,
        )
        payload = sio.loadmat(mat_path)
        trials = ordered_trial_keys(payload)
        subject_dir = processed_dir / str(subject_id)
        subject_dir.mkdir(parents=True, exist_ok=True)
        session_windows = 0

        for trial_id, key in trials:
            filtered = filter_trial(
                payload[key], args.l_freq, args.h_freq, args.notch_freq
            )
            num_windows = filtered.shape[-1] // window_samples
            if drop_last_full_window:
                num_windows = max(0, num_windows - 1)
            label = labels[trial_id - 1]
            for window_id in range(1, num_windows + 1):
                start = (window_id - 1) * window_samples
                sample = np.ascontiguousarray(
                    filtered[:, start : start + window_samples], dtype=np.float32
                )
                filename = (
                    f"S{subject_id}_{session_id}_{trial_id}_{window_id}.pkl"
                )
                destination = subject_dir / filename
                if destination.exists() and args.resume:
                    total_skipped += 1
                else:
                    atomic_pickle_dump({"X": sample, "Y": label}, destination)
                    total_written += 1
                label_counts[label] += 1
                session_windows += 1

        session_summaries.append(
            {
                "subject_id": subject_id,
                "session_id": session_id,
                "source_file": str(mat_path),
                "windows": session_windows,
            }
        )

    metadata = {
        "dataset": "SEED",
        "source_dir": str(input_dir),
        "processed_dir": str(processed_dir),
        "sampling_rate": SAMPLING_RATE,
        "window_seconds": args.window_seconds,
        "window_samples": window_samples,
        "sample_shape": [len(SEED_62_CHANNELS), window_samples],
        "dtype": "float32",
        "channels": SEED_62_CHANNELS,
        "filter": {
            "l_freq": args.l_freq,
            "h_freq": args.h_freq,
            "notch_freq": args.notch_freq,
        },
        "source_to_class_id": {str(key): value for key, value in LABEL_MAP.items()},
        "class_names": {str(key): value for key, value in LABEL_NAMES.items()},
        "total_samples": int(sum(label_counts.values())),
        "expected_samples": expected_windows,
        "written_samples": total_written,
        "skipped_samples": total_skipped,
        "label_counts": {
            str(key): int(label_counts[key]) for key in sorted(label_counts)
        },
        "sessions": session_summaries,
    }
    atomic_json_dump(metadata, output_root / "preprocess_metadata.json")
    if metadata["total_samples"] != expected_windows:
        raise RuntimeError(
            f"Wrote/accounted for {metadata['total_samples']} samples, "
            f"but dry-run structure predicted {expected_windows}"
        )
    print(
        f"Completed: total={metadata['total_samples']}, written={total_written}, "
        f"skipped={total_skipped}, labels={metadata['label_counts']}"
    )
    print(f"Samples: {processed_dir}")
    print(f"Metadata: {output_root / 'preprocess_metadata.json'}")


def main() -> None:
    preprocess(parse_args())


if __name__ == "__main__":
    main()
