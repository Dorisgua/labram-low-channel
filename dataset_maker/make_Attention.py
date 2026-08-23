"""Build BrainPro-style Attention/Rest windows for LaBraM.

The extracted raw dataset contains VP001--VP026, each with dsr, nback, and wg
continuous EEG recordings plus marker files.  BrainPro reports 26 subjects,
4 s non-overlapping windows at 200 Hz, and 4680 samples.  The local raw data
matches that total as 26 subjects x 3 tasks x 60 marker-aligned windows.

Labels are intentionally recorded in the manifest because the public raw
markers do not expose an explicit Rest class for dsr/nback:
  * wg: WG -> attention(1), BL -> rest(0)
  * dsr/nback: first 60 non-session task markers -> attention(1)
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.io import loadmat

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Channels_definition import ATTENTION_26_CHANNELS


DEFAULT_INPUT_ROOT = Path(
    "/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/"
    "global_user/7461-chenxinhe/Attention"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/"
    "global_user/7461-chenxinhe/Attention/processed_data_4s_200hz"
)

EXPECTED_SUBJECTS = tuple(range(1, 27))
TASKS = ("dsr", "nback", "wg")
TARGET_RATE = 200
WINDOW_SECONDS = 4
TARGET_SAMPLES = TARGET_RATE * WINDOW_SECONDS
WINDOWS_PER_TASK = 60
CLASS_NAMES = {0: "rest", 1: "attention"}

RAW_TO_STANDARD = {
    "Fp1": "FP1",
    "AFz": "AFZ",
    "F1": "F1",
    "FC5": "FC5",
    "FC1": "FC1",
    "T7": "T7",
    "C3": "C3",
    "Cz": "CZ",
    "CP5": "CP5",
    "CP1": "CP1",
    "P7": "P7",
    "P3": "P3",
    "Pz": "PZ",
    "POz": "POZ",
    "O1": "O1",
    "Fp2": "FP2",
    "F2": "F2",
    "FC2": "FC2",
    "FC6": "FC6",
    "C4": "C4",
    "T8": "T8",
    "CP2": "CP2",
    "CP6": "CP6",
    "P4": "P4",
    "P8": "P8",
    "O2": "O2",
}


def parse_subjects(value: str) -> tuple[int, ...]:
    subjects: set[int] = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start_text, stop_text = item.split("-", 1)
            start, stop = int(start_text), int(stop_text)
            if start > stop:
                raise argparse.ArgumentTypeError(f"Invalid range: {item}")
            subjects.update(range(start, stop + 1))
        else:
            subjects.add(int(item))
    invalid = sorted(subjects - set(EXPECTED_SUBJECTS))
    if not subjects or invalid:
        raise argparse.ArgumentTypeError(
            f"Subjects must be within 1--26; invalid={invalid}"
        )
    return tuple(sorted(subjects))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build 26-channel, 4-second, 200 Hz Attention/Rest arrays."
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--subjects", type=parse_subjects, default=EXPECTED_SUBJECTS)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _load_recording(subject_dir: Path, task: str):
    cnt = loadmat(
        subject_dir / f"cnt_{task}.mat", squeeze_me=True, struct_as_record=False
    )[f"cnt_{task}"]
    mrk = loadmat(
        subject_dir / f"mrk_{task}.mat", squeeze_me=True, struct_as_record=False
    )[f"mrk_{task}"]
    if int(cnt.fs) != TARGET_RATE:
        raise ValueError(f"{subject_dir.name} {task} expected 200 Hz, got {cnt.fs}")
    raw_channels = [str(value) for value in cnt.clab]
    raw_to_index = {name: idx for idx, name in enumerate(raw_channels)}
    selected_indices = []
    for raw_name, standard_name in RAW_TO_STANDARD.items():
        if standard_name not in ATTENTION_26_CHANNELS:
            continue
        if raw_name not in raw_to_index:
            raise ValueError(f"{subject_dir.name} {task} lacks channel {raw_name}")
        selected_indices.append(raw_to_index[raw_name])
    if len(selected_indices) != len(ATTENTION_26_CHANNELS):
        raise ValueError("Attention selected channel count mismatch")
    return np.asarray(cnt.x), mrk, np.asarray(selected_indices, dtype=np.int64)


def _one_hot_label(mrk, event_index: int) -> str | None:
    y = np.asarray(mrk.y)
    if y.ndim == 1:
        y = y[None, :]
    active = np.flatnonzero(y[:, event_index])
    if len(active) != 1:
        return None
    return str(np.asarray(mrk.className, dtype=object)[active[0]])


def _task_events(mrk, task: str) -> list[tuple[int, int]]:
    times = np.asarray(mrk.time, dtype=np.float64)
    events: list[tuple[int, int]] = []
    for event_index, time_ms in enumerate(times):
        class_name = _one_hot_label(mrk, event_index)
        if class_name is None:
            continue
        if task == "wg":
            if class_name == "WG":
                label = 1
            elif class_name == "BL":
                label = 0
            else:
                continue
        else:
            if "session" in class_name.lower():
                continue
            label = 1
        sample_index = int(round(float(time_ms) * TARGET_RATE / 1000.0))
        events.append((sample_index, label))
        if len(events) == WINDOWS_PER_TASK:
            break
    if len(events) != WINDOWS_PER_TASK:
        raise ValueError(
            f"{task} expected {WINDOWS_PER_TASK} usable events, got {len(events)}"
        )
    return events


def _process_subject(input_root: Path, subject: int) -> dict:
    subject_dir = input_root / f"VP{subject:03d}-EEG"
    if not subject_dir.is_dir():
        raise FileNotFoundError(f"Missing Attention subject directory: {subject_dir}")
    windows = []
    labels = []
    task_counts = Counter()
    skipped_oob = 0
    for task in TASKS:
        data, mrk, selected_indices = _load_recording(subject_dir, task)
        events = _task_events(mrk, task)
        for sample_index, label in events:
            stop = sample_index + TARGET_SAMPLES
            if sample_index < 0 or stop > data.shape[0]:
                skipped_oob += 1
                continue
            window = data[sample_index:stop, selected_indices].T
            windows.append(np.asarray(window, dtype=np.float32))
            labels.append(int(label))
            task_counts[task] += 1
    if skipped_oob:
        raise ValueError(f"{subject_dir.name} has out-of-bounds windows: {skipped_oob}")
    expected = len(TASKS) * WINDOWS_PER_TASK
    if len(windows) != expected:
        raise ValueError(f"{subject_dir.name} expected {expected} windows, got {len(windows)}")
    eeg = np.stack(windows).astype(np.float32, copy=False)
    labels_array = np.asarray(labels, dtype=np.int64)
    if eeg.shape != (expected, len(ATTENTION_26_CHANNELS), TARGET_SAMPLES):
        raise ValueError(f"Unexpected Attention shape for {subject_dir.name}: {eeg.shape}")
    if not np.isfinite(eeg).all():
        raise ValueError(f"NaN or Inf in Attention subject {subject}")
    channel_sum = eeg.sum(axis=(0, 2), dtype=np.float64)
    channel_squared_sum = np.square(eeg, dtype=np.float64).sum(axis=(0, 2))
    return {
        "eeg": eeg,
        "labels": labels_array,
        "record": {
            "subject_id": subject,
            "eeg_file": f"VP{subject:03d}_eeg.npy",
            "label_file": f"VP{subject:03d}_labels.npy",
            "trials": int(eeg.shape[0]),
            "task_counts": {key: int(task_counts[key]) for key in TASKS},
            "label_counts": {
                str(key): int(value) for key, value in Counter(labels_array.tolist()).items()
            },
            "channel_sum": channel_sum.tolist(),
            "channel_squared_sum": channel_squared_sum.tolist(),
            "sample_points_per_channel": int(eeg.shape[0] * eeg.shape[2]),
        },
    }


def preprocess(args: argparse.Namespace) -> None:
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    if not input_root.is_dir():
        raise FileNotFoundError(f"Missing Attention input root: {input_root}")
    if output_root.exists() and args.overwrite:
        for path in sorted(output_root.glob("*")):
            if path.is_file():
                path.unlink()
    output_root.mkdir(parents=True, exist_ok=True)

    subject_records = []
    total_label_counts = Counter()
    for index, subject in enumerate(args.subjects, start=1):
        eeg_path = output_root / f"VP{subject:03d}_eeg.npy"
        label_path = output_root / f"VP{subject:03d}_labels.npy"
        if args.resume and eeg_path.is_file() and label_path.is_file():
            labels = np.load(label_path, allow_pickle=False)
            record = {
                "subject_id": subject,
                "eeg_file": eeg_path.name,
                "label_file": label_path.name,
                "trials": int(labels.shape[0]),
                "task_counts": {task: WINDOWS_PER_TASK for task in TASKS},
                "label_counts": {
                    str(key): int(value) for key, value in Counter(labels.tolist()).items()
                },
            }
            eeg = np.load(eeg_path, mmap_mode="r", allow_pickle=False)
            record["channel_sum"] = eeg.sum(axis=(0, 2), dtype=np.float64).tolist()
            record["channel_squared_sum"] = np.square(eeg, dtype=np.float64).sum(axis=(0, 2)).tolist()
            record["sample_points_per_channel"] = int(eeg.shape[0] * eeg.shape[2])
        else:
            print(f"[{index:02d}/{len(args.subjects)}] VP{subject:03d}: processing", flush=True)
            result = _process_subject(input_root, subject)
            record = result["record"]
            if not args.dry_run:
                np.save(eeg_path, result["eeg"], allow_pickle=False)
                np.save(label_path, result["labels"], allow_pickle=False)
        subject_records.append(record)
        total_label_counts.update({int(k): int(v) for k, v in record["label_counts"].items()})

    manifest = {
        "dataset": "Attention",
        "schema_version": 1,
        "source": "EEG_01-26_MATLAB",
        "source_root": str(input_root),
        "target_sampling_rate": TARGET_RATE,
        "window_seconds": WINDOW_SECONDS,
        "window_overlap": 0.0,
        "sample_shape": [len(ATTENTION_26_CHANNELS), TARGET_SAMPLES],
        "dtype": "float32",
        "unit": "microvolt",
        "channels": list(ATTENTION_26_CHANNELS),
        "raw_excluded_channels": ["AFF5h", "AFF6h", "HEOG", "VEOG"],
        "tasks": list(TASKS),
        "windows_per_task_per_subject": WINDOWS_PER_TASK,
        "class_names": {str(k): v for k, v in CLASS_NAMES.items()},
        "label_policy": (
            "BrainPro total 4680 matched as 26 subjects x 3 tasks x 60 windows. "
            "WG markers use WG=attention and BL=rest; DSR/NBACK use the first "
            "60 non-session task markers as attention because no explicit Rest "
            "class is present in those marker files."
        ),
        "split": {
            "train_subjects": list(range(1, 21)),
            "val_subjects": [21, 22, 23],
            "test_subjects": [24, 25, 26],
        },
        "subject_records": subject_records,
        "total_samples": int(sum(record["trials"] for record in subject_records)),
        "label_counts": {str(k): int(v) for k, v in sorted(total_label_counts.items())},
    }
    if not args.dry_run:
        with (output_root / "manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)
    print(
        f"Completed: samples={manifest['total_samples']}, labels={manifest['label_counts']}, "
        f"manifest={output_root / 'manifest.json'}"
    )


def main() -> None:
    preprocess(parse_args())


if __name__ == "__main__":
    main()
