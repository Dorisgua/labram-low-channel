"""Preprocess Schirrmeister2017 HGD EDF files for LaBraM.

For every annotated four-second trial this script:

1. reads all 128 EEG channels in microvolts;
2. applies the official trial rejection rule ``max(abs(x)) < 800 uV``;
3. retains the 78 channels directly represented by LaBraM's position table;
4. resamples 500 Hz to 200 Hz with a polyphase anti-aliasing filter; and
5. stores one EEG and one label array per subject and official split.

The official train/test recording boundary is preserved.  Validation splitting
and train-only normalization are deliberately left to ``data_processor/hgd.py``.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import mne
import numpy as np
from scipy.signal import resample_poly


DEFAULT_INPUT_ROOT = Path(
    "/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/"
    "global_user/7461-chenxinhe/HGD/data"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/"
    "global_user/7461-chenxinhe/HGD/processed_data_4s_200hz"
)

EXPECTED_SUBJECTS = tuple(range(1, 15))
OFFICIAL_SPLITS = ("train", "test")
SOURCE_RATE = 500
TARGET_RATE = 200
WINDOW_SECONDS = 4
SOURCE_SAMPLES = SOURCE_RATE * WINDOW_SECONDS
TARGET_SAMPLES = TARGET_RATE * WINDOW_SECONDS
EXPECTED_EEG_CHANNELS = 128
ARTIFACT_THRESHOLD_UV = 800.0
RESAMPLE_CHUNK_TRIALS = 64
LABEL_MAP = {"right_hand": 0, "left_hand": 1, "rest": 2, "feet": 3}
CLASS_NAMES = {0: "right_hand", 1: "left_hand", 2: "rest", 3: "feet"}
HGD_78_CHANNELS = [
    "FP1", "FP2", "FPZ", "F7", "F3", "FZ", "F4", "F8",
    "FC5", "FC1", "FC2", "FC6", "M1", "T7", "C3", "CZ",
    "C4", "T8", "M2", "CP5", "CP1", "CP2", "CP6", "P7",
    "P3", "PZ", "P4", "P8", "POZ", "O1", "OZ", "O2",
    "AF7", "AF3", "AF4", "AF8", "F5", "F1", "F2", "F6",
    "FC3", "FCZ", "FC4", "C5", "C1", "C2", "C6", "CP3",
    "CPZ", "CP4", "P5", "P1", "P2", "P6", "PO5", "PO3",
    "PO4", "PO6", "FT7", "FT8", "TP7", "TP8", "PO7", "PO8",
    "FT9", "FT10", "TPP9h", "TPP10h", "PO9", "PO10", "P9", "P10",
    "AFZ", "IZ", "FTT9h", "FTT10h", "TTP7h", "TPP8h",
]


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
                raise argparse.ArgumentTypeError(f"Invalid subject range: {item}")
            subjects.update(range(start, stop + 1))
        else:
            subjects.add(int(item))
    invalid = sorted(subjects - set(EXPECTED_SUBJECTS))
    if not subjects or invalid:
        raise argparse.ArgumentTypeError(
            f"Subjects must be within 1--14; invalid={invalid}"
        )
    return tuple(sorted(subjects))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build 78-channel, four-second, 200 Hz HGD arrays."
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--subjects", type=parse_subjects, default=EXPECTED_SUBJECTS)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def source_path(root: Path, split: str, subject: int) -> Path:
    return root / split / f"{subject}.edf"


def eeg_name(raw_name: str) -> str:
    if not raw_name.startswith("EEG "):
        raise ValueError(f"Not an HGD EEG channel: {raw_name}")
    return raw_name[4:]


def selected_raw_indices(raw: mne.io.BaseRaw) -> tuple[list[int], list[int]]:
    eeg_indices = [
        index for index, name in enumerate(raw.ch_names) if name.startswith("EEG ")
    ]
    if len(eeg_indices) != EXPECTED_EEG_CHANNELS:
        raise ValueError(
            f"Expected {EXPECTED_EEG_CHANNELS} HGD EEG channels, found {len(eeg_indices)}"
        )
    canonical_by_upper = {name.upper(): name for name in HGD_78_CHANNELS}
    raw_index_by_canonical = {}
    for index in eeg_indices:
        stripped = eeg_name(raw.ch_names[index])
        canonical = canonical_by_upper.get(stripped.upper())
        if canonical is not None:
            if canonical in raw_index_by_canonical:
                raise ValueError(f"Duplicate HGD EEG channel: {canonical}")
            raw_index_by_canonical[canonical] = index
    missing = [name for name in HGD_78_CHANNELS if name not in raw_index_by_canonical]
    if missing:
        raise ValueError(f"HGD file lacks LaBraM-compatible channels: {missing}")
    selected_indices = [raw_index_by_canonical[name] for name in HGD_78_CHANNELS]
    eeg_positions = {raw_index: position for position, raw_index in enumerate(eeg_indices)}
    selected_positions = [eeg_positions[index] for index in selected_indices]
    return eeg_indices, selected_positions


def inspect_edf(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"HGD EDF not found: {path}")
    raw = mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
    if not np.isclose(raw.info["sfreq"], SOURCE_RATE):
        raise ValueError(f"Unexpected sampling rate in {path}: {raw.info['sfreq']}")
    selected_raw_indices(raw)
    descriptions = [str(value) for value in raw.annotations.description]
    unknown = set(descriptions) - set(LABEL_MAP)
    if unknown:
        raise ValueError(f"Unknown HGD annotations in {path}: {sorted(unknown)}")
    if not descriptions:
        raise ValueError(f"No HGD annotations in {path}")
    durations = np.asarray(raw.annotations.duration, dtype=np.float64)
    if not np.allclose(durations, WINDOW_SECONDS, atol=1e-3):
        raise ValueError(f"Non-four-second HGD annotations in {path}")
    starts = raw.time_as_index(raw.annotations.onset, use_rounding=True)
    if np.any(starts < 0) or np.any(starts + SOURCE_SAMPLES > raw.n_times):
        raise ValueError(f"HGD annotation falls outside recording: {path}")
    return {
        "trials": len(descriptions),
        "label_counts": Counter(LABEL_MAP[value] for value in descriptions),
        "duration_seconds": raw.n_times / SOURCE_RATE,
    }


def resample_buffer(buffer: list[np.ndarray]) -> np.ndarray:
    source = np.stack(buffer, axis=0)
    output = resample_poly(source, up=2, down=5, axis=-1)
    output = np.ascontiguousarray(output, dtype=np.float32)
    expected = (len(buffer), len(HGD_78_CHANNELS), TARGET_SAMPLES)
    if output.shape != expected:
        raise ValueError(f"Unexpected HGD resampled shape: {output.shape}, expected {expected}")
    return output


def transform_edf(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    raw = mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
    eeg_indices, selected_positions = selected_raw_indices(raw)
    starts = raw.time_as_index(raw.annotations.onset, use_rounding=True)
    output_chunks: list[np.ndarray] = []
    selected_buffer: list[np.ndarray] = []
    kept_labels: list[int] = []
    rejected_labels: Counter[int] = Counter()

    for start, description in zip(starts, raw.annotations.description):
        label = LABEL_MAP[str(description)]
        trial_uv = raw.get_data(
            picks=eeg_indices,
            start=int(start),
            stop=int(start) + SOURCE_SAMPLES,
        )
        trial_uv = np.asarray(trial_uv * 1e6, dtype=np.float32)
        if trial_uv.shape != (EXPECTED_EEG_CHANNELS, SOURCE_SAMPLES):
            raise ValueError(f"Unexpected HGD trial shape in {path}: {trial_uv.shape}")
        if not np.isfinite(trial_uv).all():
            raise ValueError(f"NaN or Inf in HGD trial from {path}")
        if float(np.max(np.abs(trial_uv))) >= ARTIFACT_THRESHOLD_UV:
            rejected_labels[label] += 1
            continue
        selected_buffer.append(np.ascontiguousarray(trial_uv[selected_positions]))
        kept_labels.append(label)
        if len(selected_buffer) == RESAMPLE_CHUNK_TRIALS:
            output_chunks.append(resample_buffer(selected_buffer))
            selected_buffer.clear()

    if selected_buffer:
        output_chunks.append(resample_buffer(selected_buffer))
    if not output_chunks:
        raise ValueError(f"All HGD trials rejected in {path}")
    eeg = np.concatenate(output_chunks, axis=0)
    labels = np.asarray(kept_labels, dtype=np.int64)
    if eeg.shape != (len(labels), len(HGD_78_CHANNELS), TARGET_SAMPLES):
        raise ValueError(f"HGD EEG/label mismatch in {path}: {eeg.shape}, {labels.shape}")
    metadata = {
        "source_trials": len(raw.annotations),
        "kept_trials": len(labels),
        "rejected_trials": int(sum(rejected_labels.values())),
        "kept_label_counts": Counter(int(value) for value in labels),
        "rejected_label_counts": rejected_labels,
    }
    return eeg, labels, metadata


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


def validate_saved_arrays(
    eeg_path: Path, label_path: Path
) -> tuple[np.ndarray, np.ndarray]:
    eeg = np.load(eeg_path, mmap_mode="r", allow_pickle=False)
    labels = np.load(label_path, mmap_mode="r", allow_pickle=False)
    if eeg.ndim != 3 or eeg.shape[1:] != (len(HGD_78_CHANNELS), TARGET_SAMPLES):
        raise ValueError(f"Invalid saved HGD EEG shape: {eeg_path} {eeg.shape}")
    if eeg.dtype != np.float32:
        raise ValueError(f"Invalid saved HGD EEG dtype: {eeg_path} {eeg.dtype}")
    if labels.shape != (eeg.shape[0],) or labels.dtype != np.int64:
        raise ValueError(f"Invalid saved HGD labels: {label_path} {labels.shape} {labels.dtype}")
    if set(np.asarray(labels).tolist()) - set(CLASS_NAMES):
        raise ValueError(f"Unknown class ids in {label_path}")
    return eeg, labels


def preprocess(args: argparse.Namespace) -> None:
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    subjects_dir = output_root / "subjects"
    config_path = output_root / "preprocess_config.json"
    manifest_path = output_root / "manifest.json"
    subjects = tuple(args.subjects)
    source_audit = {}
    for subject in subjects:
        for split in OFFICIAL_SPLITS:
            path = source_path(input_root, split, subject)
            source_audit[(subject, split)] = inspect_edf(path)
    source_total = sum(value["trials"] for value in source_audit.values())
    print(
        f"Validated subjects={len(subjects)}, EDFs={len(source_audit)}, "
        f"source_trials={source_total}, EEG=128, selected=78, "
        f"window=0-4 s, target=200 Hz",
        flush=True,
    )
    if args.dry_run:
        print("Dry run completed successfully; no output was written.")
        return

    run_config = {
        "dataset": "Schirrmeister2017-HGD",
        "schema_version": 1,
        "source_root": str(input_root),
        "subjects": list(subjects),
        "official_splits": list(OFFICIAL_SPLITS),
        "source_sampling_rate": SOURCE_RATE,
        "target_sampling_rate": TARGET_RATE,
        "selected_seconds_relative_to_annotation": [0, 4],
        "source_eeg_channels": EXPECTED_EEG_CHANNELS,
        "selected_channels": HGD_78_CHANNELS,
        "sample_shape": [len(HGD_78_CHANNELS), TARGET_SAMPLES],
        "dtype": "float32",
        "unit": "microvolt",
        "artifact_rule": "max(abs(all_128_EEG_0_to_4s)) < 800 microvolt",
        "resampling": "scipy.signal.resample_poly(up=2, down=5)",
        "label_map": LABEL_MAP,
    }
    if config_path.is_file():
        with config_path.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing != run_config:
            raise ValueError(
                f"Existing HGD preprocessing config differs: {config_path}. "
                "Use a new --output-root."
            )
    elif subjects_dir.exists() and any(subjects_dir.glob("*.npy")):
        raise FileNotFoundError(
            f"Found HGD arrays without {config_path}; use a new --output-root"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    subjects_dir.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(run_config, config_path)

    manifest_subjects = []
    global_kept: Counter[int] = Counter()
    total_kept = 0
    total_rejected = 0
    for subject_position, subject in enumerate(subjects, start=1):
        subject_record = {"subject_id": subject, "splits": {}}
        for split in OFFICIAL_SPLITS:
            source = source_path(input_root, split, subject)
            eeg_path = subjects_dir / f"sub{subject:02d}_{split}_eeg.npy"
            label_path = subjects_dir / f"sub{subject:02d}_{split}_labels.npy"
            both_exist = eeg_path.is_file() and label_path.is_file()
            if both_exist and args.resume:
                print(
                    f"[{subject_position:02d}/{len(subjects)}] sub{subject:02d} "
                    f"{split}: validating existing",
                    flush=True,
                )
                eeg, labels = validate_saved_arrays(eeg_path, label_path)
                source_trials = source_audit[(subject, split)]["trials"]
                metadata = {
                    "source_trials": source_trials,
                    "kept_trials": len(labels),
                    "rejected_trials": source_trials - len(labels),
                    "kept_label_counts": Counter(int(value) for value in labels),
                    "rejected_label_counts": Counter(),
                }
            elif (eeg_path.exists() or label_path.exists()) and not args.overwrite:
                raise FileExistsError(
                    f"HGD output exists for sub{subject:02d} {split}; "
                    "use --resume or --overwrite"
                )
            else:
                print(
                    f"[{subject_position:02d}/{len(subjects)}] sub{subject:02d} "
                    f"{split}: processing",
                    flush=True,
                )
                eeg, labels, metadata = transform_edf(source)
                atomic_numpy_save(eeg, eeg_path)
                atomic_numpy_save(labels, label_path)
                eeg, labels = validate_saved_arrays(eeg_path, label_path)
            kept_counts = Counter(int(value) for value in labels)
            global_kept.update(kept_counts)
            total_kept += len(labels)
            total_rejected += metadata["rejected_trials"]
            subject_record["splits"][split] = {
                "source_file": str(source),
                "eeg_file": str(eeg_path.relative_to(output_root)),
                "label_file": str(label_path.relative_to(output_root)),
                "source_trials": int(metadata["source_trials"]),
                "kept_trials": len(labels),
                "rejected_trials": int(metadata["rejected_trials"]),
                "label_counts": {
                    str(key): int(kept_counts[key]) for key in sorted(kept_counts)
                },
            }
        manifest_subjects.append(subject_record)

    if total_kept + total_rejected != source_total:
        raise RuntimeError(
            f"HGD accounting mismatch: kept={total_kept}, rejected={total_rejected}, "
            f"source={source_total}"
        )
    manifest = {
        **run_config,
        "processed_root": str(output_root),
        "class_names": {str(key): value for key, value in CLASS_NAMES.items()},
        "source_trials": source_total,
        "kept_trials": total_kept,
        "rejected_trials": total_rejected,
        "label_counts": {
            str(key): int(global_kept[key]) for key in sorted(global_kept)
        },
        "subject_records": manifest_subjects,
        "notes": [
            "Official train/test EDF boundary is preserved.",
            "Subject 14 test has known bad sensors; no subject-specific repair is applied.",
            "No additional high-pass or exponential running standardization is applied offline.",
        ],
    }
    atomic_json_dump(manifest, manifest_path)
    print(
        f"Completed: source={source_total}, kept={total_kept}, "
        f"rejected={total_rejected}, labels={manifest['label_counts']}, "
        f"manifest={manifest_path}"
    )


def main() -> None:
    preprocess(parse_args())


if __name__ == "__main__":
    main()
