"""Convert AdaBrain-Bench Siena windows into LaBraM-ready subject arrays.

The copied ``Siene/processed_data`` directory contains the audited output of
AdaBrain-Bench's Siena preprocessing: 10-second, 512 Hz, 29-channel pickle
windows in volts with labels non-seizure=0 and seizure=1.  That first-stage
pipeline has already applied a 0.1--75 Hz band-pass, 50 Hz notch filter, fixed
channel selection, standard non-overlapping windows, and extra overlapping
seizure windows.

This second-stage maker preserves every benchmark sample and its label, changes
only the representation required by LaBraM, and writes memory-mappable arrays:

* resample 512 Hz -> 200 Hz with ``resample_poly(25, 64)``;
* convert volts -> microvolts;
* store ``[trials, 29, 2000]`` float32 EEG per patient;
* store labels and per-trial sufficient statistics; and
* create an auditable manifest used by ``data_processor/siena.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly


DEFAULT_INPUT_DIR = Path(
    "/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/"
    "global_user/7461-chenxinhe/Siene/processed_data"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/"
    "global_user/7461-chenxinhe/Siene/processed_data_10s_200hz_adabrain_normstats"
)

PATIENTS = (
    "PN00", "PN01", "PN03", "PN05", "PN06", "PN07", "PN09",
    "PN10", "PN11", "PN12", "PN13", "PN14", "PN16", "PN17",
)
SOURCE_RATE = 512
TARGET_RATE = 200
WINDOW_SECONDS = 10
SOURCE_SAMPLES = SOURCE_RATE * WINDOW_SECONDS
TARGET_SAMPLES = TARGET_RATE * WINDOW_SECONDS
EXPECTED_CHANNELS = 29
EXPECTED_TOTAL_SAMPLES = 51349
EXPECTED_GLOBAL_LABEL_COUNTS = {0: 50665, 1: 684}
BATCH_SIZE = 32
CHANNELS = [
    "FP1", "F3", "C3", "P3", "O1", "F7", "T3", "T5",
    "FC1", "FC5", "CP1", "CP5", "F9", "FZ", "CZ", "PZ",
    "FP2", "F4", "C4", "P4", "O2", "F8", "T4", "T6",
    "FC2", "FC6", "CP2", "CP6", "F10",
]
FILENAME_PATTERN = re.compile(
    r"^(?P<patient>PN\d{2})-(?P<record>.+)_(?P<index>\d+)\.pkl$"
)


def natural_key(value: str) -> tuple:
    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", value)
    )


def parse_patients(value: str) -> tuple[str, ...]:
    requested = []
    for item in value.split(","):
        item = item.strip().upper()
        if not item:
            continue
        if item.isdigit():
            item = f"PN{int(item):02d}"
        if item not in PATIENTS:
            raise argparse.ArgumentTypeError(
                f"Unknown Siena patient {item}; valid={list(PATIENTS)}"
            )
        requested.append(item)
    if not requested or len(requested) != len(set(requested)):
        raise argparse.ArgumentTypeError("Patient list is empty or contains duplicates")
    return tuple(sorted(requested, key=natural_key))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Siena 10-second benchmark pickles to 200 Hz arrays."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--patients", type=parse_patients, default=PATIENTS)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def parse_source_path(path: Path, expected_patient: str) -> tuple[str, int]:
    match = FILENAME_PATTERN.fullmatch(path.name)
    if match is None or match.group("patient") != expected_patient:
        raise ValueError(f"Unexpected Siena pickle filename: {path}")
    return match.group("record"), int(match.group("index"))


def discover_patient_files(input_dir: Path, patient: str) -> list[Path]:
    patient_dir = input_dir / patient
    if not patient_dir.is_dir():
        raise FileNotFoundError(f"Siena patient directory not found: {patient_dir}")
    paths = [path for path in patient_dir.iterdir() if path.suffix == ".pkl"]
    paths.sort(key=lambda path: natural_key(path.name))
    if not paths:
        raise ValueError(f"No Siena pickle files below {patient_dir}")
    for path in paths:
        parse_source_path(path, patient)
    return paths


def read_label_from_tail(path: Path) -> int:
    """Read the final protocol-4 integer label without loading the large X."""

    with path.open("rb") as handle:
        handle.seek(-64, os.SEEK_END)
        tail = handle.read()
    marker = tail.rfind(b"Y\x94K")
    if marker < 0 or marker + 3 >= len(tail):
        raise ValueError(f"Cannot audit Siena label in pickle tail: {path}")
    label = int(tail[marker + 3])
    if label not in (0, 1):
        raise ValueError(f"Unexpected Siena label in {path}: {label}")
    return label


def inspect_sources(input_dir: Path, patients: tuple[str, ...]) -> dict[str, dict]:
    audit = {}
    for patient in patients:
        paths = discover_patient_files(input_dir, patient)
        labels = [read_label_from_tail(path) for path in paths]
        record_counts: Counter[str] = Counter()
        for path in paths:
            record, _ = parse_source_path(path, patient)
            record_counts[record] += 1
        audit[patient] = {
            "paths": paths,
            "label_counts": Counter(labels),
            "record_counts": record_counts,
        }
    return audit


def load_sample(path: Path) -> tuple[np.ndarray, int]:
    with path.open("rb") as handle:
        sample = pickle.load(handle)
    if not isinstance(sample, dict) or set(sample) != {"X", "Y"}:
        raise ValueError(f"Invalid Siena pickle schema: {path}")
    eeg = np.asarray(sample["X"])
    label = int(sample["Y"])
    if eeg.shape != (EXPECTED_CHANNELS, SOURCE_SAMPLES):
        raise ValueError(f"Unexpected Siena sample shape in {path}: {eeg.shape}")
    if eeg.dtype != np.float64:
        raise ValueError(f"Unexpected Siena sample dtype in {path}: {eeg.dtype}")
    if label not in (0, 1):
        raise ValueError(f"Unexpected Siena label in {path}: {label}")
    if label != read_label_from_tail(path):
        raise ValueError(f"Siena pickle/tail label mismatch: {path}")
    if not np.isfinite(eeg).all():
        raise ValueError(f"NaN or Inf in Siena sample: {path}")
    return eeg, label


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


def process_patient(
    paths: list[Path], eeg_path: Path, label_path: Path, stats_path: Path
) -> None:
    count = len(paths)
    temporary_eeg = temporary_path(eeg_path)
    temporary_labels = temporary_path(label_path)
    temporary_stats = temporary_path(stats_path)
    try:
        eeg_output = np.lib.format.open_memmap(
            temporary_eeg,
            mode="w+",
            dtype=np.float32,
            shape=(count, EXPECTED_CHANNELS, TARGET_SAMPLES),
        )
        label_output = np.lib.format.open_memmap(
            temporary_labels, mode="w+", dtype=np.int64, shape=(count,)
        )
        # stats[:, :, 0/1] stores per-trial channel sum/squared sum for the
        # legacy pointwise normalization path.  stats[:, :, 2/3] stores
        # per-trial channel mean/std, matching AdaBrain's normalization stats.
        stats_output = np.lib.format.open_memmap(
            temporary_stats,
            mode="w+",
            dtype=np.float64,
            shape=(count, EXPECTED_CHANNELS, 4),
        )
        for start in range(0, count, BATCH_SIZE):
            stop = min(start + BATCH_SIZE, count)
            source_batch = []
            labels = []
            for path in paths[start:stop]:
                source, label = load_sample(path)
                source_batch.append(source)
                labels.append(label)
            source = np.stack(source_batch, axis=0)
            resampled = resample_poly(source, up=25, down=64, axis=-1)
            stored = np.asarray(resampled * 1e6, dtype=np.float32)
            expected_shape = (stop - start, EXPECTED_CHANNELS, TARGET_SAMPLES)
            if stored.shape != expected_shape or not np.isfinite(stored).all():
                raise ValueError(
                    f"Invalid Siena resampled batch: {stored.shape}, expected {expected_shape}"
                )
            eeg_output[start:stop] = stored
            label_output[start:stop] = np.asarray(labels, dtype=np.int64)
            stored64 = np.asarray(stored, dtype=np.float64)
            stats_output[start:stop, :, 0] = np.sum(stored64, axis=-1)
            stats_output[start:stop, :, 1] = np.sum(np.square(stored64), axis=-1)
            stats_output[start:stop, :, 2] = np.mean(stored64, axis=-1)
            stats_output[start:stop, :, 3] = np.std(stored64, axis=-1)
        eeg_output.flush()
        label_output.flush()
        stats_output.flush()
        del eeg_output, label_output, stats_output
        os.replace(temporary_eeg, eeg_path)
        os.replace(temporary_labels, label_path)
        os.replace(temporary_stats, stats_path)
    finally:
        for path in (temporary_eeg, temporary_labels, temporary_stats):
            if path.exists():
                path.unlink()


def validate_outputs(
    eeg_path: Path, label_path: Path, stats_path: Path, expected_count: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    eeg = np.load(eeg_path, mmap_mode="r", allow_pickle=False)
    labels = np.load(label_path, mmap_mode="r", allow_pickle=False)
    stats = np.load(stats_path, mmap_mode="r", allow_pickle=False)
    expected_eeg = (expected_count, EXPECTED_CHANNELS, TARGET_SAMPLES)
    if eeg.shape != expected_eeg or eeg.dtype != np.float32:
        raise ValueError(f"Invalid Siena EEG output: {eeg_path} {eeg.shape} {eeg.dtype}")
    if labels.shape != (expected_count,) or labels.dtype != np.int64:
        raise ValueError(
            f"Invalid Siena label output: {label_path} {labels.shape} {labels.dtype}"
        )
    if stats.shape != (expected_count, EXPECTED_CHANNELS, 4) or stats.dtype != np.float64:
        raise ValueError(
            f"Invalid Siena statistics output: {stats_path} {stats.shape} {stats.dtype}"
        )
    if set(np.asarray(labels).tolist()) - {0, 1}:
        raise ValueError(f"Unknown labels in {label_path}")
    return eeg, labels, stats


def preprocess(args: argparse.Namespace) -> None:
    input_dir = args.input_dir.resolve()
    output_root = args.output_root.resolve()
    subjects_dir = output_root / "subjects"
    config_path = output_root / "preprocess_config.json"
    manifest_path = output_root / "manifest.json"
    patients = tuple(args.patients)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Siena benchmark pickle directory not found: {input_dir}")
    audit = inspect_sources(input_dir, patients)
    total_samples = sum(len(value["paths"]) for value in audit.values())
    global_counts = sum(
        (value["label_counts"] for value in audit.values()), Counter()
    )
    if patients == PATIENTS:
        if total_samples != EXPECTED_TOTAL_SAMPLES:
            raise ValueError(
                f"Expected {EXPECTED_TOTAL_SAMPLES} Siena samples, found {total_samples}"
            )
        if dict(global_counts) != EXPECTED_GLOBAL_LABEL_COUNTS:
            raise ValueError(
                f"Unexpected Siena label counts: {dict(global_counts)}, "
                f"expected {EXPECTED_GLOBAL_LABEL_COUNTS}"
            )
    print(
        f"Validated patients={len(patients)}, samples={total_samples}, "
        f"labels={dict(sorted(global_counts.items()))}, source=[29,5120] at 512 Hz, "
        f"target=[29,2000] at 200 Hz",
        flush=True,
    )
    if args.dry_run:
        print("Dry run completed successfully; no output was written.")
        return

    run_config = {
        "dataset": "Siena",
        "schema_version": 1,
        "source_dir": str(input_dir),
        "patients": list(patients),
        "source_sampling_rate": SOURCE_RATE,
        "target_sampling_rate": TARGET_RATE,
        "window_seconds": WINDOW_SECONDS,
        "source_sample_shape": [EXPECTED_CHANNELS, SOURCE_SAMPLES],
        "sample_shape": [EXPECTED_CHANNELS, TARGET_SAMPLES],
        "source_dtype": "float64",
        "source_unit": "volt",
        "dtype": "float32",
        "unit": "microvolt",
        "channels": CHANNELS,
        "label_names": {"0": "non_seizure", "1": "seizure"},
        "resampling": "scipy.signal.resample_poly(up=25, down=64)",
        "first_stage_preprocessing": [
            "AdaBrain-Bench 0.1-75 Hz band-pass",
            "AdaBrain-Bench 50 Hz notch",
            "AdaBrain-Bench 10-second standard windows",
            "AdaBrain-Bench seizure-enhanced windows with 5-second stride",
        ],
        "normalization_statistics": [
            "stats[:, :, 0] = per-window channel sum",
            "stats[:, :, 1] = per-window channel squared sum",
            "stats[:, :, 2] = per-window channel mean",
            "stats[:, :, 3] = per-window channel std",
            "reader uses average(stats[:, :, 2/3]) over train windows for z_score",
        ],
    }
    if config_path.is_file():
        with config_path.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing != run_config:
            raise ValueError(
                f"Existing Siena preprocessing config differs: {config_path}. "
                "Use a new --output-root."
            )
    elif subjects_dir.exists() and any(subjects_dir.glob("*.npy")):
        raise FileNotFoundError(
            f"Found Siena arrays without {config_path}; use a new --output-root"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    subjects_dir.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(run_config, config_path)

    subject_records = []
    verified_global: Counter[int] = Counter()
    for position, patient in enumerate(patients, start=1):
        paths = audit[patient]["paths"]
        eeg_path = subjects_dir / f"{patient}_eeg.npy"
        label_path = subjects_dir / f"{patient}_labels.npy"
        stats_path = subjects_dir / f"{patient}_stats.npy"
        outputs_exist = all(path.is_file() for path in (eeg_path, label_path, stats_path))
        any_output = any(path.exists() for path in (eeg_path, label_path, stats_path))
        if outputs_exist and args.resume:
            print(f"[{position:02d}/{len(patients)}] {patient}: validating existing", flush=True)
        elif any_output and not args.overwrite and not args.resume:
            raise FileExistsError(
                f"Siena output exists for {patient}; use --resume or --overwrite"
            )
        else:
            print(f"[{position:02d}/{len(patients)}] {patient}: processing", flush=True)
            process_patient(paths, eeg_path, label_path, stats_path)
        _, labels, _ = validate_outputs(
            eeg_path, label_path, stats_path, len(paths)
        )
        label_counts = Counter(int(value) for value in np.asarray(labels))
        if label_counts != audit[patient]["label_counts"]:
            raise ValueError(f"Siena output labels differ from source for {patient}")
        verified_global.update(label_counts)
        subject_records.append(
            {
                "patient_id": patient,
                "subject_id": int(patient[2:]),
                "eeg_file": str(eeg_path.relative_to(output_root)),
                "label_file": str(label_path.relative_to(output_root)),
                "stats_file": str(stats_path.relative_to(output_root)),
                "samples": len(paths),
                "label_counts": {
                    str(key): int(label_counts[key]) for key in sorted(label_counts)
                },
                "source_record_counts": {
                    key: int(value)
                    for key, value in sorted(
                        audit[patient]["record_counts"].items(),
                        key=lambda item: natural_key(item[0]),
                    )
                },
            }
        )
    manifest = {
        **run_config,
        "processed_root": str(output_root),
        "total_samples": total_samples,
        "label_counts": {
            str(key): int(verified_global[key]) for key in sorted(verified_global)
        },
        "benchmark_split": {
            "train_validation_patients": list(PATIENTS[:-2]),
            "test_patients": list(PATIENTS[-2:]),
            "validation_fraction_per_class": 0.2,
            "split_seed": 42,
        },
        "subject_records": subject_records,
    }
    atomic_json_dump(manifest, manifest_path)
    print(
        f"Completed: samples={total_samples}, labels={manifest['label_counts']}, "
        f"manifest={manifest_path}"
    )


def main() -> None:
    preprocess(parse_args())


if __name__ == "__main__":
    main()
