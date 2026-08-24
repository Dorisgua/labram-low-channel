"""Preprocess Ultra-high density AAD Curry recordings for LaBraM.

The raw dataset stores one ``S*.tar.gz`` archive per subject.  Each archive
contains four Curry recordings named like ``S0_AAD_1L`` and ``S0_AAD_1R``.
This maker streams one recording at a time through a temporary directory,
selects the 84 LaBraM-compatible high-density AAD channels, crops the AAD
period using trigger changes when present, resamples 1000 Hz to 200 Hz, and
writes non-overlapping four-second windows.

Labels are derived from the recording suffix: ``L -> 0`` and ``R -> 1``.
Validation/test splitting is left to ``data_processor/aad.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tarfile
import tempfile
from collections import Counter
from pathlib import Path
import sys

import numpy as np
from scipy.signal import find_peaks, resample_poly

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Channels_definition import HIGH_DENSITY_AAD_84_CHANNELS


DEFAULT_INPUT_ROOT = Path(
    "/inspire/alluxio/project/sais-medical/public/share_medical/EEG/"
    "Ultra-high_density_AAD"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/"
    "global_user/7461-chenxinhe/AAD/processed_data_4s_200hz"
)

EXPECTED_SUBJECTS = tuple(range(30))
SOURCE_RATE = 1000
TARGET_RATE = 200
WINDOW_SECONDS = 4
SOURCE_WINDOW_SAMPLES = SOURCE_RATE * WINDOW_SECONDS
TARGET_SAMPLES = TARGET_RATE * WINDOW_SECONDS
EXPECTED_TOTAL_CHANNELS = 256
EXPECTED_EEG_CHANNELS = 253
LABEL_MAP = {"L": 0, "R": 1}
CLASS_NAMES = {0: "attend_left", 1: "attend_right"}
RECORDING_PATTERN = re.compile(r"^S(?P<subject>\d+)_AAD_(?P<trial>\d+)(?P<side>[LR])$")


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
            f"Subjects must be within 0--29; invalid={invalid}"
        )
    return tuple(sorted(subjects))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build 84-channel, four-second, 200 Hz AAD arrays."
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--subjects", type=parse_subjects, default=EXPECTED_SUBJECTS)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def archive_path(root: Path, subject: int) -> Path:
    return root / f"S{subject}.tar.gz"


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig", errors="replace")


def parse_dap(path: Path) -> dict:
    text = _read_text(path)

    def value(name: str) -> str:
        match = re.search(rf"^\s*{re.escape(name)}\s*=\s*(.+?)\s*$", text, re.M)
        if match is None:
            raise ValueError(f"Missing Curry DAP field {name}: {path}")
        return match.group(1).strip()

    return {
        "num_samples": int(float(value("NumSamples"))),
        "num_trials": int(float(value("NumTrials"))),
        "num_channels": int(float(value("NumChannels"))),
        "sample_rate": int(float(value("SampleFreqHz"))),
        "data_format": value("DataFormat").upper(),
        "data_sample_order": value("DataSampOrder").upper(),
    }


def _list_block(text: str, start_label: str) -> list[str]:
    pattern = (
        rf"{re.escape(start_label)} START_LIST.*?\n"
        rf"(?P<body>.*?)"
        rf"{re.escape(start_label)} END_LIST"
    )
    match = re.search(pattern, text, flags=re.S)
    if match is None:
        raise ValueError(f"Missing Curry label block {start_label}")
    return [
        line.strip()
        for line in match.group("body").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def parse_rs3(path: Path) -> tuple[list[str], list[str]]:
    text = _read_text(path)
    eeg_names = _list_block(text, "LABELS")
    other_names = _list_block(text, "LABELS_OTHERS")
    if len(eeg_names) != EXPECTED_EEG_CHANNELS:
        raise ValueError(f"Expected 253 AAD EEG labels, got {len(eeg_names)}")
    return eeg_names, other_names


def recording_stems(archive: Path, subject: int) -> list[str]:
    with tarfile.open(archive, "r:gz") as tar:
        stems = {
            Path(member.name).with_suffix("").name
            for member in tar.getmembers()
            if member.isfile() and member.name.endswith(".dat")
        }
    output = []
    for stem in sorted(stems):
        match = RECORDING_PATTERN.fullmatch(stem)
        if match is None or int(match.group("subject")) != subject:
            raise ValueError(f"Unexpected AAD recording name in {archive}: {stem}")
        output.append(stem)
    if not output:
        raise ValueError(f"No AAD recordings found in {archive}")
    return output


def extract_recording(archive: Path, stem: str, destination: Path) -> dict[str, Path]:
    wanted = {f"{stem}.dat", f"{stem}.dap", f"{stem}.rs3"}
    output: dict[str, Path] = {}
    with tarfile.open(archive, "r:gz") as tar:
        by_name = {Path(member.name).name: member for member in tar.getmembers()}
        missing = wanted - set(by_name)
        if missing:
            raise FileNotFoundError(f"{archive} lacks {sorted(missing)}")
        for filename in sorted(wanted):
            member = by_name[filename]
            member.name = filename
            tar.extract(member, path=destination)
            output[Path(filename).suffix[1:]] = destination / filename
    return output


def load_recording(paths: dict[str, Path]) -> tuple[np.ndarray, list[str], int, tuple[int, int]]:
    info = parse_dap(paths["dap"])
    if info["data_format"] != "FLOAT":
        raise ValueError(f"Unsupported AAD DataFormat: {info['data_format']}")
    if info["num_trials"] != 1:
        raise ValueError(f"Unsupported AAD NumTrials: {info['num_trials']}")
    if info["num_channels"] != EXPECTED_TOTAL_CHANNELS:
        raise ValueError(f"Expected 256 total channels, got {info['num_channels']}")
    if info["sample_rate"] != SOURCE_RATE:
        raise ValueError(f"Expected 1000 Hz AAD source, got {info['sample_rate']}")

    eeg_names, other_names = parse_rs3(paths["rs3"])
    all_names = eeg_names + other_names
    uppercase_to_index = {}
    duplicate_names = set()
    for index, name in enumerate(all_names):
        key = name.upper()
        if key in uppercase_to_index:
            duplicate_names.add(key)
        uppercase_to_index[key] = index
    if duplicate_names:
        raise ValueError(
            f"AAD recording has duplicate channel labels after uppercasing: "
            f"{sorted(duplicate_names)}"
        )
    selected_indices = []
    missing = []
    for channel in HIGH_DENSITY_AAD_84_CHANNELS:
        index = uppercase_to_index.get(channel.upper())
        if index is None:
            missing.append(channel)
        else:
            selected_indices.append(index)
    if missing:
        raise ValueError(f"AAD recording lacks selected channels: {missing}")

    item_count = info["num_samples"] * info["num_channels"]
    expected_bytes = item_count * np.dtype("<f4").itemsize
    if paths["dat"].stat().st_size != expected_bytes:
        raise ValueError(
            f"Unexpected AAD dat size: {paths['dat']} has {paths['dat'].stat().st_size}, "
            f"expected {expected_bytes}"
        )
    raw = np.memmap(paths["dat"], dtype="<f4", mode="r")
    if info["data_sample_order"] == "SAMP":
        raw = raw.reshape(info["num_samples"], info["num_channels"])
        eeg = np.asarray(raw[:, selected_indices].T, dtype=np.float32)
        trigger = np.asarray(raw[:, all_names.index("Trigger")], dtype=np.float32)
    elif info["data_sample_order"] == "CHAN":
        raw = raw.reshape(info["num_channels"], info["num_samples"])
        eeg = np.asarray(raw[selected_indices, :], dtype=np.float32)
        trigger = np.asarray(raw[all_names.index("Trigger"), :], dtype=np.float32)
    else:
        raise ValueError(f"Unsupported AAD DataSampOrder: {info['data_sample_order']}")

    if not np.isfinite(eeg).all():
        raise ValueError(f"NaN or Inf in AAD data: {paths['dat']}")
    crop_start, crop_stop = aad_crop_from_trigger(trigger)
    return eeg[:, crop_start:crop_stop], list(HIGH_DENSITY_AAD_84_CHANNELS), info["sample_rate"], (crop_start, crop_stop)


def aad_crop_from_trigger(trigger: np.ndarray) -> tuple[int, int]:
    baseline = trigger[0] if np.all(np.diff(trigger[: min(len(trigger), 5 * SOURCE_RATE)]) == 0) else np.median(trigger)
    marker = np.zeros_like(trigger)
    changes = np.flatnonzero(np.diff(trigger) != 0) + 1
    if changes.size:
        marker[changes] = trigger[changes] - baseline
    peaks, _ = find_peaks(marker)
    if peaks.size >= 2:
        start, stop = int(peaks[0]), int(peaks[-1])
    else:
        start, stop = 0, int(trigger.shape[0])
    usable = ((stop - start) // SOURCE_WINDOW_SAMPLES) * SOURCE_WINDOW_SAMPLES
    if usable <= 0:
        raise ValueError(f"AAD crop is shorter than {WINDOW_SECONDS}s: {start}:{stop}")
    return start, start + usable


def windows_from_recording(eeg: np.ndarray) -> np.ndarray:
    if eeg.shape[0] != len(HIGH_DENSITY_AAD_84_CHANNELS):
        raise ValueError(f"Unexpected AAD channel count: {eeg.shape}")
    usable = (eeg.shape[1] // SOURCE_WINDOW_SAMPLES) * SOURCE_WINDOW_SAMPLES
    source = eeg[:, :usable].reshape(
        eeg.shape[0], -1, SOURCE_WINDOW_SAMPLES
    ).transpose(1, 0, 2)
    output = resample_poly(source, up=1, down=5, axis=-1)
    output = np.ascontiguousarray(output, dtype=np.float32)
    if output.shape != (source.shape[0], len(HIGH_DENSITY_AAD_84_CHANNELS), TARGET_SAMPLES):
        raise ValueError(f"Unexpected AAD output shape: {output.shape}")
    if not np.isfinite(output).all():
        raise ValueError("NaN or Inf after AAD resampling")
    return output


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


def subject_statistics(eeg: np.ndarray) -> tuple[list[float], list[float], int]:
    channel_sum = np.sum(eeg, axis=(0, 2), dtype=np.float64)
    channel_squared_sum = np.sum(
        np.square(eeg, dtype=np.float64), axis=(0, 2), dtype=np.float64
    )
    count = int(eeg.shape[0] * eeg.shape[2])
    return channel_sum.tolist(), channel_squared_sum.tolist(), count


def validate_saved_arrays(eeg_path: Path, label_path: Path) -> tuple[np.ndarray, np.ndarray]:
    eeg = np.load(eeg_path, mmap_mode="r", allow_pickle=False)
    labels = np.load(label_path, mmap_mode="r", allow_pickle=False)
    if eeg.ndim != 3 or eeg.shape[1:] != (
        len(HIGH_DENSITY_AAD_84_CHANNELS),
        TARGET_SAMPLES,
    ):
        raise ValueError(f"Invalid saved AAD EEG shape: {eeg_path} {eeg.shape}")
    if eeg.dtype != np.float32:
        raise ValueError(f"Invalid saved AAD EEG dtype: {eeg_path} {eeg.dtype}")
    if labels.shape != (eeg.shape[0],) or labels.dtype != np.int64:
        raise ValueError(f"Invalid saved AAD labels: {label_path} {labels.shape} {labels.dtype}")
    if set(np.asarray(labels).tolist()) - set(CLASS_NAMES):
        raise ValueError(f"Unknown AAD class ids in {label_path}")
    return eeg, labels


def process_subject(input_root: Path, output_root: Path, subject: int, overwrite: bool, resume: bool) -> dict:
    archive = archive_path(input_root, subject)
    if not archive.is_file():
        raise FileNotFoundError(f"AAD archive not found: {archive}")
    eeg_path = output_root / "subjects" / f"S{subject}_eeg.npy"
    label_path = output_root / "subjects" / f"S{subject}_labels.npy"
    if eeg_path.is_file() and label_path.is_file() and resume:
        eeg, labels = validate_saved_arrays(eeg_path, label_path)
        channel_sum, channel_squared_sum, sample_points = subject_statistics(eeg)
        return make_subject_record(output_root, subject, eeg_path, label_path, labels, [], channel_sum, channel_squared_sum, sample_points)
    if (eeg_path.exists() or label_path.exists()) and not overwrite:
        raise FileExistsError(f"Output exists for S{subject}; use --resume or --overwrite")

    all_windows = []
    all_labels = []
    recording_records = []
    with tempfile.TemporaryDirectory(prefix=f"aad_S{subject}_") as temp_dir:
        temp_root = Path(temp_dir)
        for stem in recording_stems(archive, subject):
            match = RECORDING_PATTERN.fullmatch(stem)
            assert match is not None
            label = LABEL_MAP[match.group("side")]
            paths = extract_recording(archive, stem, temp_root)
            cropped, _, _, crop = load_recording(paths)
            windows = windows_from_recording(cropped)
            labels = np.full(windows.shape[0], label, dtype=np.int64)
            all_windows.append(windows)
            all_labels.append(labels)
            recording_records.append(
                {
                    "recording": stem,
                    "label": label,
                    "crop_source_samples": list(crop),
                    "windows": int(windows.shape[0]),
                }
            )
            for path in paths.values():
                path.unlink(missing_ok=True)

    eeg = np.concatenate(all_windows, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    order = np.lexsort((np.arange(len(labels)), labels))
    eeg = np.ascontiguousarray(eeg[order], dtype=np.float32)
    labels = np.ascontiguousarray(labels[order], dtype=np.int64)
    atomic_numpy_save(eeg, eeg_path)
    atomic_numpy_save(labels, label_path)
    eeg, labels = validate_saved_arrays(eeg_path, label_path)
    channel_sum, channel_squared_sum, sample_points = subject_statistics(eeg)
    return make_subject_record(
        output_root,
        subject,
        eeg_path,
        label_path,
        labels,
        recording_records,
        channel_sum,
        channel_squared_sum,
        sample_points,
    )


def make_subject_record(
    output_root: Path,
    subject: int,
    eeg_path: Path,
    label_path: Path,
    labels: np.ndarray,
    recording_records: list[dict],
    channel_sum: list[float],
    channel_squared_sum: list[float],
    sample_points: int,
) -> dict:
    label_counts = Counter(int(value) for value in np.asarray(labels))
    return {
        "subject_id": subject,
        "eeg_file": str(eeg_path.relative_to(output_root)),
        "label_file": str(label_path.relative_to(output_root)),
        "trials": int(labels.shape[0]),
        "label_counts": {str(key): int(label_counts[key]) for key in sorted(label_counts)},
        "recordings": recording_records,
        "channel_sum": channel_sum,
        "channel_squared_sum": channel_squared_sum,
        "sample_points_per_channel": sample_points,
    }


def preprocess(args: argparse.Namespace) -> None:
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    subjects = tuple(args.subjects)
    if not input_root.is_dir():
        raise FileNotFoundError(f"AAD raw root not found: {input_root}")
    missing = [subject for subject in subjects if not archive_path(input_root, subject).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing AAD subject archives: {missing}")
    print(f"Validated AAD raw archives for subjects={subjects}", flush=True)
    if args.dry_run:
        for subject in subjects:
            stems = recording_stems(archive_path(input_root, subject), subject)
            print(f"S{subject}: {stems}", flush=True)
        return

    subjects_dir = output_root / "subjects"
    subjects_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "dataset": "Ultra-high_density_AAD",
        "schema_version": 1,
        "source_root": str(input_root),
        "subjects": list(subjects),
        "source_sampling_rate": SOURCE_RATE,
        "target_sampling_rate": TARGET_RATE,
        "window_seconds": WINDOW_SECONDS,
        "sample_shape": [len(HIGH_DENSITY_AAD_84_CHANNELS), TARGET_SAMPLES],
        "dtype": "float32",
        "unit": "microvolt",
        "channels": list(HIGH_DENSITY_AAD_84_CHANNELS),
        "label_map": LABEL_MAP,
        "resampling": "scipy.signal.resample_poly(up=1, down=5)",
    }
    atomic_json_dump(run_config, output_root / "preprocess_config.json")

    records = []
    global_label_counts: Counter[int] = Counter()
    for position, subject in enumerate(subjects, start=1):
        print(f"[{position:02d}/{len(subjects)}] S{subject}: processing", flush=True)
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
            f"S{subject}: windows={record['trials']}, labels={record['label_counts']}",
            flush=True,
        )

    manifest = {
        **run_config,
        "processed_root": str(output_root),
        "class_names": {str(key): value for key, value in CLASS_NAMES.items()},
        "total_trials": int(sum(record["trials"] for record in records)),
        "label_counts": {
            str(key): int(global_label_counts[key]) for key in sorted(global_label_counts)
        },
        "subject_records": records,
        "default_split": {
            "train_subjects": list(range(24)),
            "val_subjects": [24, 25, 26],
            "test_subjects": [27, 28, 29],
        },
    }
    atomic_json_dump(manifest, output_root / "manifest.json")
    print(f"Completed AAD preprocessing: {output_root / 'manifest.json'}", flush=True)


def main() -> None:
    preprocess(parse_args())


if __name__ == "__main__":
    main()
