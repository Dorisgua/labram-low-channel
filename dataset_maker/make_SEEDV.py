import argparse
import csv
import os
import pickle
import re
import shutil
from pathlib import Path

import mne
import numpy as np
from tqdm import tqdm


SEEDV_CHANNELS = [
    "FP1", "FPZ", "FP2",
    "AF3", "AF4",
    "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8",
    "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8",
    "T7", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8",
    "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8",
    "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8",
    "PO7", "PO5", "PO3", "POZ", "PO4", "PO6", "PO8",
    "CB1", "O1", "OZ", "O2", "CB2",
]

SKIP_SESSIONS = {(7, 1)}

LABEL_NAMES = ["Disgust", "Fear", "Sad", "Neutral", "Happy"]

LABEL_META = np.array([
    [4, 1, 3, 2, 0, 4, 1, 3, 2, 0, 4, 1, 3, 2, 0],
    [2, 1, 3, 0, 4, 4, 0, 3, 2, 1, 3, 4, 1, 2, 0],
    [2, 1, 3, 0, 4, 4, 0, 3, 2, 1, 3, 4, 1, 2, 0],
], dtype=np.int64)

TIME_META = np.array([
    [[30, 132, 287, 555, 773, 982, 1271, 1628, 1730, 2025, 2227, 2435, 2667, 2932, 3204],
     [102, 228, 524, 742, 920, 1240, 1568, 1697, 1994, 2166, 2401, 2607, 2901, 3172, 3359]],
    [[30, 299, 548, 646, 836, 1000, 1091, 1392, 1657, 1809, 1966, 2186, 2333, 2490, 2741],
     [267, 488, 614, 773, 967, 1059, 1331, 1622, 1777, 1908, 2153, 2302, 2428, 2709, 2817]],
    [[30, 353, 478, 674, 825, 908, 1200, 1346, 1451, 1711, 2055, 2307, 2457, 2726, 2888],
     [321, 418, 643, 764, 877, 1147, 1284, 1418, 1679, 1996, 2275, 2425, 2664, 2857, 3066]],
], dtype=np.int64)


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess SEED-V cnt files for LaBraM finetuning.")
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/SEED_V/SEED-V-origin/EEG_raw"),
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/SEED_V/SEED-V-labram"),
    )
    parser.add_argument("--sampling-rate", type=float, default=200.0)
    parser.add_argument("--l-freq", type=float, default=0.1)
    parser.add_argument("--h-freq", type=float, default=75.0)
    parser.add_argument("--notch-freq", type=float, default=50.0)
    parser.add_argument("--window-sec", type=float, default=1.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_cnt_file(path: Path):
    match = re.match(r"^(\d+)_(\d+)_(\d{8})(?:_repaired)?\.cnt$", path.name)
    if not match:
        return None
    subject, session, date = match.groups()
    return int(subject), int(session), date


def collect_cnt_files(raw_root: Path):
    grouped = {}
    for path in sorted(raw_root.glob("*.cnt")):
        parsed = resolve_cnt_file(path)
        if parsed is None:
            continue
        subject, session, _ = parsed
        if (subject, session) in SKIP_SESSIONS:
            continue
        grouped.setdefault((subject, session), []).append(path)

    selected = []
    for key in sorted(grouped):
        candidates = sorted(grouped[key], key=lambda p: ("repaired" in p.stem, p.name))
        selected.append(candidates[0])
    return selected


def split_name_from_trial(trial_idx: int):
    if trial_idx < 5:
        return "train"
    if trial_idx < 10:
        return "eval"
    return "test"


def prepare_output_dirs(out_root: Path, overwrite: bool):
    if out_root.exists() and overwrite:
        shutil.rmtree(out_root)
    if out_root.exists() and any(out_root.iterdir()) and not overwrite:
        raise FileExistsError(f"{out_root} is not empty. Pass --overwrite to rebuild it.")

    for split in ["processed_train", "processed_eval", "processed_test"]:
        (out_root / split).mkdir(parents=True, exist_ok=True)


def load_preprocessed_raw(path: Path, args):
    raw = mne.io.read_raw_cnt(path, preload=True, verbose=False)
    missing = [ch for ch in SEEDV_CHANNELS if ch not in raw.ch_names]
    if missing:
        raise RuntimeError(f"{path} missing channels: {missing}")

    raw.pick_channels(SEEDV_CHANNELS, ordered=True)
    raw.filter(l_freq=args.l_freq, h_freq=args.h_freq, verbose=False)

    if args.notch_freq > 0:
        freqs = np.arange(args.notch_freq, raw.info["sfreq"] / 2, args.notch_freq)
        if len(freqs) > 0:
            raw.notch_filter(freqs=freqs, verbose=False)

    if raw.info["sfreq"] != args.sampling_rate:
        raw.resample(args.sampling_rate, verbose=False)

    # MNE returns uV here. LaBraM normalizes by 0.1 mV, i.e. divide uV by 100.
    return raw.get_data(units="uV").astype(np.float32) / 100.0


def save_pickle(path: Path, sample: dict):
    with open(path, "wb") as f:
        pickle.dump(sample, f, protocol=pickle.HIGHEST_PROTOCOL)


def process_file(path: Path, args):
    subject, session, date = resolve_cnt_file(path)
    if not 1 <= session <= 3:
        raise ValueError(f"Unexpected session in {path}")

    data = load_preprocessed_raw(path, args)
    samples_per_window = int(round(args.window_sec * args.sampling_rate))
    labels = LABEL_META[session - 1]
    times = TIME_META[session - 1]
    rows = []

    for trial_idx, label_id in enumerate(labels):
        split_name = split_name_from_trial(trial_idx)
        split_dir = args.out_root / f"processed_{split_name}"
        start = int(round(times[0, trial_idx] * args.sampling_rate))
        end = int(round(times[1, trial_idx] * args.sampling_rate))
        end = min(end, data.shape[1])

        n_windows = max(0, (end - start) // samples_per_window)
        for win_idx in range(n_windows):
            left = start + win_idx * samples_per_window
            right = left + samples_per_window
            signal = np.ascontiguousarray(data[:, left:right])
            file_name = (
                f"sub{subject:02d}_sess{session}_trial{trial_idx + 1:02d}_"
                f"win{win_idx:04d}_{LABEL_NAMES[label_id].lower()}.pkl"
            )
            out_path = split_dir / file_name
            save_pickle(out_path, {
                "signal": signal,
                "label": np.array([label_id], dtype=np.int64),
                "subject": subject,
                "session": session,
                "trial": trial_idx + 1,
                "date": date,
                "source": str(path),
                "ch_names": SEEDV_CHANNELS,
                "sampling_rate": int(args.sampling_rate),
            })

        rows.append({
            "source": str(path),
            "subject": subject,
            "session": session,
            "date": date,
            "trial": trial_idx + 1,
            "split": split_name,
            "label": int(label_id),
            "label_name": LABEL_NAMES[label_id],
            "start_sec": int(times[0, trial_idx]),
            "end_sec": int(times[1, trial_idx]),
            "n_windows": n_windows,
        })

    return rows


def write_metadata(out_root: Path, rows: list[dict]):
    metadata_path = out_root / "metadata.csv"
    fieldnames = [
        "source", "subject", "session", "date", "trial", "split",
        "label", "label_name", "start_sec", "end_sec", "n_windows",
    ]
    with open(metadata_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    if not args.raw_root.exists():
        raise FileNotFoundError(args.raw_root)
    if len(SEEDV_CHANNELS) != 62:
        raise RuntimeError(f"Expected 62 SEED-V channels, got {len(SEEDV_CHANNELS)}")

    prepare_output_dirs(args.out_root, args.overwrite)
    cnt_files = collect_cnt_files(args.raw_root)
    if not cnt_files:
        raise RuntimeError(f"No cnt files found in {args.raw_root}")

    all_rows = []
    for path in tqdm(cnt_files, desc="Processing SEED-V"):
        all_rows.extend(process_file(path, args))

    write_metadata(args.out_root, all_rows)

    counts = {}
    for row in all_rows:
        counts[row["split"]] = counts.get(row["split"], 0) + row["n_windows"]
    print(f"Processed files: {len(cnt_files)}")
    print(f"Subjects: {len({resolve_cnt_file(p)[0] for p in cnt_files})}")
    print(f"Windows: {counts}")
    print(f"Output: {args.out_root}")


if __name__ == "__main__":
    main()
