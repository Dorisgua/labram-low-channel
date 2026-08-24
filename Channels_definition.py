"""Shared channel definitions for preexp12 reconstruction experiments.

Keep prototype generation, dataset channel selection, completion mapping, and
inspection scripts on the same channel order.
"""

try:
    standard_1020
except NameError:
    from utils import standard_1020


TUEV_23_CHANNELS = [
    "FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4",
    "O1", "O2", "F7", "F8", "T3", "T4", "T5", "T6",
    "A1", "A2", "FZ", "CZ", "PZ", "T1", "T2",
]

TUEV_13_CHANNELS = [
    "FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4",
    "O1", "O2", "T3", "T4", "CZ",
]

BCIIV2A_22_CHANNELS = [
    "FZ", "FC3", "FC1", "FCZ", "FC2", "FC4", "C5", "C3",
    "C1", "CZ", "C2", "C4", "C6", "CP3", "CP1", "CPZ",
    "CP2", "CP4", "P1", "PZ", "P2", "POZ",
]

# PreExp33N: fixed, left-right symmetric 13-channel motor-imagery subset.
# Keep this order aligned with BCIIV2A_22_CHANNELS and the dataset tensor.
BCIIV2A_13_CHANNELS = [
    "FZ", "FC3", "FCZ", "FC4", "C5", "C3", "CZ",
    "C4", "C6", "CP3", "CPZ", "CP4", "PZ",
]

BCIIV2A_MASKED_9_CHANNELS = [
    "FC1", "FC2", "C1", "C2", "CP1", "CP2", "P1", "P2", "POZ",
]


# EEGMAT mental-arithmetic dataset.  Keep the legacy T3/T4/T5/T6 names used
# by the EDF preprocessing and AdaBrain-Bench manifests; LaBraM's channel
# position table contains these names directly.
EEGMAT_19_CHANNELS = [
    "FP1", "FP2", "F3", "F4", "F7", "F8", "T3", "T4",
    "C3", "C4", "T5", "T6", "P3", "P4", "O1", "O2",
    "FZ", "CZ", "PZ",
]

EEGMAT_8_CHANNELS = [
    "FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4",
]

FACED_32_CHANNELS = [
    "FP1", "FP2", "FZ", "F3", "F4", "F7", "F8", "FC1",
    "FC2", "FC5", "FC6", "CZ", "C3", "C4", "T3", "T4",
    "A1", "A2", "CP1", "CP2", "CP5", "CP6", "PZ", "P3",
    "P4", "T5", "T6", "PO3", "PO4", "OZ", "O1", "O2",
]


# Zuo2025 lower-limb motor-imagery dataset.  The derivative MAT files contain
# the first 30 channels from ZhenTec-10-10-Cap32.locs; HEOL/HEOR are excluded.
ZUO2025_30_CHANNELS = [
    "FP1", "FP2", "FZ", "F3", "F4", "F7", "F8", "FCZ",
    "FC3", "FC4", "FT7", "FT8", "CZ", "C3", "C4", "T3",
    "T4", "CPZ", "CP3", "CP4", "TP7", "TP8", "PZ", "P3",
    "P4", "T5", "T6", "OZ", "O1", "O2",
]


# Schirrmeister2017 High Gamma Dataset (HGD).  These are the 78 EEG channels
# whose physical names occur directly in LaBraM's standard_1020 position table,
# kept in the order used by the HGD EDF files.  EOG/EMG and unsupported
# half-grid electrodes are excluded.
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

# Direct LaBraM-compatible intersection of HGD's official 44 motor-cortex
# sensors.  This is an ablation subset of HGD_78_CHANNELS, not the first
# full-channel baseline.
HGD_MOTOR_20_CHANNELS = [
    "FC5", "FC1", "FC2", "FC6", "C3", "C4", "CP5", "CP1",
    "CP2", "CP6", "FC3", "FCZ", "FC4", "C5", "C1", "C2",
    "C6", "CP3", "CPZ", "CP4",
]


# Siena Scalp EEG seizure-detection benchmark.  These 29 channels are selected
# and ordered by AdaBrain-Bench's Siena preprocessing.  The legacy T3/T4/T5/T6
# names are retained because LaBraM's position table represents them directly.
SIENA_29_CHANNELS = [
    "FP1", "F3", "C3", "P3", "O1", "F7", "T3", "T5",
    "FC1", "FC5", "CP1", "CP5", "F9", "FZ", "CZ", "PZ",
    "FP2", "F4", "C4", "P4", "O2", "F8", "T4", "T6",
    "FC2", "FC6", "CP2", "CP6", "F10",
]

SIENA_13_CHANNELS = [
    "FP1", "F3", "C3", "P3", "O1", "FZ", "CZ", "PZ",
    "FP2", "F4", "C4", "P4", "O2",
]


# Fatigue Characterization of EEG under Mixed Reality Stereo Vision.
# Keep the 30-channel order from data_eeg_FATIG_FTG/dataset_info.pkl.
FATIG_30_CHANNELS = [
    "FP1", "FP2", "F7", "F3", "FZ", "F4", "F8", "FT7",
    "FC3", "FCZ", "FC4", "FT8", "T3", "C3", "CZ", "C4",
    "T4", "TP7", "CP3", "CPZ", "CP4", "TP8", "T5", "P3",
    "PZ", "P4", "T6", "O1", "OZ", "O2",
]


# Shin et al. EEG_01-26_MATLAB attention/rest dataset.  The raw files contain
# 30 channels, but LaBraM's standard_1020 table does not include AFF5h, AFF6h,
# HEOG, or VEOG, so the runnable subset keeps the 26 compatible EEG channels.
ATTENTION_26_CHANNELS = [
    "FP1", "AFZ", "F1", "FC5", "FC1", "T7", "C3", "CZ",
    "CP5", "CP1", "P7", "P3", "PZ", "POZ", "O1", "FP2",
    "F2", "FC2", "FC6", "C4", "T8", "CP2", "CP6", "P4",
    "P8", "O2",
]

ATTENTION_10_CHANNELS = [
    "FP1", "F1", "C3", "P3", "O1",
    "FP2", "F2", "C4", "P4", "O2",
]


# ERP CORE simple_data.pt channel order after CSLP-AE preprocessing.
# HEOG/VEOG are present in the source tensor but are excluded from the
# LaBraM-compatible 28-channel target space because they are not in
# standard_1020.
ERPCORE_30_CHANNELS = [
    "FP1", "F3", "F7", "FC3", "C3", "C5", "P3", "P7",
    "PO7", "PO3", "O1", "OZ", "PZ", "CPZ", "FP2", "FZ",
    "F4", "F8", "FC4", "FCZ", "CZ", "C4", "C6", "P4",
    "P8", "PO8", "PO4", "O2", "HEOG", "VEOG",
]

ERPCORE_28_CHANNELS = [
    channel for channel in ERPCORE_30_CHANNELS
    if channel not in {"HEOG", "VEOG"}
]

ERPCORE_12_CHANNELS = [
    "FP1", "FP2", "F3", "F4", "F7", "F8",
    "C3", "C4", "P3", "P4", "O1", "O2",
]


# OpenNeuro ds005416 24-channel subset retained for older artifacts.
FATIG_24_CHANNELS = [
    "FP1", "FP2", "AF3", "AF4", "F7", "FZ", "F8", "FC5",
    "FC6", "FT7", "FT8", "C3", "CZ", "C4", "CP3", "CP4",
    "P3", "PZ", "P4", "PO3", "PO4", "O1", "OZ", "O2",
]


# PhysioNet EEG Motor Movement/Imagery Database (EEGMMIDB).
# Keep this order aligned with EEG-FM-Bench's motor_mv_img 10-10 montage.
PHYSIONET_64_CHANNELS = [
    "FP1", "FPZ", "FP2", "AF7", "AF3", "AFZ", "AF4", "AF8",
    "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8",
    "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8",
    "T9", "T7", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8", "T10",
    "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8",
    "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8",
    "PO7", "PO3", "POZ", "PO4", "PO8", "O1", "OZ", "O2", "IZ",
]

PHYSIONET_23_CHANNELS = [
    "F3", "F4",
    "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6",
    "C5", "C3", "C1", "CZ", "C2", "C4", "C6",
    "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6",
]

PHYSIONET_32_CHANNELS = [
    "F5", "F3", "F4", "F6",
    "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8",
    "T7", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8",
    "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6",
    "P3", "PZ", "P4",
]

PHYSIONET_MASKED_41_CHANNELS = [
    channel for channel in PHYSIONET_64_CHANNELS
    if channel not in PHYSIONET_23_CHANNELS
]

PHYSIONET_MASKED_32_CHANNELS = [
    channel for channel in PHYSIONET_64_CHANNELS
    if channel not in PHYSIONET_32_CHANNELS
]


HIGH_DENSITY_AAD_84_CHANNELS = [
    "FP1", "FPZ", "FP2", "AF7", "AF3", "AF1", "AF2", "AF4",
    "AF8", "F9", "F7", "F5", "F3", "F1", "FZ", "F2",
    "F4", "F6", "F8", "F10", "FT9", "FT7", "FC5", "FC3",
    "FC1", "FC2", "FC4", "FC6", "FT8", "FT10", "T7", "C5",
    "C3", "C1", "CZ", "C2", "C4", "C6", "T8", "TP9",
    "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6",
    "TP8", "TP10", "P9", "P7", "P5", "P3", "P1", "PZ",
    "P2", "P4", "P6", "P8", "P10", "PO9", "PO7", "PO3",
    "PO1", "POZ", "PO2", "PO4", "PO8", "PO10", "O1", "OZ",
    "O2", "IZ", "CCP1", "CCP2", "CCP3", "CCP4", "CCP5",
    "CCP6", "TTP7h", "TPP9h", "TPP8h", "TPP10h",
]

SEEDV_62_CHANNELS = [
    "FP1", "FPZ", "FP2", "AF3", "AF4", "F7", "F5", "F3",
    "F1", "FZ", "F2", "F4", "F6", "F8", "FT7", "FC5",
    "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8", "T7",
    "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8",
    "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6",
    "TP8", "P7", "P5", "P3", "P1", "PZ", "P2", "P4",
    "P6", "P8", "PO7", "PO5", "PO3", "POZ", "PO4", "PO6",
    "PO8", "CB1", "O1", "OZ", "O2", "CB2",
]

# SEED and SEED-V use the same 62-channel ESI NeuroScan montage.  Keep a
# dataset-specific name so launchers/loaders cannot accidentally conflate the
# two datasets even though their physical channel order is identical.
SEED_62_CHANNELS = list(SEEDV_62_CHANNELS)

TUEV23_SEEDV62_EXTRA_CHANNELS = TUEV_23_CHANNELS + [
    ch for ch in SEEDV_62_CHANNELS
    if ch not in TUEV_23_CHANNELS and ch in standard_1020
]

SEEDV_23_CHANNELS = [
    "FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4",
    "O1", "O2", "F7", "F8", "T7", "T8", "P7", "P8",
    "FZ", "CZ", "PZ", "FCZ", "CPZ", "CB1", "CB2",
]

# PreExp36N: fixed 23-channel SEED subset. SEED and SEED-V share the same
# 62-channel montage, so reuse the audited, left-right symmetric 23-channel
# layout rather than defining a second ordering for the same electrodes.
SEED_23_CHANNELS = list(SEEDV_23_CHANNELS)

SEED_MASKED_39_CHANNELS = [
    channel for channel in SEED_62_CHANNELS
    if channel not in SEED_23_CHANNELS
]
