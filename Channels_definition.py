"""ERP-Core channel orders used by the standalone Stage1/Stage2 pipeline."""

# Other dataset channel tables from the upstream multi-dataset repository are
# intentionally omitted: this package currently trains only on ERP-Core.
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
