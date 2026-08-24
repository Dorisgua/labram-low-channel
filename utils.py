"""Small compatibility helpers used by the ERP-Core Stage1/Stage2 code."""

from pyhealth.metrics import binary_metrics_fn, multiclass_metrics_fn

# Distributed training, TensorBoard, pretraining datasets, AMP helpers and
# legacy TUAB/TUEV loaders are intentionally omitted from this standalone tree.
standard_1020 = [
    "FP1", "FPZ", "FP2",
    "AF9", "AF7", "AF5", "AF3", "AF1", "AFZ", "AF2", "AF4", "AF6", "AF8", "AF10",
    "F9", "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8", "F10",
    "FT9", "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8", "FT10",
    "T9", "T7", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8", "T10",
    "TP9", "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8", "TP10",
    "P9", "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8", "P10",
    "PO9", "PO7", "PO5", "PO3", "PO1", "POZ", "PO2", "PO4", "PO6", "PO8", "PO10",
    "O1", "OZ", "O2", "O9", "CB1", "CB2",
    "IZ", "O10", "T3", "T5", "T4", "T6", "M1", "M2", "A1", "A2",
    "CFC1", "CFC2", "CFC3", "CFC4", "CFC5", "CFC6", "CFC7", "CFC8",
    "CCP1", "CCP2", "CCP3", "CCP4", "CCP5", "CCP6", "CCP7", "CCP8",
    "T1", "T2", "FTT9h", "TTP7h", "TPP9h", "FTT10h", "TPP8h", "TPP10h",
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1", "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
]


def load_state_dict(model, state_dict, prefix="", ignore_missing="relative_position_index"):
    """Load a checkpoint while preserving LaBraM's missing-key reporting."""
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    metadata = getattr(state_dict, "_metadata", None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, current_prefix=""):
        local_metadata = (
            {} if metadata is None else metadata.get(current_prefix[:-1], {})
        )
        module._load_from_state_dict(
            state_dict,
            current_prefix,
            local_metadata,
            True,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        for name, child in module._modules.items():
            if child is not None:
                load(child, current_prefix + name + ".")

    load(model, current_prefix=prefix)

    warn_missing_keys = []
    ignored_missing_keys = []
    for key in missing_keys:
        if any(ignore_key in key for ignore_key in ignore_missing.split("|")):
            ignored_missing_keys.append(key)
        else:
            warn_missing_keys.append(key)

    if warn_missing_keys:
        print(
            f"Weights of {model.__class__.__name__} not initialized from "
            f"pretrained model: {warn_missing_keys}"
        )
    if unexpected_keys:
        print(
            f"Weights from pretrained model not used in "
            f"{model.__class__.__name__}: {unexpected_keys}"
        )
    if ignored_missing_keys:
        print(
            f"Ignored weights of {model.__class__.__name__} not initialized "
            f"from pretrained model: {ignored_missing_keys}"
        )
    if error_msgs:
        print("\n".join(error_msgs))


def get_input_chans(ch_names):
    """Map channel names to LaBraM positions; position 0 is the CLS token."""
    return [0, *(standard_1020.index(name) + 1 for name in ch_names)]


def get_metrics(output, target, metrics, is_binary, threshold=0.5):
    """Keep the metric behavior used by the original Stage2 runner."""
    if is_binary:
        target_sum = sum(target)
        if "roc_auc" not in metrics or target_sum * (len(target) - target_sum) != 0:
            return binary_metrics_fn(
                target,
                output,
                metrics=metrics,
                threshold=threshold,
            )
        return {
            "accuracy": 0.0,
            "balanced_accuracy": 0.0,
            "pr_auc": 0.0,
            "roc_auc": 0.0,
        }
    return multiclass_metrics_fn(target, output, metrics=metrics)
