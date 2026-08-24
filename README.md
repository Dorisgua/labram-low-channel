# ERP-Core LaBraM disentanglement

This repository is a small standalone extraction of the LaBraM-based ERP-Core
Stage1/Stage2 pipeline. The training implementation lives in
[`clean_disentangle/`](clean_disentangle/README.md).

Retained upstream compatibility files:

- `modeling_finetune.py`: LaBraM backbone and temporal patch embedding.
- `modeling_adabrain.py`: Stage2 max-norm linear layer.
- `utils.py`: checkpoint loading, channel mapping, and metrics only.
- `Channels_definition.py`: ERP-Core channel orders only.
- `data_processor/erpcore.py` and `erpcore_cslp.py`: ERP-Core data loading and pair sampling.

The dataset is external and is selected with `DATA_PATH` in the Stage1/Stage2
launchers. The required LaBraM checkpoint is expected at
`checkpoints/labram-base.pth`.

The retained LaBraM-derived code remains under the repository's original
[MIT license](LICENSE).
