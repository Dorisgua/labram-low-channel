# LaBraM-unified-AON 运行命令

## 1. 进入仓库并激活环境

```bash
cd /inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/eeg-main/LaBraM-unified-AON

export MAMBA_ROOT_PREFIX=/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/micromamba-root
eval "$(/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/bin/micromamba shell hook -s bash)"
micromamba activate labram
```

确认 PyTorch 能识别 GPU：

```bash
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))'
```

当前可运行环境应显示 PyTorch `2.0.1`、CUDA `11.8`，并且
`torch.cuda.is_available()` 为 `True`。

## 2. O、N、A 和 freeze/full 的含义

- `O`：使用完整导联，不做通道补全。
- `N`：使用少导联，不做通道补全。
- `A`：使用少导联，并通过 prototype 补全到目标导联。
- `freeze_cnn.sh`：冻结 CNN/patch embedding，训练 Transformer 和分类头。
- `full_finetune.sh`：CNN、Transformer 和分类头一起微调。

## 3. 运行单个实验

所有脚本均从仓库根目录用 `bash` 启动，不要求脚本本身具有 executable bit。

以 TUEV 为例：

```bash
# O：完整导联，冻结 CNN
bash scripts/tuev/O/freeze_cnn.sh

# O：完整导联，全量微调
bash scripts/tuev/O/full_finetune.sh

# N：少导联，冻结 CNN
bash scripts/tuev/N/freeze_cnn.sh

# N：少导联，全量微调
bash scripts/tuev/N/full_finetune.sh

# A：少导联通过 prototype 补全，冻结 CNN
bash scripts/tuev/A/freeze_cnn.sh
```

其他数据集使用相同格式，例如：

```bash
bash scripts/erp_core/O/freeze_cnn.sh
bash scripts/physionet/A/freeze_cnn.sh
bash scripts/seed/N/full_finetune.sh
bash scripts/seedv/O/freeze_cnn.sh
bash scripts/eegmat/A/freeze_cnn.sh
bash scripts/hgd/O/full_finetune.sh
bash scripts/siena/N/freeze_cnn.sh
bash scripts/attention/A/freeze_cnn.sh
```

`aad`、`faced` 和 `zuo2025` 当前只有 O/freeze wrapper：

```bash
bash scripts/aad/O/freeze_cnn.sh
bash scripts/faced/O/freeze_cnn.sh
bash scripts/zuo2025/O/freeze_cnn.sh
```

## 4. 指定 GPU、前台运行和端口

默认使用 GPU 0，并在后台启动。为了直接在终端看到输出，建议调试时使用前台模式：

```bash
CUDA_VISIBLE_DEVICES=0 RUN_FOREGROUND=1 bash scripts/tuev/A/freeze_cnn.sh
```

如果同时运行多个实验，每个实验必须使用不同的 `MASTER_PORT`：

```bash
CUDA_VISIBLE_DEVICES=0 MASTER_PORT=29501 bash scripts/tuev/O/freeze_cnn.sh
CUDA_VISIBLE_DEVICES=1 MASTER_PORT=29502 bash scripts/tuev/N/freeze_cnn.sh
```

不要在同一张 GPU 上同时启动多个大模型实验，否则容易出现 CUDA OOM。

## 5. 只检查命令，不启动训练

新 O/N/A wrapper 支持 `DRY_RUN=1`：

```bash
DRY_RUN=1 bash scripts/tuev/A/freeze_cnn.sh
```

它会打印最终的 `torchrun` 命令，但不会创建训练进程。正式运行前建议先执行一次。

## 6. 临时覆盖数据路径

无需修改脚本，可以在命令前设置 `DATA_PATH`：

```bash
DATA_PATH=/path/to/dataset \
RUN_FOREGROUND=1 \
bash scripts/tuev/A/freeze_cnn.sh
```

BCI-IV-2a 的 base 当前没有启用默认 `DATA_PATH`，运行时需要显式指定 JSON
索引目录：

```bash
DATA_PATH=/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/eeg-main/LaBraM-unified-AON/preprocessing/BCI-IV-2A/multi_subject_json \
RUN_FOREGROUND=1 \
bash scripts/bciiv2a/O/freeze_cnn.sh
```

这些 JSON 索引中的实际 `.pkl` 文件路径位于 HDD：

```text
/inspire/hdd/project/sais-medical/public/share_medical/EEG/BCI-IV-2A/processed_data
```

## 7. 运行旧版长文件名脚本

旧脚本同样使用 `bash`。例如 TUEV 17Ah：

```bash
bash scripts/tuev/17Ah.finetune_tuev_labrambase_freeze_cnn_train_transformer_head_eval.sh
```

17Ah 会使用 `nohup` 在后台运行，并打印日志位置。它也可以临时覆盖数据路径：

```bash
DATA_PATH=/inspire/hdd/project/sais-medical/public/share_medical/EEG/TUEZ/v2.0.1/processed_labram/processed \
bash scripts/tuev/17Ah.finetune_tuev_labrambase_freeze_cnn_train_transformer_head_eval.sh
```

新实验优先使用 `scripts/<dataset>/{O,N,A}/...sh`；旧版长文件名脚本主要用于复现实验。


## 8. 查看日志

后台运行时，脚本会打印 `PID` 和 `Log`。根据打印出的路径查看日志：

```bash
tail -f /path/to/run.log
```

新 wrapper 的日志通常位于：

```text
outputs/<dataset>/<script_name>/run_logs/
```

检查 GPU 进程和显存：

```bash
nvidia-smi
```

## 9. Resume 和只评估

从 checkpoint 恢复训练：

```bash
RESUME=/path/to/checkpoint.pth bash scripts/tuev/A/freeze_cnn.sh
```

只进行评估：

```bash
RESUME=/path/to/checkpoint-best.pth EVAL_ONLY=1 RUN_FOREGROUND=1 \
bash scripts/tuev/A/freeze_cnn.sh
```

## 10. 批量运行

先只打印计划：

```bash
DRY_RUN=1 bash scripts/run_all_non_others.sh
```

该调度器默认每个数据集最多并发 3 个任务。单卡环境建议改为顺序运行：

```bash
MAX_CONCURRENT=1 CUDA_VISIBLE_DEVICES=0 bash scripts/run_all_non_others.sh
```

只顺序运行 BCI-IV-2a 和 TUEV：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_bciiv2a_tuev_sequential.sh
```

## 11. 运行前注意事项

1. 确认 `checkpoints/labram-base.pth` 存在。
2. 确认所用数据目录、manifest/JSON 索引和 prototype 文件存在。
3. 先用 `DRY_RUN=1` 检查最终命令和数据路径。
4. 并行任务使用不同 `MASTER_PORT`，并确认 GPU 显存足够。
5. `freeze_cnn` 和 `full_finetune` 是不同实验，不要混用输出目录或 checkpoint。
