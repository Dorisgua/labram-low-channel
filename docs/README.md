# LaBraM-unified-AON 运行命令

## 1. 进入仓库并激活环境（先按照labram的readme激活环境）

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

## 12. 各数据集旧实验、新实验与运行命令

每个数据集分为“旧实验结果”和“新实验结果”两部分：旧实验整理自
`scripts/<dataset>/HISTORICAL_RESULTS.md`；新实验整理自本机
`outputs/<dataset>/*/run_logs/` 中 2026-08-23 至 2026-08-25 完成的日志。新表只记录
日志末尾能够核对到 `Test metrics distribution at best val epoch` 的 seed 0 结果。
当前标准 O/N/A wrapper 使用 `adabrain_all_token`、scope=`real`；TUEV 的 17 系列
重跑仍使用 `mean_pool`、scope=`all`，已在表内单独标明。

旧实验结果内部再按 classifier 拆成 `adabrain_all_token` 与 `mean_pool` 两张表；
没有对应实验时保留“暂无结果”的占位表。FACED 历史记录还包含独立的
`adabrain_mlp_token`，为避免误归类，额外放在“其他 classifier”表中。

表格后的命令是当前推荐 wrapper，并已通过 `DRY_RUN=1` 检查。旧实验可能使用
不同 classifier、token scope 或 seed，因此新旧表的数值不能直接视为同一配置复现。

命令默认后台运行；需要在终端直接查看训练过程时，在命令前加
`RUN_FOREGROUND=1`。

### 当前标准 wrapper 的训练与 checkpoint 配置

下表记录当前 `scripts/<dataset>/base.sh` 与共享 `scripts/base.sh` 共同形成的默认值。
各数据集 base 负责设置 `BEST_METRIC`，随后调用共享 base；`BATCH_SIZE`、`LR` 和
`CLASSIFIER_MODE` 当前均采用共享 base 的默认值，其中标准 O/N/A wrapper 又显式固定
`CLASSIFIER_MODE=adabrain_all_token`、scope=`real`。`BEST_METRIC`、`BATCH_SIZE` 和
`LR` 可以通过同名环境变量临时覆盖。

| 数据集 | best checkpoint 选择依据 | Batch size | LR | Classifier mode |
|---|---|---:|---:|---|
| BCI-IV-2a | Val Balanced Accuracy | 64 | `5e-4` | `adabrain_all_token`，scope=`real` |
| ERP-Core | Val Balanced Accuracy | 64 | `5e-4` | `adabrain_all_token`，scope=`real` |
| TUEV | Val Balanced Accuracy | 64 | `5e-4` | `adabrain_all_token`，scope=`real` |
| PhysioNet | Val Balanced Accuracy | 64 | `5e-4` | `adabrain_all_token`，scope=`real` |
| SEED | Val Balanced Accuracy | 64 | `5e-4` | `adabrain_all_token`，scope=`real` |
| SEED-V | Val Accuracy | 64 | `5e-4` | `adabrain_all_token`，scope=`real` |
| EEGMAT | Val Balanced Accuracy | 64 | `5e-4` | `adabrain_all_token`，scope=`real` |
| HGD | Val Balanced Accuracy | 64 | `5e-4` | `adabrain_all_token`，scope=`real` |
| Siena | Val Balanced Accuracy | 64 | `5e-4` | `adabrain_all_token`，scope=`real` |
| Attention | Val Balanced Accuracy | 64 | `5e-4` | `adabrain_all_token`，scope=`real` |
| AAD | Val Balanced Accuracy | 64 | `5e-4` | `adabrain_all_token`，scope=`real` |
| FACED | Val Balanced Accuracy | 64 | `5e-4` | `adabrain_all_token`，scope=`real` |
| Zuo2025 | Val Balanced Accuracy | 64 | `5e-4` | `adabrain_all_token`，scope=`real` |

这里的 TUEV 行描述当前标准 O/N/A wrapper。历史 17O/17N/17Ah 入口使用
`mean_pool`、scope=`all`，并按 Val Cohen's Kappa 选择 checkpoint，不应与该行混用。

<!-- ### 旧实验的训练与 checkpoint 配置

下表对应后文“旧实验结果”中的实验族。参数根据生成这些历史结果时保留的长脚本
快照核对，不使用当前共享 `scripts/base.sh` 的默认值反推。一个数据集的多种 classifier
若使用相同 best metric、batch size 和 LR，则合并在同一行；FACED 两种 classifier 的
LR 不同，因此分成两行。

| 数据集 | 对应旧实验族 | best checkpoint 选择依据 | Batch size | LR | Classifier mode | 结果 |
|---|---|---|---:|---:|---|---|
| BCI-IV-2a | 33A/33N/33O | Val Balanced Accuracy | 64 | `5e-4` | `adabrain_all_token` |  |
| ERP-Core | 45A/45N/45O | Val Balanced Accuracy | 64 | `5e-4` | `adabrain_all_token` / `mean_pool` | 65.29% / 47.83%（45O all-token，full finetune） |
| TUEV | 17O/17N/17Ah | Val Cohen's Kappa | 64 | `5e-4` | `mean_pool` | 81.68% / 63.90%（17O，freeze CNN） |
| PhysioNet | 34A/34N/34O，23/32/64 导联 | Val Balanced Accuracy | 64 | `5e-4` | `adabrain_all_token` | 63.22% / 63.23%（34O，freeze CNN） |
| SEED | 36A/36N/36O | Val Accuracy | 64 | `5e-4` | `adabrain_all_token` / `mean_pool` | 55.70% / 55.24%（36N all-token，freeze CNN） |
| SEED-V | 35A/35N/35O | Val Accuracy | 64 | `5e-4` | `mean_pool` | 40.81% / 41.08%（35O，full finetune） |
| EEGMAT | 37A/37N/37O | Val Accuracy | 64 | `5e-4` | `adabrain_all_token` / `mean_pool` | 83.33% / 83.33%（37O all-token，freeze CNN） |
| HGD | 39A/39N/39O | Val Accuracy | 64 | `5e-4` | `mean_pool` | 74.35% / 74.35%（39O，freeze CNN） |
| Siena | 40A/40N/40O | Val Balanced Accuracy | 32 | `5e-4` | `adabrain_all_token` / `mean_pool` | Acc 97.69%（40O mean-pool，freeze）；BAcc 79.28%（40O all-token，full） |
| Attention | 44A/44N/44O | Val Balanced Accuracy | 64 | `5e-4` | `mean_pool` | Acc 85.00%（44O-10 seed2，full）；BAcc 76.89%（44O-26 seed0，full） |
| AAD | 43O | Val Accuracy | 64 | `5e-4` | `mean_pool` | 暂无可核对的结果日志 |
| FACED | 42O mean-pool | Val Accuracy | 64 | `5e-4` | `mean_pool` | 18.12% / 17.12%（run2，freeze CNN） |
| FACED | 42O MLP head | Val Accuracy | 64 | `1e-4` | `adabrain_mlp_token` | 54.09% / 54.19%（full finetune） |
| Zuo2025 | 38O | Val Accuracy | 64 | `5e-4` | `mean_pool` | 暂无可核对的结果日志 |

这里的 Batch size 是每个进程的 `--batch_size`，不是乘上 `UPDATE_FREQ` 和 GPU 进程数
后的 global/effective batch size。旧结果表中的 Val BAcc 只是报告指标时，也不代表该实验
一定按 Val BAcc 选择 checkpoint；应以上表的 `best checkpoint 选择依据` 为准。 -->


### 整体结论
0. 旧实验里同时满足 Test Acc 和 Test BAcc 都是 A（freeze CNN） > N（freeze CNN） 的数据集

  - BCI-IV-2a：仅 33Arealada > 33Nada；普通 33Aada 不满足。
  - TUEV：17Ah > 17N。
  - PhysioNet：32 导联配置满足；23 导联配置不满足。
  - SEED：mean_pool 的 Ah、Al 均高于 N，但有效 seed 集合不完全一致。
  - SEED-V：mean_pool 的 Ah、Al 均高于 N。
  - EEGMAT：仅 mean_pool 满足；adabrain_all_token 三 seed 均值不满足。
  - Siena：40Aada > 40Nada。

  另外，Attention 只在 Test Acc 均值上满足：

  - Acc：A 78.27% > N 76.79%
  - BAcc：A 63.70% < N 68.00%

  因此 Attention 不属于“两项都满足”的名单。ERP-Core 没有同为 freeze CNN 的 A/N 旧实验可直接比较。

1. 按旧结果表，同时满足 Test Acc 和 Test BAcc 都是 O（freeze CNN）> A（freeze CNN）> N（freeze CNN）的数据集有：

  - BCI-IV-2a：仅使用 33Arealada 时满足；普通 33Aada 不满足。
  - TUEV
  - PhysioNet：仅 32 导联配置满足，23 导联配置不满足。
  - SEED-V：Ah 和 Al 两种 A 都满足。
  - Siena

  具体数值：

   数据集          Test Acc：O > A > N         Test BAcc：O > A > N
  ━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━
   BCI-IV-2a       55.48% > 54.71% > 52.62%    55.48% > 54.71% > 52.62%
  ──────────────  ──────────────────────────  ──────────────────────────
   TUEV            81.68% > 81.30% > 79.87%    63.90% > 61.48% > 60.62%
  ──────────────  ──────────────────────────  ──────────────────────────
   PhysioNet-32    63.22% > 55.26% > 54.97%    63.23% > 55.27% > 55.00%
  ──────────────  ──────────────────────────  ──────────────────────────
   SEED-V（Ah）    40.18% > 39.63% > 39.18%    40.53% > 39.51% > 39.10%
  ──────────────  ──────────────────────────  ──────────────────────────
   SEED-V（Al）    40.18% > 39.46% > 39.18%    40.53% > 39.44% > 39.10%
  ──────────────  ──────────────────────────  ──────────────────────────
   Siena           97.53% > 97.31% > 97.01%    51.64% > 50.38% > 49.65%

  其中 TUEV 的 N 只有 seed 0，而 O/A 是多 seed 汇总，因此属于“按 README 当前汇总值满足”，并不是严格
  的相同 seed 对比。

3. 新实验里同时满足 Test Acc 和 Test BAcc 都是 A（freeze CNN） > N（freeze CNN） 的数据集

       数据集          Test Acc：A > N    Test BAcc：A > N
  ━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━
   BCI-IV-2a       54.71% > 52.70%    54.71% > 52.70%
  ──────────────  ─────────────────  ──────────────────
   ERP-Core        61.38% > 60.59%    40.29% > 39.73%
  ──────────────  ─────────────────  ──────────────────
   PhysioNet-32    54.36% > 53.62%    54.37% > 53.65%
  ──────────────  ─────────────────  ──────────────────
   SEED            55.08% > 54.32%    54.60% > 53.90%
  ──────────────  ─────────────────  ──────────────────
   HGD             80.03% > 78.37%    80.02% > 78.37%

  共 5 个数据集。这里只比较 A/N 的 freeze CNN，没有使用 full finetune。

4. 新实验里同时满足 Test Acc 和 Test BAcc 都是 O（freeze CNN） > A（freeze CNN） > N（freeze CNN） 的数据集：

  - BCI-IV-2a
  - PhysioNet
  - SEED
  - HGD

  仅 Test Acc 满足：

  - Siena

  仅 Test BAcc 满足：
  - seedv

  TUEV、EEGMAT、Attention 两项都不满足。（tuev可能有这个数据集自己的特征）
  AAD、FACED、Zuo2025 缺少完整的 A/N 新实验，无法比较。

5. 如果要求 Test Acc 和 Test BAcc 都满足 A（freeze CNN）> N（full finetune），新实验共有 2 个数据集：

   数据集              A/freeze             N/full    A − N
  ━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━  ━━━━━━━
   BCI-IV-2a    Acc/BAcc 54.71%    Acc/BAcc 52.39%    +2.32
  ───────────  ─────────────────  ─────────────────  ───────
   EEGMAT       Acc/BAcc 74.17%    Acc/BAcc 73.33%    +0.84

  另外 Siena 只有 Test Acc 满足，但 Test BAcc 不满足，因此未计入。

  相差不大的：
     数据集          Test Acc：A − N    Test BAcc：A − N
  ━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━
   SEED-V                    -0.19               -0.16
  ──────────────  ─────────────────  ──────────────────
   SEED                      -0.26               -0.24
  ──────────────  ─────────────────  ──────────────────
   EEGMAT                    +0.84               +0.84
  ──────────────  ─────────────────  ──────────────────
   PhysioNet-32              -0.95               -0.96

待做：tuev和seedv的base.sh需要换选checkpoint的方式（tuev按照cohen kappa选择）和修改batchsize；需要mean pool的classifier的方式

### 12.1 BCI-IV-2a

来源：[scripts/bciiv2a/HISTORICAL_RESULTS.md](../scripts/bciiv2a/HISTORICAL_RESULTS.md)

#### 旧实验结果

##### classifier = `adabrain_all_token`

| 历史实验 | O/N/A | 输入/目标导联 | completion | classifier | 训练方式 | 最佳 epoch | Val BAcc | Test Acc | Test BAcc | Test Kappa | Test F1 |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|
| 33Aada | A | 13 → 22 | `bciiv2a13_with_bciiv2a22`，high pool | `adabrain_all_token` | freeze CNN | 8 | 49.54% | 49.23% | 49.23% | 0.3230 | 49.09% |
| 33Arealada | A | 13 → 22 | `bciiv2a13_with_bciiv2a22`，high pool | `adabrain_all_token`，scope=`real` | freeze CNN | 9 | 53.32% | 54.71% | 54.71% | 0.3961 | 54.58% |
| 33Nada | N | 13 | `none` | `adabrain_all_token` | freeze CNN | 21 | 52.70% | 52.62% | 52.62% | 0.3683 | 52.49% |
| 33Nada | N | 13 | `none` | `adabrain_all_token` | full finetune | 9 | 52.93% | 52.31% | 52.31% | 0.3642 | 52.29% |
| 33Oada | O | 22 | `none` | `adabrain_all_token` | freeze CNN | 36 | 57.56% | 55.48% | 55.48% | 0.4064 | 55.50% |
| 33Oada | O | 22 | `none` | `adabrain_all_token` | full finetune | 36 | **63.12%** | **57.95%** | **57.95%** | **0.4393** | **57.92%** |

**Test Acc 排序（从高到低）：** 33Oada（full finetune）57.95% > 33Oada（freeze CNN）55.48% > 33Arealada（freeze CNN）54.71% > 33Nada（freeze CNN）52.62% > 33Nada（full finetune）52.31% > 33Aada（freeze CNN）49.23%。

**Test BAcc 排序（从高到低）：** 33Oada（full finetune）57.95% > 33Oada（freeze CNN）55.48% > 33Arealada（freeze CNN）54.71% > 33Nada（freeze CNN）52.62% > 33Nada（full finetune）52.31% > 33Aada（freeze CNN）49.23%。

##### classifier = `mean_pool`

| 状态 | 说明 |
|---|---|
| — | 暂无可核对的 `mean_pool` 旧实验结果 |

**Test Acc 排序（从高到低）：** 暂无可排序结果。

**Test BAcc 排序（从高到低）：** 暂无可排序结果。

#### 新实验结果

| 新实验 | 训练方式 | 最佳 epoch | Val 选择指标 | Test Acc | Test BAcc | Test Kappa | Test F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| A | freeze CNN | 9 | BAcc 53.32% | 54.71% | 54.71% | 0.3961 | 54.58% |
| N | freeze CNN | 39 | BAcc 54.01% | 52.70% | 52.70% | 0.3693 | 52.49% |
| N | full finetune | 10 | BAcc 52.70% | 52.39% | 52.39% | 0.3652 | 52.42% |
| O | freeze CNN | 36 | BAcc 57.56% | 55.48% | 55.48% | 0.4064 | 55.50% |
| O | full finetune | 36 | BAcc 61.42% | **58.87%** | **58.87%** | **0.4516** | **58.88%** |

**Test Acc 排序（从高到低）：** O（full finetune）58.87% > O（freeze CNN）55.48% > A（freeze CNN）54.71% > N（freeze CNN）52.70% > N（full finetune）52.39%。

**Test BAcc 排序（从高到低）：** O（full finetune）58.87% > O（freeze CNN）55.48% > A（freeze CNN）54.71% > N（freeze CNN）52.70% > N（full finetune）52.39%。

运行命令：

```bash
BCI_DATA_PATH="$PWD/preprocessing/BCI-IV-2A/multi_subject_json"

DATA_PATH="$BCI_DATA_PATH" bash scripts/bciiv2a/O/freeze_cnn.sh
DATA_PATH="$BCI_DATA_PATH" bash scripts/bciiv2a/O/full_finetune.sh
DATA_PATH="$BCI_DATA_PATH" bash scripts/bciiv2a/N/freeze_cnn.sh
DATA_PATH="$BCI_DATA_PATH" bash scripts/bciiv2a/N/full_finetune.sh
DATA_PATH="$BCI_DATA_PATH" bash scripts/bciiv2a/A/freeze_cnn.sh
```

### 12.2 ERP-Core

来源：[scripts/erp_core/HISTORICAL_RESULTS.md](../scripts/erp_core/HISTORICAL_RESULTS.md)

#### 旧实验结果

##### classifier = `adabrain_all_token`

| 历史实验 | O/N/A | 输入/目标导联 | completion | classifier | 训练方式 | Test Acc | Test BAcc | Test Kappa | Test F1 |
|---|---|---|---|---|---|---:|---:|---:|---:|
| 45Oada，erpcore28 | O | 28 | `none` | `adabrain_all_token`，scope=`all` | full finetune | **65.29%** | **47.83%** | 0.5656 | 62.85% |
| 45Nada，erpcore12 | N | 12 | `none` | `adabrain_all_token`，scope=`all` | full finetune | 63.31% | 42.20% | 0.5326 | 58.07% |
| 45Aada，erpcore12→28 | A | 12 → 28 | prototype completion | `adabrain_all_token`，scope=`all` | freeze CNN | 60.05% | 39.37% | 0.4959 | 56.62% |

**Test Acc 排序（从高到低）：** 45Oada，erpcore28（full finetune）65.29% > 45Nada，erpcore12（full finetune）63.31% > 45Aada，erpcore12→28（freeze CNN）60.05%。

**Test BAcc 排序（从高到低）：** 45Oada，erpcore28（full finetune）47.83% > 45Nada，erpcore12（full finetune）42.20% > 45Aada，erpcore12→28（freeze CNN）39.37%。

##### classifier = `mean_pool`

| 历史实验 | O/N/A | 输入/目标导联 | completion | classifier | 训练方式 | Test Acc | Test BAcc | Test Kappa | Test F1 |
|---|---|---|---|---|---|---:|---:|---:|---:|
| 45Nmeanpool，erpcore12，seed0 | N | 12 | `none` | `mean_pool` | full finetune | 59.12% | 43.99% | 0.4948 | 58.41% |
| 45Nmeanpool，erpcore12，seed1 | N | 12 | `none` | `mean_pool` | full finetune | 62.53% | 44.38% | 0.5326 | 58.76% |
| 45Nmeanpool，erpcore12，seed2 | N | 12 | `none` | `mean_pool` | full finetune | 60.77% | 43.30% | 0.5096 | 58.26% |
| 45Omeanpool，erpcore28 | O | 28 | `none` | `mean_pool` | full finetune | 62.46% | 47.30% | 0.5374 | 61.56% |
| 45Omeanpool，erpcore28 | O | 28 | `none` | `mean_pool` | freeze CNN | 57.26% | 43.03% | 0.4787 | 56.91% |
| 45Omeanpool，erpcore28，seed0 | O | 28 | `none` | `mean_pool` | freeze CNN | 56.29% | 42.78% | 0.4720 | 56.14% |
| 45Omeanpool，erpcore28，seed1 | O | 28 | `none` | `mean_pool` | freeze CNN | 54.01% | 41.59% | 0.4475 | 54.15% |
| 45Omeanpool，erpcore28，seed2 | O | 28 | `none` | `mean_pool` | freeze CNN | 57.25% | 43.01% | 0.4785 | 56.89% |
| 45Ahmeanpool，erpcore12→28，seed0 | A | 12 → 28 | prototype completion | `mean_pool` | freeze CNN | 54.71% | 38.48% | 0.4414 | 53.80% |
| 45Ahmeanpool，erpcore12→28，seed1 | A | 12 → 28 | prototype completion | `mean_pool` | freeze CNN | 53.52% | 34.06% | 0.4185 | 49.89% |
| 45Ahmeanpool，erpcore12→28，seed2 | A | 12 → 28 | prototype completion | `mean_pool` | freeze CNN | 56.33% | 39.03% | 0.4605 | 54.20% |

排序规则：同一配置的 seed0/1/2 先计算算术平均值，再参与排序。未标 seed 的两条
`45Omeanpool，erpcore28` 历史记录无法确认属于哪一个 seed，因此作为独立单次记录，
不混入 seed0/1/2 均值。

**Test Acc 均值排序（从高到低）：** 45Omeanpool，erpcore28（full finetune，单次记录）62.46% > 45Nmeanpool，erpcore12（full finetune，seed0/1/2 均值）60.81% > 45Omeanpool，erpcore28（freeze CNN，未标 seed 的单次记录）57.26% > 45Omeanpool，erpcore28（freeze CNN，seed0/1/2 均值）55.85% > 45Ahmeanpool，erpcore12→28（freeze CNN，seed0/1/2 均值）54.85%。

**Test BAcc 均值排序（从高到低）：** 45Omeanpool，erpcore28（full finetune，单次记录）47.30% > 45Nmeanpool，erpcore12（full finetune，seed0/1/2 均值）43.89% > 45Omeanpool，erpcore28（freeze CNN，未标 seed 的单次记录）43.03% > 45Omeanpool，erpcore28（freeze CNN，seed0/1/2 均值）42.46% > 45Ahmeanpool，erpcore12→28（freeze CNN，seed0/1/2 均值）37.19%。

#### 新实验结果

| 新实验 | 训练方式 | 最佳 epoch | Val 选择指标 | Test Acc | Test BAcc | Test Kappa | Test F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| A | freeze CNN | 1 | BAcc 38.46% | 61.38% | 40.29% | 0.5065 | 56.27% |
| N | freeze CNN | 1 | BAcc 38.67% | 60.59% | 39.73% | 0.5008 | 55.62% |
| N | full finetune | 4 | BAcc 45.53% | 64.42% | 45.42% | 0.5488 | 59.78% |
| O | freeze CNN | 6 | BAcc 44.74% | 57.60% | 44.36% | 0.4907 | 57.54% |
| O | full finetune | 1 | BAcc 53.99% | **64.61%** | **48.91%** | **0.5612** | **62.53%** |

**Test Acc 排序（从高到低）：** O（full finetune）64.61% > N（full finetune）64.42% > A（freeze CNN）61.38% > N（freeze CNN）60.59% > O（freeze CNN）57.60%。

**Test BAcc 排序（从高到低）：** O（full finetune）48.91% > N（full finetune）45.42% > O（freeze CNN）44.36% > A（freeze CNN）40.29% > N（freeze CNN）39.73%。

运行命令：

```bash
bash scripts/erp_core/O/freeze_cnn.sh
bash scripts/erp_core/O/full_finetune.sh
bash scripts/erp_core/N/freeze_cnn.sh
bash scripts/erp_core/N/full_finetune.sh
bash scripts/erp_core/A/freeze_cnn.sh
```

### 12.3 TUEV

来源：[scripts/tuev/HISTORICAL_RESULTS.md](../scripts/tuev/HISTORICAL_RESULTS.md)

#### 旧实验结果

##### classifier = `adabrain_all_token`

| 状态 | 说明 |
|---|---|
| — | 暂无可核对的 `adabrain_all_token` 旧实验结果 |

**Test Acc 排序（从高到低）：** 暂无可排序结果。

**Test BAcc 排序（从高到低）：** 暂无可排序结果。

##### classifier = `mean_pool`

| 历史实验 | O/N/A | 输入/目标导联 | completion | classifier | 训练方式 | 有效 seeds | Val BAcc 均值 | Test Acc 均值 | Test BAcc 均值 | Test Kappa 均值 | Test F1 均值 |
|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|
| 17N | N | 13 | `none` | `mean_pool` | freeze CNN，50 epochs | 0 | 66.60% | 79.87% | 60.62% | 0.5958 | 80.34% |
| 17Ah | A | 13 → 23 | `tuev13_with_tuev23`，high pool | `mean_pool` | freeze CNN，50 epochs | 0/1/2 | 63.60% | 81.30% | 61.48% | 0.6319 | 81.74% |
| 17O | O | 23 | `none` | `mean_pool` | freeze CNN，50 epochs | 0/1/2 | **68.03%** | **81.68%** | **63.90%** | **0.6391** | **82.01%** |

**Test Acc 排序（从高到低）：** 17O（freeze CNN，50 epochs）81.68% > 17Ah（freeze CNN，50 epochs）81.30% > 17N（freeze CNN，50 epochs）79.87%。

**Test BAcc 排序（从高到低）：** 17O（freeze CNN，50 epochs）63.90% > 17Ah（freeze CNN，50 epochs）61.48% > 17N（freeze CNN，50 epochs）60.62%。

#### 新实验结果

##### classifier / scope = `all-token / real`

| 新实验 | classifier / scope | 训练方式 | 最佳 epoch | Val 选择指标 | Test Acc | Test BAcc | Test Kappa | Test F1 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| A | all-token / real | freeze CNN | 35 | BAcc 69.22% | 71.58% | 54.37% | 0.4760 | 73.61% |
| N | all-token / real | freeze CNN | 4 | BAcc 68.30% | 74.19% | 61.67% | 0.5391 | 76.54% |
| N | all-token / real | full finetune | 30 | BAcc 66.93% | 74.63% | 60.47% | 0.5363 | 76.31% |
| O | all-token / real | freeze CNN | 5 | BAcc 67.67% | 71.64% | 53.27% | 0.4535 | 72.83% |
| O | all-token / real | full finetune | 4 | BAcc 66.94% | 74.19% | 61.92% | 0.5377 | 76.14% |

**Test Acc 排序（从高到低）：** N（full finetune）74.63% > N（freeze CNN）74.19% = O（full finetune）74.19% > O（freeze CNN）71.64% > A（freeze CNN）71.58%。

**Test BAcc 排序（从高到低）：** O（full finetune）61.92% > N（freeze CNN）61.67% > N（full finetune）60.47% > A（freeze CNN）54.37% > O（freeze CNN）53.27%。

##### classifier / scope = `mean-pool / all`

| 新实验 | classifier / scope | 训练方式 | 最佳 epoch | Val 选择指标 | Test Acc | Test BAcc | Test Kappa | Test F1 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 17N 重跑 | mean-pool / all | freeze CNN | 4 | Kappa 0.7742 | 79.87% | 60.62% | 0.5958 | 80.34% |
| 17Ah 重跑 | mean-pool / all | freeze CNN | 2 | Kappa 0.7836 | **83.97%** | 58.75% | **0.6651** | **83.34%** |

**Test Acc 排序（从高到低）：** 17Ah 重跑（freeze CNN）83.97% > 17N 重跑（freeze CNN）79.87%。

**Test BAcc 排序（从高到低）：** 17N 重跑（freeze CNN）60.62% > 17Ah 重跑（freeze CNN）58.75%。

运行命令：

```bash
bash scripts/tuev/O/freeze_cnn.sh
bash scripts/tuev/O/full_finetune.sh
bash scripts/tuev/N/freeze_cnn.sh
bash scripts/tuev/N/full_finetune.sh
bash scripts/tuev/A/freeze_cnn.sh
```

严格运行历史 17O/17N/17Ah 入口：

```bash
bash scripts/tuev/17O.finetune_tuev_labrambase_freeze_cnn_train_transformer_head_eval.sh
bash scripts/tuev/17N.finetune_tuev_labrambase_freeze_cnn_train_transformer_head_eval.sh
bash scripts/tuev/17Ah.finetune_tuev_labrambase_freeze_cnn_train_transformer_head_eval.sh
```

### 12.4 PhysioNet

来源：[scripts/physionet/HISTORICAL_RESULTS.md](../scripts/physionet/HISTORICAL_RESULTS.md)

#### 旧实验结果

##### classifier = `adabrain_all_token`

| 历史实验 | O/N/A | 输入/目标导联 | completion | classifier | 训练方式 | 有效 seeds | Val BAcc 均值 | Test Acc 均值 | Test BAcc 均值 | Test Kappa 均值 | Test F1 均值 |
|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|
| 34Neegfm-23 | N | 23 | `none` | `adabrain_all_token`，scope=`all` | freeze CNN，30 epochs | 0/1/2 | 48.79% | 51.49% | 51.50% | 0.3532 | 51.17% |
| 34Aeegfm-23 | A | 23 → 64 | `physionet23_with_physionet64`，high pool | `adabrain_all_token`，scope=`real` | freeze CNN，30 epochs | 0/1/2 | 47.52% | 51.22% | 51.23% | 0.3497 | 51.26% |
| 34Neegfm-32 | N | 32 | `none` | `adabrain_all_token`，scope=`all` | freeze CNN，30 epochs | 0/1/2 | 52.00% | 54.97% | 55.00% | 0.3998 | 54.65% |
| 34Aeegfm-32 | A | 32 → 64 | `physionet32_with_physionet64`，high pool | `adabrain_all_token`，scope=`real` | freeze CNN，30 epochs | 0/1 | 51.41% | 55.26% | 55.27% | 0.4034 | 55.62% |
| 34Oeegfm | O | 64 | `none` | `adabrain_all_token`，scope=`all` | freeze CNN，30 epochs | 0/1/2 | **58.04%** | **63.22%** | **63.23%** | **0.5095** | **63.28%** |
| 34Oeegfm | O | 64 | `none` | `adabrain_all_token`，scope=`all` | full finetune，50 epochs | 0 | 57.96% | 63.07% | 63.09% | 0.5077 | 63.11% |

**Test Acc 排序（从高到低）：** 34Oeegfm（freeze CNN，30 epochs）63.22% > 34Oeegfm（full finetune，50 epochs）63.07% > 34Aeegfm-32（freeze CNN，30 epochs）55.26% > 34Neegfm-32（freeze CNN，30 epochs）54.97% > 34Neegfm-23（freeze CNN，30 epochs）51.49% > 34Aeegfm-23（freeze CNN，30 epochs）51.22%。

**Test BAcc 排序（从高到低）：** 34Oeegfm（freeze CNN，30 epochs）63.23% > 34Oeegfm（full finetune，50 epochs）63.09% > 34Aeegfm-32（freeze CNN，30 epochs）55.27% > 34Neegfm-32（freeze CNN，30 epochs）55.00% > 34Neegfm-23（freeze CNN，30 epochs）51.50% > 34Aeegfm-23（freeze CNN，30 epochs）51.23%。

##### classifier = `mean_pool`

| 状态 | 说明 |
|---|---|
| — | 暂无可核对的 `mean_pool` 旧实验结果 |

**Test Acc 排序（从高到低）：** 暂无可排序结果。

**Test BAcc 排序（从高到低）：** 暂无可排序结果。

#### 新实验结果-32导联

| 新实验 | 训练方式 | 最佳 epoch | Val 选择指标 | Test Acc | Test BAcc | Test Kappa | Test F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| A | freeze CNN | 1 | BAcc 51.21% | 54.36% | 54.37% | 0.3914 | 54.96% |
| N | freeze CNN | 19 | BAcc 52.54% | 53.62% | 53.65% | 0.3817 | 53.25% |
| N | full finetune | 7 | BAcc 50.87% | 55.31% | 55.33% | 0.4042 | 55.24% |
| O | freeze CNN | 19 | BAcc 58.13% | 62.86% | 62.88% | 0.5049 | 62.75% |
| O | full finetune | 25 | BAcc 57.96% | **63.07%** | **63.09%** | **0.5077** | **63.11%** |

**Test Acc 排序（从高到低）：** O（full finetune）63.07% > O（freeze CNN）62.86% > N（full finetune）55.31% > A（freeze CNN）54.36% > N（freeze CNN）53.62%。

**Test BAcc 排序（从高到低）：** O（full finetune）63.09% > O（freeze CNN）62.88% > N（full finetune）55.33% > A（freeze CNN）54.37% > N（freeze CNN）53.65%。

运行命令：

```bash
bash scripts/physionet/O/freeze_cnn.sh
bash scripts/physionet/O/full_finetune.sh
bash scripts/physionet/N/freeze_cnn.sh
bash scripts/physionet/N/full_finetune.sh
bash scripts/physionet/A/freeze_cnn.sh
```

### 12.5 SEED

来源：[scripts/seed/HISTORICAL_RESULTS.md](../scripts/seed/HISTORICAL_RESULTS.md)

#### 旧实验结果

##### classifier = `adabrain_all_token`

| 历史实验 | O/N/A | 输入/目标导联 | completion | classifier | 训练方式 | 有效 seeds | Val BAcc 均值 | Test Acc 均值 | Test BAcc 均值 | Test Kappa 均值 | Test F1 均值 |
|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|
| 36Oada | O | 62 | `none` | `adabrain_all_token`，scope=`all` | freeze CNN，50 epochs | 0/1/2 | **90.57%** | 53.75% | 53.41% | 0.3058 | 53.52% |
| 36Nada | N | 23 | `none` | `adabrain_all_token`，scope=`all` | freeze CNN，50 epochs | 0/1/2 | 87.42% | **55.70%** | **55.24%** | **0.3341** | **54.50%** |

**Test Acc 排序（从高到低）：** 36Nada（freeze CNN，50 epochs）55.70% > 36Oada（freeze CNN，50 epochs）53.75%。

**Test BAcc 排序（从高到低）：** 36Nada（freeze CNN，50 epochs）55.24% > 36Oada（freeze CNN，50 epochs）53.41%。

##### classifier = `mean_pool`

| 历史实验 | O/N/A | 输入/目标导联 | completion | classifier | 训练方式 | 有效 seeds | Val BAcc 均值 | Test Acc 均值 | Test BAcc 均值 | Test Kappa 均值 | Test F1 均值 |
|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|
| 36Omeanpool | O | 62 | `none` | `mean_pool` | freeze CNN，50 epochs | 0/1/2 | 89.02% | 52.81% | 52.50% | 0.2918 | 52.60% |
| 36Nmeanpool | N | 23 | `none` | `mean_pool` | freeze CNN，50 epochs | 0/1 | 85.81% | 54.56% | 54.16% | 0.3175 | 53.64% |
| 36Ahmeanpool | A | 23 → 62 | `seed23_with_seed62`，high pool | `mean_pool` | freeze CNN，50 epochs | 0/2 | 84.61% | 54.90% | 54.47% | 0.3224 | 53.72% |
| 36Almeanpool | A | 23 → 62 | `seed23_with_seed62`，low pool | `mean_pool` | freeze CNN，50 epochs | 1 | 85.76% | 55.09% | 54.67% | 0.3253 | 54.05% |

**Test Acc 排序（从高到低）：** 36Almeanpool（freeze CNN，50 epochs）55.09% > 36Ahmeanpool（freeze CNN，50 epochs）54.90% > 36Nmeanpool（freeze CNN，50 epochs）54.56% > 36Omeanpool（freeze CNN，50 epochs）52.81%。

**Test BAcc 排序（从高到低）：** 36Almeanpool（freeze CNN，50 epochs）54.67% > 36Ahmeanpool（freeze CNN，50 epochs）54.47% > 36Nmeanpool（freeze CNN，50 epochs）54.16% > 36Omeanpool（freeze CNN，50 epochs）52.50%。

#### 新实验结果

| 新实验 | 训练方式 | 最佳 epoch | Val 选择指标 | Test Acc | Test BAcc | Test Kappa | Test F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| A | freeze CNN | 46 | BAcc 87.03% | 55.08% | 54.60% | 0.3249 | 53.96% |
| N | freeze CNN | 42 | BAcc 87.21% | 54.32% | 53.90% | 0.3137 | 53.31% |
| N | full finetune | 49 | BAcc 86.55% | 55.34% | 54.84% | 0.3287 | 54.08% |
| O | freeze CNN | 42 | BAcc 90.54% | **56.16%** | **55.84%** | **0.3421** | **56.09%** |
| O | full finetune | 48 | BAcc 91.03% | 55.36% | 55.01% | 0.3297 | 54.97% |

**Test Acc 排序（从高到低）：** O（freeze CNN）56.16% > O（full finetune）55.36% > N（full finetune）55.34% > A（freeze CNN）55.08% > N（freeze CNN）54.32%。

**Test BAcc 排序（从高到低）：** O（freeze CNN）55.84% > O（full finetune）55.01% > N（full finetune）54.84% > A（freeze CNN）54.60% > N（freeze CNN）53.90%。

运行命令：

```bash
bash scripts/seed/O/freeze_cnn.sh
bash scripts/seed/O/full_finetune.sh
bash scripts/seed/N/freeze_cnn.sh
bash scripts/seed/N/full_finetune.sh
bash scripts/seed/A/freeze_cnn.sh
```

### 12.6 SEED-V

来源：[scripts/seedv/HISTORICAL_RESULTS.md](../scripts/seedv/HISTORICAL_RESULTS.md)

#### 旧实验结果

##### classifier = `adabrain_all_token`

| 状态 | 说明 |
|---|---|
| — | 暂无可核对的 `adabrain_all_token` 旧实验结果 |

**Test Acc 排序（从高到低）：** 暂无可排序结果。

**Test BAcc 排序（从高到低）：** 暂无可排序结果。

##### classifier = `mean_pool`

| 历史实验 | O/N/A | 输入/目标导联 | completion | classifier | 训练方式 | 有效 seeds | Val BAcc 均值 | Test Acc 均值 | Test BAcc 均值 | Test Kappa 均值 | Test F1 均值 |
|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|
| 35N | N | 23 | `none` | `mean_pool` | freeze CNN | 0/1/2 | 37.22% | 39.18% | 39.10% | 0.2381 | 39.40% |
| 35Ah | A | 23 → 62 | `seedv23_with_seedv62`，high pool | `mean_pool` | freeze CNN | 0/1/2 | 37.27% | 39.63% | 39.51% | 0.2438 | 40.03% |
| 35Al | A | 23 → 62 | `seedv23_with_seedv62`，low pool | `mean_pool` | freeze CNN | 0/1/2 | 36.89% | 39.46% | 39.44% | 0.2424 | 39.72% |
| 35O | O | 62 | `none` | `mean_pool` | freeze CNN | 0/1/2 | **39.51%** | 40.18% | 40.53% | 0.2537 | 40.45% |
| 35O | O | 62 | `none` | `mean_pool` | full finetune | 0/1/2 | 39.36% | **40.81%** | **41.08%** | **0.2610** | **41.28%** |

**Test Acc 排序（从高到低）：** 35O（full finetune）40.81% > 35O（freeze CNN）40.18% > 35Ah（freeze CNN）39.63% > 35Al（freeze CNN）39.46% > 35N（freeze CNN）39.18%。

**Test BAcc 排序（从高到低）：** 35O（full finetune）41.08% > 35O（freeze CNN）40.53% > 35Ah（freeze CNN）39.51% > 35Al（freeze CNN）39.44% > 35N（freeze CNN）39.10%。

#### 新实验结果

SEED-V 当前以 Val Accuracy 选择最佳 checkpoint，因此这一列不是 Val BAcc。

| 新实验 | 训练方式 | 最佳 epoch | Val 选择指标 | Test Acc | Test BAcc | Test Kappa | Test F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| A | freeze CNN | 33 | Acc 36.71% | 38.86% | 38.22% | 0.2289 | 38.75% |
| N | freeze CNN | 27 | Acc 36.30% | 38.93% | 38.10% | 0.2289 | 38.80% |
| N | full finetune | 26 | Acc 36.12% | 39.05% | 38.38% | 0.2318 | 39.05% |
| O | freeze CNN | 35 | Acc 39.48% | **42.25%** | **42.04%** | **0.2738** | **42.46%** |
| O | full finetune | 35 | Acc 38.71% | 41.94% | 41.51% | 0.2691 | 42.15% |

**Test Acc 排序（从高到低）：** O（freeze CNN）42.25% > O（full finetune）41.94% > N（full finetune）39.05% > N（freeze CNN）38.93% > A（freeze CNN）38.86%。

**Test BAcc 排序（从高到低）：** O（freeze CNN）42.04% > O（full finetune）41.51% > N（full finetune）38.38% > A（freeze CNN）38.22% > N（freeze CNN）38.10%。

运行命令：

```bash
bash scripts/seedv/O/freeze_cnn.sh
bash scripts/seedv/O/full_finetune.sh
bash scripts/seedv/N/freeze_cnn.sh
bash scripts/seedv/N/full_finetune.sh
bash scripts/seedv/A/freeze_cnn.sh
```

### 12.7 EEGMAT

来源：[scripts/eegmat/HISTORICAL_RESULTS.md](../scripts/eegmat/HISTORICAL_RESULTS.md)

#### 旧实验结果

##### classifier = `adabrain_all_token`

| 历史实验 | O/N/A | 输入/目标导联 | completion | classifier | 训练方式 | Test Acc | Test BAcc | Test Kappa | Test F1 |
|---|---|---|---|---|---|---:|---:|---:|---:|
| 37Oada，eegmat19 | O | 19 | `none` | `adabrain_all_token` | freeze CNN | **83.33%** | **83.33%** | 0.6667 | 83.29% |
| 37Oada，eegmat19 | O | 19 | `none` | `adabrain_all_token` | full finetune | 76.67% | 76.67% | 0.5333 | 76.66% |
| 37Nada，eegmat8，seed0 | N | 8 | `none` | `adabrain_all_token` | freeze CNN | 75.83% | 75.83% | 0.5167 | 75.70% |
| 37Nada，eegmat8，seed1 | N | 8 | `none` | `adabrain_all_token` | freeze CNN | 82.50% | 82.50% | 0.6500 | 82.50% |
| 37Nada，eegmat8，seed2 | N | 8 | `none` | `adabrain_all_token` | freeze CNN | 75.83% | 75.83% | 0.5167 | 75.70% |
| 37Aada，eegmat8→19，seed0 | A | 8 → 19 | `eegmat8_with_eegmat19` | `adabrain_all_token` | freeze CNN | 75.83% | 75.83% | 0.5167 | 75.70% |
| 37Aada，eegmat8→19，seed1 | A | 8 → 19 | `eegmat8_with_eegmat19` | `adabrain_all_token` | freeze CNN | 78.33% | 78.33% | 0.5667 | 78.33% |
| 37Aada，eegmat8→19，seed2 | A | 8 → 19 | `eegmat8_with_eegmat19` | `adabrain_all_token` | freeze CNN | 74.17% | 74.17% | 0.4833 | 74.04% |

**Test Acc 排序（从高到低）：** 37Oada，eegmat19（freeze CNN）83.33% > 37Nada，eegmat8，seed1（freeze CNN）82.50% > 37Aada，eegmat8→19，seed1（freeze CNN）78.33% > 37Oada，eegmat19（full finetune）76.67% > 37Nada，eegmat8，seed0（freeze CNN）75.83% = 37Nada，eegmat8，seed2（freeze CNN）75.83% = 37Aada，eegmat8→19，seed0（freeze CNN）75.83% > 37Aada，eegmat8→19，seed2（freeze CNN）74.17%。

**Test BAcc 排序（从高到低）：** 37Oada，eegmat19（freeze CNN）83.33% > 37Nada，eegmat8，seed1（freeze CNN）82.50% > 37Aada，eegmat8→19，seed1（freeze CNN）78.33% > 37Oada，eegmat19（full finetune）76.67% > 37Nada，eegmat8，seed0（freeze CNN）75.83% = 37Nada，eegmat8，seed2（freeze CNN）75.83% = 37Aada，eegmat8→19，seed0（freeze CNN）75.83% > 37Aada，eegmat8→19，seed2（freeze CNN）74.17%。

##### classifier = `mean_pool`

| 历史实验 | O/N/A | 输入/目标导联 | completion | classifier | 训练方式 | Test Acc | Test BAcc | Test Kappa | Test F1 |
|---|---|---|---|---|---|---:|---:|---:|---:|
| 37Omeanpool，eegmat19 | O | 19 | `none` | `mean_pool` | full finetune | 69.17% | 69.17% | 0.3833 | 69.06% |
| 37Omeanpool，eegmat19 | O | 19 | `none` | `mean_pool` | freeze CNN | 70.00% | 70.00% | 0.4000 | 69.46% |
| 37Nmeanpool，eegmat8 | N | 8 | `none` | `mean_pool` | freeze CNN | 60.00% | 60.00% | 0.2000 | 57.33% |
| 37Ameanpool，eegmat8→19 | A | 8 → 19 | `eegmat8_with_eegmat19` | `mean_pool` | freeze CNN | 70.00% | 70.00% | 0.4000 | 68.00% |

**Test Acc 排序（从高到低）：** 37Omeanpool，eegmat19（freeze CNN）70.00% = 37Ameanpool，eegmat8→19（freeze CNN）70.00% > 37Omeanpool，eegmat19（full finetune）69.17% > 37Nmeanpool，eegmat8（freeze CNN）60.00%。

**Test BAcc 排序（从高到低）：** 37Omeanpool，eegmat19（freeze CNN）70.00% = 37Ameanpool，eegmat8→19（freeze CNN）70.00% > 37Omeanpool，eegmat19（full finetune）69.17% > 37Nmeanpool，eegmat8（freeze CNN）60.00%。

#### 新实验结果

| 新实验 | 训练方式 | 最佳 epoch | Val 选择指标 | Test Acc | Test BAcc | Test Kappa | Test F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| A | freeze CNN | 5 | BAcc 71.88% | 74.17% | 74.17% | 0.4833 | 74.12% |
| N | freeze CNN | 8 | BAcc 76.56% | 75.00% | 75.00% | 0.5000 | 75.00% |
| N | full finetune | 11 | BAcc 75.00% | 73.33% | 73.33% | 0.4667 | 73.33% |
| O | freeze CNN | 19 | BAcc 79.69% | **83.33%** | **83.33%** | **0.6667** | **83.29%** |
| O | full finetune | 28 | BAcc 79.69% | 80.00% | 80.00% | 0.6000 | 80.00% |

**Test Acc 排序（从高到低）：** O（freeze CNN）83.33% > O（full finetune）80.00% > N（freeze CNN）75.00% > A（freeze CNN）74.17% > N（full finetune）73.33%。

**Test BAcc 排序（从高到低）：** O（freeze CNN）83.33% > O（full finetune）80.00% > N（freeze CNN）75.00% > A（freeze CNN）74.17% > N（full finetune）73.33%。

运行命令：

```bash
bash scripts/eegmat/O/freeze_cnn.sh
bash scripts/eegmat/O/full_finetune.sh
bash scripts/eegmat/N/freeze_cnn.sh
bash scripts/eegmat/N/full_finetune.sh
bash scripts/eegmat/A/freeze_cnn.sh
```

### 12.8 HGD

来源：[scripts/hgd/HISTORICAL_RESULTS.md](../scripts/hgd/HISTORICAL_RESULTS.md)

#### 旧实验结果

##### classifier = `adabrain_all_token`

| 状态 | 说明 |
|---|---|
| — | 暂无可核对的 `adabrain_all_token` 旧实验结果 |

**Test Acc 排序（从高到低）：** 暂无可排序结果。

**Test BAcc 排序（从高到低）：** 暂无可排序结果。

##### classifier = `mean_pool`

| 历史实验 | O/N/A | 输入/目标导联 | completion | classifier | 训练方式 | Test Acc | Test BAcc | Test Kappa | Test F1 |
|---|---|---|---|---|---|---:|---:|---:|---:|
| 39Omeanpool，hgd78 | O | 78 | `none` | `mean_pool` | freeze CNN | **74.35%** | **74.35%** | 0.6580 | 74.42% |
| 39Nmeanpool，hgd20 | N | 20 | `none` | `mean_pool` | freeze CNN | 71.49% | 71.49% | 0.6199 | 71.58% |
| 39Ameanpool，hgd20→78 | A | 20 → 78 | `hgd20_with_hgd78` | `mean_pool` | freeze CNN | 71.40% | 71.40% | 0.6187 | 71.72% |
| 39Ameanpool，smoke | A | 20 → 78 | `hgd20_with_hgd78` | `mean_pool` | freeze CNN，1 epoch smoke | 25.02% | 25.00% | 0.0000 | 10.02% |

**Test Acc 排序（从高到低）：** 39Omeanpool，hgd78（freeze CNN）74.35% > 39Nmeanpool，hgd20（freeze CNN）71.49% > 39Ameanpool，hgd20→78（freeze CNN）71.40% > 39Ameanpool，smoke（freeze CNN，1 epoch smoke）25.02%。

**Test BAcc 排序（从高到低）：** 39Omeanpool，hgd78（freeze CNN）74.35% > 39Nmeanpool，hgd20（freeze CNN）71.49% > 39Ameanpool，hgd20→78（freeze CNN）71.40% > 39Ameanpool，smoke（freeze CNN，1 epoch smoke）25.00%。

#### 新实验结果

| 新实验 | 训练方式 | 最佳 epoch | Val 选择指标 | Test Acc | Test BAcc | Test Kappa | Test F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| A | freeze CNN | 9 | BAcc 80.92% | 80.03% | 80.02% | 0.7337 | 80.07% |
| N | freeze CNN | 10 | BAcc 81.28% | 78.37% | 78.37% | 0.7116 | 78.37% |
| N | full finetune | 9 | BAcc 86.21% | 83.33% | 83.33% | 0.7778 | 83.39% |
| O | freeze CNN | 8 | BAcc 83.88% | 81.86% | 81.86% | 0.7581 | 81.92% |
| O | full finetune | 8 | BAcc 87.32% | **84.36%** | **84.36%** | **0.7915** | **84.37%** |

**Test Acc 排序（从高到低）：** O（full finetune）84.36% > N（full finetune）83.33% > O（freeze CNN）81.86% > A（freeze CNN）80.03% > N（freeze CNN）78.37%。

**Test BAcc 排序（从高到低）：** O（full finetune）84.36% > N（full finetune）83.33% > O（freeze CNN）81.86% > A（freeze CNN）80.02% > N（freeze CNN）78.37%。

运行命令：

```bash
bash scripts/hgd/O/freeze_cnn.sh
bash scripts/hgd/O/full_finetune.sh
bash scripts/hgd/N/freeze_cnn.sh
bash scripts/hgd/N/full_finetune.sh
bash scripts/hgd/A/freeze_cnn.sh
```

### 12.9 Siena

来源：[scripts/siena/HISTORICAL_RESULTS.md](../scripts/siena/HISTORICAL_RESULTS.md)

#### 旧实验结果

Siena 类别不均衡明显，Accuracy 与 BAcc 差距较大，应同时报告 BAcc、ROC-AUC
和 PR-AUC，不能只看 Accuracy。

##### classifier = `adabrain_all_token`

| 历史实验 | O/N/A | 输入/目标导联 | completion | classifier | 训练方式 | Test Acc | Test BAcc | Test ROC-AUC | Test PR-AUC |
|---|---|---|---|---|---|---:|---:|---:|---:|
| 40Oada，siena29 | O | 29 | `none` | `adabrain_all_token` | full finetune | **97.66%** | **79.28%** | 0.9091 | 0.4337 |
| 40Oada，siena29 | O | 29 | `none` | `adabrain_all_token` | freeze CNN | 97.53% | 51.64% | — | — |
| 40Aada，siena13→29 | A | 13 → 29 | `siena13_with_siena29` | `adabrain_all_token` | freeze CNN | 97.31% | 50.38% | — | — |
| 40Nada，siena13 | N | 13 | `none` | `adabrain_all_token` | freeze CNN | 97.01% | 49.65% | — | — |

**Test Acc 排序（从高到低）：** 40Oada，siena29（full finetune）97.66% > 40Oada，siena29（freeze CNN）97.53% > 40Aada，siena13→29（freeze CNN）97.31% > 40Nada，siena13（freeze CNN）97.01%。

**Test BAcc 排序（从高到低）：** 40Oada，siena29（full finetune）79.28% > 40Oada，siena29（freeze CNN）51.64% > 40Aada，siena13→29（freeze CNN）50.38% > 40Nada，siena13（freeze CNN）49.65%。

##### classifier = `mean_pool`

| 历史实验 | O/N/A | 输入/目标导联 | completion | classifier | 训练方式 | Test Acc | Test BAcc | Test ROC-AUC | Test PR-AUC |
|---|---|---|---|---|---|---:|---:|---:|---:|
| 40Omeanpool，siena29 | O | 29 | `none` | `mean_pool` | freeze CNN | 97.69% | 50.00% | — | — |

**Test Acc 排序（从高到低）：** 40Omeanpool，siena29（freeze CNN）97.69%。

**Test BAcc 排序（从高到低）：** 40Omeanpool，siena29（freeze CNN）50.00%。

#### 新实验结果

| 新实验 | 训练方式 | 最佳 epoch | Val 选择指标 | Test Acc | Test BAcc | Test ROC-AUC | Test PR-AUC |
|---|---|---:|---:|---:|---:|---:|---:|
| A | freeze CNN | 25 | BAcc 68.93% | 97.55% | 49.93% | 0.6419 | 0.0422 |
| N | freeze CNN | 4 | BAcc 66.36% | 97.42% | 53.31% | 0.7523 | 0.1096 |
| N | full finetune | 17 | BAcc 73.32% | 97.36% | 58.45% | 0.9024 | 0.2453 |
| O | freeze CNN | 2 | BAcc 64.91% | 97.93% | 57.59% | 0.8646 | 0.3212 |
| O | full finetune | 17 | BAcc 77.10% | 93.56% | **82.92%** | **0.9158** | **0.4691** |

**Test Acc 排序（从高到低）：** O（freeze CNN）97.93% > A（freeze CNN）97.55% > N（freeze CNN）97.42% > N（full finetune）97.36% > O（full finetune）93.56%。

**Test BAcc 排序（从高到低）：** O（full finetune）82.92% > N（full finetune）58.45% > O（freeze CNN）57.59% > N（freeze CNN）53.31% > A（freeze CNN）49.93%。

运行命令：

```bash
bash scripts/siena/O/freeze_cnn.sh
bash scripts/siena/O/full_finetune.sh
bash scripts/siena/N/freeze_cnn.sh
bash scripts/siena/N/full_finetune.sh
bash scripts/siena/A/freeze_cnn.sh
```

### 12.10 Attention

来源：[scripts/attention/HISTORICAL_RESULTS.md](../scripts/attention/HISTORICAL_RESULTS.md)

#### 旧实验结果

##### classifier = `adabrain_all_token`

| 状态 | 说明 |
|---|---|
| — | 暂无可核对的 `adabrain_all_token` 旧实验结果 |

**Test Acc 排序（从高到低）：** 暂无可排序结果。

**Test BAcc 排序（从高到低）：** 暂无可排序结果。

##### classifier = `mean_pool`

| 历史实验 | O/N/A | 输入/目标导联 | completion | classifier | 训练方式 | Test Acc | Test BAcc | Test Kappa | Test F1 |
|---|---|---|---|---|---|---:|---:|---:|---:|
| 44Omeanpool，attention26，seed0 | O | 26 | `none` | `mean_pool` | full finetune | 81.48% | **76.89%** | 0.4465 | 82.87% |
| 44Omeanpool，attention26，seed1 | O | 26 | `none` | `mean_pool` | full finetune | 81.85% | 75.78% | 0.4411 | 83.03% |
| 44Omeanpool，attention26，seed2 | O | 26 | `none` | `mean_pool` | full finetune | 81.11% | 69.11% | 0.3598 | 81.64% |
| 44Omeanpool，attention10，seed0 | O | 10 | `none` | `mean_pool` | full finetune | 81.85% | 69.11% | 0.3691 | 82.16% |
| 44Omeanpool，attention10，seed1 | O | 10 | `none` | `mean_pool` | full finetune | 84.63% | 69.44% | 0.4127 | 84.14% |
| 44Omeanpool，attention10，seed2 | O | 10 | `none` | `mean_pool` | full finetune | 85.00% | 74.56% | 0.4763 | 85.22% |
| 44Omeanpool，attention26 | O | 26 | `none` | `mean_pool` | freeze CNN | 74.81% | 68.44% | 0.2892 | 77.11% |
| 44Omeanpool，attention26，seed1 | O | 26 | `none` | `mean_pool` | freeze CNN | 73.33% | 65.33% | 0.2421 | 75.71% |
| 44Omeanpool，attention26，seed2 | O | 26 | `none` | `mean_pool` | freeze CNN | 73.33% | 63.56% | 0.2202 | 75.50% |
| 44Nmeanpool，attention10 | N | 10 | `none` | `mean_pool` | freeze CNN | 73.70% | 63.78% | 0.2255 | 75.79% |
| 44Nmeanpool，attention10，seed1 | N | 10 | `none` | `mean_pool` | freeze CNN | 79.26% | 68.89% | 0.3360 | 80.32% |
| 44Nmeanpool，attention10，seed2 | N | 10 | `none` | `mean_pool` | freeze CNN | 77.41% | 71.33% | 0.3441 | 79.29% |
| 44Ameanpool，attention10→26 | A | 10 → 26 | `attention10_with_attention26` | `mean_pool` | freeze CNN | 76.30% | 64.00% | 0.2471 | 77.58% |
| 44Ameanpool，attention10→26，seed1 | A | 10 → 26 | `attention10_with_attention26` | `mean_pool` | freeze CNN | 78.52% | 61.33% | 0.2267 | 78.52% |
| 44Ameanpool，attention10→26，seed2 | A | 10 → 26 | `attention10_with_attention26` | `mean_pool` | freeze CNN | 80.00% | 65.78% | 0.3047 | 80.34% |

**Test Acc 排序（从高到低）：** 44Omeanpool，attention10，seed2（full finetune）85.00% > 44Omeanpool，attention10，seed1（full finetune）84.63% > 44Omeanpool，attention26，seed1（full finetune）81.85% = 44Omeanpool，attention10，seed0（full finetune）81.85% > 44Omeanpool，attention26，seed0（full finetune）81.48% > 44Omeanpool，attention26，seed2（full finetune）81.11% > 44Ameanpool，attention10→26，seed2（freeze CNN）80.00% > 44Nmeanpool，attention10，seed1（freeze CNN）79.26% > 44Ameanpool，attention10→26，seed1（freeze CNN）78.52% > 44Nmeanpool，attention10，seed2（freeze CNN）77.41% > 44Ameanpool，attention10→26（freeze CNN）76.30% > 44Omeanpool，attention26（freeze CNN）74.81% > 44Nmeanpool，attention10（freeze CNN）73.70% > 44Omeanpool，attention26，seed1（freeze CNN）73.33% = 44Omeanpool，attention26，seed2（freeze CNN）73.33%。

**Test BAcc 排序（从高到低）：** 44Omeanpool，attention26，seed0（full finetune）76.89% > 44Omeanpool，attention26，seed1（full finetune）75.78% > 44Omeanpool，attention10，seed2（full finetune）74.56% > 44Nmeanpool，attention10，seed2（freeze CNN）71.33% > 44Omeanpool，attention10，seed1（full finetune）69.44% > 44Omeanpool，attention26，seed2（full finetune）69.11% = 44Omeanpool，attention10，seed0（full finetune）69.11% > 44Nmeanpool，attention10，seed1（freeze CNN）68.89% > 44Omeanpool，attention26（freeze CNN）68.44% > 44Ameanpool，attention10→26，seed2（freeze CNN）65.78% > 44Omeanpool，attention26，seed1（freeze CNN）65.33% > 44Ameanpool，attention10→26（freeze CNN）64.00% > 44Nmeanpool，attention10（freeze CNN）63.78% > 44Omeanpool，attention26，seed2（freeze CNN）63.56% > 44Ameanpool，attention10→26，seed1（freeze CNN）61.33%。

#### 新实验结果

| 新实验 | 训练方式 | 最佳 epoch | Val 选择指标 | Test Acc | Test BAcc | Test Kappa | Test F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| A | freeze CNN | 5 | BAcc 73.67% | 84.07% | 73.11% | 0.4464 | 84.34% |
| N | freeze CNN | 6 | BAcc 72.44% | 85.00% | 69.22% | 0.4159 | 84.36% |
| N | full finetune | 6 | BAcc 77.33% | **85.19%** | **77.78%** | **0.5102** | **85.76%** |
| O | freeze CNN | 31 | BAcc 74.22% | 84.63% | 66.78% | 0.3775 | 83.60% |
| O | full finetune | 3 | BAcc 78.22% | 83.52% | 72.33% | 0.4295 | 83.83% |

**Test Acc 排序（从高到低）：** N（full finetune）85.19% > N（freeze CNN）85.00% > O（freeze CNN）84.63% > A（freeze CNN）84.07% > O（full finetune）83.52%。

**Test BAcc 排序（从高到低）：** N（full finetune）77.78% > A（freeze CNN）73.11% > O（full finetune）72.33% > N（freeze CNN）69.22% > O（freeze CNN）66.78%。

运行命令：

```bash
bash scripts/attention/O/freeze_cnn.sh
bash scripts/attention/O/full_finetune.sh
bash scripts/attention/N/freeze_cnn.sh
bash scripts/attention/N/full_finetune.sh
bash scripts/attention/A/freeze_cnn.sh
```

### 12.11 AAD

来源：[scripts/aad/HISTORICAL_RESULTS.md](../scripts/aad/HISTORICAL_RESULTS.md)

#### 旧实验结果

##### classifier = `adabrain_all_token`

| 状态 | 说明 |
|---|---|
| — | 暂无可核对的 `adabrain_all_token` 旧实验结果 |

**Test Acc 排序（从高到低）：** 暂无可排序结果。

**Test BAcc 排序（从高到低）：** 暂无可排序结果。

##### classifier = `mean_pool`

| 历史实验 | O/N/A | 输入/目标导联 | completion | classifier | 训练方式 | Test Acc | Test BAcc | 备注 |
|---|---|---|---|---|---|---:|---:|---|
| 43Omeanpool | O | 84 | `none` | `mean_pool` | freeze CNN | — | — | 暂未找到可核对的结果日志 |

**Test Acc 排序（从高到低）：** 暂无可排序结果。

**Test BAcc 排序（从高到低）：** 暂无可排序结果。

#### 新实验结果

AAD 是二分类任务，除 Accuracy/BAcc 外同时列出 ROC-AUC 与 PR-AUC。

| 新实验 | 训练方式 | 最佳 epoch | Val 选择指标 | Test Acc | Test BAcc | Test ROC-AUC | Test PR-AUC |
|---|---|---:|---:|---:|---:|---:|---:|
| O | freeze CNN | 15 | BAcc 54.91% | 44.63% | 44.30% | 0.4748 | 0.4631 |

**Test Acc 排序（从高到低）：** O（freeze CNN）44.63%。

**Test BAcc 排序（从高到低）：** O（freeze CNN）44.30%。

运行命令：

```bash
bash scripts/aad/O/freeze_cnn.sh
```

### 12.12 FACED

来源：[scripts/faced/HISTORICAL_RESULTS.md](../scripts/faced/HISTORICAL_RESULTS.md)

#### 旧实验结果

##### classifier = `adabrain_all_token`

| 状态 | 说明 |
|---|---|
| — | 暂无可核对的 `adabrain_all_token` 旧实验结果 |

**Test Acc 排序（从高到低）：** 暂无可排序结果。

**Test BAcc 排序（从高到低）：** 暂无可排序结果。

##### classifier = `mean_pool`

| 历史实验 | O/N/A | 输入/目标导联 | completion | classifier | 训练方式 | Test Acc | Test BAcc | Test Kappa | Test F1 |
|---|---|---|---|---|---|---:|---:|---:|---:|
| 42Omeanpool，faced32，run1 | O | 32 | `none` | `mean_pool` | freeze CNN | 14.29% | 11.11% | 0.0000 | 3.57% |
| 42Omeanpool，faced32，run2 | O | 32 | `none` | `mean_pool` | freeze CNN | 18.12% | 17.12% | 0.0719 | 16.20% |

**Test Acc 排序（从高到低）：** 42Omeanpool，faced32，run2（freeze CNN）18.12% > 42Omeanpool，faced32，run1（freeze CNN）14.29%。

**Test BAcc 排序（从高到低）：** 42Omeanpool，faced32，run2（freeze CNN）17.12% > 42Omeanpool，faced32，run1（freeze CNN）11.11%。

##### 其他 classifier：`adabrain_mlp_token`

| 历史实验 | O/N/A | 输入/目标导联 | completion | classifier | 训练方式 | Test Acc | Test BAcc | Test Kappa | Test F1 |
|---|---|---|---|---|---|---:|---:|---:|---:|
| 42Oada，faced32 | O | 32 | `none` | `adabrain_mlp_token` | full finetune | **54.09%** | **54.19%** | 0.4820 | 54.39% |
| 42Oada，faced32 | O | 32 | `none` | `adabrain_mlp_token` | freeze CNN | 29.61% | 29.36% | 0.2056 | 29.39% |

**Test Acc 排序（从高到低）：** 42Oada，faced32（full finetune）54.09% > 42Oada，faced32（freeze CNN）29.61%。

**Test BAcc 排序（从高到低）：** 42Oada，faced32（full finetune）54.19% > 42Oada，faced32（freeze CNN）29.36%。

#### 新实验结果

| 新实验 | 训练方式 | 最佳 epoch | Val 选择指标 | Test Acc | Test BAcc | Test Kappa | Test F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| O | freeze CNN | 38 | BAcc 51.62% | 44.31% | 44.50% | 0.3726 | 44.32% |

**Test Acc 排序（从高到低）：** O（freeze CNN）44.31%。

**Test BAcc 排序（从高到低）：** O（freeze CNN）44.50%。

运行命令：

```bash
bash scripts/faced/O/freeze_cnn.sh
```

### 12.13 Zuo2025

来源：[scripts/zuo2025/HISTORICAL_RESULTS.md](../scripts/zuo2025/HISTORICAL_RESULTS.md)

#### 旧实验结果

##### classifier = `adabrain_all_token`

| 状态 | 说明 |
|---|---|
| — | 暂无可核对的 `adabrain_all_token` 旧实验结果 |

**Test Acc 排序（从高到低）：** 暂无可排序结果。

**Test BAcc 排序（从高到低）：** 暂无可排序结果。

##### classifier = `mean_pool`

| 历史实验 | O/N/A | 输入/目标导联 | completion | classifier | 训练方式 | Test Acc | Test BAcc | 备注 |
|---|---|---|---|---|---|---:|---:|---|
| 38Omeanpool | O | 30 | `none` | `mean_pool` | freeze CNN | — | — | 暂未找到可核对的结果日志 |

**Test Acc 排序（从高到低）：** 暂无可排序结果。

**Test BAcc 排序（从高到低）：** 暂无可排序结果。

#### 新实验结果

| 新实验 | 训练方式 | 最佳 epoch | Val 选择指标 | Test Acc | Test BAcc | Test Kappa | Test F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| O | freeze CNN | 1 | BAcc 73.10% | 74.38% | 74.37% | 0.4874 | 74.34% |

**Test Acc 排序（从高到低）：** O（freeze CNN）74.38%。

**Test BAcc 排序（从高到低）：** O（freeze CNN）74.37%。

运行命令：

```bash
bash scripts/zuo2025/O/freeze_cnn.sh
```
