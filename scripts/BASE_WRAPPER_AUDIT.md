# scripts 数据集级 base.sh / wrapper 只读审计

审计日期：2026-08-23

审计分支：`develop/aon-v1`

审计目标：设计“每个数据集一个 `base.sh`，其余普通实验均为 wrapper”的迁移方案。
审计方式：读取脚本实际 `CMD=(...)`、wrapper `export`/`exec bash` 链、`run_class_finetuning.py` 的 argparse 与 `DATASET_CONFIGS`；只执行 `bash -n` 和路径存在性检查，未启动训练。

## 1. 总体结论

| 项目 | 结果 |
|---|---:|
| 数据集目录 | 12 |
| 普通实验脚本 | 54 |
| 数据集内批量调度器 | 3 |
| 根目录跨数据集调度器 | 2 |
| 批量调度器总数 | 5 |
| 命令模板 | 1（`0.example.sh`，不计入普通实验） |
| 自行组装 `CMD=(...)` 的普通脚本 | 21 |
| 现有 wrapper | 33 |
| `bash -n` | 60/60 通过 |
| checkpoint | `checkpoints/labram-base.pth` 存在，只检查元数据 |
| 数据路径 | 12 个脚本实际使用的数据根均存在 |
| prototype | 除 SEED-V 外均存在 |
| 审计时 Git 状态 | clean |

| 分类 | 数据集 |
|---|---|
| 适合先直接统一 | `bciiv2a`（首选）、`aad`、`hgd`、`tuev`、`zuo2025` |
| base 可统一，但 wrapper 命名需容纳旧变体 | `eegmat`、`faced`、`physionet`、`seed` |
| 存在明确运行阻塞 | `attention`、`seedv`、`siena` |

新实验的推荐默认分类路径：

```bash
CLASSIFIER_MODE="${CLASSIFIER_MODE:-adabrain_all_token}"
CLASSIFIER_TOKEN_SCOPE="${CLASSIFIER_TOKEN_SCOPE:-all}"
```

旧脚本迁移时必须通过 wrapper 显式保留其真实 `mean_pool`、`adabrain_mlp_token` 或 `classifier_token_scope=real` 行为。

## 2. 目标目录与职责

```text
scripts/<dataset>/
├── base.sh
├── O/
│   ├── full_finetune.sh
│   └── freeze_cnn.sh
├── N/
│   ├── full_finetune.sh
│   └── freeze_cnn.sh
└── A/
    ├── prototype_fixed_full_finetune.sh
    └── prototype_fixed_freeze_cnn.sh
```

| 层级 | 应承担的职责 | 不应承担的职责 |
|---|---|---|
| `base.sh` | 定位仓库根目录；读取稳定参数和 wrapper 变量；验证路径；唯一组装 `CMD=(...)`；按布尔值加入 `--freeze_cnn`；追加 `"$@"` | 不硬编码 O/N/A 某一种实验；不使用 `eval` |
| O/N/A wrapper | 只设置实验变量、旧行为兼容变量和少量例外超参数，然后 `exec bash ../base.sh "$@"` | 不重复 Python 参数，不自行组装 CMD |
| `run_*.sh` | 只调度 wrapper、设置 seed/GPU/端口、维护 batch 日志 | 不直接维护 Python 参数，不绕过 wrapper |

O/N/A 定义：

| 组别 | 语义 |
|---|---|
| O | 全导联真实输入，不补全 |
| N | 少导联真实输入，不补全 |
| A | 少导联真实输入，通过 prototype 补到目标导联 |

## 3. 逐数据集脚本映射

### 3.1 AAD

| 当前脚本 | O/N/A | freeze/full | completion | classifier | 建议 wrapper 名称 | 无行为变化迁移 |
|---|---|---|---|---|---|---|
| `43Omeanpool.finetune_aad84_labrambase_freeze_cnn.sh` | O | freeze | none | mean_pool | `O/freeze_cnn.sh` | 是；显式保留 mean_pool |

缺少：O full、N、A。

### 3.2 Attention

| 当前脚本 | O/N/A | freeze/full | completion | classifier | 建议 wrapper 名称 | 无行为变化迁移 |
|---|---|---|---|---|---|---|
| `44Omeanpool.finetune_attention26_labrambase_freeze_cnn.sh` | O | freeze | none | mean_pool | `O/freeze_cnn.sh` | 是；显式保留 mean_pool |
| `44Omeanpool.finetune_attention26_labrambase_full_finetune.sh` | O | full | none | mean_pool | `O/full_finetune.sh` | 是；显式保留 mean_pool |
| `44Nmeanpool.finetune_attention10_labrambase_freeze_cnn.sh` | N | freeze | none | mean_pool | `N/freeze_cnn.sh` | 是；显式保留 mean_pool |
| `44Nmeanpool.finetune_attention10_labrambase_full_finetune.sh` | N | full | none | mean_pool | `N/full_finetune.sh` | 是；显式保留 mean_pool |
| `44Ameanpool.finetune_attention10_with_attention26_prototype_labrambase_freeze_cnn.sh` | A | freeze | `attention10_with_attention26`，low | mean_pool | `A/prototype_fixed_freeze_cnn.sh` | 是；显式保留 mean_pool |

缺少：`A/prototype_fixed_full_finetune.sh` 对应的旧脚本。

### 3.3 BCI-IV-2a

| 当前脚本 | O/N/A | freeze/full | completion | classifier | 建议 wrapper 名称 | 无行为变化迁移 |
|---|---|---|---|---|---|---|
| `33Oada.finetune_bciiv2a_labrambase_freeze_cnn.sh` | O | freeze | none | all-token/all | `O/freeze_cnn.sh` | 是 |
| `33Oada.finetune_bciiv2a_labrambase_full_finetuen.sh` | O | full | none | all-token/all | `O/full_finetune.sh` | 是 |
| `33Nada.finetune_bciiv2a_labrambase_freeze_cnn.sh` | N | freeze | none | all-token/all | `N/freeze_cnn.sh` | 是 |
| `33Nada.finetune_bciiv2a_labrambase_full_finetuen.sh` | N | full | none | all-token/all | `N/full_finetune.sh` | 是 |
| `33Aada.finetune_bciiv2a_labrambase_freeze_cnn.sh` | A | freeze | `bciiv2a13_with_bciiv2a22`，high | all-token/all | `A/prototype_fixed_freeze_cnn.sh` | 是 |
| `33Aada.finetune_bciiv2a_labrambase_full_finetuen.sh` | A | full | 同上 | all-token/all | `A/prototype_fixed_full_finetune.sh` | 是 |
| `33Arealada.finetune_bciiv2a_labrambase_freeze_cnn.sh` | A | freeze | 同上 | all-token/real | `A/prototype_fixed_real_token_freeze_cnn.sh` | 是；需额外 wrapper |

这是最适合第一个迁移的数据集。A full-finetune 已改为复用 A freeze CMD 的 wrapper；其余 5 个脚本仍重复维护 CMD。

Dynamic D 另使用两阶段 wrapper：`D/stage1.sh` 训练 13 → 22 通道 corrector，
`D/stage2.sh` 加载并冻结该 corrector 后执行分类；它们不是上表旧脚本的无行为变化迁移。

### 3.4 EEGMAT

| 当前脚本 | O/N/A | freeze/full | completion | classifier | 建议 wrapper 名称 | 无行为变化迁移 |
|---|---|---|---|---|---|---|
| `37Oada.finetune_eegmat19_labrambase_freeze_cnn.sh` | O | freeze | none | all-token/all | `O/freeze_cnn.sh` | 是 |
| `37Oada.finetune_eegmat19_labrambase_full_finetune.sh` | O | full | none | all-token/all | `O/full_finetune.sh` | 是 |
| `37Omeanpool.finetune_eegmat19_labrambase_freeze_cnn.sh` | O | freeze | none | mean_pool | `O/legacy_mean_pool_freeze_cnn.sh` | 是；需额外 wrapper |
| `37Omeanpool.finetune_eegmat19_labrambase_full_finetune.sh` | O | full | none | mean_pool | `O/legacy_mean_pool_full_finetune.sh` | 是；需额外 wrapper |
| `37Nada.finetune_eegmat8_labrambase_freeze_cnn.sh` | N | freeze | none | all-token/all | `N/freeze_cnn.sh` | 是 |
| `37Nmeanpool.finetune_eegmat8_labrambase_freeze_cnn.sh` | N | freeze | none | mean_pool | `N/legacy_mean_pool_freeze_cnn.sh` | 是；需额外 wrapper |
| `37Aada.finetune_eegmat8_with_eegmat19_prototype_labrambase_freeze_cnn.sh` | A | freeze | `eegmat8_with_eegmat19`，low | all-token/all | `A/prototype_fixed_freeze_cnn.sh` | 是 |
| `37Ameanpool.finetune_eegmat8_with_eegmat19_prototype_labrambase_freeze_cnn.sh` | A | freeze | 同上 | mean_pool | `A/legacy_mean_pool_prototype_fixed_freeze_cnn.sh` | 是；需额外 wrapper |

缺少：N full、A full。严格只有一个 O/N/A wrapper 会丢失 mean-pool 对照。

### 3.5 FACED

| 当前脚本 | O/N/A | freeze/full | completion | classifier | 建议 wrapper 名称 | 无行为变化迁移 |
|---|---|---|---|---|---|---|
| `42Omeanpool.finetune_faced32_labrambase_freeze_cnn.sh` | O | freeze | none | mean_pool | `O/legacy_mean_pool_freeze_cnn.sh` | 是 |
| `42Oada.finetune_faced32_labrambase_mlp_freeze_cnn.sh` | O | freeze | none | MLP-token/all | `O/legacy_mlp_token_freeze_cnn.sh` | 是 |
| `42Oada.finetune_faced32_labrambase_mlp_full_finetune.sh` | O | full | none | MLP-token/all | `O/legacy_mlp_token_full_finetune.sh` | 是 |

没有旧 all-token 实验；不能把 MLP 脚本改写成 all-token。

### 3.6 HGD

| 当前脚本 | O/N/A | freeze/full | completion | classifier | 建议 wrapper 名称 | 无行为变化迁移 |
|---|---|---|---|---|---|---|
| `39Omeanpool.finetune_hgd78_labrambase_freeze_cnn.sh` | O | freeze | none | mean_pool | `O/freeze_cnn.sh` | 是；显式保留 mean_pool |
| `39Nmeanpool.finetune_hgd20_labrambase_freeze_cnn.sh` | N | freeze | none | mean_pool | `N/freeze_cnn.sh` | 是；显式保留 mean_pool |
| `39Ameanpool.finetune_hgd20_with_hgd78_prototype_labrambase_freeze_cnn.sh` | A | freeze | `hgd20_with_hgd78`，low | mean_pool | `A/prototype_fixed_freeze_cnn.sh` | 是；显式保留 mean_pool |

缺少：全部 full-finetune wrapper。

### 3.7 PhysioNet

| 当前脚本 | O/N/A | freeze/full | completion | classifier | 建议 wrapper 名称 | 无行为变化迁移 |
|---|---|---|---|---|---|---|
| `34Oeegfm.finetune_physionet_labrambase_freeze_cnn.sh` | O | freeze | none | all-token/all | `O/freeze_cnn.sh` | 是 |
| `34Oeegfm.finetune_physionet_labrambase_full_finetuen.sh` | O | full | none | all-token/all | `O/full_finetune.sh` | 是 |
| `34Neegfm.finetune_physionet23_labrambase_freeze_cnn.sh` | N-23 | freeze | none | all-token/all | `N/physionet23_freeze_cnn.sh` | 是；需通道数后缀 |
| `34Neegfm.finetune_physionet32_labrambase_freeze_cnn.sh` | N-32 | freeze | none | all-token/all | `N/physionet32_freeze_cnn.sh` | 是；需通道数后缀 |
| `34Aeegfm.finetune_physionet23_with_physionet64_prototype_labrambase_freeze_cnn.sh` | A-23 | freeze | `physionet23_with_physionet64`，high | all-token/real | `A/physionet23_prototype_fixed_freeze_cnn.sh` | 是；需扩展名称 |
| `34Aeegfm.finetune_physionet32_with_physionet64_prototype_labrambase_freeze_cnn.sh` | A-32 | freeze | `physionet32_with_physionet64`，high | all-token/real | `A/physionet32_prototype_fixed_freeze_cnn.sh` | 是；需扩展名称 |

真实超参数差异：freeze 路径默认 `EPOCHS=30`，full 路径默认 `EPOCHS=50`。统一后 wrapper 必须保留。

### 3.8 SEED

| 当前脚本 | O/N/A | freeze/full | completion | classifier | 建议 wrapper 名称 | 无行为变化迁移 |
|---|---|---|---|---|---|---|
| `36Oada.finetune_seed62_labrambase_freeze_cnn.sh` | O | freeze | none | all-token/all | `O/freeze_cnn.sh` | 是 |
| `36Omeanpool.finetune_seed62_labrambase_freeze_cnn.sh` | O | freeze | none | mean_pool | `O/legacy_mean_pool_freeze_cnn.sh` | 是 |
| `36Nada.finetune_seed23_labrambase_freeze_cnn.sh` | N | freeze | none | all-token/all | `N/freeze_cnn.sh` | 是 |
| `36Nmeanpool.finetune_seed23_labrambase_freeze_cnn.sh` | N | freeze | none | mean_pool | `N/legacy_mean_pool_freeze_cnn.sh` | 是 |
| `36Ahmeanpool.finetune_seed23_with_seed62_prototype_labrambase_freeze_cnn.sh` | A | freeze | `seed23_with_seed62`，high | mean_pool | `A/legacy_mean_pool_high_freeze_cnn.sh` | 是 |
| `36Almeanpool.finetune_seed23_with_seed62_prototype_labrambase_freeze_cnn.sh` | A | freeze | 同上，low | mean_pool | `A/legacy_mean_pool_low_freeze_cnn.sh` | 是 |
| `36Alada.finetune_seed23_with_seed62_prototype_labrambase_freeze_cnn.sh` | A | freeze | 同上，low | all-token/real | `A/prototype_fixed_real_token_freeze_cnn.sh` | 是 |

没有 A all-token/all，也没有 full-finetune 实验。

### 3.9 SEED-V

| 当前脚本 | O/N/A | freeze/full | completion | classifier | 建议 wrapper 名称 | 无行为变化迁移 |
|---|---|---|---|---|---|---|
| `35O.finetune_seedv62_labrambase_freeze_cnn.sh` | O | freeze | none | mean_pool | `O/freeze_cnn.sh` | 静态可迁移；当前不可运行 |
| `35O.finetune_seedv62_labrambase_full_finetune.sh` | O | full | none | mean_pool | `O/full_finetune.sh` | 静态可迁移；当前不可运行 |
| `35N.finetune_seedv23_labrambase_freeze_cnn.sh` | N | freeze | none | mean_pool | `N/freeze_cnn.sh` | 静态可迁移；当前不可运行 |
| `35Ah.finetune_seedv23_with_seedv62_prototype_high_pool_labrambase_freeze_cnn.sh` | A | freeze | `seedv23_with_seedv62`，high | mean_pool | `A/prototype_fixed_high_pool_freeze_cnn.sh` | 静态可迁移；资源阻塞 |
| `35Al.finetune_seedv23_with_seedv62_prototype_low_pool_labrambase_freeze_cnn.sh` | A | freeze | 同上，low | mean_pool | `A/prototype_fixed_freeze_cnn.sh` | 静态可迁移；资源阻塞 |

阻塞：`DATASET_CONFIGS` 无 `SEEDV`；`docs/prototypes/01_seedv62_cnn_patch_embed_mean.pth` 不存在。

### 3.10 Siena

| 当前脚本 | O/N/A | freeze/full | completion | classifier | 建议 wrapper 名称 | 无行为变化迁移 |
|---|---|---|---|---|---|---|
| `40Oada.finetune_siena29_labrambase_full_finetune.sh` | O | full | none | all-token/all | `O/full_finetune.sh` | 是 |
| `40Oada.finetune_siena29_labrambase_freeze_cnn.sh` | O | freeze | none | all-token/all | `O/freeze_cnn.sh` | 是 |
| `40Omeanpool.finetune_siena29_labrambase_freeze_cnn.sh` | O | freeze | none | mean_pool | `O/legacy_mean_pool_freeze_cnn.sh` | 是 |
| `40Nada.finetune_siena13_labrambase_freeze_cnn.sh` | N | freeze | none | all-token/all | `N/freeze_cnn.sh` | 是 |
| `40Aada.finetune_siena13_with_siena29_prototype_labrambase_freeze_cnn.sh` | **N（文件名为 A）** | freeze | **none（实际）** | all-token/all | `N/legacy_40A_no_completion_freeze_cnn.sh` | 只能保留当前错误行为；不能直接作为 A |

40A wrapper 导出的 `COMPLETION_SCOPE` 和 `CHANNEL_PROTOTYPE_PATH` 没有被 base 读取；最终 CMD 写死 `--completion_scope none --pooling_scope low`，且没有 prototype 参数或检查。

### 3.11 TUEV

| 当前脚本 | O/N/A | freeze/full | completion | classifier | 建议 wrapper 名称 | 无行为变化迁移 |
|---|---|---|---|---|---|---|
| `17O.finetune_tuev_labrambase_freeze_cnn_train_transformer_head_eval.sh` | O | freeze | none | mean_pool（argparse 默认） | `O/freeze_cnn.sh` | 是；显式保留 mean_pool |
| `17N.finetune_tuev_labrambase_freeze_cnn_train_transformer_head_eval.sh` | N | freeze | none | mean_pool（argparse 默认） | `N/freeze_cnn.sh` | 是；显式保留 mean_pool |
| `17Ah.finetune_tuev_labrambase_freeze_cnn_train_transformer_head_eval.sh` | A | freeze | `tuev13_with_tuev23`，high | mean_pool（argparse 默认） | `A/prototype_fixed_freeze_cnn.sh` | 是；显式保留 mean_pool |

三个脚本都没有显式传入 `--classifier_mode`，实际取 argparse 默认 `mean_pool`。三个文件没有 executable bit，但 `bash script.sh` 可调用。

### 3.12 Zuo2025

| 当前脚本 | O/N/A | freeze/full | completion | classifier | 建议 wrapper 名称 | 无行为变化迁移 |
|---|---|---|---|---|---|---|
| `38Omeanpool.finetune_zuo2025_30_labrambase_freeze_cnn.sh` | O | freeze | none | mean_pool | `O/freeze_cnn.sh` | 是；显式保留 mean_pool |

当前真实默认 `EPOCHS=2`，迁移时不能自动改成 50。

## 4. 每个数据集建议的 base 变量

路径缩写：`$U=/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe`。

### 4.1 稳定参数

| 数据集目录 | `DATASET` | `DATA_PATH` 默认值 | batch / epochs / lr / wd / workers | checkpoint | output 数据集根 | 来源证据 |
|---|---|---|---|---|---|---|
| aad | `AAD` | `$U/AAD/processed_data_4s_200hz` | 64 / 50 / 5e-4 / 0.05 / 4 | `./checkpoints/labram-base.pth` | `./outputs/aad` | `43O`: 17-65 |
| attention | `Attention` | `$U/Attention/processed_data_4s_200hz` | 64 / 50 / 5e-4 / 0.05 / 4 | 同上 | `./outputs/attention` | `44O`: 18-71 |
| bciiv2a | `bciiv2a` | `$U/eeg-test/AdaBrain-PreExp34-35-repro/AdaBrain-Bench-main_film/preprocessing/BCI-IV-2A/multi_subject_json` | 64 / 50 / 5e-4 / 0.05 / 4 | 同上 | `./outputs/bciiv2a` | `33O`: 16-76 |
| eegmat | `EEGMAT` | `$U/EEGMAT` | 64 / 50 / 5e-4 / 0.05 / 4 | 同上 | `./outputs/eegmat` | `37Oada`: 17-74 |
| faced | `FACED` | `$U/FACED/processed_data_10s_200hz` | 64 / 50 / 5e-4 / 0.05 / 4 | 同上 | `./outputs/faced` | `42O`: 17-68 |
| hgd | `HGD` | `$U/HGD/processed_data_4s_200hz` | 64 / 50 / 5e-4 / 0.05 / 4 | 同上 | `./outputs/hgd` | `39O`: 28-113 |
| physionet | `physionet` | `$U/physionet/physionet.org/files/eegmmidb/processed_eegfmbench/processed/fs_200/motor_mv_img/finetune/1.0.0` | 64 / **30 freeze、50 full** / 5e-4 / 0.05 / 4 | 同上 | `./outputs/physionet` | `34O-freeze`: 16-84；`34O-full`: 16-84 |
| seed | `SEED` | `$U/SEED/processed_data` | 64 / 50 / 5e-4 / 0.05 / 4 | 同上 | `./outputs/seed` | `36Oada`: 17-81 |
| seedv | `SEEDV`（当前非法） | `$U/SEED_V/SEED-V-labram` | 64 / 50 / 5e-4 / 0.05 / 4 | 同上 | `./outputs/seedv` | `35O`: 16-82 |
| siena | `Siena` | `$U/Siene/processed_data_10s_200hz_adabrain_normstats` | **32** / 50 / 5e-4 / 0.05 / 4 | 同上 | `./outputs/siena` | `40Oada`: 17-66 |
| tuev | `TUEV` | `$U/TUEZ/v2.0.1/processed_labram/processed` | 64 / 50 / 5e-4 / 0.05 / 4 | 同上 | `./outputs/tuev` | `17O`: 28-169；data_path 当前来自配置默认 |
| zuo2025 | `Zuo2025` | `$U/Zuo2025/processed_data_4s_200hz` | 64 / **2** / 5e-4 / 0.05 / 4 | 同上 | `./outputs/zuo2025` | `38O`: 17-62 |

### 4.2 变量归属

| 变量 | base 默认/规则 | wrapper 是否覆盖 | 说明 |
|---|---|---|---|
| `PYTHON_ENTRY` | `run_class_finetuning.py` | 否 | 所有数据集统一入口 |
| `DATASET` | 每个数据集 base 固定 | 否 | 不允许 wrapper 改到另一数据集 |
| `DATA_PATH` | 上表默认 | 通常否 | 必须允许外部环境覆盖 |
| `CHECKPOINT` | `./checkpoints/labram-base.pth` | 通常否 | 传给 `--finetune`，允许环境覆盖 |
| `BATCH_SIZE` | 64；Siena 32 | 少数情况 | 环境可覆盖 |
| `EPOCHS` | 见上表 | PhysioNet freeze 等旧 wrapper 覆盖 | 环境可覆盖 |
| `LR` | 5e-4 | FACED MLP wrapper 覆盖为 1e-4 | 环境可覆盖 |
| `WEIGHT_DECAY` | 0.05 | 通常否 | 环境可覆盖 |
| `NUM_WORKERS` | 4 | 通常否 | 环境可覆盖 |
| `SEED` | 0 | batch 或环境覆盖 | 输出目录必须显式包含 seed |
| `CUDA_VISIBLE_DEVICES` | `${GPU_IDS:-0}` | batch 或环境覆盖 | base 显式读取 |
| `OUTPUT_ROOT` | `./outputs` | 环境可覆盖 | base 追加 dataset/O-N-A/train-mode/seed |
| `EXPERIMENT_GROUP` | 必填：O/N/A | 是 | wrapper 必须设置 |
| `RUN_LABEL` | 必填 | 是 | 用于区分 classifier、pooling、通道变体 |
| `CHANNEL_SUBSET` | 必填 | 是 | O/N/A 的核心差异 |
| `COMPLETION_SCOPE` | `none` | A wrapper 覆盖 | 非 none 时 prototype 必须存在 |
| `PROTOTYPE_PATH` | 空 | A wrapper 覆盖 | 传给 `--channel_prototype_path` |
| `POOLING_SCOPE` | `low` | high-pool wrapper 覆盖 | 旧行为必须保留 |
| `FREEZE_CNN` | 1 | full wrapper 设 0 | base 根据布尔值决定是否加入 flag |
| `CLASSIFIER_MODE` | `adabrain_all_token` | 旧 mean/MLP wrapper 覆盖 | 不得根据文件名推断 |
| `CLASSIFIER_TOKEN_SCOPE` | `all` | real-token wrapper 覆盖 | 即使 mean_pool 不使用也建议显式传入 |

## 5. base.sh 设计要点

| 要求 | 建议实现 |
|---|---|
| 严格 Bash | `set -euo pipefail` |
| 定位 repo | base 位于 `scripts/<dataset>/base.sh`，使用 `${SCRIPT_DIR}/../..` |
| wrapper 定位 base | wrapper 位于 O/N/A 子目录，`${SCRIPT_DIR}/../base.sh` 层级正确 |
| 唯一 CMD | 只有 base 包含 `CMD=(...)` |
| 禁止 eval | 最终使用 `exec "${CMD[@]}"` |
| prototype 检查 | `COMPLETION_SCOPE != none` 时要求非空且 `-f "$PROTOTYPE_PATH"` |
| freeze | 验证 `FREEZE_CNN` 只能为 0/1；1 时 `CMD+=(--freeze_cnn)` |
| 额外参数 | wrapper `exec bash ... "$@"`；base 在执行前 `CMD+=("$@")` |
| 输出隔离 | `outputs/<dataset>/<O-N-A>/<freeze-full>/seed<seed>/<run_label>_<timestamp>` |
| batch | 只调用 wrapper；不重复 Python 参数 |

推荐输出计算：

```bash
case "${FREEZE_CNN}" in
    0) TRAIN_MODE="full_finetune" ;;
    1) TRAIN_MODE="freeze_cnn" ;;
    *) echo "FREEZE_CNN must be 0 or 1" >&2; exit 2 ;;
esac

RUN_ID="$(date +%Y%m%d_%H%M%S)"
DATASET_OUTPUT_ROOT="${OUTPUT_ROOT%/}/${DATASET_SLUG}"
OUTPUT_DIR="${DATASET_OUTPUT_ROOT}/${EXPERIMENT_GROUP}/${TRAIN_MODE}/seed${SEED}/${RUN_LABEL}_${RUN_ID}"
```

推荐 wrapper 骨架：

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export EXPERIMENT_GROUP="A"
export RUN_LABEL="prototype_fixed_all_token"
export CHANNEL_SUBSET="..."
export COMPLETION_SCOPE="..."
export PROTOTYPE_PATH="..."
export POOLING_SCOPE="low"
export FREEZE_CNN="1"
export CLASSIFIER_MODE="adabrain_all_token"
export CLASSIFIER_TOKEN_SCOPE="all"

exec bash "${SCRIPT_DIR}/../base.sh" "$@"
```

## 6. 当前必须先修复的问题

| 优先级 | 问题 | 证据 | 影响 |
|---:|---|---|---|
| 1 | Siena 40A 的 completion/prototype 未进入最终 CMD | wrapper 导出变量；被调 base 写死 `--completion_scope none --pooling_scope low` | 40A 实际是 N，不是 A |
| 1 | `SEEDV` 不在 `DATASET_CONFIGS` | `run_class_finetuning.py` 配置表无此 key | 5 个 SEED-V 脚本都会被入口拒绝 |
| 1 | SEED-V prototype 缺失 | `docs/prototypes/01_seedv62_cnn_patch_embed_mean.pth` 不存在 | 两个 A 脚本无法通过路径检查 |
| 1 | Attention A full-finetune 子脚本不存在 | 两个 batch 引用 `44Ameanpool...full_finetune.sh` | 两个 batch 在启动前失败 |
| 2 | 现有 `"$@"` 未真正进入 CMD | 18 个 wrapper 转发、14 个不转发；所有旧 CMD 都未追加位置参数 | 用户额外参数被忽略 |
| 2 | 目标树无法容纳所有旧行为 | mean/all/MLP/real/high/low/23/32 存在同槽位冲突 | 需要额外 legacy/variant wrapper |
| 2 | TUEV 多项值硬编码 | batch、epochs、lr、wd、seed、checkpoint | 不满足环境变量覆盖要求 |
| 3 | TUEV 3 个脚本无 executable bit | 文件权限检查 | 必须用 `bash` 调用或迁移时统一权限 |

其余静态资源结果：

| 项目 | 结果 |
|---|---|
| `docs/prototypes/01_attention26_cnn_patch_embed_mean.pth` | 存在 |
| `docs/prototypes/01_bciiv2a22_cnn_patch_embed_mean.pth` | 存在 |
| `docs/prototypes/01_eegmat19_cnn_patch_embed_mean.pth` | 存在 |
| `docs/prototypes/01_hgd78_cnn_patch_embed_mean.pth` | 存在 |
| `docs/prototypes/01_physionet64_cnn_patch_embed_mean.pth` | 存在 |
| `docs/prototypes/01_seed62_cnn_patch_embed_mean.pth` | 存在 |
| `docs/prototypes/01_siena29_cnn_patch_embed_mean.pth` | 存在 |
| `docs/prototypes/01_tuev23_cnn_patch_embed_mean.pth` | 存在 |
| `docs/prototypes/01_seedv62_cnn_patch_embed_mean.pth` | **缺失** |
| `${SCRIPT_DIR}` 相对脚本引用 | 全部存在 |
| 实际 `source` 命令 | 无 |
| 数据集目录内 `REPO_DIR=.../../..` | 正确 |
| 根目录 batch 的 `REPO_DIR=.../..` | 正确 |

## 7. 批量调度器审计

| 调度器 | 当前路径是否已更新 | 当前问题 | 迁移后建议 |
|---|---|---|---|
| `seed/run_36_experiments_seed012.sh` | 是 | 调度 6 种 mean/all/high/low 旧行为 | 调用标准及 legacy wrapper |
| `attention/run_44_attention_freeze_seed1_2.sh` | 是 | 无缺失子脚本 | 调用 O/N/A freeze wrapper |
| `attention/run_44_attention_3seed.sh` | 部分 | A full 子脚本不存在 | 先补齐或移除 A full 任务 |
| `run_37_40_44_selected.sh` | 部分 | 同样引用缺失的 Attention A full；还包含当前错误的 Siena 40A | 修复后改调 wrapper |
| `run_37_42_selected_two_gpus.sh` | 是 | 当前子脚本均存在 | 改调 FACED/EEGMAT wrapper |

## 8. 建议迁移顺序

| 顺序 | 数据集 | 原因/前置条件 |
|---:|---|---|
| 1 | BCI-IV-2a | O/N/A、freeze/full、prototype、all-token 最完整；作为 base 设计试点 |
| 2 | HGD、TUEV、AAD、Zuo2025 | 结构简单；验证旧 mean-pool wrapper 兼容 |
| 3 | Attention | 先处理缺失的 A full-finetune |
| 4 | PhysioNet | 先确定 23/32 wrapper 命名 |
| 5 | EEGMAT、SEED、FACED | 验证 mean/all/MLP/real/high/low 多变体兼容 |
| 6 | Siena | 先单独决定是否修复 40A 的实际行为 |
| 7 | SEED-V | 先增加合法 dataset 配置并生成 prototype |

## 9. 迁移验收标准

| 检查 | 通过条件 |
|---|---|
| Bash 语法 | 所有 base、wrapper、batch 均通过 `bash -n` |
| CMD 唯一性 | 每个数据集只有 `base.sh` 包含 Python CMD |
| wrapper 纯度 | wrapper 只有变量设置和 `exec bash ... "$@"` |
| 变量穿透 | wrapper 设置的 subset/completion/prototype/freeze/classifier/scope/pooling 全部出现在展开后的 CMD |
| prototype | completion 非 none 时必须存在且传入 Python |
| freeze/full | 仅 freeze wrapper 的 CMD 包含 `--freeze_cnn` |
| 旧行为 | mean_pool、MLP、real-token、high/low 旧实验静态展开结果不变 |
| 输出隔离 | 路径包含 dataset、O/N/A、freeze/full、seed 和 run label |
| batch | 只调用 wrapper；所有引用存在 |
| smoke test | 静态迁移审核通过后再做 1-batch，不在结构迁移阶段启动完整训练 |
