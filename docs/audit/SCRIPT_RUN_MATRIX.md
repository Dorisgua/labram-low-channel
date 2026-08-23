# Bash 实验脚本第一阶段预检矩阵

审计日期：2026-08-23  
审计分支：`develop/aon-v1`  
审计范围：`scripts/` 下递归共 60 个 `.sh`；只做静态预检，未启动训练、未读取 checkpoint 权重内容、未分析 `outputs`。

## 判定口径

- 所有 60 个脚本均通过 `bash -n`，且均未发现 CRLF。
- Python 入口统一为仓库根目录的 `run_class_finetuning.py`；批量调度器通过子脚本间接进入该入口。
- 实验命令中的 Python 参数均已与 `get_args()` 的 `argparse` 定义核对；未发现未定义参数、重复参数或非法 `choices` 值。批量调度器中的 `--query-gpu`、`--format` 属于 `nvidia-smi`，不属于 Python 参数。
- 大 checkpoint `./checkpoints/labram-base.pth` 存在，大小 96,612,769 bytes；只检查文件元数据，未加载权重。
- `run_class_finetuning.py` 和默认 `torchrun` 均存在；输出目录全部位于 `./outputs` 下，本阶段未创建或改写任何输出。
- 静态检查通过但尚未实际启动的实验脚本统一标为 `NEEDS_SMOKE_TEST`；因此第一阶段 `READY=0`。
- `0.example.sh` 是只打印命令的模板，不实际进入 Python；5 个 `run_*.sh` 是多实验调度器，无法归入单一执行路径。

路径缩写：

- `$CKPT` = `./checkpoints/labram-base.pth`
- `$PROTO` = `docs/prototypes`
- `$OUT` = `./outputs`
- `$TUEV` = `/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/TUEZ/v2.0.1/processed_labram/processed`
- `$BCI` = `/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/eeg-test/AdaBrain-PreExp34-35-repro/AdaBrain-Bench-main_film/preprocessing/BCI-IV-2A/multi_subject_json`
- `$PHYSIONET` = `/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/physionet/physionet.org/files/eegmmidb/processed_eegfmbench/processed/fs_200/motor_mv_img/finetune/1.0.0`
- `$SEEDV` = `/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/SEED_V/SEED-V-labram`
- `$SEED` = `/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/SEED/processed_data`
- `$EEGMAT` = `/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/EEGMAT`
- `$ZUO` = `/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/Zuo2025/processed_data_4s_200hz`
- `$HGD` = `/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/HGD/processed_data_4s_200hz`
- `$SIENA` = `/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/Siene/processed_data_10s_200hz_adabrain_normstats`
- `$FACED` = `/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/FACED/processed_data_10s_200hz`
- `$AAD` = `/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/AAD/processed_data_4s_200hz`
- `$ATTN` = `/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/Attention/processed_data_4s_200hz`

## 逐脚本矩阵

| 脚本 | 数据集 | Python入口 | channel_subset | completion_scope | prototype | classifier_mode | 实际 AdaBrain 模式 / wrapper | freeze_cnn | checkpoint | data_path | output_dir | 预检状态 | 问题 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `0.example.sh` | TUEV | `run_class_finetuning.py`（仅打印） | tuev13 | tuev13_with_tuev23 | `$PROTO/01_tuev23_cnn_patch_embed_mean.pth` | mean_pool（默认） | 不使用 AdaBrain（mean_pool） | 是 | `$CKPT` | `$TUEV`（入口默认） | `$OUT/tuev/preexp12/checkpoints/${EXP_NAME}` | AMBIGUOUS | 模板明确声明不可原样运行；只打印命令；无 executable bit |
| `tuev/17Ah.finetune_tuev_labrambase_freeze_cnn_train_transformer_head_eval.sh` | TUEV | `run_class_finetuning.py` | tuev13 | tuev13_with_tuev23 | `$PROTO/01_tuev23_cnn_patch_embed_mean.pth` | mean_pool（默认） | 不使用 AdaBrain（mean_pool） | 是 | `$CKPT` | `$TUEV`（入口默认） | `$OUT/tuev/preexp17_tuev/checkpoints/${RUN_NAME}` | NEEDS_SMOKE_TEST | 无 executable bit，需使用 `bash scripts/...sh` |
| `tuev/17N.finetune_tuev_labrambase_freeze_cnn_train_transformer_head_eval.sh` | TUEV | `run_class_finetuning.py` | tuev13 | none | — | mean_pool（默认） | 不使用 AdaBrain（mean_pool） | 是 | `$CKPT` | `$TUEV`（入口默认） | `$OUT/tuev/preexp17_tuev/checkpoints/${RUN_NAME}` | NEEDS_SMOKE_TEST | 无 executable bit，需使用 `bash scripts/...sh` |
| `tuev/17O.finetune_tuev_labrambase_freeze_cnn_train_transformer_head_eval.sh` | TUEV | `run_class_finetuning.py` | tuev23 | none | — | mean_pool（默认） | 不使用 AdaBrain（mean_pool） | 是 | `$CKPT` | `$TUEV`（入口默认） | `$OUT/tuev/preexp17_tuev/checkpoints/${RUN_NAME}` | NEEDS_SMOKE_TEST | 无 executable bit，需使用 `bash scripts/...sh` |
| `bciiv2a/33Aada.finetune_bciiv2a_labrambase_freeze_cnn.sh` | bciiv2a | `run_class_finetuning.py` | bciiv2a13 | bciiv2a13_with_bciiv2a22 | `$PROTO/01_bciiv2a22_cnn_patch_embed_mean.pth` | adabrain_all_token | adabrain_all_token（`AdaBrainLaBraMWrapper`） | 是 | `$CKPT` | `$BCI` | `$OUT/bciiv2a/preexp33_bciiv2a_multisession/checkpoints/${RUN_NAME}` | NEEDS_SMOKE_TEST | — |
| `bciiv2a/33Aada.finetune_bciiv2a_labrambase_full_finetuen.sh` | bciiv2a | `run_class_finetuning.py` | bciiv2a13 | bciiv2a13_with_bciiv2a22 | `$PROTO/01_bciiv2a22_cnn_patch_embed_mean.pth` | adabrain_all_token | adabrain_all_token（`AdaBrainLaBraMWrapper`） | 否 | `$CKPT` | `$BCI` | `$OUT/bciiv2a/preexp33_bciiv2a_multisession/checkpoints/${RUN_NAME}` | NEEDS_SMOKE_TEST | — |
| `bciiv2a/33Arealada.finetune_bciiv2a_labrambase_freeze_cnn.sh` | bciiv2a | `run_class_finetuning.py` | bciiv2a13 | bciiv2a13_with_bciiv2a22 | `$PROTO/01_bciiv2a22_cnn_patch_embed_mean.pth` | adabrain_all_token（real tokens） | adabrain_all_token（`AdaBrainLaBraMWrapper`） | 是 | `$CKPT` | `$BCI` | `$OUT/bciiv2a/preexp33_bciiv2a_multisession/checkpoints/${RUN_NAME}` | NEEDS_SMOKE_TEST | — |
| `bciiv2a/33Nada.finetune_bciiv2a_labrambase_freeze_cnn.sh` | bciiv2a | `run_class_finetuning.py` | bciiv2a13 | none | — | adabrain_all_token | adabrain_all_token（`AdaBrainLaBraMWrapper`） | 是 | `$CKPT` | `$BCI` | `$OUT/bciiv2a/preexp33_bciiv2a_multisession/checkpoints/${RUN_NAME}` | NEEDS_SMOKE_TEST | — |
| `bciiv2a/33Nada.finetune_bciiv2a_labrambase_full_finetuen.sh` | bciiv2a | `run_class_finetuning.py` | bciiv2a13 | none | — | adabrain_all_token | adabrain_all_token（`AdaBrainLaBraMWrapper`） | 否 | `$CKPT` | `$BCI` | `$OUT/bciiv2a/preexp33_bciiv2a_multisession/checkpoints/${RUN_NAME}` | NEEDS_SMOKE_TEST | — |
| `bciiv2a/33Oada.finetune_bciiv2a_labrambase_freeze_cnn.sh` | bciiv2a | `run_class_finetuning.py` | bciiv2a22 | none | — | adabrain_all_token | adabrain_all_token（`AdaBrainLaBraMWrapper`） | 是 | `$CKPT` | `$BCI` | `$OUT/bciiv2a/preexp33_bciiv2a_multisession/checkpoints/${RUN_NAME}` | NEEDS_SMOKE_TEST | — |
| `bciiv2a/33Oada.finetune_bciiv2a_labrambase_full_finetuen.sh` | bciiv2a | `run_class_finetuning.py` | bciiv2a22 | none | — | adabrain_all_token | adabrain_all_token（`AdaBrainLaBraMWrapper`） | 否 | `$CKPT` | `$BCI` | `$OUT/bciiv2a/preexp33_bciiv2a_multisession/checkpoints/${RUN_NAME}` | NEEDS_SMOKE_TEST | — |
| `physionet/34Aeegfm.finetune_physionet23_with_physionet64_prototype_labrambase_freeze_cnn.sh` | physionet | `run_class_finetuning.py` | physionet23 | physionet23_with_physionet64 | `$PROTO/01_physionet64_cnn_patch_embed_mean.pth` | adabrain_all_token（real tokens） | adabrain_all_token（`AdaBrainLaBraMWrapper`） | 是 | `$CKPT` | `$PHYSIONET` | `$OUT/physionet/${EXP_GROUP}/checkpoints/${RUN_NAME}` | NEEDS_SMOKE_TEST | — |
| `physionet/34Aeegfm.finetune_physionet32_with_physionet64_prototype_labrambase_freeze_cnn.sh` | physionet | `run_class_finetuning.py` | physionet32 | physionet32_with_physionet64 | `$PROTO/01_physionet64_cnn_patch_embed_mean.pth` | adabrain_all_token（real tokens） | adabrain_all_token（`AdaBrainLaBraMWrapper`） | 是 | `$CKPT` | `$PHYSIONET` | `$OUT/physionet/${EXP_GROUP}/checkpoints/${RUN_NAME}` | NEEDS_SMOKE_TEST | — |
| `physionet/34Neegfm.finetune_physionet23_labrambase_freeze_cnn.sh` | physionet | `run_class_finetuning.py` | physionet23 | none | — | adabrain_all_token | adabrain_all_token（`AdaBrainLaBraMWrapper`） | 是 | `$CKPT` | `$PHYSIONET` | `$OUT/physionet/${EXP_GROUP}/checkpoints/${RUN_NAME}` | NEEDS_SMOKE_TEST | — |
| `physionet/34Neegfm.finetune_physionet32_labrambase_freeze_cnn.sh` | physionet | `run_class_finetuning.py` | physionet32 | none | — | adabrain_all_token | adabrain_all_token（`AdaBrainLaBraMWrapper`） | 是 | `$CKPT` | `$PHYSIONET` | `$OUT/physionet/${EXP_GROUP}/checkpoints/${RUN_NAME}` | NEEDS_SMOKE_TEST | — |
| `physionet/34Oeegfm.finetune_physionet_labrambase_freeze_cnn.sh` | physionet | `run_class_finetuning.py` | physionet64 | none | — | adabrain_all_token | adabrain_all_token（`AdaBrainLaBraMWrapper`） | 是 | `$CKPT` | `$PHYSIONET` | `$OUT/physionet/${EXP_GROUP}/checkpoints/${RUN_NAME}` | NEEDS_SMOKE_TEST | — |
| `physionet/34Oeegfm.finetune_physionet_labrambase_full_finetuen.sh` | physionet | `run_class_finetuning.py` | physionet64 | none | — | adabrain_all_token | adabrain_all_token（`AdaBrainLaBraMWrapper`） | 否 | `$CKPT` | `$PHYSIONET` | `$OUT/physionet/${EXP_GROUP}/checkpoints/${RUN_NAME}` | NEEDS_SMOKE_TEST | — |
| `seedv/35Ah.finetune_seedv23_with_seedv62_prototype_high_pool_labrambase_freeze_cnn.sh` | SEEDV | `run_class_finetuning.py` | seedv23 | seedv23_with_seedv62 | `$PROTO/01_seedv62_cnn_patch_embed_mean.pth`（缺失） | mean_pool | 不使用 AdaBrain（mean_pool） | 是 | `$CKPT` | `$SEEDV` | `$OUT/seedv/${EXP_GROUP}/checkpoints/${RUN_NAME}` | MISSING_PROTOTYPE | prototype 缺失；且入口 `DATASET_CONFIGS` 不支持 `SEEDV` |
| `seedv/35Al.finetune_seedv23_with_seedv62_prototype_low_pool_labrambase_freeze_cnn.sh` | SEEDV | `run_class_finetuning.py` | seedv23 | seedv23_with_seedv62 | `$PROTO/01_seedv62_cnn_patch_embed_mean.pth`（缺失） | mean_pool | 不使用 AdaBrain（mean_pool） | 是 | `$CKPT` | `$SEEDV` | `$OUT/seedv/${EXP_GROUP}/checkpoints/${RUN_NAME}` | MISSING_PROTOTYPE | prototype 缺失；且入口 `DATASET_CONFIGS` 不支持 `SEEDV` |
| `seedv/35N.finetune_seedv23_labrambase_freeze_cnn.sh` | SEEDV | `run_class_finetuning.py` | seedv23 | none | — | mean_pool | 不使用 AdaBrain（mean_pool） | 是 | `$CKPT` | `$SEEDV` | `$OUT/seedv/${EXP_GROUP}/checkpoints/${RUN_NAME}` | INVALID_ARGUMENT | `--dataset SEEDV` 会被 `get_dataset()` 拒绝 |
| `seedv/35O.finetune_seedv62_labrambase_freeze_cnn.sh` | SEEDV | `run_class_finetuning.py` | seedv62 | none | — | mean_pool | 不使用 AdaBrain（mean_pool） | 是 | `$CKPT` | `$SEEDV` | `$OUT/seedv/${EXP_GROUP}/checkpoints/${RUN_NAME}` | INVALID_ARGUMENT | `--dataset SEEDV` 会被 `get_dataset()` 拒绝 |
| `seedv/35O.finetune_seedv62_labrambase_full_finetune.sh` | SEEDV | `run_class_finetuning.py` | seedv62 | none | — | mean_pool | 不使用 AdaBrain（mean_pool） | 否 | `$CKPT` | `$SEEDV` | `$OUT/seedv/${EXP_GROUP}/checkpoints/${RUN_NAME}` | INVALID_ARGUMENT | `--dataset SEEDV` 会被 `get_dataset()` 拒绝 |
| `seed/36Ahmeanpool.finetune_seed23_with_seed62_prototype_labrambase_freeze_cnn.sh` | SEED | `run_class_finetuning.py` | seed23 | seed23_with_seed62 | `$PROTO/01_seed62_cnn_patch_embed_mean.pth` | mean_pool | 不使用 AdaBrain（mean_pool） | 是 | `$CKPT` | `$SEED` | `$OUT/seed/${EXP_GROUP}/checkpoints/${RUN_NAME}` | NEEDS_SMOKE_TEST | — |
| `seed/36Alada.finetune_seed23_with_seed62_prototype_labrambase_freeze_cnn.sh` | SEED | `run_class_finetuning.py` | seed23 | seed23_with_seed62 | `$PROTO/01_seed62_cnn_patch_embed_mean.pth` | adabrain_all_token（real tokens） | adabrain_all_token（`AdaBrainLaBraMWrapper`） | 是 | `$CKPT` | `$SEED` | `$OUT/seed/${EXP_GROUP}/checkpoints/${RUN_NAME}` | NEEDS_SMOKE_TEST | — |
| `seed/36Almeanpool.finetune_seed23_with_seed62_prototype_labrambase_freeze_cnn.sh` | SEED | `run_class_finetuning.py` | seed23 | seed23_with_seed62 | `$PROTO/01_seed62_cnn_patch_embed_mean.pth` | mean_pool | 不使用 AdaBrain（mean_pool） | 是 | `$CKPT` | `$SEED` | `$OUT/seed/${EXP_GROUP}/checkpoints/${RUN_NAME}` | NEEDS_SMOKE_TEST | — |
| `seed/36Nada.finetune_seed23_labrambase_freeze_cnn.sh` | SEED | `run_class_finetuning.py` | seed23 | none | — | adabrain_all_token | adabrain_all_token（`AdaBrainLaBraMWrapper`） | 是 | `$CKPT` | `$SEED` | `$OUT/seed/${EXP_GROUP}/checkpoints/${RUN_NAME}` | NEEDS_SMOKE_TEST | — |
| `seed/36Nmeanpool.finetune_seed23_labrambase_freeze_cnn.sh` | SEED | `run_class_finetuning.py` | seed23 | none | — | mean_pool | 不使用 AdaBrain（mean_pool） | 是 | `$CKPT` | `$SEED` | `$OUT/seed/${EXP_GROUP}/checkpoints/${RUN_NAME}` | NEEDS_SMOKE_TEST | — |
| `seed/36Oada.finetune_seed62_labrambase_freeze_cnn.sh` | SEED | `run_class_finetuning.py` | seed62 | none | — | adabrain_all_token | adabrain_all_token（`AdaBrainLaBraMWrapper`） | 是 | `$CKPT` | `$SEED` | `$OUT/seed/${EXP_GROUP}/checkpoints/${RUN_NAME}` | NEEDS_SMOKE_TEST | — |
| `seed/36Omeanpool.finetune_seed62_labrambase_freeze_cnn.sh` | SEED | `run_class_finetuning.py` | seed62 | none | — | mean_pool | 不使用 AdaBrain（mean_pool） | 是 | `$CKPT` | `$SEED` | `$OUT/seed/${EXP_GROUP}/checkpoints/${RUN_NAME}` | NEEDS_SMOKE_TEST | — |
| `eegmat/37Aada.finetune_eegmat8_with_eegmat19_prototype_labrambase_freeze_cnn.sh` | EEGMAT | `run_class_finetuning.py` | eegmat8 | eegmat8_with_eegmat19 | `$PROTO/01_eegmat19_cnn_patch_embed_mean.pth` | adabrain_all_token | adabrain_all_token（`AdaBrainLaBraMWrapper`） | 是 | `$CKPT` | `$EEGMAT` | `$OUT/eegmat/${EXP_GROUP}/checkpoints/${RUN_NAME}` | NEEDS_SMOKE_TEST | — |
| `eegmat/37Ameanpool.finetune_eegmat8_with_eegmat19_prototype_labrambase_freeze_cnn.sh` | EEGMAT | `run_class_finetuning.py` | eegmat8 | eegmat8_with_eegmat19 | `$PROTO/01_eegmat19_cnn_patch_embed_mean.pth` | mean_pool | 不使用 AdaBrain（mean_pool） | 是 | `$CKPT` | `$EEGMAT` | `$OUT/eegmat/${EXP_GROUP}/checkpoints/${RUN_NAME}` | NEEDS_SMOKE_TEST | — |
| `eegmat/37Nada.finetune_eegmat8_labrambase_freeze_cnn.sh` | EEGMAT | `run_class_finetuning.py` | eegmat8 | none | — | adabrain_all_token | adabrain_all_token（`AdaBrainLaBraMWrapper`） | 是 | `$CKPT` | `$EEGMAT` | `$OUT/eegmat/${EXP_GROUP}/checkpoints/${RUN_NAME}` | NEEDS_SMOKE_TEST | — |
| `eegmat/37Nmeanpool.finetune_eegmat8_labrambase_freeze_cnn.sh` | EEGMAT | `run_class_finetuning.py` | eegmat8 | none | — | mean_pool | 不使用 AdaBrain（mean_pool） | 是 | `$CKPT` | `$EEGMAT` | `$OUT/eegmat/${EXP_GROUP}/checkpoints/${RUN_NAME}` | NEEDS_SMOKE_TEST | — |
| `eegmat/37Oada.finetune_eegmat19_labrambase_freeze_cnn.sh` | EEGMAT | `run_class_finetuning.py` | eegmat19 | none | — | adabrain_all_token | adabrain_all_token（`AdaBrainLaBraMWrapper`） | 是 | `$CKPT` | `$EEGMAT` | `$OUT/eegmat/${EXP_GROUP}/checkpoints/${RUN_NAME}` | NEEDS_SMOKE_TEST | — |
| `eegmat/37Oada.finetune_eegmat19_labrambase_full_finetune.sh` | EEGMAT | `run_class_finetuning.py` | eegmat19 | none | — | adabrain_all_token | adabrain_all_token（`AdaBrainLaBraMWrapper`） | 否 | `$CKPT` | `$EEGMAT` | `$OUT/eegmat/${EXP_GROUP}/checkpoints/${RUN_NAME}` | NEEDS_SMOKE_TEST | — |
| `eegmat/37Omeanpool.finetune_eegmat19_labrambase_freeze_cnn.sh` | EEGMAT | `run_class_finetuning.py` | eegmat19 | none | — | mean_pool | 不使用 AdaBrain（mean_pool） | 是 | `$CKPT` | `$EEGMAT` | `$OUT/eegmat/${EXP_GROUP}/checkpoints/${RUN_NAME}` | NEEDS_SMOKE_TEST | — |
| `eegmat/37Omeanpool.finetune_eegmat19_labrambase_full_finetune.sh` | EEGMAT | `run_class_finetuning.py` | eegmat19 | none | — | mean_pool | 不使用 AdaBrain（mean_pool） | 否 | `$CKPT` | `$EEGMAT` | `$OUT/eegmat/${EXP_GROUP}/checkpoints/${RUN_NAME}` | NEEDS_SMOKE_TEST | — |
| `zuo2025/38Omeanpool.finetune_zuo2025_30_labrambase_freeze_cnn.sh` | Zuo2025 | `run_class_finetuning.py` | zuo2025_30 | none | — | mean_pool | 不使用 AdaBrain（mean_pool） | 是 | `$CKPT` | `$ZUO` | `$OUT/zuo2025/${EXP_GROUP}/checkpoints/${RUN_NAME}` | NEEDS_SMOKE_TEST | — |
| `hgd/39Ameanpool.finetune_hgd20_with_hgd78_prototype_labrambase_freeze_cnn.sh` | HGD | `run_class_finetuning.py` | hgd20 | hgd20_with_hgd78 | `$PROTO/01_hgd78_cnn_patch_embed_mean.pth` | mean_pool | 不使用 AdaBrain（mean_pool） | 是 | `$CKPT` | `$HGD` | `$OUT/hgd/${EXP_GROUP}/checkpoints/${RUN_NAME}` | NEEDS_SMOKE_TEST | — |
| `hgd/39Nmeanpool.finetune_hgd20_labrambase_freeze_cnn.sh` | HGD | `run_class_finetuning.py` | hgd20 | none | — | mean_pool | 不使用 AdaBrain（mean_pool） | 是 | `$CKPT` | `$HGD` | `$OUT/hgd/${EXP_GROUP}/checkpoints/${RUN_NAME}` | NEEDS_SMOKE_TEST | — |
| `hgd/39Omeanpool.finetune_hgd78_labrambase_freeze_cnn.sh` | HGD | `run_class_finetuning.py` | hgd78 | none | — | mean_pool | 不使用 AdaBrain（mean_pool） | 是 | `$CKPT` | `$HGD` | `$OUT/hgd/${EXP_GROUP}/checkpoints/${RUN_NAME}` | NEEDS_SMOKE_TEST | — |
| `siena/40Aada.finetune_siena13_with_siena29_prototype_labrambase_freeze_cnn.sh` | Siena | `run_class_finetuning.py` | siena13 | **none（实际）** | **未传入（实际）** | adabrain_all_token | adabrain_all_token（`AdaBrainLaBraMWrapper`） | 是 | `$CKPT` | `$SIENA` | `$OUT/siena/${EXP_GROUP}/checkpoints/${RUN_NAME}` | AMBIGUOUS | wrapper 导出的 `siena13_with_siena29` 和 prototype 未被基脚本 CMD 使用；实际无 completion |
| `siena/40Nada.finetune_siena13_labrambase_freeze_cnn.sh` | Siena | `run_class_finetuning.py` | siena13 | none | — | adabrain_all_token | adabrain_all_token（`AdaBrainLaBraMWrapper`） | 是 | `$CKPT` | `$SIENA` | `$OUT/siena/${EXP_GROUP}/checkpoints/${RUN_NAME}` | NEEDS_SMOKE_TEST | — |
| `siena/40Oada.finetune_siena29_labrambase_freeze_cnn.sh` | Siena | `run_class_finetuning.py` | siena29 | none | — | adabrain_all_token | adabrain_all_token（`AdaBrainLaBraMWrapper`） | 是 | `$CKPT` | `$SIENA` | `$OUT/siena/${EXP_GROUP}/checkpoints/${RUN_NAME}` | NEEDS_SMOKE_TEST | — |
| `siena/40Oada.finetune_siena29_labrambase_full_finetune.sh` | Siena | `run_class_finetuning.py` | siena29 | none | — | adabrain_all_token | adabrain_all_token（`AdaBrainLaBraMWrapper`） | 否 | `$CKPT` | `$SIENA` | `$OUT/siena/${EXP_GROUP}/checkpoints/${RUN_NAME}` | NEEDS_SMOKE_TEST | — |
| `siena/40Omeanpool.finetune_siena29_labrambase_freeze_cnn.sh` | Siena | `run_class_finetuning.py` | siena29 | none | — | mean_pool | 不使用 AdaBrain（mean_pool） | 是 | `$CKPT` | `$SIENA` | `$OUT/siena/${EXP_GROUP}/checkpoints/${RUN_NAME}` | NEEDS_SMOKE_TEST | — |
| `faced/42Oada.finetune_faced32_labrambase_mlp_freeze_cnn.sh` | FACED | `run_class_finetuning.py` | faced32 | none | — | adabrain_mlp_token | adabrain_mlp_token（`AdaBrainLaBraMMLPWrapper`） | 是 | `$CKPT` | `$FACED` | `$OUT/faced/${EXP_GROUP}/checkpoints/${RUN_NAME}` | NEEDS_SMOKE_TEST | — |
| `faced/42Oada.finetune_faced32_labrambase_mlp_full_finetune.sh` | FACED | `run_class_finetuning.py` | faced32 | none | — | adabrain_mlp_token | adabrain_mlp_token（`AdaBrainLaBraMMLPWrapper`） | 否 | `$CKPT` | `$FACED` | `$OUT/faced/${EXP_GROUP}/checkpoints/${RUN_NAME}` | NEEDS_SMOKE_TEST | — |
| `faced/42Omeanpool.finetune_faced32_labrambase_freeze_cnn.sh` | FACED | `run_class_finetuning.py` | faced32 | none | — | mean_pool | 不使用 AdaBrain（mean_pool） | 是 | `$CKPT` | `$FACED` | `$OUT/faced/${EXP_GROUP}/checkpoints/${RUN_NAME}` | NEEDS_SMOKE_TEST | — |
| `aad/43Omeanpool.finetune_aad84_labrambase_freeze_cnn.sh` | AAD | `run_class_finetuning.py` | aad84 | none | — | mean_pool | 不使用 AdaBrain（mean_pool） | 是 | `$CKPT` | `$AAD` | `$OUT/aad/${EXP_GROUP}/checkpoints/${RUN_NAME}` | NEEDS_SMOKE_TEST | — |
| `attention/44Ameanpool.finetune_attention10_with_attention26_prototype_labrambase_freeze_cnn.sh` | Attention | `run_class_finetuning.py` | attention10 | attention10_with_attention26 | `$PROTO/01_attention26_cnn_patch_embed_mean.pth` | mean_pool | 不使用 AdaBrain（mean_pool） | 是 | `$CKPT` | `$ATTN` | `$OUT/attention/${EXP_GROUP}/checkpoints/${RUN_NAME}` | NEEDS_SMOKE_TEST | — |
| `attention/44Nmeanpool.finetune_attention10_labrambase_freeze_cnn.sh` | Attention | `run_class_finetuning.py` | attention10 | none | — | mean_pool | 不使用 AdaBrain（mean_pool） | 是 | `$CKPT` | `$ATTN` | `$OUT/attention/${EXP_GROUP}/checkpoints/${RUN_NAME}` | NEEDS_SMOKE_TEST | — |
| `attention/44Nmeanpool.finetune_attention10_labrambase_full_finetune.sh` | Attention | `run_class_finetuning.py` | attention10 | none | — | mean_pool | 不使用 AdaBrain（mean_pool） | 否 | `$CKPT` | `$ATTN` | `$OUT/attention/${EXP_GROUP}/checkpoints/${RUN_NAME}` | NEEDS_SMOKE_TEST | — |
| `attention/44Omeanpool.finetune_attention26_labrambase_freeze_cnn.sh` | Attention | `run_class_finetuning.py` | attention26 | none | — | mean_pool | 不使用 AdaBrain（mean_pool） | 是 | `$CKPT` | `$ATTN` | `$OUT/attention/${EXP_GROUP}/checkpoints/${RUN_NAME}` | NEEDS_SMOKE_TEST | — |
| `attention/44Omeanpool.finetune_attention26_labrambase_full_finetune.sh` | Attention | `run_class_finetuning.py` | attention26 | none | — | mean_pool | 不使用 AdaBrain（mean_pool） | 否 | `$CKPT` | `$ATTN` | `$OUT/attention/${EXP_GROUP}/checkpoints/${RUN_NAME}` | NEEDS_SMOKE_TEST | — |
| `seed/run_36_experiments_seed012.sh` | SEED（多实验） | 子脚本间接进入 | mixed | mixed | `$PROTO/01_seed62_cnn_patch_embed_mean.pth`（部分任务） | mixed | 混合：mean_pool / adabrain_all_token | 是 | `$CKPT`（子脚本） | `$SEED`（子脚本） | 子脚本 output；批量日志 `$OUT/seed/batch_logs` | AMBIGUOUS | 批量调度 6 个实验，不能归入单一路径；不作为 1-batch 代表脚本 |
| `run_37_40_44_selected.sh` | EEGMAT/Siena/Attention | 子脚本间接进入 | mixed | mixed | mixed | mixed | 混合：mean_pool / adabrain_all_token | mixed | `$CKPT`（子脚本） | mixed | 子脚本 output；批量日志 `$OUT/selected_batch_logs` | AMBIGUOUS | 引用不存在的 `44Ameanpool...prototype...full_finetune.sh`；且包含有歧义的 40A |
| `run_37_42_selected_two_gpus.sh` | FACED/EEGMAT | 子脚本间接进入 | faced32/eegmat19 | none | — | adabrain_mlp_token / adabrain_all_token | 混合：adabrain_mlp_token / adabrain_all_token | mixed | `$CKPT`（子脚本） | `$FACED` / `$EEGMAT` | 子脚本 output；调度日志 `$OUT/launcher_logs` | AMBIGUOUS | 多路径异步调度器，不适合作为单一 1-batch smoke 入口 |
| `attention/run_44_attention_3seed.sh` | Attention（多实验） | 子脚本间接进入 | attention26/attention10 | none / completion | `$PROTO/01_attention26_cnn_patch_embed_mean.pth`（部分任务） | mean_pool | 不使用 AdaBrain（mean_pool） | 否 | `$CKPT`（子脚本） | `$ATTN` | 子脚本 output；批量日志 `$OUT/attention/batch_logs` | AMBIGUOUS | 引用不存在的 `44Ameanpool...prototype...full_finetune.sh` |
| `attention/run_44_attention_freeze_seed1_2.sh` | Attention（多实验） | 子脚本间接进入 | attention26/attention10 | none / completion | `$PROTO/01_attention26_cnn_patch_embed_mean.pth`（部分任务） | mean_pool | 不使用 AdaBrain（mean_pool） | 是 | `$CKPT`（子脚本） | `$ATTN` | 子脚本 output；批量日志 `$OUT/attention/batch_logs` | AMBIGUOUS | 多配置批量调度器，不作为单一 1-batch 代表脚本 |

## 汇总

| 状态 | 数量 |
|---|---:|
| READY | 0 |
| SYNTAX_ERROR | 0 |
| INVALID_ARGUMENT | 3 |
| MISSING_DATA | 0 |
| MISSING_CHECKPOINT | 0 |
| MISSING_PROTOTYPE | 2 |
| AMBIGUOUS | 7 |
| NEEDS_SMOKE_TEST | 48 |
| **总计** | **60** |

明确问题：

- PreExp35 的 5 个 SEED-V 脚本最终都传入 `--dataset SEEDV`，但当前入口的 `DATASET_CONFIGS` 不含 `SEEDV`；其中 `35Ah`、`35Al` 还缺少 `docs/prototypes/01_seedv62_cnn_patch_embed_mean.pth`。
- `40Aada.finetune_siena13_with_siena29_prototype_labrambase_freeze_cnn.sh` 的 completion/prototype 导出没有进入最终 CMD，实际执行 `completion_scope=none`。
- `run_37_40_44_selected.sh` 与 `run_44_attention_3seed.sh` 引用了不存在的 Attention prototype full-finetune 子脚本。
- `0.example.sh` 是打印模板；3 个 PreExp17 脚本无 executable bit，但可明确用 `bash` 调用。

## 最小执行路径分类

以下分类依据最终传给 Python 的参数，而不是脚本文件名或注释。

### LaBraM + mean_pool（20）

`17N`、`17O`、`35N`、`35O-freeze`、`35O-full`、`36Nmeanpool`、`36Omeanpool`、`37Nmeanpool`、`37Omeanpool-freeze`、`37Omeanpool-full`、`38Omeanpool`、`39Nmeanpool`、`39Omeanpool`、`40Omeanpool`、`42Omeanpool`、`43Omeanpool`、`44Nmeanpool-freeze`、`44Nmeanpool-full`、`44Omeanpool-freeze`、`44Omeanpool-full`。

其中 3 个 PreExp35 脚本当前为 `INVALID_ARGUMENT`，不进入 smoke test。

### LaBraM + prototype + mean_pool（8）

`17Ah`、`35Ah`、`35Al`、`36Ahmeanpool`、`36Almeanpool`、`37Ameanpool`、`39Ameanpool`、`44Ameanpool`。

其中 `35Ah`、`35Al` 当前缺 prototype 且数据集 key 不受支持，不进入 smoke test。

### LaBraM backbone + AdaBrain all-token wrapper（24）

全部 7 个 PreExp33；全部 6 个 PreExp34；`36Alada`、`36Nada`、`36Oada`；`37Aada`、`37Nada`、`37Oada-freeze`、`37Oada-full`；`40Aada`、`40Nada`、`40Oada-freeze`、`40Oada-full`。

`40Aada` 的实际路径是 AdaBrain all-token 但没有 prototype completion，与 wrapper 声明冲突，暂不进入 smoke test。

### LaBraM backbone + AdaBrain MLP-token wrapper（2）

`42Oada.finetune_faced32_labrambase_mlp_freeze_cnn.sh`、`42Oada.finetune_faced32_labrambase_mlp_full_finetune.sh`。

### 无法确认（6）

`0.example.sh` 以及 5 个 `run_*.sh` 批量调度器。批量调度器的子实验路径可以确认，但调度器自身不能归入一个单一路径。

## 下一阶段 smoke test 候选

共有 48 个 `NEEDS_SMOKE_TEST` 实验脚本：PreExp17 全部 3 个、PreExp33 全部 7 个、PreExp34 全部 6 个、PreExp36 全部 7 个、PreExp37 全部 8 个、PreExp38 的 1 个、PreExp39 全部 3 个、PreExp40 除 `40Aada` 外的 4 个、PreExp42 全部 3 个、PreExp43 的 1 个、PreExp44 全部 5 个。

建议第一轮每条路径各选一个代表脚本：

1. LaBraM + mean_pool：`scripts/seed/36Omeanpool.finetune_seed62_labrambase_freeze_cnn.sh`
2. LaBraM + prototype + mean_pool：`scripts/seed/36Almeanpool.finetune_seed23_with_seed62_prototype_labrambase_freeze_cnn.sh`
3. LaBraM backbone + AdaBrain all-token wrapper：`scripts/seed/36Alada.finetune_seed23_with_seed62_prototype_labrambase_freeze_cnn.sh`
4. LaBraM backbone + AdaBrain MLP-token wrapper：`scripts/faced/42Oada.finetune_faced32_labrambase_mlp_freeze_cnn.sh`

这 4 个代表脚本的入口、默认数据路径、checkpoint、所需 prototype 和 argparse 静态检查均已通过；本阶段未实际启动。
