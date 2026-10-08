# SHU 只重建两阶段对照

目的：保留当前模型的两个修正分支，关闭解耦对比损失和交换损失，比较“只重建”和“重建＋解耦”经过 Stage2 后的分类效果。

## 启动命令

在终端复制执行以下命令。每次启动会创建独立的时间戳目录，默认后台依次运行 Stage1 和 Stage2。

```bash
cd /inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/eeg-main/LaBraM-unified-AON-dynamic_general

SHU_RECON_RUN_DIR="$PWD/outputs/shu/recon_only/seed0_$(date +%Y%m%d_%H%M%S)_$$"

RUN_ROOT="$SHU_RECON_RUN_DIR" \
SEED=0 \
MISSING_WEIGHT=500 \
REG_WEIGHT=0 \
SUBJECT_SUMMARY_CONTRA_WEIGHT=0 \
TASK_SUMMARY_CONTRA_WEIGHT=0 \
SUBJECT_CORRECTION_CONTRA_WEIGHT=0 \
TASK_CORRECTION_CONTRA_WEIGHT=0 \
PERMUTE_SUB_WEIGHT=0 \
PERMUTE_TASK_WEIGHT=0 \
bash scripts/shu/D/stage1_then_stage2.sh
```

Stage1 成功结束后，串联脚本会将本次 `stage1/checkpoint-best.pth` 传给 Stage2。

命令假定终端没有遗留的 `FINETUNE`、`RESUME`、`STAGE1_OUTPUT_DIR`、`STAGE2_OUTPUT_DIR` 等环境覆盖。若之前手动 export 过实验参数，建议在新终端执行，并检查日志开头的最终命令。

## 保存位置与日志

```text
outputs/shu/recon_only/seed0_时间戳_进程号/
├── stage1/
│   └── checkpoint-best.pth
├── stage2/
│   └── checkpoint-best.pth
└── pipeline.log
```

在执行启动命令的同一个终端查看日志：

```bash
tail -f "$SHU_RECON_RUN_DIR/pipeline.log"
```

退出 `tail -f` 用 Ctrl+C；后台训练会继续运行。新终端查看时，将变量替换为实际运行目录。

## 对照时保持一致的设置

- 重建权重仍为 500；只关闭对比、交换和正则项。
- 模型结构、导联布局、数据划分、归一化、seed、学习率、batch、更新频率、训练轮数及 Stage2 设置，与对应的解耦实验保持一致。
- 以上命令沿用 SHU 脚本中的其余默认值。若原解耦实验覆盖过这些默认值，应给本次对照传入同样的覆盖值。
- 以预先确定的验证指标选择 Stage2 checkpoint，再比较对应测试结果；Stage1 重建误差和最终分类指标分别记录。

本文档只提供命令；创建文档不会启动训练。
