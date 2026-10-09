# SEED-V 三阶段动态补全

保留 SEED-V 数据路径、23→62 导联补全、prototype 及分类 accuracy 选模口径。

| 阶段 | 入口 | 默认轮数 | 训练目标 |
|---|---|---:|---|
| 1 | stage1.sh | 50 | missing_weight=0；保留现有对比、交换及正则配置 |
| 2 | stage1_2.sh | 50 | 加载阶段 1；仅 missing_weight=500，其余损失权重为 0 |
| 3 | stage2.sh | 30 | 加载阶段 2，冻结 CNN 和 corrector，训练分类器 |

阶段 1 沿用 BCI 模板的拆分方式：关闭直接重建，但交换损失仍涉及重建目标，并非完全无重建约束。

从项目根目录启动（默认后台运行）：

```bash
CUDA_VISIBLE_DEVICES=0 SEED=0 bash scripts/seedv/D_3stage/stage1_then_stage2.sh
```

只预览阶段 1 命令及后续交接路径，不启动训练：

```bash
DRY_RUN=1 bash scripts/seedv/D_3stage/stage1_then_stage2.sh
```

可通过 STAGE1_EPOCHS、STAGE2_EPOCHS、STAGE3_EPOCHS 覆盖轮数。RUN_BACKGROUND=0 为前台运行。
输出默认位于 outputs/seedv/D_pipeline_3stage/<run_id>/{stage1,stage2,stage3}。
指定已有 RUN_ROOT 重启时，会检查最后一轮日志、训练结束记录和最佳权重后跳过完成阶段；发现未完成结果则停止，不自动覆盖。相同 RUN_ROOT 使用 flock 防止重复启动。

手动运行阶段 2 需要 STAGE1_CHECKPOINT；阶段 3 需要 STAGE2_CHECKPOINT。两者均使用前一阶段的 checkpoint-best.pth。stage2.sh 为匹配 BCI 模板保留的文件名，实际执行第三阶段分类。
