# 五个数据集串行运行 recon-only

默认顺序：SHU → SEEDV → TUEV → SEED → ERP-Core。每个数据集完成 Stage1 和 Stage2 后才进入下一个；任一阶段失败，总队列停止。

## 启动

建议在没有遗留实验参数的新终端中运行：

```bash
cd /inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/eeg-main/LaBraM-unified-AON-dynamic_general
CUDA_VISIBLE_DEVICES=0 SEED=0 bash scripts/run_recon_only_serial.sh
```

总队列自动后台运行，终端会打印总日志路径。指定其他 GPU 时替换 `CUDA_VISIBLE_DEVICES`。

只跑部分数据集或修改顺序：

```bash
CUDA_VISIBLE_DEVICES=0 SEED=0 bash scripts/run_recon_only_serial.sh seed erp_core
```

只检查路径和 Stage1 最终命令，不启动训练：

```bash
DRY_RUN=1 bash scripts/run_recon_only_serial.sh
```

现有子脚本的 DRY_RUN 只展开 Stage1，并报告 Stage2 checkpoint 交接路径；它不等于已完成 Stage2 加载和训练验证。

## 输出

每次队列使用共同的、带时间戳的运行编号：

```text
outputs/recon_only_serial/运行编号/serial.log
outputs/recon_only_serial/运行编号/status.tsv
outputs/shu/recon_only/运行编号/{stage1,stage2,pipeline.log}
outputs/seedv/recon_only/运行编号/{stage1,stage2,pipeline.log}
outputs/tuev/recon_only/运行编号/{stage1,stage2,pipeline.log}
outputs/seed/recon_only/运行编号/{stage1,stage2,pipeline.log}
outputs/erpcore/recon_only/运行编号/{stage1,stage2,pipeline.log}
```

用 `tail -f` 加启动时打印的 `serial.log` 路径查看队列。`status.tsv` 记录完成或失败的数据集。

## 对照设置

总脚本将两个 summary 对比权重、两个 correction 对比权重、两个交换权重及正则权重设为 0。保留双分支模型，仅用重建目标训练 Stage1。

本队列明确设置 `MISSING_WEIGHT=500`，可以用环境变量覆盖为其他正数。SHU、SEEDV、TUEV、SEED 的当前默认值也是 500；ERP-Core 原脚本默认值为 0，所以本次 ERP-Core 会新增重建训练目标，不能称为仅删除解耦损失的严格消融。

学习率、batch、更新频率、归一化、各阶段轮数和选择指标沿用各数据集现有脚本。若已有解耦实验使用了不同的覆盖参数，应先对齐；本脚本不会自动读取历史实验参数。

Stage2 自动使用同一次运行的 Stage1 最佳 checkpoint。最终按预先确定的验证指标选择 Stage2 checkpoint，再比较测试分类结果。

失败后可以指定尚未完成的数据集重新启动，它会创建新运行目录；不会自动恢复半途训练。
