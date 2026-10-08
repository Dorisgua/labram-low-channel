# 五个数据集的 A/N/O 串行实验

在项目根目录运行。复用各数据集的 `A/N/O/freeze_cnn.sh`：冻结 CNN，训练 Transformer 和分类头。
A 为静态原型补全，N 为少导联直接分类，O 为全导联分类。

```bash
CUDA_VISIBLE_DEVICES=0 SEED=0 bash scripts/run_aon_serial.sh
```

默认后台运行，依次 SHU → SEEDV → TUEV → SEED → ERP-Core，每个数据集 A → N → O，共 15 次训练。任意一次失败就停止。自动选择端口。若 recon 队列仍在运行，要选择另一张空闲 GPU，或者等它结束；本脚本不会自动等待 recon 队列。

沿用现有入口的训练参数（当前默认 epochs=50）。若要与 recon-only Stage2 的默认 30 轮对齐：

```bash
CUDA_VISIBLE_DEVICES=0 SEED=0 EPOCHS=30 bash scripts/run_aon_serial.sh
```

比较前还需对齐实际 lr、batch size、梯度累积、数据划分和验证集选模指标；串行脚本不会自动读取历史实验配置。

只跑部分数据集／模式：

```bash
AON_MODES='A N' CUDA_VISIBLE_DEVICES=0 bash scripts/run_aon_serial.sh shu seedv
```

仅检查最终命令和必要路径，不启动训练：

```bash
DRY_RUN=1 bash scripts/run_aon_serial.sh
```

数据路径默认沿用各数据集 base.sh。可分别设置 `SHU_DATA_PATH`、`SEEDV_DATA_PATH`、`TUEV_DATA_PATH`、`SEED_DATA_PATH`、`ERP_CORE_DATA_PATH`，例如改用 TUEV 的 SSD 副本：

```bash
TUEV_DATA_PATH=/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/TUEZ/v2.0.1/processed_labram/processed \
CUDA_VISIBLE_DEVICES=0 SEED=0 EPOCHS=30 bash scripts/run_aon_serial.sh
```

每次运行生成独立时间戳目录：

- 总日志：`outputs/aon_serial/<run_id>/serial.log`
- 完成／失败记录：同目录 `status.tsv`
- 模型及单次日志：`outputs/<dataset>/aon/<run_id>/<A或N或O>/`（ERP-Core 的目录名为 erpcore）

启动后会打印总日志路径，用 `tail -f <打印的路径>` 查看。脚本不会自动恢复失败任务，可指定剩余数据集和模式重新运行；重新运行会创建新目录。
