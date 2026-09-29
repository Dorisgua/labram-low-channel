# BCI-IV-2A CBraMod 跨被试划分（LaBraM D）

本目录从相邻的 `LabraM-Git-Diff-disengle_3stage_changeronghe_deletep_bciiv2a` checkout 中经核对后复制；三份 JSON 的 trial 路径已指向当前用户 SSD 上的 `BCI-IV-2A/processed_data/A01` 至 `A09`。三个 split 的 5184 个对应文件均存在且与原路径文件大小一致，首尾抽样文件的 SHA256 也一致。

| split | 被试 | trial 数 |
| --- | --- | ---: |
| train | A01–A05 | 2880 |
| val | A06–A07 | 1152 |
| test | A08–A09 | 1152 |

每名被试 576 个 trial，四分类各类数量相等。JSON 中的 mean/std 来自 train，被三个 split 共用。本实验是 13 个观测导联到 22 个完整导联，信号从 250 Hz 重采样到 200 Hz，长度为 `[13,800]` 与 `[22,800]`。训练集原型在 `docs/prototypes/01_bciiv2a22_cbramod_train_cnn_patch_embed_mean.pth`，其来源元数据记载 2880 个 train trial、每个 trial 4 个 patch，源 LaBraM checkpoint 与本仓库的 `checkpoints/labram-base.pth` SHA256 一致。

此配置会在训练启动时把 train/val/test 的归一化完整信号预载入 CPU 内存（合计约 348 MiB）。DataLoader 和 Stage1 配对抽样随后都从内存读取；其他 BCI 数据配置不启用这一模式。已启动的进程不会因修改代码或 JSON 自动切换数据源。

使用本仓库的 `dataset_maker/make_BCIIV2A.py --protocol cbramod` 可重新生成划分；使用 `docs/prototypes/01_generate_bciiv2a_cnn_patch_prototypes.py --protocol cbramod` 可重新生成训练集原型。生成脚本会写入目标文件，运行前可用 `--dry-run` 检查划分脚本。

按顺序跑完整 D（Stage1 结束并生成 best 后才启动 Stage2）：

```bash
bash scripts/bciiv2a/D/cbramod_stage1_then_stage2.sh
```

默认把整个流程放在后台，启动时输出 PID 和 `pipeline.log` 路径；查看日志可用 `tail -f <运行目录>/pipeline.log`。需要前台运行时设置 `RUN_BACKGROUND=0`。

也可以分别运行两阶段：

```bash
bash scripts/bciiv2a/D/cbramod_stage1.sh
STAGE1_CHECKPOINT=/实际的Stage1输出目录/checkpoint-best.pth bash scripts/bciiv2a/D/cbramod_stage2.sh
```

Stage1 按验证集 loss 选 best；Stage2 按验证集 balanced accuracy 选 best。Test 不参与选 checkpoint。这里的 CBraMod 指跨被试数据划分；模型仍为 LaBraM Dynamic。
