# clean_disentangle

ERP-Core 两阶段实验代码：Stage1 学习 subject/task token 修正量，Stage2 使用不同通道输入完成 12 分类。

## 主流程

```text
ERP-Core x_full/x_obs
  -> LaBraM TemporalConv token
  -> Stage1: StableCore + D_sub/D_task + token reconstruction
  -> Stage2: full/observed/prototype/dynamic 输入对照
  -> ERP-Core 12 分类
```

Stage1 重建的是 LaBraM token，不是原始 EEG 波形。Missing + Prototype 模式的预测为：

```text
P_miss + D_sub + D_task
```

## 目录

| 路径 | 作用 |
|---|---|
| `modeling.py` | Stage1 模型、重建配置和 token 组合 |
| `stage1/` | Stage1 正式训练入口和启动脚本 |
| `losses.py` | contrastive、swap reconstruction 和 MSE |
| `prototype.py` | 固定通道 prototype 的加载和选择 |
| `engine.py` | batch 搬运及简单单步重建接口 |
| `stage2/` | 下游分类模型、训练入口和脚本 |
| `evaluation/` | latent probe、绘图和专项诊断 |
| `tests/` | smoke test 和单 batch audit |

## Stage1

先检查：

```bash
DRY_RUN=1 BACKGROUND=0 bash clean_disentangle/stage1/scripts/train_missing_prototype_d.sh
```

正式运行：

```bash
bash clean_disentangle/stage1/scripts/train_missing_prototype_d.sh
```

## Stage2

不依赖 Stage1 的基线可直接检查：

```bash
DRY_RUN=1 BACKGROUND=0 bash clean_disentangle/stage2/scripts/wrapper_full.sh
```

动态补全模式需要指定一个 Stage1 运行目录，其中必须包含 `config.json` 和 `checkpoints/checkpoint-last.pth`：

```bash
STAGE1_RUN_DIR=/path/to/stage1/run \
bash clean_disentangle/stage2/scripts/wrapper_dynamic.sh
```

## 测试与评估

```bash
bash clean_disentangle/stage1/scripts/smoke_reconstruction.sh
```

正式 probe 和绘图位于 `evaluation/`，针对具体 checkpoint 的专项检查位于 `evaluation/diagnostics/`。

## 输出

训练结果默认写入仓库根目录下的 `outputs/`。数据保留在仓库外部，由 Shell 脚本中的 `DATA_PATH` 指定。
