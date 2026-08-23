# COMMAND_LOG

## 2026-08-23 — 建立统一 AON 基线

目的：
从旧工作区保存独立代码副本，不修改原目录，不复制旧实验结果。

源 commit：
3342fd5b1b42d0e1cead59e8ae9db9bdb71a11aa

源工作区状态：
287 条修改、删除或未跟踪记录，详情见 BASELINE_SOURCE_STATUS.txt。

执行结果：
- 已生成排除旧 outputs、Git 历史、缓存和日志的源归档。
- 归档大小约 185M。
- 已解压到 LaBraM-unified-AON。
- 新目录共 151 个文件。
- 已保存源路径、源 commit 和源工作区状态。
- 已修订 .gitignore。
- checkpoints 保留在本地，但不进入 Git。
- docs/prototypes 下的小型 prototype 允许进入 Git。

## 2026-08-23 — scripts Bash 第一阶段静态预检

范围与限制：
- 分支确认：`develop/aon-v1`。
- 只读检查 `scripts/*.sh`；未启动训练，未读取 checkpoint 内容，未改 `.py`、`.sh` 或 `outputs`。
- 只新增 `docs/audit/SCRIPT_RUN_MATRIX.md` 并追加本节。

执行的检查：
- `find scripts -maxdepth 1 -type f -name '*.sh'`：枚举 60 个 Bash 脚本。
- 对每个脚本执行 `bash -n`：60/60 通过。
- 对每个脚本用字节级 `grep` 检查 `\r`：未发现 CRLF。
- 用 `rg`、`sed` 和只读 Python AST/正则脚本核对最终入口、wrapper 继承链、CMD 参数、argparse 定义与 choices。
- 检查所有脚本引用的 `.sh`/`.py`、默认 `torchrun`、数据路径及数据清单、prototype、output_dir 展开方式。
- 用 `stat` 检查 `checkpoints/labram-base.pth`：存在，96,612,769 bytes；未加载权重。
- 检查 CMD 内重复参数：未发现重复或互斥参数。

结果：
- `READY=0`，`NEEDS_SMOKE_TEST=48`，`INVALID_ARGUMENT=3`，`MISSING_PROTOTYPE=2`，`AMBIGUOUS=7`，其余状态为 0。
- 5 个 PreExp35 脚本使用入口不支持的 `SEEDV` dataset key；其中 2 个还缺 SEED-V prototype。
- `40Aada` wrapper 的 completion/prototype 未进入最终 CMD。
- 2 个批量调度器引用不存在的 Attention prototype full-finetune 脚本。
- 未运行任何 smoke test，等待用户审核。
