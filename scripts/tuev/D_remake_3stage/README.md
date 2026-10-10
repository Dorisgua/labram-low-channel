# TUEV remake 三阶段 D

参照 ERP-Core 三阶段：全导联分支训练→少导联适配补全→分类。

| 脚本 | 轮数 | 输入及损失 |
|---|---:|---|
| stage1_remake.sh | 50 | 23真实导联，fullchannel；直接重建0，对比各1，交换各1，其余0 |
| stage1_2_remake.sh | 20 | 13真实导联+10Prototype；重建50000，对比各1，交换各1，其余0 |
| stage2_remake.sh | 50 | 继承补全checkpoint，冻结CNN/corrector，训练Transformer与分类头 |

保留TUEV remake：SSD数据、5时间片、batch64、update_freq8、LR5e-4、weight_decay0.05、layer_decay0.65、warmup5、mean_pool/all、按验证Kappa选分类checkpoint、seed0。补全阶段按验证总loss选模。第一步并非没有重建约束：交换项包含重建。

```bash
SEED=0 bash scripts/tuev/D_remake_3stage/stage1_then_stage2_remake.sh
```

默认后台严格串行、端口auto，输出 `outputs/tuev/D_pipeline_remake_3stage/seed0_时间戳_PID/{stage1,stage2,stage3}`。SEED已export，沿用完成检查和同RUN_ROOT恢复逻辑；未完成阶段不覆盖。

单独补全要求STAGE1_CHECKPOINT，单独分类要求STAGE2_CHECKPOINT。RECON_MISSING_WEIGHT默认50000可覆盖。第二步保留对比与交换，不是仅重建。历史ERP-Core39.54%的成绩不能推广为TUEV结果。
