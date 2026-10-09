# ERP-Core 历史 39.54% 配置

| 脚本 | 默认轮数 | 输入及损失 |
|---|---:|---|
| stage1.sh | 50 | 全部28真实导联；直接重建0，correction对比各1，交换各1，其余0 |
| stage1_2.sh | 20 | 12真实导联+16Prototype；重建50000，correction对比各1，交换各1，其余0 |
| stage2.sh | 50 | 冻结CNN/corrector，训练Transformer及分类头；验证BAcc选模 |

沿用历史 batch64、update_freq1、LR5e-4、weight_decay0.05、layer_decay1.0、warmup5、z-score、200Hz、seed0。模型为LaBraM动态补全；各补全阶段按验证总loss选模。

```bash
SEED=0 bash scripts/erp_core/D_3stage/stage1_then_stage2.sh
```

默认后台运行，端口auto，输出 `outputs/erpcore/D_pipeline_3stage/seed0_时间戳_PID/{stage1,stage2,stage3}`。使用新RUN_ROOT，避免将此前不同配置的运行当成已完成阶段跳过。完成检查和恢复逻辑保留。

单独第二步需STAGE1_CHECKPOINT，分类需STAGE2_CHECKPOINT。第一步FULLCHANNEL默认1，第二步固定0。第二步不是仅重建，保留对比与交换约束。

这些是历史39.54%实验的训练设置；不保证在当前代码环境下精确重现该数值。
