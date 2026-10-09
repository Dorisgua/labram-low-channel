# ERP-Core 三阶段 D

沿用 SEED-V 的三阶段组织和恢复检查，保留 ERP-Core 数据路径、12→28 导联、z-score、batch64、LR5e-4、update_freq1、layer_decay1.0。

| 脚本 | 默认轮数 | 目标 |
|---|---:|---|
| stage1.sh | 50 | 冻结 CNN 训练 corrector：直接重建0，对比各50，交换各1，其余0 |
| stage1_2.sh | 50 | 继承解耦权重，仅重建500，其余损失0 |
| stage2.sh | 30 | 继承重建权重，冻结 CNN/corrector，训练 Transformer 和分类头；验证 BAcc 选模 |

交换约束包含重建，第一步关闭的是直接缺失特征重建。第二步继续更新整个 corrector，不保证之前的解耦结构保持。

```bash
SEED=0 bash scripts/erp_core/D_3stage/stage1_then_stage2.sh
```

默认后台严格串行，端口 auto。结果为 `outputs/erpcore/D_pipeline_3stage/seed0_时间戳_PID/{stage1,stage2,stage3}`，分别对应上述三个脚本。

相同 RUN_ROOT 重启会检查并跳过已完成阶段。训练结束后退出崩溃，仅当最终 epoch、结束日志和可读最佳 checkpoint 验证通过才继续；中途失败停止且不覆盖。

单独重建要求 STAGE1_CHECKPOINT；单独分类要求 STAGE2_CHECKPOINT，不再回退到历史 checkpoint。SEED 已 export。RECON_MISSING_WEIGHT 默认500，可覆盖。

这是少导联解耦→仅重建→分类流程，与相邻历史三阶段仓库的全导联预训练方案不同，结果不能直接混用。
