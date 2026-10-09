# SHU 三阶段 D

按 SEED-V 三阶段入口组织，保留 SHU 的 13→32 导联、四时间片、95 归一化、batch64、LR5e-4、update_freq1、layer_decay1.0。

| 脚本 | 默认轮数 | 训练目标 |
|---|---:|---|
| stage1.sh | 50 | 冻结 CNN，训练 corrector；直接重建0，subject/task correction 对比各50，交换各1，其他0 |
| stage1_2.sh | 50 | 加载解耦 checkpoint，继续训练 corrector；仅重建500，其他权重全部0 |
| stage2.sh | 30 | 加载重建 checkpoint，冻结 CNN/corrector，训练 Transformer 和分类头；按验证 BAcc 选模 |

第一步的交换损失仍包含重建约束；关闭的是直接缺失特征重建。

```bash
SEED=1 bash scripts/shu/D_3stage/stage1_then_stage2.sh
```

默认后台严格串行，端口 auto。输出为 `outputs/shu/D_pipeline_3stage/seed1_时间戳_PID/{stage1,stage2,stage3}`。脚本名 stage1_2 对应输出 stage2，分类脚本 stage2 对应输出 stage3。

用相同 RUN_ROOT 重启可跳过完成的阶段。退出崩溃后，仅当最终 epoch、训练结束记录和可读取 checkpoint 都通过检查才继续；不完整阶段停止且不会覆盖。SEED 已 export 到子阶段及后台进程。

单独重建需设置 STAGE1_CHECKPOINT；单独分类需设置 STAGE2_CHECKPOINT。补全阶段加载权重时关闭 student. 前缀过滤。RECON_MISSING_WEIGHT 可覆盖第二步重建权重。
