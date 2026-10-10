# 实验分析工具

通用工具从 checkpoint 读取数据集、导联与预处理配置。旧脚本名保留为兼容入口。

| 工具 | 用途 |
|---|---|
| `evaluate_fixed_prototype_mse.py` | 固定 Prototype 的缺失特征 MSE；`--split test/val/both` |
| `plot_dynamic_z_d.py` | z/d 解耦可视化；`--views full/few/all`；`--render-only` 从缓存重绘 |
| `plot_channel_feature_comparison.py` | 指定缺失导联的真实特征、Prototype、动态预测对照；按 subject/class 取首个样本 |
| `plot_disentanglement_four_column.py` | 旧格式坐标的四列排版 |
| `plot_erpcore_channel_layouts.py` | ERP-Core 导联布局 |

```bash
python tools/prototype_mse/evaluate_fixed_prototype_mse.py --checkpoint <checkpoint-best.pth> --split both --output <metrics.json>
python tools/plot_dynamic_z_d.py --checkpoint <checkpoint-best.pth> --split test --views all
python tools/plot_dynamic_z_d.py --checkpoint <checkpoint-best.pth> --render-only --output-dir <已有绘图目录>
python tools/plot_channel_feature_comparison.py --checkpoint <checkpoint-best.pth> --channel C5 --group-by subject
python tools/plot_channel_feature_comparison.py --checkpoint <checkpoint-best.pth> --channel FZ --group-by class
```

缓存重绘输出 `*_tsne_redraw.png/pdf`，不加载模型、不重新计算 t-SNE。
ERP-Core 使用任务名称，其他数据集显示 Task ID。
导联特征对比默认画第 0 个 patch，可通过 `--patch-index` 修改；横轴是 CNN 特征维度，不是时间。
动态预测始终使用观测导联，完整 EEG 只用于提取目标特征。
按组取首个 loader 样本，不按误差挑选，不代表完整数据集性能。

兼容入口：
- `evaluate_erpcore_prototype_mse.py`：默认 val + test、num-workers=0。
- `plot_erpcore_dynamic_stage1_disentanglement.py`：默认少导联视图，输出 `disentanglement_best/`；新计算的文件命名采用通用版本。
- `plot_erpcore_c5_first_trials.py`：默认 C5、按 subject。
- `plot_tuev_fz_first_trials.py`：默认 FZ、按 class。

已有结果无需重跑。新版本文件名、布局、采样方式可能与历史脚本不同；精确复现旧图需使用历史 Git 版本。
