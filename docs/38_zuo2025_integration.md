# Zuo2025 第一版接入说明

## 已确认的数据语义

- 任务：左腿与右腿运动想象二分类。
- 衍生文件：30 名被试，共 14,034 个 trial。
- 原始 trial：500 Hz、30 个 EEG 通道、10 秒，时间范围为提示前 2 秒到提示后 8 秒。
- 第一版模型输入：提示出现后的 0--4 秒，即 Python 切片 `1000:3000`。
- 输出：200 Hz、每个 trial 800 点，不使用滑窗，不把一个 trial 拆成多个样本。
- 标签：源标签 1（左腿）映射为 0；源标签 2（右腿）映射为 1。
- 单位：μV。
- 作者衍生数据已经进行 2--45 Hz 带通、49--51 Hz 陷波、平均重参考和伪迹剔除，本项目不重复滤波。

## 预处理

先检查源文件，不写数据：

```bash
python dataset_maker/make_Zuo2025.py --dry-run
```

生成完整数据：

```bash
python dataset_maker/make_Zuo2025.py
```

默认输出目录为：

```text
/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/Zuo2025/processed_data_4s_200hz
```

每个被试保存一个 `[trials, 30, 800]` 的 float32 EEG 数组和一个标签数组。`manifest.json` 记录时间范围、通道、标签、每被试样本数以及计算训练集归一化所需的统计量。

## 读取与划分

读取入口是：

```python
from data_processor.zuo2025 import prepare_Zuo2025_cross_subject_dataset
```

第一版默认采用 subject-disjoint 的 24/3/3 划分：

- train：subject 1--24
- validation：subject 25--27
- test：subject 28--30

这个固定划分用于先完成可重复基线，并不声称是论文唯一指定的 benchmark。读取函数允许显式传入三组被试编号，后续找到需要复现的公开 protocol 时可以直接替换。`z_score` 的均值和标准差只由训练被试的样本计算，不读取验证或测试被试统计量。

Zuo2025 已注册到 `run_class_finetuning.py`。第一版启动脚本为：

```text
scripts/38Omeanpool.finetune_zuo2025_30_labrambase_freeze_cnn.sh
```

它使用全 30 通道、`mean_pool`、冻结 CNN、无 prototype，默认 seed 0 和 2 epochs，作为第一轮 smoke test。运行前必须先完成上述预处理。
