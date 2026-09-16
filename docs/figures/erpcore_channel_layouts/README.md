# ERP-Core 导联位置图

名单读取自 `Channels_definition.py`。蓝色：原始 12 导联；橙色：14 导联新增 C5/C6；绿色：21 导联进一步新增；灰色：28 导联目标空间内仍缺失的导联。

位置使用 MNE `standard_1020` 模板进行二维投影，并统一缩放。俯视图，鼻尖向上，左侧为受试者左侧；并非个体实测坐标。所有面板使用相同坐标和比例。

21 导联新增：P8、P7、PO7、FC3、PO3、FC4、PO4；PO7/PO8 的不对称来自当前实验配置。

PNG 用于展示，SVG 用于矢量编辑。`channels.json` 保存观测与缺失名单。

复现：使用装有 matplotlib、numpy、mne 的 Python 运行 `tools/plot_erpcore_channel_layouts.py`。
