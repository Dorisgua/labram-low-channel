#!/usr/bin/env python3
"""Draw ERP-Core observed subsets using MNE standard_1020 template positions."""
import ast
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Ellipse
import mne
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs' / 'figures' / 'erpcore_channel_layouts'
COLORS = ['#2676B8', '#E58A23', '#24947C', '#E7EBF0']


def load_channels():
    # Read literal lists without importing training dependencies from utils.
    tree = ast.parse((ROOT / 'Channels_definition.py').read_text())
    names = {f'ERPCORE_{n}_CHANNELS' for n in (12, 14, 21, 30)}
    lists = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in names:
                    lists[target.id] = ast.literal_eval(node.value)
    full = [c for c in lists['ERPCORE_30_CHANNELS'] if c not in ('HEOG', 'VEOG')]
    subsets = {n: lists[f'ERPCORE_{n}_CHANNELS'] for n in (12, 14, 21)}
    for n, channels in subsets.items():
        assert len(set(channels)) == n and set(channels) <= set(full)
    assert set(subsets[12]) < set(subsets[14]) < set(subsets[21])
    return full, subsets


def positions(full):
    template = mne.channels.make_standard_montage('standard_1020')
    xyz = {k.upper(): v for k, v in template.get_positions()['ch_pos'].items()}
    # Azimuthal equidistant projection: top view, anterior up, left on left.
    # Template coordinates are illustrative, not subject-specific digitization.
    result = {}
    for name in full:
        x, y, z = xyz[name]
        theta = np.arctan2(np.hypot(x, y), z)
        azimuth = np.arctan2(y, x)
        radius = theta / (np.pi / 2)
        result[name] = np.array([radius * np.cos(azimuth), radius * np.sin(azimuth)])
    scale = max(np.linalg.norm(p) for p in result.values()) / .88
    return {name: p / scale for name, p in result.items()}


def panel(ax, n, full, subsets, coords):
    ax.add_patch(Circle((0, 0), 1, fill=False, lw=1.7, ec='#536273'))
    for sign in (-1, 1):
        ax.add_patch(Ellipse((sign * 1.015, 0), .12, .32, fill=False, ec='#536273', lw=1.5))
    ax.plot([-.12, 0, .12], [.993, 1.13, .993], color='#536273', lw=1.7)
    ax.text(0, 1.20, 'Front / nose', ha='center', fontsize=10, color='#536273')
    ax.text(-1.16, 0, 'L', ha='center', va='center', color='#536273')
    ax.text(1.16, 0, 'R', ha='center', va='center', color='#536273')
    for name in full:
        group = (0 if name in subsets[12] else 1 if name in subsets[14] else 2) if name in subsets[n] else 3
        x, y = coords[name]
        ax.scatter(x, y, s=590, c=COLORS[group], edgecolors='white', linewidths=1.5, zorder=3)
        ax.text(x, y, name, fontsize=9, weight='bold', ha='center', va='center',
                color='white' if group < 3 else '#647080', zorder=4)
    ax.set_title(f'{n} observed channels\n{28-n} missing of 28', fontsize=17, weight='bold', pad=16)
    ax.set(xlim=(-1.26, 1.26), ylim=(-1.12, 1.30), aspect='equal')
    ax.axis('off')


def save(fig, stem):
    for extension in ('png', 'svg'):
        fig.savefig(OUT / f'{stem}.{extension}', dpi=220, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    full, subsets = load_channels()
    coords = positions(full)
    legend = [Line2D([], [], marker='o', linestyle='', markersize=11, color=color, label=label)
              for color, label in zip(COLORS, ['Original 12', 'Added at 14: C5 / C6', 'Added at 21', 'Missing'])]
    fig, axes = plt.subplots(1, 3, figsize=(17, 6))
    for ax, n in zip(axes, subsets):
        panel(ax, n, full, subsets, coords)
    fig.legend(handles=legend, loc='lower center', ncol=4, frameon=False, fontsize=12, bbox_to_anchor=(.5, .035))
    fig.text(.5, .01, 'ERP-Core | Standard 10-20 template, top view | Schematic positions; not individual electrode digitization', ha='center', fontsize=10, color='#647080')
    fig.subplots_adjust(bottom=.17, wspace=.09, top=.86)
    save(fig, 'erpcore_12_14_21_comparison')
    for n in subsets:
        fig, ax = plt.subplots(figsize=(6.4, 7.3))
        panel(ax, n, full, subsets, coords)
        fig.legend(handles=legend, loc='lower center', ncol=2, frameon=False, fontsize=10, bbox_to_anchor=(.5, .025))
        fig.subplots_adjust(bottom=.17, top=.86)
        save(fig, f'erpcore_{n}_observed')
    metadata = {str(n): {'observed': channels, 'missing': [c for c in full if c not in channels]}
                for n, channels in subsets.items()}
    (OUT / 'channels.json').write_text(json.dumps(metadata, indent=2) + '\n')
    (OUT / 'README.md').write_text(
        '# ERP-Core 导联位置图\n\n'
        '名单读取自 `Channels_definition.py`。蓝色：原始 12 导联；橙色：14 导联新增 C5/C6；'
        '绿色：21 导联进一步新增；灰色：28 导联目标空间内仍缺失的导联。\n\n'
        '位置使用 MNE `standard_1020` 模板进行二维投影，并统一缩放。俯视图，鼻尖向上，'
        '左侧为受试者左侧；并非个体实测坐标。所有面板使用相同坐标和比例。\n\n'
        '21 导联新增：P8、P7、PO7、FC3、PO3、FC4、PO4；PO7/PO8 的不对称来自当前实验配置。\n\n'
        'PNG 用于展示，SVG 用于矢量编辑。`channels.json` 保存观测与缺失名单。\n\n'
        '复现：使用装有 matplotlib、numpy、mne 的 Python 运行 '
        '`tools/plot_erpcore_channel_layouts.py`。\n')
    print(OUT)


if __name__ == '__main__':
    main()
