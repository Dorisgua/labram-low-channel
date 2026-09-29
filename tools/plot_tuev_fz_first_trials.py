#!/usr/bin/env python3
"""Plot FZ missing-token features for the first Test trial of each TUEV class."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    checkpoint_path = args.checkpoint.resolve()
    out = args.output_dir or checkpoint_path.parent / 'test_fz_first_trial_by_class'
    out.mkdir(parents=True, exist_ok=True)
    os.chdir(ROOT)
    import utils
    from Channels_definition import TUEV_13_CHANNELS, TUEV_23_CHANNELS
    from data_processor.tuev import prepare_TUEV_dynamic_dataset
    from run_dynamic_stage1 import get_models, _validate_completion_prototype

    torch.set_num_threads(4)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    train_args = checkpoint['args']
    assert train_args.dataset == 'TUEV' and train_args.completion_scope == 'tuev13_with_tuev23'
    assert train_args.channel_subset == 'tuev13'
    _, test, _ = prepare_TUEV_dynamic_dataset(train_args.data_path, TUEV_13_CHANNELS)
    prototype_ckpt = torch.load(train_args.channel_prototype_path, map_location='cpu', weights_only=False)
    prototypes = prototype_ckpt['channel_prototypes']
    target_names = prototype_ckpt['ch_names']
    target_indices = [int(v) for v in prototype_ckpt['input_chans_index']]
    _validate_completion_prototype(train_args, TUEV_13_CHANNELS, target_names, target_indices, prototypes)
    assert list(target_names) == TUEV_23_CHANNELS == test.full_channel_names
    assert 'FZ' not in TUEV_13_CHANNELS

    model = get_models(train_args)
    model.completion_scope = train_args.completion_scope
    model.pooling_scope = train_args.pooling_scope
    model.tuev23_channel_prototypes.copy_(prototypes)
    model.target_input_chans_index = target_indices
    model.real_input_chans_index = [int(v) for v in utils.get_input_chans(TUEV_13_CHANNELS)]
    model.load_state_dict(checkpoint['model'], strict=True)
    device = torch.device(args.device)
    model.to(device).eval().requires_grad_(False)

    selected = [(label, indices[0]) for label, indices in sorted(test.task_indices.items())]
    records, arrays = [], []
    with torch.inference_mode():
        for label, index in selected:
            batch = test[index]
            x_obs = batch[2].reshape(1, len(TUEV_13_CHANNELS), 5, 200).to(device) * float(train_args.input_scale)
            x_full = batch[3].reshape(1, len(TUEV_23_CHANNELS), 5, 200).to(device) * float(train_args.input_scale)
            result = model.forward_stage1(x_obs, x_full)
            fz_index = result['miss_indices'].tolist().index(target_names.index('FZ'))
            values = [result[key][0, fz_index, 0].cpu().numpy() for key in ('h_miss_target', 'p_miss', 'h_pred_miss')]
            assert all(value.shape == (200,) and np.isfinite(value).all() for value in values)
            target, prototype, prediction = values
            assert np.allclose(result['h_pred_miss'].cpu().numpy(),
                               (result['p_miss'] + result['d_sub'] + result['d_task']).cpu().numpy(), atol=1e-5)
            records.append(dict(task_label=label, test_local_index=index, filename=test.files[index],
                                prototype_mse=float(np.mean((prototype-target)**2)),
                                dynamic_mse=float(np.mean((prediction-target)**2))))
            arrays.append(values)
    arrays = np.asarray(arrays)
    lower, upper = float(arrays.min()), float(arrays.max())
    pad = max((upper-lower)*.06, 1e-6)

    def draw(axes, record, values):
        target, prototype, prediction = values
        for ax, estimate, name, color, metric in zip(axes, (prototype, prediction),
                ('Static prototype', 'Dynamic prediction'), ('#777777', '#e68122'),
                ('prototype_mse', 'dynamic_mse')):
            ax.plot(np.arange(1, 201), target, color='#2676b8', lw=1.15, label='True FZ token')
            ax.plot(np.arange(1, 201), estimate, color=color, lw=1.05, alpha=.9, label=name)
            ax.set_title(f"Class {record['task_label']} | {name} | MSE {record[metric]:.6f}", fontsize=10)
            ax.set(xlim=(1, 200), ylim=(lower-pad, upper+pad),
                   xlabel='Token feature dimension (not time)', ylabel='Feature value')
            ax.grid(alpha=.15)
            ax.legend(fontsize=8, loc='upper right')

    for record, values in zip(records, arrays):
        fig, axes = plt.subplots(1, 2, figsize=(14, 3.8))
        draw(axes, record, values)
        fig.suptitle(f"Test first trial in file order | FZ patch 1/5 | class {record['task_label']} | index {record['test_local_index']}")
        fig.tight_layout()
        fig.savefig(out / f"class_{record['task_label']}_FZ.png", dpi=180)
        plt.close(fig)
    fig, axes = plt.subplots(len(records), 2, figsize=(16, 3*len(records)), squeeze=False)
    for axis, record, values in zip(axes, records, arrays):
        draw(axis, record, values)
    fig.suptitle('FZ missing token | First Test trial per class | Patch 1/5 | Same axis scales', fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, .98))
    fig.savefig(out / 'all_classes_FZ_comparison.png', dpi=160)
    fig.savefig(out / 'all_classes_FZ_comparison.pdf')
    plt.close(fig)

    with (out / 'samples_and_metrics.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    np.savez(out / 'FZ_features.npz', target=arrays[:, 0], prototype=arrays[:, 1],
             prediction=arrays[:, 2], task_labels=[r['task_label'] for r in records],
             test_local_indices=[r['test_local_index'] for r in records])
    with checkpoint_path.open('rb') as handle:
        digest = hashlib.file_digest(handle, 'sha256').hexdigest()
    metadata = dict(checkpoint=str(checkpoint_path), checkpoint_sha256=digest,
                    epoch=checkpoint.get('epoch'), checkpoint_args=vars(train_args),
                    split='test', selection='First file-sorted Test trial per class',
                    channel='FZ', patch_index=0, feature_shape=list(arrays[:, 0].shape),
                    records=records, device=str(device), dtype='float32', autocast=False)
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2, default=str))
    (out / 'README.md').write_text(
        '# TUEV Test：FZ 缺失导联 token 特征对比\n\n'
        'Test 文件没有经核实的 subject ID，因此按类别各取文件名排序后的首个 trial；不按误差挑选。'
        'FZ 是 13→23 设置的缺失导联。每个 trial 有 5 个 200 点 patch，这里只画第 1 个 patch 的 200 维 CNN token 特征；横轴不是时间。\n\n'
        '左图：静态 prototype 与真实 token；右图：动态预测与同一真实 token。所有图使用相同纵轴。'
        '真实 token 从完整 EEG 经冻结 patch_embed 提取，完整 EEG 不输入动态预测分支。'
        '模型以 checkpoint strict=True 加载，数据、原型和 input_scale 使用 checkpoint 参数。\n\n'
        '这些是每类一个 Test trial 的示例图，不能代表整体性能。MSE 是这个 patch 的特征均方误差。'
        '复现：labram 环境 Python 运行 tools/plot_tuev_fz_first_trials.py --checkpoint <checkpoint-best.pth>。\n')
    print(json.dumps(records, indent=2), flush=True)
    print(f'Saved plots: {out}', flush=True)


if __name__ == '__main__':
    main()
