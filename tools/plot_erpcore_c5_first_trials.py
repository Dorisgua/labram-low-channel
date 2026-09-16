#!/usr/bin/env python3
"""C5 token comparison: first retained Test trial per subject, no error-based selection."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'outputs/erpcore/erp_core_D_stage1/checkpoint-best.pth')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    args.checkpoint = args.checkpoint.resolve()
    out = args.output_dir or args.checkpoint.parent / 'test_c5_first_trial_by_subject'
    out.mkdir(parents=True, exist_ok=True)
    os.chdir(ROOT)
    import utils
    from Channels_definition import ERPCORE_12_CHANNELS
    from data_processor.erpcore import prepare_ERPCORE_pt_dataset
    from run_dynamic_stage1 import get_models, _validate_completion_prototype
    print('Loading checkpoint and original data preprocessing...', flush=True)
    torch.set_num_threads(args.threads)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    train_args = checkpoint["args"]
    original_args = vars(train_args).copy()
    train, test, val = prepare_ERPCORE_pt_dataset(
        train_args.data_path,
        sampling_rate=train_args.sampling_rate,
        normalize_method=train_args.norm_method,
        channel_names=ERPCORE_12_CHANNELS,
    )
    del train, val

    prototype_ckpt = torch.load(
        train_args.channel_prototype_path, map_location="cpu", weights_only=False
    )
    prototypes = prototype_ckpt["channel_prototypes"]
    target_names = prototype_ckpt["ch_names"]
    target_indices = [int(v) for v in prototype_ckpt["input_chans_index"]]
    real_indices = [int(v) for v in utils.get_input_chans(ERPCORE_12_CHANNELS)]
    _validate_completion_prototype(
        args=train_args,
        ch_names=ERPCORE_12_CHANNELS,
        target_ch_names=target_names,
        target_input_chans_index=target_indices,
        prototypes=prototypes,
    )

    model = get_models(train_args)
    model.completion_scope = train_args.completion_scope
    model.pooling_scope = train_args.pooling_scope
    model.erpcore28_channel_prototypes.copy_(prototypes)
    model.target_input_chans_index = target_indices
    model.real_input_chans_index = real_indices
    model.load_state_dict(checkpoint["model"], strict=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(device).eval().requires_grad_(False)

    assert train_args.completion_scope == 'erpcore12_with_erpcore28', train_args.completion_scope
    assert list(target_names) == list(test.full_channel_names)
    subjects = sorted(test.subject_indices)
    selected = [test.subject_indices[subject][0] for subject in subjects]
    records, arrays = [], []
    print(f'Test subjects: {subjects}; selected local indices: {selected}', flush=True)
    with torch.inference_mode():
        for subject, index in zip(subjects, selected):
            batch = test[index]
            x_obs = batch[2].float().unsqueeze(0).unsqueeze(2).to(device) * float(train_args.input_scale)
            x_full = batch[3].float().unsqueeze(0).unsqueeze(2).to(device) * float(train_args.input_scale)
            result = model.forward_stage1(x_obs, x_full)
            missing = result['miss_indices'].cpu().tolist()
            c5_index = missing.index(target_names.index('C5'))
            values = [result[key][0, c5_index, 0].cpu().numpy() for key in ('h_miss_target', 'p_miss', 'h_pred_miss')]
            assert all(v.shape == (200,) and np.isfinite(v).all() for v in values)
            target, prototype, prediction = values
            assert np.allclose(result['h_pred_miss'].cpu().numpy(), (result['p_miss'] + result['d_sub'] + result['d_task']).cpu().numpy())
            record = dict(subject=subject, test_local_index=index, payload_index=int(test.indices[index]),
                          task_remapped=int(batch[1]), task_original=int(test.tasks[test.indices[index]]),
                          prototype_mse=float(np.mean((prototype-target)**2)),
                          dynamic_mse=float(np.mean((prediction-target)**2)))
            records.append(record)
            arrays.append(values)
    arrays = np.asarray(arrays)
    lower, upper = float(arrays.min()), float(arrays.max())
    padding = (upper-lower)*.06
    def draw(axes, record, values):
        target, prototype, prediction = values
        for ax, estimate, label, color, metric in zip(axes, (prototype, prediction),
                ('Static prototype', 'Dynamic prediction'), ('#777777', '#e68122'), ('prototype_mse', 'dynamic_mse')):
            ax.plot(np.arange(1,201), target, color='#2676b8', lw=1.15, label='True C5 token')
            ax.plot(np.arange(1,201), estimate, color=color, lw=1.05, alpha=.9, label=label)
            ax.set_title(f"Subject {record['subject']} | {label} | MSE {record[metric]:.6f}", fontsize=10)
            ax.set(xlim=(1,200), ylim=(lower-padding,upper+padding), xlabel='Token feature dimension (not time)', ylabel='Feature value')
            ax.grid(alpha=.15)
            ax.legend(fontsize=8, loc='upper right')
    for record, values in zip(records, arrays):
        fig, axes = plt.subplots(1,2,figsize=(14,3.8))
        draw(axes, record, values)
        fig.suptitle(f"Test first retained trial | C5 | local index {record['test_local_index']} | task {record['task_remapped']}")
        fig.tight_layout()
        fig.savefig(out / f"subject_{record['subject']:02d}_C5.png", dpi=180)
        plt.close(fig)
    fig, axes = plt.subplots(len(records),2,figsize=(16,3*len(records)), squeeze=False)
    for axis, record, values in zip(axes, records, arrays):
        draw(axis, record, values)
    fig.suptitle('C5 missing token | First retained Test trial per subject | Same axis scales', fontsize=16)
    fig.tight_layout(rect=(0,0,1,.98))
    fig.savefig(out / 'all_subjects_C5_comparison.png', dpi=160)
    fig.savefig(out / 'all_subjects_C5_comparison.pdf')
    plt.close(fig)
    with (out / 'samples_and_metrics.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    np.savez(out / 'C5_features.npz', target=arrays[:,0], prototype=arrays[:,1], prediction=arrays[:,2], subjects=subjects, test_local_indices=selected)
    with args.checkpoint.open('rb') as handle:
        digest = hashlib.file_digest(handle, 'sha256').hexdigest()
    metadata = dict(checkpoint=str(args.checkpoint), checkpoint_sha256=digest, epoch=checkpoint.get('epoch'),
                    checkpoint_args=original_args, split='test', selection='First retained trial in loader order per subject; not chronological onset',
                    feature_shape=list(arrays[:,0].shape), target_channel_names=target_names, records=records,
                    device=str(device), dtype='float32', autocast=False)
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2, default=str))
    (out / 'README.md').write_text(
        '# Test 各 subject 首个 trial 的 C5 特征对比\n\n'
        '每个 Test subject 按 loader 顺序取第一个保留的 trial（排除 N170 后），不按重建误差挑选；不表示原始实验时间顺序的首个 trial。\n\n'
        '每行一个 subject，左图为静态 prototype 与真实 C5 token，右图为动态预测与同一个真实 token。蓝线为真实值，灰线为 prototype，橙线为动态预测。'
        '横轴为 200 个 CNN token 特征维度，不是时间；全部图片使用相同纵轴范围。\n\n'
        'prototype 采用训练时的静态原型，checkpoint strict=True 加载；真实 token 通过冻结的 patch_embed 从完整 EEG 提取，完整 EEG 不作为动态预测的输入。'
        '归一化、采样率和 input_scale 使用 checkpoint 保存的参数，推理采用 eval、float32、CPU。\n\n'
        '单图 subject_XX_C5.png；汇总图 all_subjects_C5_comparison.png/pdf；样本索引及 MSE 见 samples_and_metrics.csv；特征数组见 C5_features.npz；配置和校验和见 metadata.json。\n\n'
        '这些是每人一个 trial 的 Test 示例，只用于可视化，不支持整体性能或同一被试 trial 间恢复能力的结论，也不据此选择 checkpoint。\n\n'
        '复现：在仓库根目录，使用 labram 环境 Python 运行 tools/plot_erpcore_c5_first_trials.py。\n')
    print(json.dumps(records, indent=2), flush=True)
    print(f'Saved plots: {out}', flush=True)


if __name__ == '__main__':
    main()
