#!/usr/bin/env python3
"""Plot z/d from one Dynamic checkpoint: all full positions, matched missing positions, and few-channel completion."""
import argparse
import csv
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys

EXPERIMENT = Path(__file__).resolve().parent
ROOT = next(parent for parent in EXPERIMENT.parents if (parent / 'run_dynamic_stage1.py').is_file())
CKPT = EXPERIMENT / 'checkpoint-best.pth'

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sample-mode', choices=['all', 'random'], default='all')
    p.add_argument('--max-samples', type=int, default=2000, help='Number of random samples; ignored in all mode')
    p.add_argument('--sample-seed', type=int, default=42)
    p.add_argument('--split', choices=['test', 'train'], default='test')
    p.add_argument('--views', choices=['full', 'full32', 'all'], default='all',
                   help='full: full-input all positions only; all: also compare full/few input at missing positions')
    p.add_argument('--max-iter', type=int, default=1000)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--device', default='cuda')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--d-pca-dim', type=int, default=50, help='0: direct t-SNE of flattened D (slower)')
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, default=None,
                   help='Default: checkpoint directory / tsne_<split>_<sample-mode>_<views>')
    a = p.parse_args()
    if (a.sample_mode == 'random' and a.max_samples < 32) or a.max_iter < 300 or a.threads < 1 or a.d_pca_dim < 0:
        p.error('Require max-samples >=32, max-iter >=300, threads >=1, d-pca-dim >=0')
    a.checkpoint = a.checkpoint.resolve()
    if not a.checkpoint.is_file():
        p.error(f'Checkpoint not found: {a.checkpoint}')
    sample_tag = 'all' if a.sample_mode == 'all' else f'random{a.max_samples}_seed{a.sample_seed}'
    out = (a.output_dir or a.checkpoint.parent / f'tsne_{a.split}_{sample_tag}_{a.views}').resolve()
    out.mkdir(parents=True, exist_ok=True)
    os.environ['OMP_NUM_THREADS'] = str(a.threads)
    os.environ['OPENBLAS_NUM_THREADS'] = str(a.threads)
    os.environ['MKL_NUM_THREADS'] = str(a.threads)
    import numpy as np
    import torch
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from torch.utils.data import DataLoader, Subset
    sys.path.insert(0, str(ROOT))
    os.chdir(ROOT)
    from run_dynamic_stage1 import get_models, _validate_completion_prototype, DATASET_CONFIGS
    import utils
    torch.set_num_threads(a.threads)
    torch.manual_seed(42)
    ck = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    train_args = ck['args']
    original_args = vars(train_args).copy()
    config = DATASET_CONFIGS[train_args.dataset]
    observed_channels = config['ch_names']
    if isinstance(observed_channels, dict):
        observed_channels = observed_channels[train_args.channel_subset]
    kwargs = {key: getattr(train_args, value)
              for key, value in config.get('prepare_kwargs_from_args', {}).items()}
    if config.get('pass_channel_names'):
        kwargs['channel_names'] = observed_channels
    train, test, val = config['prepare_fn'](train_args.data_path or config['root'], **kwargs)
    n_obs = len(observed_channels)
    dataset = train if a.split == 'train' else test
    del train, test, val
    n = len(dataset) if a.sample_mode == 'all' else min(a.max_samples, len(dataset))
    if n < 32:
        raise ValueError('Need at least 32 samples for perplexity=30')
    selected = (np.arange(len(dataset)) if a.sample_mode == 'all' else
                np.sort(np.random.default_rng(a.sample_seed).choice(len(dataset), n, replace=False)))
    subject_batches, task_batches = [], []
    print(f'Split={a.split}, samples={n}/{len(dataset)}, mode={a.sample_mode}, views={a.views}', flush=True)
    device = torch.device(a.device if torch.cuda.is_available() else 'cpu')
    proto = torch.load(train_args.channel_prototype_path, map_location='cpu', weights_only=False)
    n_full = len(proto['ch_names'])
    n_missing = n_full - n_obs
    if n_missing <= 0:
        raise ValueError('Checkpoint must describe observed-to-full channel completion')
    target_indices = [int(i) for i in proto['input_chans_index']]
    _validate_completion_prototype(train_args, observed_channels, proto['ch_names'],
                                   target_indices, proto['channel_prototypes'])
    model = get_models(train_args)
    model.load_state_dict(ck['model'], strict=True)
    model.completion_scope = train_args.completion_scope
    model.pooling_scope = getattr(train_args, 'pooling_scope', 'real')
    model.real_input_chans_index = list(utils.get_input_chans(observed_channels))
    model.target_input_chans_index = target_indices
    model.to(device).eval().requires_grad_(False)
    loader = DataLoader(Subset(dataset, selected.tolist()), batch_size=a.batch_size,
                        num_workers=a.num_workers, shuffle=False, pin_memory=device.type == 'cuda')
    features = {}
    offset = 0
    # Save raw features incrementally; all views have exactly the same sample order.
    with torch.no_grad():
        for batch in loader:
            if len(batch) != 6:
                raise ValueError('Dataset must return input, label, observed, full, subject, task')
            subject_batches.append(np.asarray(batch[4]).reshape(-1))
            task_batches.append(np.asarray(batch[5]).reshape(-1))
            scale = float(getattr(train_args, 'input_scale', config.get('input_scale', 0.01)))
            x_obs, x_full = [batch[i].float().to(device) * scale for i in (2, 3)]
            if x_obs.shape[1] != n_obs or x_full.shape[1] != n_full or x_full.shape[-1] % 200:
                raise ValueError(f'Unexpected observed/full shapes: {x_obs.shape}, {x_full.shape}')
            x_obs = x_obs.reshape(x_obs.shape[0], n_obs, -1, 200)
            x_full = x_full.reshape(x_full.shape[0], n_full, -1, 200)
            full = model._encode_dynamic_tokens(model._patch_tokens(x_full), fullchannel=True)
            values = {}
            for branch in ('sub', 'task'):
                values[f'full_all_z_{branch}'] = full[f'z_{branch}']
                values[f'full_all_d_{branch}'] = full[f'd_{branch}'].flatten(1)
            if a.views == 'all':
                few = model._encode_dynamic_tokens(model._patch_tokens(x_obs), fullchannel=False)
                miss_indices = few['miss_indices']
                if len(miss_indices) != n_missing:
                    raise ValueError(f'Expected {n_missing} missing channels')
                for branch in ('sub', 'task'):
                    full_missing_d = full[f'd_{branch}'].index_select(1, miss_indices)
                    values[f'full_missing_z_{branch}'] = full_missing_d.flatten(1, 2).mean(1)
                    values[f'full_missing_d_{branch}'] = full_missing_d.flatten(1)
                    values[f'few_missing_z_{branch}'] = few[f'z_{branch}']
                    values[f'few_missing_d_{branch}'] = few[f'd_{branch}'].flatten(1)
            count = len(batch[1])
            for key, tensor in values.items():
                x = tensor.cpu().numpy()
                if not np.isfinite(x).all():
                    raise ValueError(f'Nonfinite features: {key}')
                if key not in features:
                    features[key] = np.lib.format.open_memmap(out / f'{key}.npy', mode='w+',
                                                              dtype='float32', shape=(n, x.shape[1]))
                features[key][offset:offset+count] = x
            offset += count
            print(f'Extracted {offset}/{n}', flush=True)
    assert offset == n
    subjects = np.concatenate(subject_batches).astype(int)
    tasks = np.concatenate(task_batches).astype(int)
    task_ids = sorted(np.unique(tasks))
    reference = {'TASK_NAMES': {int(v): f'Task {v}' for v in task_ids},
                 'TASK_COLORS': dict(zip(task_ids, plt.get_cmap('tab20')(np.linspace(0, 1, len(task_ids)))))}
    for x in features.values():
        x.flush()
    with (out / 'sample_alignment.csv').open('w') as f:
        writer = csv.writer(f)
        writer.writerow(['split_local_index', 'subject', 'task'])
        writer.writerows(zip(selected, subjects, tasks))
    metadata = {'checkpoint': str(a.checkpoint.resolve()), 'epoch': ck.get('epoch'),
        'checkpoint_sha256': hashlib.file_digest(a.checkpoint.open('rb'), 'sha256').hexdigest(),
        'checkpoint_args': original_args, 'cli': vars(a), 'split': a.split, 'samples': n, 'sample_mode': a.sample_mode, 'views': a.views,
        'sample_seed': a.sample_seed, 'tsne_seed': 1968125571, 'perplexity': 30,
        'device_used': str(device), 'features': {},
        'training_fullchannel': bool(getattr(train_args, 'fullchannel', False)),
        'target_channels': proto['ch_names'],
        'observed_channels': observed_channels,
        'correction_definition': 'Raw branch output; no tanh or correction_scale multiplication',
        'note': 'Same frozen checkpoint (default: this experiment Stage1 best), eval/no_grad; full_all/full_missing/few_missing differ in input or selected positions, not checkpoint. Independent t-SNE fits; axes across views are not comparable.'}
    coordinates = {}
    for key, x in features.items():
        print(f'{key}: {x.shape} -> t-SNE', flush=True)
        info = {'shape': list(x.shape), 'pca_dim': None}
        if '_d_' in key and a.d_pca_dim:
            dim = min(a.d_pca_dim, n-1, x.shape[1])
            reducer = PCA(n_components=dim, svd_solver='randomized', random_state=42)
            tx = reducer.fit_transform(x)
            info.update(pca_dim=dim, pca_explained_variance=float(reducer.explained_variance_ratio_.sum()))
        else:
            tx = x
        kwargs = dict(n_components=2, perplexity=30, init='pca', learning_rate='auto',
                      random_state=1968125571, n_jobs=a.threads, verbose=1)
        iteration_key = 'max_iter' if 'max_iter' in inspect.signature(TSNE).parameters else 'n_iter'
        kwargs[iteration_key] = a.max_iter
        estimator = TSNE(**kwargs)
        coordinates[key] = estimator.fit_transform(tx)
        if not np.isfinite(coordinates[key]).all() or not np.isfinite(estimator.kl_divergence_):
            raise ValueError(f'Nonfinite t-SNE output: {key}')
        np.save(out / f'{key}_tsne.npy', coordinates[key])
        info['kl_divergence'] = float(estimator.kl_divergence_)
        metadata['features'][key] = info
        (out / 'metadata.json').write_text(json.dumps(metadata, indent=2, default=str))
    np.savez_compressed(out / 'tsne_coordinates.npz', **coordinates, subjects=subjects, tasks=tasks,
                        selected_split_indices=selected)
    sub_ids = sorted(np.unique(subjects))
    sub_colors = dict(zip(sub_ids, plt.get_cmap('nipy_spectral')(np.linspace(0, 1, len(sub_ids)))))
    task_colors = reference['TASK_COLORS']
    handles_sub = [Line2D([], [], marker='o', linestyle='', color=sub_colors[v], label=f'Subject {v}') for v in sub_ids]
    handles_task = [Line2D([], [], marker='o', linestyle='', color=task_colors[int(v)],
                          label=reference['TASK_NAMES'][int(v)]) for v in sorted(np.unique(tasks))]
    titles = {'full_all': f'Full input: all {n_full} positions',
              'full_missing': f'Full input: matched missing {n_missing} positions',
              'few_missing': f'{n_obs} real + {n_missing} prototypes: missing {n_missing} positions'}
    if a.views != 'all':
        titles = {'full_all': titles['full_all']}
    def panels(axes, view, kind):
        for axis, (branch, label) in zip(axes, [('sub','subject'),('sub','task'),('task','task'),('task','subject')]):
            key = f'{view}_{kind}_{branch}'
            vals, colors = (subjects, sub_colors) if label == 'subject' else (tasks, task_colors)
            xy = coordinates[key]
            axis.scatter(xy[:,0], xy[:,1], c=[colors[int(v)] for v in vals], s=3, alpha=.65, linewidths=0, rasterized=True)
            axis.set_title(f'{kind}_{branch} ({features[key].shape[1]}D), by {label}', fontsize=10)
            axis.set_xticks([]); axis.set_yticks([])
    def save(fig, name):
        fig.legend(handles=handles_sub, title='Subject', loc='upper right', bbox_to_anchor=(.998,.90), fontsize=8)
        fig.legend(handles=handles_task, title='Task', loc='lower right', bbox_to_anchor=(.998,.07), fontsize=8)
        fig.subplots_adjust(left=.03, right=.83, bottom=.04, top=.90, wspace=.10, hspace=.28)
        fig.savefig(out / f'{name}.png', dpi=180)
        fig.savefig(out / f'{name}.pdf', dpi=180)
        plt.close(fig)
    for kind in ('z', 'd'):
        for view in titles:
            fig, axes = plt.subplots(2, 2, figsize=(18,12))
            panels(axes.flat, view, kind)
            fig.suptitle(f'{titles[view]} | same checkpoint {a.checkpoint.parent.name} epoch {ck.get("epoch")} | {a.split.capitalize()} n={n}')
            save(fig, f'{view}_{kind}_tsne')
        if a.views != 'all':
            continue
        fig, axes = plt.subplots(3,4,figsize=(24,15))
        for row, view in enumerate(titles):
            panels(axes[row], view, kind)
            axes[row,0].set_ylabel(titles[view], fontsize=9)
        fig.suptitle(f'{kind}: full vs few input | same frozen checkpoint {a.checkpoint.parent.name} epoch {ck.get("epoch")} | {a.split.capitalize()} n={n}')
        save(fig, f'comparison_{kind}_tsne')
    print(f'Done: {out}', flush=True)

if __name__ == '__main__':
    main()
