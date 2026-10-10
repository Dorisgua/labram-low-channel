#!/usr/bin/env python3
"""Same frozen checkpoint (default: Stage2): full28, full-input missing16, few-input missing16."""
import argparse
import csv
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys

OUT = Path(__file__).resolve().parent
ROOT = next(parent for parent in OUT.parents if (parent / 'run_preexp16_erpcore_cslp.py').is_file())
EXPERIMENT = OUT.parent
CKPT = EXPERIMENT / 'stage2/checkpoint-last.pth'

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--max-samples', type=int, default=99999)
    p.add_argument('--max-iter', type=int, default=1000)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--device', default='cuda')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--d-pca-dim', type=int, default=50, help='0: direct t-SNE of flattened D (slower)')
    p.add_argument('--checkpoint', type=Path, default=CKPT)
    a = p.parse_args()
    if a.max_samples < 32 or a.max_iter < 250 or a.threads < 1 or a.d_pca_dim < 0:
        p.error('Require max-samples >=32, max-iter >=250, threads >=1, d-pca-dim >=0')
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
    from run_preexp16_erpcore_cslp import build_model
    from data_processor.erpcore_cslp import prepare_ERPCORE_cslp_dataset
    # Keep the same labels/colors without depending on another experiment script.
    reference = {'TASK_NAMES': {0: 'ERN/Incorrect', 1: 'ERN/Correct', 2: 'LRP/Contralateral', 3: 'LRP/Ipsilateral', 4: 'MMN/Deviants', 5: 'MMN/Standards', 6: 'N2pc/Contralateral', 7: 'N2pc/Ipsilateral', 8: 'N400/Unrelated', 9: 'N400/Related', 10: 'P3/Rare', 11: 'P3/Frequent'}, 'TASK_COLORS': {0: '#1479D1', 1: '#73B7F2', 2: '#F07818', 3: '#FFB15C', 4: '#159447', 5: '#78C86A', 6: '#D62828', 7: '#F18181', 8: '#7441A8', 9: '#B28BD0', 10: '#8C564B', 11: '#D59B8B'}}
    torch.set_num_threads(a.threads)
    torch.manual_seed(42)
    ck = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    train_args = ck['args']
    original_args = vars(train_args).copy()
    train_args.oracle_missing = False
    train_args.g_sub = train_args.g_task = 1.0
    train_args.gate_mode = 'fixed'
    _, test, _ = prepare_ERPCORE_cslp_dataset(train_args.data_path,
        sampling_rate=train_args.sampling_rate, normalize_method=train_args.norm_method)
    n = min(a.max_samples, len(test))
    selected = np.sort(np.random.default_rng(42).choice(len(test), n, replace=False))
    subjects, tasks = np.asarray(test.subjects)[selected], np.asarray(test.labels)[selected]
    device = torch.device(a.device if torch.cuda.is_available() else 'cpu')
    model = build_model(train_args)
    model.load_state_dict(ck['model'], strict=True)
    model.to(device).eval().requires_grad_(False)
    loader = DataLoader(Subset(test, selected.tolist()), batch_size=a.batch_size,
                        num_workers=a.num_workers, shuffle=False, pin_memory=device.type == 'cuda')
    features = {}
    offset = 0
    # Save raw features incrementally; all views have exactly the same sample order.
    with torch.no_grad():
        for batch in loader:
            mb = {k: (v.float().to(device) * float(train_args.input_scale)
                      if k.startswith('x_') else v.to(device)) for k, v in batch.items()}
            full = model.forward_fullchannel_contrastive(mb)
            few = model.encode_personal(mb)
            values = {}
            for branch in ('sub', 'task'):
                values[f'full28_z_{branch}'] = full[f'z_{branch}']
                values[f'full28_d_{branch}'] = full[f'd_{branch}'].flatten(1)
                values[f'full16_z_{branch}'] = full[f'{branch}_tokens'].index_select(1, model.miss_indices).mean(1)
                values[f'full16_d_{branch}'] = full[f'd_{branch}'].index_select(1, model.miss_indices).flatten(1)
                values[f'few16_z_{branch}'] = few[f'z_{branch}']
                values[f'few16_d_{branch}'] = few[f'd_{branch}'].flatten(1)
            count = len(batch['label'])
            for key, tensor in values.items():
                x = tensor.cpu().numpy()
                if not np.isfinite(x).all():
                    raise ValueError(f'Nonfinite features: {key}')
                if key not in features:
                    features[key] = np.lib.format.open_memmap(OUT / f'{key}.npy', mode='w+',
                                                              dtype='float32', shape=(n, x.shape[1]))
                features[key][offset:offset+count] = x
            offset += count
            print(f'Extracted {offset}/{n}', flush=True)
    assert offset == n
    for x in features.values():
        x.flush()
    with (OUT / 'sample_alignment.csv').open('w') as f:
        writer = csv.writer(f)
        writer.writerow(['test_local_index', 'payload_index', 'subject', 'task'])
        writer.writerows(zip(selected, np.asarray(test.indices)[selected], subjects, tasks))
    metadata = {'checkpoint': str(a.checkpoint.resolve()), 'epoch': ck.get('epoch'),
        'checkpoint_sha256': hashlib.file_digest(a.checkpoint.open('rb'), 'sha256').hexdigest(),
        'checkpoint_args': original_args, 'cli': vars(a), 'split': 'test', 'samples': n,
        'sample_seed': 42, 'tsne_seed': 1968125571, 'perplexity': 30,
        'device_used': str(device), 'features': {},
        'note': 'Same frozen checkpoint (default: this experiment Stage2), eval/no_grad; full28/full16/few16 differ in input or selected positions, not checkpoint. Independent t-SNE fits; axes across views are not comparable.'}
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
        np.save(OUT / f'{key}_tsne.npy', coordinates[key])
        info['kl_divergence'] = float(estimator.kl_divergence_)
        metadata['features'][key] = info
        (OUT / 'metadata.json').write_text(json.dumps(metadata, indent=2, default=str))
    np.savez_compressed(OUT / 'tsne_coordinates.npz', **coordinates, subjects=subjects, tasks=tasks,
                        selected_test_indices=selected)
    sub_ids = sorted(np.unique(subjects))
    sub_colors = dict(zip(sub_ids, plt.get_cmap('nipy_spectral')(np.linspace(0, 1, len(sub_ids)))))
    task_colors = reference['TASK_COLORS']
    handles_sub = [Line2D([], [], marker='o', linestyle='', color=sub_colors[v], label=f'Subject {v}') for v in sub_ids]
    handles_task = [Line2D([], [], marker='o', linestyle='', color=task_colors[int(v)],
                          label=reference['TASK_NAMES'][int(v)]) for v in sorted(np.unique(tasks))]
    titles = {'full28': 'Full input: all 28 positions',
              'full16': 'Full input: matched missing 16 positions',
              'few16': '12 real + 16 prototypes: missing 16 positions'}
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
        fig.legend(handles=handles_task, title='ERP task', loc='lower right', bbox_to_anchor=(.998,.07), fontsize=8)
        fig.subplots_adjust(left=.03, right=.83, bottom=.04, top=.90, wspace=.10, hspace=.28)
        fig.savefig(OUT / f'{name}.png', dpi=180)
        fig.savefig(OUT / f'{name}.pdf', dpi=180)
        plt.close(fig)
    for kind in ('z', 'd'):
        for view in titles:
            fig, axes = plt.subplots(2, 2, figsize=(18,12))
            panels(axes.flat, view, kind)
            fig.suptitle(f'{titles[view]} | same checkpoint {a.checkpoint.parent.name} epoch {ck.get("epoch")} | Test n={n}')
            save(fig, f'{view}_{kind}_tsne')
        fig, axes = plt.subplots(3,4,figsize=(24,15))
        for row, view in enumerate(titles):
            panels(axes[row], view, kind)
            axes[row,0].set_ylabel(titles[view], fontsize=9)
        fig.suptitle(f'{kind}: full vs few input | same frozen checkpoint {a.checkpoint.parent.name} epoch {ck.get("epoch")} | Test n={n}')
        save(fig, f'comparison_{kind}_tsne')
    print(f'Done: {OUT}', flush=True)

if __name__ == '__main__':
    main()
