"""Checkpoint-faithful BCI audit; CPU extraction, all splits, fixed plot seed."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from run_dynamic_stage1 import get_models
from data_processor.bciiv2a_cbramod import prepare_BCIIV2A_cbramod_dataset
from Channels_definition import BCIIV2A_13_CHANNELS
import utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    cli = parser.parse_args()
    out = cli.run / 'disentanglement_audit'
    out.mkdir(exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(0)
    checkpoint = cli.run / 'stage1/checkpoint-best.pth'
    ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
    args = ck['args']
    assert args.dataset == 'bciiv2a_cbramod'
    model = get_models(args)
    # 历史 checkpoint 没有时间编码；补零可复现当时的无位置编码行为。
    ck['model'].setdefault('corrector.time_embedding.weight',
                           torch.zeros_like(model.corrector['time_embedding'].weight))
    model.load_state_dict(ck['model'], strict=True)
    # Prototype buffers are nonpersistent: restore the exact training artifact.
    proto = torch.load(args.channel_prototype_path, map_location='cpu', weights_only=False)
    model.bciiv2a22_channel_prototypes.copy_(proto['channel_prototypes'])
    model.completion_scope = args.completion_scope
    model.real_input_chans_index = list(map(int, utils.get_input_chans(BCIIV2A_13_CHANNELS)))
    model.target_input_chans_index = list(map(int, proto['input_chans_index']))
    model.eval()
    train, test, val = prepare_BCIIV2A_cbramod_dataset(
        args.data_path, sampling_rate=args.sampling_rate,
        normalize_method=args.norm_method, channel_names=BCIIV2A_13_CHANNELS, preload=False)
    report = {'checkpoint': str(checkpoint.resolve()), 'epoch': ck['epoch'],
              'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              'prototype_path': args.channel_prototype_path,
              'prototype_sha256': hashlib.sha256(Path(args.channel_prototype_path).read_bytes()).hexdigest(),
              'args': vars(args), 'splits': {},
              'notes': 'All samples. PCA50 then independent t-SNE per branch/split, seed0. Linear task probe fits train only; no test tuning. Subject identity is disjoint across splits.'}
    arrays = {}
    for split, ds in [('train', train), ('val', val), ('test', test)]:
        chunks = {k: [] for k in ['d_sub', 'd_task', 'target', 'pred', 'proto', 'subject', 'task']}
        with torch.inference_mode():
            for batch in DataLoader(ds, batch_size=64, num_workers=2):
                obs, full = batch[2], batch[3]
                obs = obs.reshape(len(obs), 13, -1, 200) * args.input_scale
                full = full.reshape(len(full), 22, -1, 200) * args.input_scale
                r = model.forward_stage1(obs, full)
                for k, name in [('d_sub','d_sub'), ('d_task','d_task'), ('target','h_miss_target'), ('pred','h_pred_miss'), ('proto','p_miss')]:
                    chunks[k].append(r[name].flatten(1).numpy().copy())
                chunks['subject'].append(batch[4].numpy())
                chunks['task'].append(batch[5].numpy())
        a = {k: np.concatenate(v) for k, v in chunks.items()}
        arrays[split] = a
        mse = lambda x: float(np.mean((x-a['target'])**2))
        sub, task = a['d_sub'], a['d_task']
        metrics = {'n':len(ds), 'subjects':np.unique(a['subject']).tolist(),
                   'prototype_mse':mse(a['proto']), 'dynamic_mse':mse(a['pred']),
                   'branch_cosine':float(np.mean(np.sum(sub*task,1)/(np.linalg.norm(sub,axis=1)*np.linalg.norm(task,axis=1)))),
                   'rms': {k:float(np.sqrt(np.mean(a[k]**2))) for k in ['d_sub','d_task','target','pred']},
                   'sample_std_rms': {k:float(np.sqrt(np.mean(np.var(a[k],axis=0)))) for k in ['d_sub','d_task','target','pred']}}
        report['splits'][split] = metrics
        metrics['temporal_patch_max_difference'] = {}
        for key in ['d_sub', 'd_task', 'pred', 'target']:
            temporal = a[key].reshape(len(ds), 9, 4, 200)
            metrics['temporal_patch_max_difference'][key] = float(np.max(np.abs(temporal-temporal[:, :, :1])))
        print(split, metrics, flush=True)
        fig, axes = plt.subplots(1,4,figsize=(18,4),constrained_layout=True)
        for j,k in enumerate(['d_sub','d_task']):
            reduced = PCA(n_components=50,random_state=0,svd_solver='randomized').fit_transform(a[k])
            xy = TSNE(n_components=2,perplexity=30,init='pca',learning_rate='auto',random_state=0).fit_transform(reduced)
            for offset,label in enumerate(['subject','task']):
                ax=axes[j*2+offset]
                for value in np.unique(a[label]):
                    ix=a[label]==value
                    ax.scatter(xy[ix,0],xy[ix,1],s=3,alpha=.65,label=str(value),rasterized=True)
                ax.set_title(f'{k} by {label}')
                ax.set_xticks([]); ax.set_yticks([]); ax.legend(markerscale=3,fontsize=7)
        fig.suptitle(f'BCI-IV-2a {split}, n={len(ds)} | Stage1 epoch {ck["epoch"]} | PCA50 + t-SNE')
        fig.savefig(out/f'{split}_disentanglement.png',dpi=170); plt.close(fig)
        print('Plotted',split,flush=True)
    # Fixed probe setup; report both branches, not only the desired one.
    for k in ['d_sub','d_task']:
        probe=make_pipeline(PCA(n_components=50,random_state=0,svd_solver='randomized'),StandardScaler(),LogisticRegression(C=1,max_iter=1500))
        probe.fit(arrays['train'][k],arrays['train']['task'])
        for split in arrays:
            report['splits'][split][k+'_task_probe_bacc']=float(balanced_accuracy_score(arrays[split]['task'],probe.predict(arrays[split][k])))
    logs=[json.loads(s) for s in (cli.run/'stage1/log.txt').read_text().splitlines()]
    fig,ax=plt.subplots(1,3,figsize=(15,4),constrained_layout=True)
    for split in ['train','val','test']:
        ax[0].plot([r['epoch'] for r in logs],[r[split+'_loss_subject_correction_contra'] for r in logs],label=split)
        ax[1].plot([r['epoch'] for r in logs],[r[split+'_loss_missing'] for r in logs],label=split)
    ax[0].axhline(np.log(5),ls='--',color='gray',label='ln5 (train)')
    ax[0].axhline(np.log(2),ls=':',color='gray',label='ln2 (val/test)')
    ax[0].set_title('Subject correction contrastive loss'); ax[0].legend()
    ax[1].set_yscale('log'); ax[1].set_title('Missing-token MSE'); ax[1].legend()
    x=np.arange(3)
    for i,k in enumerate(['prototype_mse','dynamic_mse']):
        ax[2].bar(x+(i-.5)*.35,[report['splits'][s][k] for s in arrays],width=.35,label=k)
    ax[2].set_xticks(x,list(arrays)); ax[2].set_title('Best checkpoint reconstruction'); ax[2].legend()
    fig.savefig(out/'loss_and_reconstruction.png',dpi=170); plt.close(fig)
    # Fixed first validation example; no selection for visual quality.
    a=arrays['val']; fig,ax=plt.subplots(2,1,figsize=(12,6),constrained_layout=True)
    for k in ['target','proto','pred']:
        ax[0].plot(a[k][0,:200],label=k,alpha=.8)
    for k in ['d_sub','d_task']:
        ax[1].plot(a[k][0,:200],label=k,alpha=.8)
    ax[0].set_title('First validation trial, first missing channel/patch: 200 CNN features')
    ax[1].set_title('Raw branch corrections (not EEG waveform)')
    for axis in ax: axis.legend()
    fig.savefig(out/'completion_example.png',dpi=170); plt.close(fig)
    (out/'metrics.json').write_text(json.dumps(report,indent=2,default=str)+'\n')
    print('Saved',out,flush=True)


if __name__=='__main__': main()
