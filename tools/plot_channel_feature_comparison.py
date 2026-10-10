#!/usr/bin/env python3
"""Compare true CNN tokens, fixed prototypes and dynamic predictions on first samples."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--channel', required=True)
    p.add_argument('--group-by', choices=['subject', 'class'], default='subject')
    p.add_argument('--split', choices=['train', 'val', 'test'], default='test')
    p.add_argument('--patch-index', type=int, default=0)
    p.add_argument('--output-dir', type=Path)
    p.add_argument('--device', default='cpu')
    p.add_argument('--threads', type=int, default=4)
    a = p.parse_args()
    if a.patch_index < 0 or a.threads < 1: p.error('Invalid patch index or thread count')
    import numpy as np
    import torch
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from run_dynamic_stage1 import get_models, DATASET_CONFIGS, _validate_completion_prototype
    import utils
    torch.set_num_threads(a.threads)
    c = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    cfg = c['args']
    config = DATASET_CONFIGS[cfg.dataset]
    channels = config['ch_names']
    if isinstance(channels, dict): channels = channels[cfg.channel_subset]
    kwargs = {k:getattr(cfg,v) for k,v in config.get('prepare_kwargs_from_args',{}).items()}
    if config.get('pass_channel_names'): kwargs['channel_names'] = channels
    train,test,val = config['prepare_fn'](cfg.data_path or config['root'], **kwargs)
    dataset = {'train':train,'test':test,'val':val}[a.split]
    del train,test,val
    proto = torch.load(cfg.channel_prototype_path, map_location='cpu', weights_only=False)
    targets = proto['ch_names']
    target_indices = [int(i) for i in proto['input_chans_index']]
    _validate_completion_prototype(cfg,channels,targets,target_indices,proto['channel_prototypes'])
    if a.channel not in targets or a.channel in channels:
        p.error('Channel must be a missing channel in this checkpoint configuration')
    if hasattr(dataset,'full_channel_names') and list(dataset.full_channel_names) != list(targets):
        raise ValueError('Full input and prototype channel orders differ')
    model = get_models(cfg)
    model.load_state_dict(c['model'],strict=True)
    model.completion_scope = cfg.completion_scope
    model.pooling_scope = getattr(cfg,'pooling_scope','real')
    model.real_input_chans_index = list(utils.get_input_chans(channels))
    model.target_input_chans_index = target_indices
    device = torch.device(a.device if torch.cuda.is_available() else 'cpu')
    model.to(device).eval().requires_grad_(False)
    # Always predict from observed channels, even if this checkpoint trained on full input.
    model.fullchannel = False
    groups = getattr(dataset,'subject_indices' if a.group_by=='subject' else 'task_indices',None)
    if not groups:
        raise ValueError(f'Dataset has no verified {a.group_by} index map; select another grouping')
    selected = [(int(group),int(indices[0])) for group,indices in sorted(groups.items()) if len(indices)]
    records, arrays = [], []
    with torch.inference_mode():
        for group,index in selected:
            batch = dataset[index]
            scale = float(getattr(cfg,'input_scale',config.get('input_scale',0.01)))
            obs,full = [batch[i].float().reshape(1,batch[i].shape[0],-1,200).to(device)*scale for i in (2,3)]
            result = model.forward_stage1(obs,full)
            position = result['miss_indices'].tolist().index(targets.index(a.channel))
            if a.patch_index >= result['h_pred_miss'].shape[2]: p.error('Patch index exceeds sample patch count')
            values = [result[key][0,position,a.patch_index].cpu().numpy() for key in ('h_miss_target','p_miss','h_pred_miss')]
            if not all(np.isfinite(v).all() for v in values): raise ValueError('Nonfinite features')
            target,prototype,prediction = values
            records.append(dict(group=group,split_local_index=index,task=int(batch[1]),
                prototype_mse=float(np.mean((prototype-target)**2)),dynamic_mse=float(np.mean((prediction-target)**2))))
            arrays.append(values)
    if not records: raise ValueError('No selected samples')
    arrays = np.asarray(arrays)
    low,high = float(arrays.min()),float(arrays.max())
    pad = max((high-low)*.06,1e-6)
    out = a.output_dir or a.checkpoint.parent / f'{a.split}_{a.channel.lower()}_first_trial_by_{a.group_by}'
    out.mkdir(parents=True,exist_ok=True)
    def draw(axes,record,values):
        for ax,estimate,label,color,metric in zip(axes,values[1:],['Static prototype','Dynamic prediction'],['#777777','#e68122'],['prototype_mse','dynamic_mse']):
            ax.plot(values[0],color='#2676b8',label=f'True {a.channel} token')
            ax.plot(estimate,color=color,label=label)
            ax.set(title=f"{a.group_by} {record['group']} | MSE {record[metric]:.6f}",ylim=(low-pad,high+pad),xlabel='CNN feature dimension (not time)')
            ax.legend(fontsize=8)
    for record,values in zip(records,arrays):
        fig,axes = plt.subplots(1,2,figsize=(14,4))
        draw(axes,record,values)
        fig.tight_layout();fig.savefig(out/f"{a.group_by}_{record['group']}_{a.channel}.png",dpi=180);plt.close(fig)
    fig,axes = plt.subplots(len(records),2,figsize=(16,3*len(records)),squeeze=False)
    for axes_row,record,values in zip(axes,records,arrays): draw(axes_row,record,values)
    fig.suptitle(f'{cfg.dataset} {a.split} | {a.channel}, patch {a.patch_index} | first sample per {a.group_by}')
    fig.tight_layout()
    for suffix in ['png','pdf']:fig.savefig(out/f'all_{a.group_by}s_{a.channel}_comparison.{suffix}',dpi=160)
    plt.close(fig)
    with (out/'samples_and_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    np.savez(out/f'{a.channel}_features.npz',target=arrays[:,0],prototype=arrays[:,1],prediction=arrays[:,2])
    with a.checkpoint.open('rb') as f: digest=hashlib.file_digest(f,'sha256').hexdigest()
    metadata=dict(checkpoint=str(a.checkpoint.resolve()),checkpoint_sha256=digest,checkpoint_args=vars(cfg),
        cli=vars(a),records=records,selection='First retained sample in loader order per group; no error-based selection',
        prediction_input='Observed channels only; full EEG is target only',device=str(device),autocast=False)
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2,default=str))
    print(f'Saved: {out}')

if __name__=='__main__':main()
