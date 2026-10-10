#!/usr/bin/env python3
"""直接运行画全部验证样本；--split train/test 可切换。默认 CPU，不占训练 GPU。"""

"""
export MAMBA_ROOT_PREFIX=/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/micromamba-root
                         
eval "$(/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/bin/micromamba shell hook -s bash)"                                                                 
                                 
micromamba activate labram

python plot_stage1_disentanglement.py --split test
"""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '2'
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
from torch.utils.data import DataLoader

RUN = Path(__file__).resolve().parent
ROOT = next(p for p in RUN.parents if (p / 'run_dynamic_stage1.py').is_file())
sys.path.insert(0, str(ROOT))
from run_dynamic_stage1 import get_models
from data_processor.shu import SHUCrossSubjectLoader
from Channels_definition import SHU_13_CHANNELS
import utils


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='val')
    parser.add_argument('--checkpoint', type=Path, default=RUN/'stage1/checkpoint-best.pth')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=32)
    cli = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(0)
    ck = torch.load(cli.checkpoint, map_location='cpu', weights_only=False)
    args = ck['args']
    if args.dataset.upper() != 'SHU' or args.completion_scope != 'shu13_with_shu32':
        raise ValueError('需要 SHU 13→32 Stage1 checkpoint')
    model = get_models(args)
    old_checkpoint = 'corrector.time_embedding.weight' not in ck['model']
    if old_checkpoint:
        # 旧训练没有时间编码，补零才能保持它当时的行为。
        ck['model']['corrector.time_embedding.weight'] = torch.zeros_like(model.corrector['time_embedding'].weight)
    model.load_state_dict(ck['model'], strict=True)
    def resolve(path):
        path = Path(path)
        return path if path.is_absolute() else ROOT/path
    proto_path = resolve(args.channel_prototype_path)
    proto = torch.load(proto_path, map_location='cpu', weights_only=False)
    model.shu32_channel_prototypes.copy_(proto['channel_prototypes'])
    model.completion_scope = args.completion_scope
    model.real_input_chans_index = list(map(int, utils.get_input_chans(SHU_13_CHANNELS)))
    model.target_input_chans_index = list(map(int, proto['input_chans_index']))
    ds = SHUCrossSubjectLoader(resolve(args.data_path)/f'{cli.split}.json',
        sampling_rate=args.sampling_rate, normalize_method=args.norm_method, channel_names=SHU_13_CHANNELS)
    if [x.upper() for x in proto['ch_names']] != ds.manifest_channel_names:
        raise ValueError('原型和数据的完整导联顺序不一致')
    model.to(cli.device).eval()
    features = {'d_sub': [], 'd_task': []}
    subjects, tasks = [], []
    errors = {'dynamic_mse': 0., 'prototype_mse': 0.}
    print(f'提取 {cli.split} 全部 {len(ds)} 个样本；旧checkpoint补零时间编码={old_checkpoint}', flush=True)
    with torch.inference_mode():
        for i, batch in enumerate(DataLoader(ds, batch_size=cli.batch_size, shuffle=False, num_workers=0)):
            obs, full = batch[2].to(cli.device), batch[3].to(cli.device)
            r = model.forward_stage1(obs.reshape(len(obs),13,-1,200)*args.input_scale,
                                    full.reshape(len(full),32,-1,200)*args.input_scale)
            for key in features:
                features[key].append(r[key].flatten(1).cpu().numpy())
            subjects.append(batch[4].numpy()); tasks.append(batch[5].numpy())
            for key, pred in [('dynamic_mse','h_pred_miss'), ('prototype_mse','p_miss')]:
                errors[key] += float((r[pred]-r['h_miss_target']).square().mean())*len(obs)
            if (i+1)%10 == 0: print(f'已提取 {min((i+1)*cli.batch_size,len(ds))}/{len(ds)}',flush=True)
    features = {k:np.concatenate(v) for k,v in features.items()}
    subjects, tasks = np.concatenate(subjects), np.concatenate(tasks)
    coords = {}
    for key, x in features.items():
        reduced = PCA(n_components=min(50,len(x)-1,x.shape[1]),svd_solver='randomized',random_state=0).fit_transform(x)
        coords[key] = TSNE(n_components=2,perplexity=min(30,(len(x)-1)/3),init='pca',learning_rate='auto',random_state=0).fit_transform(reduced)
        print(f'{key} t-SNE 完成',flush=True)
    fig, axes = plt.subplots(1,4,figsize=(20,5),constrained_layout=True)
    for ax, (key, labels, label_name) in zip(axes,[('d_sub',subjects,'subject'),('d_sub',tasks,'task'),('d_task',tasks,'task'),('d_task',subjects,'subject')]):
        values=np.unique(labels)
        colors=plt.get_cmap('gist_rainbow' if label_name=='subject' else 'tab10')
        xy=coords[key]
        for j,value in enumerate(values):
            ix=labels==value
            color=colors(j/max(1,len(values)-1)) if label_name=='subject' else colors(j)
            ax.scatter(xy[ix,0],xy[ix,1],s=4,alpha=.65,color=color,label=str(value),linewidths=0)
        ax.set_title(f'{key} by {label_name}');ax.set_xticks([]);ax.set_yticks([])
        ax.legend(fontsize=6,ncol=2 if len(values)>10 else 1,markerscale=2)
    fig.suptitle(f'SHU {cli.split}: all {len(ds)} trials | 13 observed + 19 missing | PCA50 + t-SNE (seed=0)')
    out=RUN/'stage1_disentanglement'/cli.split
    out.mkdir(parents=True,exist_ok=True)
    fig.savefig(out/'disentanglement.png',dpi=180);plt.close(fig)
    np.savez_compressed(out/'embedding.npz',d_sub_xy=coords['d_sub'],d_task_xy=coords['d_task'],subject=subjects,task=tasks)
    metadata={'checkpoint':str(cli.checkpoint.resolve()),'checkpoint_sha256':hashlib.sha256(cli.checkpoint.read_bytes()).hexdigest(),
              'prototype':str(proto_path),'prototype_sha256':hashlib.sha256(proto_path.read_bytes()).hexdigest(),
              'split':cli.split,'samples':len(ds),'subjects':np.unique(subjects).tolist(),
              'task_counts':{int(k):int((tasks==k).sum()) for k in np.unique(tasks)},
              'norm_method':args.norm_method,'input_scale':args.input_scale,'legacy_time_embedding_zero':old_checkpoint,
              **{k:v/len(ds) for k,v in errors.items()},
              'note':'Independent t-SNE per branch; same coordinates for two label colorings. Clustering alone does not establish disentanglement. SHU val shares train subjects; test uses unseen subjects.'}
    (out/'metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n')
    print('保存到:',out,flush=True)


if __name__ == '__main__':
    main()
