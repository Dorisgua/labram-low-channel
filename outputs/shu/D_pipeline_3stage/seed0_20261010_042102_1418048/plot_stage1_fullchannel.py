#!/usr/bin/env python3
"""Full-test SHU latent visualization with all 32 real channels."""
import argparse,csv,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
REPO=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(REPO))
from run_dynamic_stage1 import get_models
from data_processor.shu import prepare_SHU_cross_subject_dataset
from Channels_definition import SHU_13_CHANNELS
import utils
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--checkpoint',type=Path,default=Path(__file__).parent/'stage1/checkpoint-best.pth')
p.add_argument('--output-dir',type=Path,default=Path(__file__).parent/'stage1/disentanglement_fullchannel_test_all')
p.add_argument('--split', choices=['train','test'], default='test')
p.add_argument('--num-workers', type=int, default=4)
p.add_argument('--device',default='cuda')
p.add_argument('--seed',type=int,default=42)
p.add_argument('--batch-size',type=int,default=64)
a=p.parse_args();torch.set_num_threads(4)
if a.split == 'train' and a.output_dir == Path(__file__).parent/'stage1/disentanglement_fullchannel_test_all':
 a.output_dir=Path(__file__).parent/'stage1/disentanglement_fullchannel_train_all'
a.output_dir.mkdir(parents=True,exist_ok=True)
c=torch.load(a.checkpoint,map_location='cpu');cfg=c['args']
train,test,val=prepare_SHU_cross_subject_dataset(cfg.data_path,sampling_rate=cfg.sampling_rate,normalize_method=cfg.norm_method,channel_names=SHU_13_CHANNELS)
dataset=train if a.split=='train' else test
del val
m=get_models(cfg);m.load_state_dict(c['model'],strict=True);m.completion_scope=cfg.completion_scope
proto=torch.load(cfg.channel_prototype_path,map_location='cpu')
m.real_input_chans_index=list(utils.get_input_chans(SHU_13_CHANNELS));m.target_input_chans_index=[int(i) for i in proto['input_chans_index']]
dev=torch.device(a.device);m.to(dev).eval().requires_grad_(False)
features={'z_sub':[],'z_task':[]};subjects=[];tasks=[]
with torch.inference_mode():
 for batch in DataLoader(dataset,batch_size=a.batch_size,num_workers=a.num_workers,shuffle=False):
  x=batch[3].float().to(dev)*float(getattr(cfg,'input_scale',1.0));x=x.reshape(x.shape[0],32,4,200)
  o=m._encode_dynamic_tokens(m._patch_tokens(x),fullchannel=True)
  for k in features:features[k].append(o[k].float().cpu().numpy())
  subjects.extend(batch[4].tolist());tasks.extend(batch[5].tolist())
  print(f'Extracted {len(tasks)}/{len(dataset)}',flush=True)
subjects=np.asarray(subjects);tasks=np.asarray(tasks)
coords={}
for k,parts in features.items():
 v=np.concatenate(parts);assert len(v)==len(dataset) and np.isfinite(v).all()
 np.save(a.output_dir/f'{k}.npy',v)
 coords[k]=TSNE(n_components=2,perplexity=min(30,(len(v)-1)/3),init='pca',learning_rate='auto',random_state=a.seed).fit_transform(v)
 np.save(a.output_dir/f'{k}_tsne.npy',coords[k]);print(k+' t-SNE complete',flush=True)
with (a.output_dir/'sample_alignment.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['split_index','subject','task','file']);w.writerows((i,int(subjects[i]),int(tasks[i]),dataset.files[i]['file']) for i in range(len(dataset)))
ids=sorted(set(subjects.tolist()));labels=sorted(set(tasks.tolist()))
sc=dict(zip(ids,plt.get_cmap('nipy_spectral')(np.linspace(0,1,len(ids)))))
tc=dict(zip(labels,plt.get_cmap('tab10')(np.arange(len(labels)))))
fig,axes=plt.subplots(2,2,figsize=(14,10))
for ax,(key,by) in zip(axes.flat,[('z_sub','subject'),('z_sub','task'),('z_task','task'),('z_task','subject')]):
 lab=subjects if by=='subject' else tasks;pal=sc if by=='subject' else tc;xy=coords[key]
 ax.scatter(xy[:,0],xy[:,1],c=[pal[int(v)] for v in lab],s=3,alpha=.7,linewidths=0,rasterized=True)
 ax.set_title(f'{key}, colored by {by}');ax.set_xticks([]);ax.set_yticks([])
for pal,title,anchor in [(sc,'Subject',(.99,.78)),(tc,'SHU task',(.99,.32))]:
 handles=[Line2D([],[],marker='o',linestyle='',color=color,label=f'{title} {i}') for i,color in pal.items()]
 fig.legend(handles=handles,title=title,loc='upper right',bbox_to_anchor=anchor,fontsize=8,frameon=False)
fig.suptitle(f'SHU Stage1 | Full 32 real channels | All {a.split} samples n={len(dataset)}')
fig.subplots_adjust(right=.82,wspace=.12,hspace=.20)
for ext in ['png','pdf']:fig.savefig(a.output_dir/f'comparison_z_tsne.{ext}',dpi=200)
plt.close(fig)
meta={'checkpoint':str(a.checkpoint.resolve()),'checkpoint_sha256':hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),'training_fullchannel':bool(getattr(cfg,'fullchannel',False)),'visualization_fullchannel':True,'samples':len(dataset),'split':a.split,'subjects':ids,'tasks':labels,'tsne_seed':a.seed,'normalization':cfg.norm_method,'input_scale':float(getattr(cfg,'input_scale',1.0)),'latent':'z_sub/z_task: mean corrections across all 32 channels and 4 patches','note':'All requested split samples, no subsampling. Independent branch t-SNE; label views within each branch share coordinates. Full-channel diagnostic does not imply checkpoint was trained on full channels.'}
(a.output_dir/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
print('Saved '+str(a.output_dir/'comparison_z_tsne.png'),flush=True)
