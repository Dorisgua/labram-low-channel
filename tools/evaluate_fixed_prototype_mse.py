#!/usr/bin/env python3
"""Full-split fixed Prototype MSE in the reference D checkpoint's CNN space."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import torch
from torch.utils.data import DataLoader
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from Channels_definition import ERPCORE_12_CHANNELS
from data_processor.erpcore import prepare_ERPCORE_pt_dataset
from run_dynamic_stage1 import get_models, DATASET_CONFIGS
import utils

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--checkpoint', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--batch-size', type=int, default=128)
a = p.parse_args()
torch.set_num_threads(4)
c = torch.load(a.checkpoint, map_location='cpu')
args = c['args']
config=DATASET_CONFIGS[args.dataset]
channels=config['ch_names'][args.channel_subset]
kwargs={k:getattr(args,v) for k,v in config.get('prepare_kwargs_from_args',{}).items()}
if config.get('pass_channel_names'): kwargs['channel_names']=channels
train,test,val=config['prepare_fn'](args.data_path,**kwargs)
del train,val
model = get_models(args)
model.load_state_dict(c['model'], strict=True)
model.completion_scope = args.completion_scope
model.real_input_chans_index = [int(i) for i in utils.get_input_chans(channels)]
prototype = torch.load(args.channel_prototype_path, map_location='cpu')
model.target_input_chans_index = [int(i) for i in prototype['input_chans_index']]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device).eval().requires_grad_(False)
_, missing = model._dynamic_channel_indices(device)
pred = prototype['channel_prototypes'].to(device).index_select(0, missing)[None, :, None, :]
result = {'reference_checkpoint': str(a.checkpoint.resolve()), 'prototype': args.channel_prototype_path, 'normalization': args.norm_method, 'input_scale': float(args.input_scale), 'missing_channels': [prototype['ch_names'][i] for i in missing.tolist()], 'precision': 'CUDA autocast (same as Stage1 evaluation); float32 squared error', 'splits': {}}
for split, dataset in [('test',test)]:
    total, n = 0.0, 0
    with torch.inference_mode():
        for batch in DataLoader(dataset,batch_size=a.batch_size,num_workers=4):
            x=batch[3].to(device).float()*float(args.input_scale)
            x=x.reshape(x.shape[0],x.shape[1],-1,200)
            with torch.cuda.amp.autocast(enabled=device.type=='cuda'):
                target=model._patch_tokens(x).index_select(1,missing)
                mse=(pred-target).square().mean()
            total+=float(mse)*len(x);n+=len(x)
    result['splits'][split]={'samples':n,'mse':total/n}
    print(split,result['splits'][split],flush=True)
for key,path in [('checkpoint_sha256',a.checkpoint),('prototype_sha256',Path(args.channel_prototype_path))]:
    with path.open('rb') as f: result[key]=hashlib.file_digest(f,'sha256').hexdigest()
a.output.parent.mkdir(parents=True,exist_ok=True)
a.output.write_text(json.dumps(result,indent=2)+'\n')
