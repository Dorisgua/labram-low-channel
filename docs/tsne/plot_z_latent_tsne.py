#!/usr/bin/env python3

"""
export MAMBA_ROOT_PREFIX=/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/micromamba-root
                         
eval "$(/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/bin/micromamba shell hook -s bash)"                                                                 
                                 
micromamba activate labram
"""
"""t-SNE for mean-pooled d_sub and d_task from Dynamic Stage 1."""
from pathlib import Path
import argparse, sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "p_miss_add_Lmissingmae_delta_bs64_lr5e4_epoch200_missingw2"
CKPT = REPO_ROOT / "outputs/erpcore/erp_core_D_stage1/p_miss_add_Lmissingmae_delta_bs64_lr5e4_epoch200_missingw2/checkpoint-best.pth"
TASK_NAMES = {0:"ERN/Incorrect",1:"ERN/Correct",2:"LRP/Contralateral",3:"LRP/Ipsilateral",4:"MMN/Deviants",5:"MMN/Standards",6:"N2pc/Contralateral",7:"N2pc/Ipsilateral",8:"N400/Unrelated",9:"N400/Related",10:"P3/Rare",11:"P3/Frequent"}
# Saturated, shared hues for the two tasks in each ERP family.
TASK_COLORS = {
    0: "#1479D1", 1: "#73B7F2",       # ERN: blue
    2: "#F07818", 3: "#FFB15C",       # LRP: orange
    4: "#159447", 5: "#78C86A",       # MMN: green
    6: "#D62828", 7: "#F18181",       # N2pc: red
    8: "#7441A8", 9: "#B28BD0",       # N400: purple
    10: "#8C564B", 11: "#D59B8B",      # P3: brown/pink
}

def args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, default=CKPT)
    p.add_argument("--split", choices=("train", "val", "test"), default="test")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--max-samples", type=int, default=999999)
    p.add_argument("--max-iter", type=int, default=1000)
    return p.parse_args()

def build_dynamic_model(a, state_dict, ch_names, device):
    import utils
    from run_dynamic_stage1 import get_models
    model = get_models(a); model.load_state_dict(state_dict, strict=True)
    model.completion_scope = a.completion_scope; model.pooling_scope = a.pooling_scope
    proto = torch.load(a.channel_prototype_path, map_location="cpu")
    model.erpcore28_channel_prototypes.copy_(proto["channel_prototypes"])
    model.target_input_chans_index = [int(v) for v in proto["input_chans_index"]]
    model.real_input_chans_index = [int(v) for v in utils.get_input_chans(ch_names)]
    return model.to(device).eval()

def mean_dynamic_latents(model, x_obs, input_scale, device):
    x_obs = x_obs.float().to(device) * float(input_scale)
    b, c, length = x_obs.shape
    x_obs = x_obs.reshape(b, c, length // model.patch_size, model.patch_size)
    out = model._encode_dynamic_tokens(model._patch_tokens(x_obs))
    dims = tuple(range(1, out["d_sub"].ndim - 1))
    return out["d_sub"].mean(dim=dims), out["d_task"].mean(dim=dims)

def main():
    cli = args(); sys.path.insert(0, str(REPO_ROOT))
    from run_dynamic_stage1 import get_dataset
    ck = torch.load(cli.checkpoint, map_location="cpu"); a = ck["args"]
    train, test, val, ch_names, _ = get_dataset(a)
    dataset = {"train": train, "val": val, "test": test}[cli.split]
    rng = np.random.default_rng(42); n = min(cli.max_samples, len(dataset)); all_i = np.arange(len(dataset)); sel = np.sort(rng.choice(all_i, n, replace=False))
    device = torch.device(cli.device if torch.cuda.is_available() else "cpu")
    model = build_dynamic_model(a, ck["model"], ch_names, device)
    loader = DataLoader(Subset(dataset, sel.tolist()), batch_size=cli.batch_size, shuffle=False, num_workers=cli.num_workers, pin_memory=device.type == "cuda")
    zs, zt, subjects, tasks = [], [], [], []
    with torch.no_grad():
        for _, _, x_obs, _, subject, task in tqdm(loader, desc=f"Extracting {cli.split} latents", unit="batch"):
            d_sub, d_task = mean_dynamic_latents(model, x_obs, a.input_scale, device)
            zs.append(d_sub.cpu().numpy()); zt.append(d_task.cpu().numpy())
            subjects.append(subject.numpy()); tasks.append(task.numpy())
    zs, zt = np.concatenate(zs), np.concatenate(zt)
    subjects, tasks = np.concatenate(subjects), np.concatenate(tasks)
    coords = {}
    for name, x in (("z_sub", zs), ("z_task", zt)):
        print(f"{name}: {x.shape} -> t-SNE", flush=True)
        coords[name] = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto", max_iter=cli.max_iter, random_state=1968125571, n_jobs=-1, verbose=1).fit_transform(x)
    colors_sub = plt.get_cmap("nipy_spectral")(np.linspace(0, 1, len(np.unique(subjects))))
    sub_map = {v: colors_sub[i] for i,v in enumerate(sorted(np.unique(subjects)))}
    task_map = {v: TASK_COLORS[int(v)] for v in sorted(np.unique(tasks))}
    fig, ax = plt.subplots(2, 2, figsize=(20, 16))
    panels = [("z_sub", subjects, sub_map, "mean(d_sub), colored by subject"), ("z_sub", tasks, task_map, "mean(d_sub), colored by task"), ("z_task", tasks, task_map, "mean(d_task), colored by task"), ("z_task", subjects, sub_map, "mean(d_task), colored by subject")]
    for axis, (latent, vals, cmap, title) in zip(ax.flat, panels):
        axis.scatter(coords[latent][:,0], coords[latent][:,1], c=[cmap[int(v)] for v in vals], s=8, alpha=.75, linewidths=0); axis.set_title(title); axis.set_xticks([]); axis.set_yticks([])
    subject_handles = [Line2D([0], [0], marker="o", linestyle="", markersize=6, color=sub_map[v], label=f"Subject {v}") for v in sorted(sub_map)]
    task_handles = [Line2D([0], [0], marker="o", linestyle="", markersize=6, color=task_map[v], label=TASK_NAMES[v]) for v in sorted(task_map)]
    fig.legend(handles=subject_handles, title="Subject", loc="upper right", bbox_to_anchor=(0.995, 0.78), frameon=False)
    fig.legend(handles=task_handles, title="ERP task", loc="lower right", bbox_to_anchor=(0.995, 0.38), frameon=False)
    fig.subplots_adjust(left=0.03, right=0.84, bottom=0.04, top=0.93, wspace=0.08, hspace=0.22)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "d_latent_mean_tsne.png", dpi=220)
    np.savez_compressed(OUT / "d_latent_mean_tsne.npz", d_sub_mean=zs, d_task_mean=zt, d_sub_tsne=coords["z_sub"], d_task_tsne=coords["z_task"], subjects=subjects, tasks=tasks)
    print(f"saved: {OUT / 'd_latent_mean_tsne.png'}")
if __name__ == "__main__": main()
