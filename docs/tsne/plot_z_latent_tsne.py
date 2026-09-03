#!/usr/bin/env python3
"""Fast CSLP-AE-style t-SNE plot for z_sub and z_task."""
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

REPO_ROOT = Path("/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/eeg-main/LabraM-Git-Diff-disengle")
OUT = Path(__file__).resolve().parent
CKPT = REPO_ROOT / "outputs/preexp16_erpcore28_fullchannel_two_contrastive_frozen_cnn/checkpoints/preexp16_erpcore28_fullchannel_two_contrastive_frozen_cnn_seed0_20260817_113200/checkpoint-last.pth"
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
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--max-samples", type=int, default=1500)
    p.add_argument("--max-iter", type=int, default=1000)
    return p.parse_args()

def main():
    cli = args(); sys.path.insert(0, str(REPO_ROOT)); from data_processor.erpcore_cslp import prepare_ERPCORE_cslp_dataset; from run_preexp16_erpcore_cslp import build_model
    ck = torch.load(CKPT, map_location="cpu"); a = ck["args"]
    _, test, _ = prepare_ERPCORE_cslp_dataset(a.data_path, sampling_rate=a.sampling_rate, normalize_method=a.norm_method)
    rng = np.random.default_rng(42); n = min(cli.max_samples, len(test)); all_i = np.arange(len(test)); sel = np.sort(rng.choice(all_i, n, replace=False))
    subjects = np.asarray(test.subjects)[sel]; tasks = np.asarray(test.labels)[sel]
    device = torch.device(cli.device if torch.cuda.is_available() else "cpu")
    a.oracle_missing = False; a.g_sub = 1.0; a.g_task = 1.0; a.gate_mode = "fixed"
    model = build_model(a); model.load_state_dict(ck["model"], strict=True); model.to(device).eval()
    loader = DataLoader(Subset(test, sel.tolist()), batch_size=cli.batch_size, shuffle=False, num_workers=cli.num_workers, pin_memory=device.type == "cuda")
    zs, zt = [], []
    with torch.no_grad():
        for batch in loader:
            mb = {k: (v.float().to(device) * float(a.input_scale) if k.startswith("x_") else v.to(device)) for k,v in batch.items()}
            out = model.forward_fullchannel_contrastive(mb); zs.append(out["z_sub"].cpu().numpy()); zt.append(out["z_task"].cpu().numpy())
    zs, zt = np.concatenate(zs), np.concatenate(zt)
    coords = {}
    for name, x in (("z_sub", zs), ("z_task", zt)):
        print(f"{name}: {x.shape} -> t-SNE", flush=True)
        coords[name] = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto", max_iter=cli.max_iter, random_state=1968125571, n_jobs=-1, verbose=1).fit_transform(x)
    colors_sub = plt.get_cmap("nipy_spectral")(np.linspace(0, 1, len(np.unique(subjects))))
    sub_map = {v: colors_sub[i] for i,v in enumerate(sorted(np.unique(subjects)))}
    task_map = {v: TASK_COLORS[int(v)] for v in sorted(np.unique(tasks))}
    fig, ax = plt.subplots(2, 2, figsize=(20, 16))
    panels = [("z_sub", subjects, sub_map, "z_sub, colored by subject"), ("z_sub", tasks, task_map, "z_sub, colored by task"), ("z_task", tasks, task_map, "z_task, colored by task"), ("z_task", subjects, sub_map, "z_task, colored by subject")]
    for axis, (latent, vals, cmap, title) in zip(ax.flat, panels):
        axis.scatter(coords[latent][:,0], coords[latent][:,1], c=[cmap[int(v)] for v in vals], s=8, alpha=.75, linewidths=0); axis.set_title(title); axis.set_xticks([]); axis.set_yticks([])
    subject_handles = [Line2D([0], [0], marker="o", linestyle="", markersize=6, color=sub_map[v], label=f"Subject {v}") for v in sorted(sub_map)]
    task_handles = [Line2D([0], [0], marker="o", linestyle="", markersize=6, color=task_map[v], label=TASK_NAMES[v]) for v in sorted(task_map)]
    fig.legend(handles=subject_handles, title="Subject", loc="upper right", bbox_to_anchor=(0.995, 0.78), frameon=False)
    fig.legend(handles=task_handles, title="ERP task", loc="lower right", bbox_to_anchor=(0.995, 0.38), frameon=False)
    fig.subplots_adjust(left=0.03, right=0.84, bottom=0.04, top=0.93, wspace=0.08, hspace=0.22)
    fig.savefig(OUT / "z_latent_tsne.png", dpi=220)
    np.savez_compressed(OUT / "z_latent_tsne.npz", z_sub=zs, z_task=zt, z_sub_tsne=coords["z_sub"], z_task_tsne=coords["z_task"], subjects=subjects, tasks=tasks)
    print(f"saved: {OUT / 'z_latent_tsne.png'}")
if __name__ == "__main__": main()
