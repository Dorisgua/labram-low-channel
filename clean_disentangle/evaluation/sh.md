
export MAMBA_ROOT_PREFIX=/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/micromamba-root
                         
eval "$(/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/bin/micromamba shell hook -s bash)"                                                                 
                                 
micromamba activate labram


### tsne
A：

  python -m clean_disentangle.evaluation.plot_latent_tsne \
    --checkpoint outputs/full_d_only/full_d_only_seed0_20260818_131233/checkpoints/checkpoint-last.pth

  B：

  python -m clean_disentangle.evaluation.plot_latent_tsne \
    --checkpoint outputs/full_prototype_d/full_prototype_d_seed0_20260818_135346/checkpoints/checkpoint-last.pth

  C：

  python -m clean_disentangle.evaluation.plot_latent_tsne \
    --checkpoint outputs/missing_prototype_d/missing_prototype_d_seed0_20260818_143337/checkpoints/checkpoint-last.pth

### probe
  python -m clean_disentangle.evaluation.probe_latents \
    --checkpoint outputs/full_d_only/full_d_only_seed0_20260818_131233/checkpoints/checkpoint-last.pth

  python -m clean_disentangle.evaluation.probe_latents \
    --checkpoint outputs/full_prototype_d/full_prototype_d_seed0_20260818_135346/checkpoints/checkpoint-last.pth

python -m clean_disentangle.evaluation.probe_latents \
    --checkpoint outputs/missing_prototype_d/missing_prototype_d_seed0_20260818_143337/checkpoints/checkpoint-last.pth


/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/eeg-main/LabraM-Git-Diff-clean/outputs/missing_prototype_d/missing_prototype_d_seed0_20260818_143337