## 0.conda
export MAMBA_ROOT_PREFIX=/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/micromamba-root
                         
eval "$(/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/bin/micromamba shell hook -s bash)"                                                                 
                                 
micromamba activate labram


## 解压
unzip file.zip -d target_dir

## 看和一个网站的链接
curl -I https://www.google.com


## api查看用度

curl -X GET "http://www.litellm.org/key/info?" -H "Authorization: Bearer sk-I6Hn9f2HHbLfjog7StaauQ"


## 实时获取可用模型清单
curl -s -H "Authorization: Bearer <your_api_key>" https://www.litellm.org/models | jq '.data[].id'


## 切换api

看当前在用哪种登录
  codex login status

切到 ChatGPT / GPT 订阅额度
  codex logout
  codex login
  codex login --device-auth
  codex --sandbox danger-full-access

切到 API 额度
  codex logout
  printenv OPENAI_API_KEY | codex login --with-api-key

  如果你还没设置 API key：

  export OPENAI_API_KEY="sk-..."
  printenv OPENAI_API_KEY | codex login --with-api-key

export OPENAI_API_KEY="sk-I6Hn9f2HHbLfjog7StaauQ"
printenv OPENAI_API_KEY | codex login --with-api-key
  我的建议：日常交互开发用 codex login，这样走 ChatGPT/Codex 订阅额度；批处
  理、脚本、CI 或想单独消耗 API 预算时，用 codex logout 后切到 --with-api-
  key。切完后一定跑一次：
  
  codex login status

  确认当前到底在用 ChatGPT 登录还是 API key 登录。

### gpt

cp ~/.codex/config.chatgpt.toml ~/.codex/config.toml
codex logout
codex login --device-auth
codex

### api

cp ~/.codex/config.litellm.toml ~/.codex/config.toml
codex


  export OPENAI_API_KEY="sk-你的OpenAI API key"

  codex logout
  printenv OPENAI_API_KEY | codex login --with-api-key
  codex login status


#### 快速切换
放到 ~/.bashrc：

alias codex-gpt='cp ~/.codex/config.chatgpt.toml ~/.codex/config.toml && codex logout && codex login --device-auth && codex'
alias codex-api='cp ~/.codex/config.litellm.toml ~/.codex/config.toml && codex'

然后执行：

source ~/.bashrc

以后你要 ChatGPT 登录版：

codex-gpt

要 API/LiteLLM 版：

codex-api


### 回到last codex
codex resume --last

## 1.ssh
```
git add .
git commit -m "加了docs整理"
GIT_SSH_COMMAND='ssh -i /inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/.ssh/id_ed25519 -o IdentitiesOnly=yes' \
git push
```


给当前仓库永久配置
git config --local core.sshCommand \
'ssh -i /inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/.ssh/id_ed25519 -o IdentitiesOnly=yes'

如果第一次推送库：
  git status
  git checkout -b preexp12
  git add .
  git commit -m "01_reconstruction_steps.md"

  GIT_SSH_COMMAND='ssh -i /inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/.ssh/id_ed25519 -o IdentitiesOnly=yes' git push -u origin preexp12

## 1.git status看改了哪里和怎么看细致的

git diff -- run_class_finetuning.py
git diff -- scripts/8Ah.run_finetune_tuev_labram_bs32_uf8_2gpu_high.sh
git diff -- utils.py


## 2. 新建labram-preexp-work分支
我是想在/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-
  chenxinhe/eeg-test这个文件夹下建立/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-
  8850f313be13/global_user/7461-chenxinhe/eeg-test/labram-preexp-work的一个分支，可以
  从github上拉么？



ssh-keygen -t ed25519 -C "2139152205@qq.com"

  一路回车即可。然后查看公钥：

  cat ~/.ssh/id_ed25519.pub

  把输出的整行复制到 GitHub：

  GitHub -> Settings -> SSH and GPG keys -> New SSH key


cd /inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/eeg-test

<!-- git clone -b preexp2-recover https://github.com/Dorisgua/labram-exp.git labram-preexp2-recover -->
<!-- git clone -b preexp2-recover git@github.com:Dorisgua/labram-exp.git  -->
git clone -b preexp2-prototype git@github.com:Dorisgua/labram-exp.git labram-preexp2-recover


git clone -b preexp3 git@github.com:Dorisgua/labram-exp.git labram-preexp8

git clone -b preexp9 git@github.com:Dorisgua/labram-exp.git labram-preexp8

  然后进去确认：

  cd labram-preexp2-recover
  git branch --show-current

  输出是preexp2-prototype

  git switch -c preexp2-recover
  git push -u origin preexp2-recover


  git config user.name "doris"
  git config user.email "2139152205@qq.com"


### 在已有目录新建分支

```
# 进入你复制出来的新目录
  cd /inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/
  eeg-test/LaBraM-origin_preexp12

  # 查看当前 Git 状态，确认有没有未提交的修改、当前在哪个分支
  git status

  # 基于当前代码创建一个新的本地分支，名字叫 preexp12，并切换到这个分支
  git checkout -b preexp12

  # 把本地 preexp12 分支上传到 GitHub 的 origin 远程仓库
  # -u 表示建立本地 preexp12 和远程 origin/preexp12 的跟踪关系
  # 以后在这个分支上可以直接用 git push / git pull
  git push -u origin preexp12

  如果 git checkout -b preexp12 报错说分支已经存在，就改用：

  # 切换到已经存在的 preexp12 分支
  git checkout preexp12
```

## 1.gpu

1.把/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe我的私人目录软链接到/inspire/ssd/project/sais-medical/public/chenxinhe/user_data，这样我打开/inspire/ssd/project/sais-medical/public/chenxinhe/user_data就相当于打开了我的私人目录

ln -s /inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe /inspire/ssd/project/sais-medical/public/chenxinhe/user_data

2.确认gpu使用率，主要看100%      Default，8小时内不能连续低于10%
nvidia-smi 
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 570.195.03             Driver Version: 570.195.03     CUDA Version: 12.8     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 4090        On  |   00000000:17:00.0 Off |                  Off |
| 62%   57C    P0            427W /  450W |   23466MiB /  24564MiB |    100%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
+-----------------------------------------------------------------------------------------+

3.在后台挂载
cd /inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/gpu_occopy
nohup python gpu_verify.py & 

4.查找进程并且kill
ps -aux | grep "python"

root         112  1.5  0.0 2923800 200480 ?      Sl   04:28   0:20 /etc/inspire/jupyter-4.3/bin/python /etc/inspire/jupyter-4.3/bin/jupyter-lab --no-browser --ip
root       10287  3.6  0.0 2439464 117788 ?      Sl   04:45   0:07 /etc/inspire/jupyter-4.3/bin/python -m pylsp
root       10361  0.2  0.0  36368 27900 ?        S    04:45   0:00 /etc/inspire/jupyter-4.3/bin/python /etc/inspire/jupyter-4.3/lib/python3.9/site-packages/jedi/
root       12344 1414  0.0 37432648 732968 pts/1 Sl   04:48   8:57 python gpu_verify.py
root       13101  0.0  0.0   3860  2044 pts/1    S+   04:49   0:00 grep --color=auto python

(gpu) [root:gpu_occupy]$ kill -9 12344

5.解压文件
python -m zipfile -e /inspire/ssd/project/sais-medical/public/chenxinhe/user_data/eeg-test/LaBraM-main.zip .

6.将虚拟环境注册到jupyter上
conda install ipykernel -y
把环境注册到 Jupyter 的菜单里
python -m ipykernel install --user --name labram --display-name "labram"


7.在

## 预实验1 由02.tuev_global_prototype.py生成checkpoint

### 运行/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/eeg-test/labram-preexp-work/scripts/8Ah.run_finetune_tuev_labram_bs32_uf8_2gpu_high.sh

```
bash /inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/eeg-test/labram-preexp-work/scripts/8Ah.run_finetune_tuev_labram_bs32_uf8_2gpu_high.sh
```

### eval
8A：13导联 + prototype补回23导联

  CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 \
  /inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/micromamba-root/envs/labram/bin/torchrun \
    --nnodes=1 --nproc_per_node=1 run_class_finetuning.py \
    --model labram_base_patch200_200 \
    --finetune ./checkpoints/labram-base.pth \
    --dataset TUEV \
    --channel_subset tuev13 \
    --completion_scope tuev23 \
    --missing_channel_mode B \
    --channel_prototype_path docs/preexp2_missing_channels/artifacts/tuev_23_channel_prototypes.pth \
    --batch_size 32 \
    --update_freq 8 \
    --disable_rel_pos_bias \
    --abs_pos_emb \
    --disable_qkv_bias \
    --num_workers 0 \
    --smoothing 0.1 \
    --seed 0 \
    --eval \
    --no_auto_resume \
    --resume ./checkpoints/8A.run_finetune_tuev_labram_bs32_uf8_2gpu_nearest_channel_20260604_184506/checkpoint-best.pth \
    2>&1 | tee /inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/eeg-test/LaBraM-main-2/docs/preexp2_missing_channels/eval_8A_best.log


### 微调查看有些参数需不需要

abs_pos_emb = True
rel_pos_bias = False
qkv_bias = False 

• abs_pos_emb = True：为每个 EEG 通道 token 加入可学习的绝对位置编码，使模型能够区分不同通道的位置。如果为false，则什么都不加  
  if use_abs_pos_emb:
      self.pos_embed = nn.Parameter(torch.zeros(1, 129, embed_dim))
  else:
      self.pos_embed = None

  if self.pos_embed is not None:
      x = x + pos_embed
- rel_pos_bias = False：不根据两个 token 之间的相对位置调整它们的 Attention分数。【什么意思？】
  开启相对位置偏置后：score(i,j) = Qi · Kj + bias(位置 i 与位置 j 的相对距离)
- qkv_bias = False：计算 Attention 的 Q、K、V 时只使用线性权重，不添加可学习的偏置项。
开启时计算：
  Q = XWq + bq
  K = XWk
  V = XWv + bv
```
/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/micromamba-root/envs/labram/bin/python -c "import torch;m=torch.load('checkpoints/labram-base.pth',map_location='cpu').get('model',{}); k=list(m); print('绝对位置编码:',any('pos_embed' in x for x in k));print('相对位置偏置:',any('relative_position_bias_table' in x for x in k));print('Q/V bias:',any('.attn.q_bias' in x or '.attn.v_bias' in x or '.attn.qkv.bias' in x for x in k))"

```

## tensorboard
需要在终端启动 TensorBoard，再通过平台提供的端口访问功能打开网页。

  先安装 TensorBoard：

  /inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/micromamba-root/envs/labram/bin/pip install tensorboard

  安装完成后，在项目根目录运行：

  /inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/micromamba-root/envs/labram/bin/tensorboard \
    --logdir ./log \
    --host 0.0.0.0 \
    --port 6006


  /inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/micromamba-root/envs/labram/bin/tensorboard --logdir ./log --host 0.0.0.0 --port 6006

  这个只能看到./log下的，或许可以用：

  /inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/micromamba-root/envs/labram/bin/tensorboard \
    --logdir ./outputs/preexp22_tuev_recon_subject_missing_delta \
    --host 0.0.0.0 \
    --port 6006

  保持该终端运行。然后在训练平台中：

  1. 找到“端口转发”“自定义服务”或“打开端口”功能。
  2. 填写端口 6006。
  3. 打开平台生成的网址。
  4. 进入 TensorBoard 的 Scalars 页面查看曲线。

  如果是通过本地电脑 SSH 连接服务器，在本地终端运行：

  ssh -L 6006:localhost:6006 用户名@服务器地址

  然后本地浏览器打开：

  http://localhost:6006

  --logdir ./log 会同时展示所有历史训练；仅查看最新实验可改为：

  --logdir ./
  log/7A.run_finetune_tuev_labram_bs32_uf8_2gpu_global_20260604_060131



  ## 预实验1 eval
  cd /inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/eeg-test/LaBraM-
  main-2
  <!-- mkdir -p run_logs -->

  8A：13导联 + prototype补回23导联

  CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 \
  /inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/micromamba-root/envs/labram/bin/torchrun \
    --nnodes=1 --nproc_per_node=1 run_class_finetuning.py \
    --model labram_base_patch200_200 \
    --finetune ./checkpoints/labram-base.pth \
    --dataset TUEV \
    --channel_subset tuev13 \
    --completion_scope tuev23 \
    --missing_channel_mode B \
    --channel_prototype_path docs/preexp2_missing_channels/artifacts/tuev_23_channel_prototypes.pth \
    --batch_size 32 \
    --update_freq 8 \
    --disable_rel_pos_bias \
    --abs_pos_emb \
    --disable_qkv_bias \
    --num_workers 0 \
    --smoothing 0.1 \
    --seed 0 \
    --eval \
    --no_auto_resume \
    --resume ./checkpoints/8A.run_finetune_tuev_labram_bs32_uf8_2gpu_nearest_channel_20260604_184506/checkpoint-best.pth \
    2>&1 | tee /inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/eeg-test/LaBraM-main-2/docs/preexp2_missing_channels/eval_8A_best.log

  8O：原始23导联，不补全

  CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 \
  /inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/micromamba-root/envs/labram/bin/torchrun \
    --nnodes=1 --nproc_per_node=1 run_class_finetuning.py \
    --model labram_base_patch200_200 \
    --finetune ./checkpoints/labram-base.pth \
    --dataset TUEV \
    --channel_subset tuev23 \
    --completion_scope none \
    --missing_channel_mode origin \
    --batch_size 32 \
    --update_freq 8 \
    --disable_rel_pos_bias \
    --abs_pos_emb \
    --disable_qkv_bias \
    --num_workers 0 \
    --smoothing 0.1 \
    --seed 0 \
    --eval \
    --no_auto_resume \
    --resume ./checkpoints/8O.run_finetune_tuev_labram_bs32_uf8_2gpu_20260604_192036/checkpoint-best.pth \
    2>&1 | tee run_logs/eval_8O_best.log

  8N：只用13导联，不补全

  CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 \
  /inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/micromamba-root/envs/labram/bin/torchrun \
    --nnodes=1 --nproc_per_node=1 run_class_finetuning.py \
    --model labram_base_patch200_200 \
    --finetune ./checkpoints/labram-base.pth \
    --dataset TUEV \
    --channel_subset tuev13 \
    --completion_scope none \
    --missing_channel_mode origin \
    --batch_size 32 \
    --update_freq 8 \
    --disable_rel_pos_bias \
    --abs_pos_emb \
    --disable_qkv_bias \
    --num_workers 0 \
    --smoothing 0.1 \
    --seed 0 \
    --eval \
    --no_auto_resume \
    --resume ./checkpoints/8N.run_finetune_tuev_labram_bs32_uf8_2gpu_13ch_no_completion_20260604_180535/checkpoint-best.pth \
    2>&1 | tee /inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/eeg-test/LaBraM-main-2/docs/preexp2_missing_channels/eval_8N_best.log



## 255数据集

### 1epoch尝试aad-84
cd /inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/eeg-test/labram-preexp-work
bash scripts/9O.run_finetune_high_density_aad_84_1epoch.sh


## seed-V数据集 跑labram

### 预处理
cd /inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/eeg-test/labram-preexp-work OVERWRITE=1 scripts/10.make_seedv_labram.sh

### 11O

cd /inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/eeg-test/labram-preexp-work
scripts/11.run_finetune_seedv_labram.sh

  看日志：

tail -f outputs/preexp4_seedV/run_logs/11.run_finetune_seedv_labram.latest.log

  停止训练：

kill "$(cat outputs/preexp4_seedV/run_logs/11.run_finetune_seedv_labram.pid)"

  我刚补了这些文件/配置：

  - utils.py：新增 SEEDV_62_CHANNELS、SEEDVLoader、prepare_SEEDV_dataset
  - run_class_finetuning.py：注册 DATASET="SEEDV"
  - scripts/11.run_finetune_seedv_labram.sh：SEED-V 微调脚本



### eval 11O 但是64 *2=128global bs

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 \
/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/micromamba-root/envs/labram/bin/torchrun \
    --nnodes=1 \
    --nproc_per_node=1 \
    run_class_finetuning.py \
    --model labram_base_patch200_200 \
    --finetune ./checkpoints/labram-base.pth \
    --dataset SEEDV \
    --completion_scope none \
    --pooling_scope high \
    --missing_channel_mode origin \
    --batch_size 64 \
    --update_freq 1 \
    --lr 5e-4 \
    --epochs 50 \
    --warmup_epochs 4 \
    --weight_decay 0.05 \
    --layer_decay 0.65 \
    --drop_path 0.1 \
    --disable_rel_pos_bias \
    --abs_pos_emb \
    --disable_qkv_bias \
    --num_workers 4 \
    --seed 0 \
    --smoothing 0.1 \
    --eval \
    --resume outputs/preexp4_seedV/checkpoints/11.run_finetune_seedv_labram_20260609_034900/checkpoint-best.pth


### eval 11O

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 \
/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/micromamba-root/envs/labram/bin/torchrun \
    --nnodes=1 \
    --nproc_per_node=1 \
    run_class_finetuning.py \
    --model labram_base_patch200_200 \
    --finetune ./checkpoints/labram-base.pth \
    --dataset SEEDV \
    --completion_scope none \
    --pooling_scope high \
    --missing_channel_mode origin \
    --batch_size 64 \
    --update_freq 4 \
    --lr 5e-4 \
    --epochs 50 \
    --warmup_epochs 4 \
    --weight_decay 0.05 \
    --layer_decay 0.65 \
    --drop_path 0.1 \
    --disable_rel_pos_bias \
    --abs_pos_emb \
    --disable_qkv_bias \
    --num_workers 4 \
    --seed 0 \
    --smoothing 0.1 \
    --eval \
    --resume outputs/preexp4_seedV/checkpoints/11.run_finetune_seedv_labram_bs64_uf4_20260609_045421/checkpoint-best.pth \
    --output_dir outputs/preexp4_seedV/eval/11.run_finetune_seedv_labram_bs64_uf4_best \
    --log_dir outputs/preexp4_seedV/eval_tb/11.run_finetune_seedv_labram_bs64_uf4_best


### 11N

scripts/11N.run_finetune_seedv_labram_bs64_uf4.sh

#### 11N eval

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 \
/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/micromamba-root/envs/labram/bin/torchrun \
    --nnodes=1 \
    --nproc_per_node=1 \
    run_class_finetuning.py \
    --model labram_base_patch200_200 \
    --finetune ./checkpoints/labram-base.pth \
    --dataset SEEDV \
    --completion_scope none \
    --pooling_scope high \
    --missing_channel_mode origin \
    --batch_size 64 \
    --update_freq 4 \
    --lr 5e-4 \
    --epochs 50 \
    --warmup_epochs 4 \
    --weight_decay 0.05 \
    --layer_decay 0.65 \
    --drop_path 0.1 \
    --disable_rel_pos_bias \
    --abs_pos_emb \
    --disable_qkv_bias \
    --num_workers 4 \
    --seed 0 \
    --smoothing 0.1 \
    --eval \
    --channel_subset seedv23 \
    --resume outputs/preexp4_seedV/checkpoints/11N.run_finetune_seedv_labram_bs64_uf4_20260609_053911/checkpoint-best.pth 


### 11A
接下来我还要写/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/
  global_user/7461-chenxinhe/eeg-test/labram-preexp-work/
  scripts/11A.run_finetune_seedv_labram_bs64_uf4.sh，模仿/inspire/ssd/
  tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/
  eeg-test/labram-preexp-work/
  scripts/8Ah.run_finetune_tuev_labram_bs32_uf8_2gpu_high.sh，

### 62的prototype

bash /inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/eeg-test/labram-preexp-work/docs/preexp4_seedV/01_generate_seedv_prototypes.sh

#### 用的是checkpoint的哪一层？
› /inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/
  global_user/7461-chenxinhe/eeg-test/labram-preexp-work/docs/
  preexp4_seedV/01_generate_seedv_prototypes.py【这个用的是checkpoint的哪
  一层？】

用的是 checkpoints/vqnsp.pth 里的 VQNSP encoder 最后一层输出。

  具体在 eeg-test/labram-preexp-work/docs/
  preexp4_seedV/01_generate_seedv_prototypes.py:111 这里：

  features = model.encoder(
      samples,
      input_chans=input_chans,
      return_patch_tokens=True,
  )

  含义是：

  1. 加载 checkpoints/vqnsp.pth
  2. 只取里面的 model.encoder
  3. 输入 SEED-V train set 的 62 导联 EEG
  4. 取 encoder 经过所有 Transformer block 之后的 patch token feature
  5. 对每个导联在所有 train 样本上求平均
  6. 得到 seedv_62_channel_prototypes.pth，形状是 [62, 200]

  不是用 quantizer/codebook 的离散 token，也不是 decoder 输出。
  它用的是 VQNSP encoder 的连续特征，200 维。

#### 跑11A
bash scripts/11A.run_finetune_seedv_labram_bs64_uf4.sh


#### 11A的eval-encoder版本

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 \
/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/micromamba-root/envs/labram/bin/torchrun \
    --nnodes=1 \
    --nproc_per_node=1 \
    run_class_finetuning.py \
    --model labram_base_patch200_200 \
    --finetune ./checkpoints/labram-base.pth \
    --dataset SEEDV \
    --completion_scope seedv62 \
    --pooling_scope high \
    --missing_channel_mode B \
    --channel_prototype_path docs/preexp4_seedV/artifacts/seedv_62_channel_prototypes.pth \
    --batch_size 64 \
    --update_freq 4 \
    --lr 5e-4 \
    --epochs 50 \
    --warmup_epochs 4 \
    --weight_decay 0.05 \
    --layer_decay 0.65 \
    --drop_path 0.1 \
    --disable_rel_pos_bias \
    --abs_pos_emb \
    --disable_qkv_bias \
    --num_workers 4 \
    --seed 0 \
    --smoothing 0.1 \
    --eval \
    --channel_subset seedv23 \
    --resume outputs/preexp4_seedV/checkpoints/11A.run_finetune_seedv_labram_bs64_uf4_20260609_063234/checkpoint-best.pth 


#### 11A的eval-patch_emb版本

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 \
/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/micromamba-root/envs/labram/bin/torchrun \
    --nnodes=1 \
    --nproc_per_node=1 \
    run_class_finetuning.py \
    --model labram_base_patch200_200 \
    --finetune ./checkpoints/labram-base.pth \
    --dataset SEEDV \
    --completion_scope seedv62 \
    --pooling_scope high \
    --missing_channel_mode B \
    --channel_prototype_path docs/preexp4_seedV/artifacts/seedv_62_channel_prototypes_patch_embed.pth \
    --batch_size 64 \
    --update_freq 4 \
    --lr 5e-4 \
    --epochs 50 \
    --warmup_epochs 4 \
    --weight_decay 0.05 \
    --layer_decay 0.65 \
    --drop_path 0.1 \
    --disable_rel_pos_bias \
    --abs_pos_emb \
    --disable_qkv_bias \
    --num_workers 4 \
    --seed 0 \
    --smoothing 0.1 \
    --eval \
    --channel_subset seedv23 \
    --resume outputs/preexp4_seedV/checkpoints/11A.run_finetune_seedv_labram_bs64_uf4_20260609_083802/checkpoint-best.pth 

## 预实验2：用seedv的prototype补充tuev23

cd /inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/eeg-test/labram-preexp-work
bash scripts/13.tuev23_with_seedv62_extra.sh


### eval high版本

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 \
/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/micromamba-root/envs/labram/bin/torchrun \
    --nnodes=1 \
    --nproc_per_node=1 \
    run_class_finetuning.py \
    --output_dir outputs/preexp5/eval/13.tuev23_with_seedv62_extra_best \
    --log_dir outputs/preexp5/eval_tb/13.tuev23_with_seedv62_extra_best \
    --model labram_base_patch200_200 \
    --finetune ./checkpoints/labram-base.pth \
    --dataset TUEV \
    --completion_scope tuev23_with_seedv62_extra \
    --pooling_scope high \
    --missing_channel_mode B \
    --channel_prototype_path docs/preexp4_seedV/artifacts/seedv_62_channel_prototypes_patch_embed.pth \
    --batch_size 32 \
    --update_freq 8 \
    --lr 5e-4 \
    --epochs 50 \
    --warmup_epochs 5 \
    --weight_decay 0.05 \
    --layer_decay 0.65 \
    --drop_path 0.1 \
    --disable_rel_pos_bias \
    --abs_pos_emb \
    --disable_qkv_bias \
    --num_workers 4 \
    --seed 0 \
    --smoothing 0.1 \
    --eval \
    --channel_subset tuev23 \
    --no_auto_resume \
    --resume outputs/preexp5/checkpoints/13.tuev23_with_seedv62_extra_20260610_033320/checkpoint-best.pth 


### eval low版本

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 \
/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/micromamba-root/envs/labram/bin/torchrun \
    --nnodes=1 \
    --nproc_per_node=1 \
    run_class_finetuning.py \
    --output_dir outputs/preexp5/eval/13.tuev23_with_seedv62_extra_best \
    --log_dir outputs/preexp5/eval_tb/13.tuev23_with_seedv62_extra_best \
    --model labram_base_patch200_200 \
    --finetune ./checkpoints/labram-base.pth \
    --dataset TUEV \
    --completion_scope tuev23_with_seedv62_extra \
    --pooling_scope low \
    --missing_channel_mode B \
    --channel_prototype_path docs/preexp4_seedV/artifacts/seedv_62_channel_prototypes_patch_embed.pth \
    --batch_size 32 \
    --update_freq 8 \
    --lr 5e-4 \
    --epochs 50 \
    --warmup_epochs 5 \
    --weight_decay 0.05 \
    --layer_decay 0.65 \
    --drop_path 0.1 \
    --disable_rel_pos_bias \
    --abs_pos_emb \
    --disable_qkv_bias \
    --num_workers 4 \
    --seed 0 \
    --smoothing 0.1 \
    --eval \
    --channel_subset tuev23 \
    --no_auto_resume \
    --resume outputs/preexp5/checkpoints/13l.tuev23_with_seedv62_extra_low_20260610_064021/checkpoint-best.pth 


## preexp9

 /inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/micromamba-root/envs/labram/bin/python \
    docs/preexp9_seedV62_subject_real/07_debug_custom_group_sampler.py \
    --limit 8 \
    --batch_size 16 \
    --group_size 2 \
    --epoch 0



#### 算只经过cnn后的tuev和seedV
/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/micromamba-root/envs/labram/bin/python docs/preexp15_prototype_zscore/03_extract_tuev_channel_sample_features.py


/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/micromamba-root/envs/labram/bin/python docs/preexp15_prototype_zscore/04_extract_seedv_channel_sample_features.py

05 我要算zscore再mean；而且 general[channel] = 0.5 * (tuev[channel] + seedv[channel])

/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/micromamba-root/envs/labram/bin/python docs/preexp15_prototype_zscore/05_zscore_then_mean_prototypes.py


cp -a \
/inspire/ssd/project/sais-medical/public/hdd_public/share_medical/EEG/TUEZ \
/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/

cp -a \
/inspire/ssd/project/sais-medical/public/hdd_public/share_medical/EEG/TUEZ/v2.0.1/processed_labram/processed \
/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/TUEZ/v2.0.1/processed_labram


nohup cp -a \
/inspire/hdd/project/sais-medical/public/share_medical/EEG/TUEZ/v2.0.1/processed_labram/processed \
/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/TUEZ/v2.0.1/processed_labram/ \
> copy_tuev_to_ssd.log 2>&1 &


#### 指定tuev数据集位置
TUEV_ROOT='/inspire/hdd/project/sais-medical/public/share_medical/EEG/TUEZ/v2.0.1/processed_labram/processed' \
bash scripts/18Ah.run_finetune_tuev_labram_bs32_uf8_2gpu_high_prototype.sh


#### 查看文件数量
find /inspire/hdd/project/sais-medical/public/share_medical/EEG/TUEZ/v2.0.1/processed_labram/processed_train -maxdepth 1 -type f | wc -l

find /inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/TUEZ/v2.0.1/processed_labram/processed/processed_train -maxdepth 1 -type f | wc -l


#### 查看效果

python experiment_logging_notes/extract_best_metrics.py "/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/eeg-test/labram-preexp-work copy/outputs/preexp17_tuev/checkpoints/17N.finetune_tuev_labrambase_freeze_cnn_train_transformer_head_eval copy 2_20260704_083655/log.txt"


17N.finetune_tuev_labrambase_freeze_cnn_train_transformer_head_eval copy 2_20260704_083655



#### 用hf下载
/root/.local/bin/hf

/root/.local/bin/hf download MER-PS/MER-PS-trainval \
    --repo-type dataset \
    --local-dir /inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/MER-PS-trainval

## 建一个分支保存快照

  当前这些未提交修改之前的版本

  git branch baseline/preexp12-clean-before-prototype HEAD

  GIT_SSH_COMMAND='ssh -i /inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/.ssh/id_ed25519 -o IdentitiesOnly=yes' \
  git push origin baseline/preexp12-clean-before-prototype


  1. 看当前工作区相对这个 baseline 改了什么

  包括你还没 commit 的改动：

  git diff baseline/preexp12-clean-before-prototype

  2. 只看文件列表

  git diff --name-status baseline/preexp12-clean-before-prototype

  3. 看统计信息

  git diff --stat baseline/preexp12-clean-before-prototype

  4. 如果你之后已经 commit 到 preexp12 了，想看 preexp12 分支相对 baseline 的全部变化

  git diff baseline/preexp12-clean-before-prototype..preexp12

  文件列表：

  git diff --name-status baseline/preexp12-clean-before-prototype..preexp12

  git diff --stat baseline/preexp12-clean-before-prototype..preexp12

  最常用你记一个就行：

  git diff baseline/preexp12-clean-before-prototype

  意思是：

  当前目录里的代码，相比 clean-before-prototype 这个参照点，改了什么。



/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/eeg-main/CSLP-AE/data_preparation/simple_data.pt

复制到/inspire/hdd/project/sais-medical/public/share_medical/EEG/erp_core/data_preparation/simple_data.pt

mkdir -p /inspire/hdd/project/sais-medical/public/share_medical/EEG/erp_core/data_preparation

cp -v \
    /inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/eeg-main/CSLP-AE/data_preparation/simple_data.pt \
    /inspire/hdd/project/sais-medical/public/share_medical/EEG/erp_core/data_preparation/simple_data.pt

stat -c '%s %n' \
    /inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/eeg-main/CSLP-AE/data_preparation/simple_data.pt \
    /inspire/hdd/project/sais-medical/public/share_medical/EEG/erp_core/data_preparation/simple_data.pt


/inspire/hdd/project/sais-medical/public/medical_agent/SR/SR_Dataset/EEG-bciiv2a/processed_data