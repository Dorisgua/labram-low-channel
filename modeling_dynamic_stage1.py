# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# By Wei-Bang Jiang
# Based on BEiT-v2, timm, DeiT, and DINO code bases
# https://github.com/microsoft/unilm/tree/master/beitv2
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# ---------------------------------------------------------

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from einops import rearrange


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5),
        **kwargs
    }


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        # commit this for the orignal BERT implement 
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_norm=None, qk_scale=None, attn_drop=0.,
            proj_drop=0., window_size=None, attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        if qk_norm is not None:
            self.q_norm = qk_norm(head_dim)
            self.k_norm = qk_norm(head_dim)
        else:
            self.q_norm = None
            self.k_norm = None

        if window_size:
            self.window_size = window_size
            self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(self.num_relative_distance, num_heads))  # 2*Wh-1 * 2*Ww-1, nH
            # cls to token & token 2 cls & cls to cls

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(window_size[0])
            coords_w = torch.arange(window_size[1])
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * window_size[1] - 1
            relative_position_index = \
                torch.zeros(size=(window_size[0] * window_size[1] + 1, ) * 2, dtype=relative_coords.dtype)
            relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            relative_position_index[0, 0:] = self.num_relative_distance - 3
            relative_position_index[0:, 0] = self.num_relative_distance - 2
            relative_position_index[0, 0] = self.num_relative_distance - 1

            self.register_buffer("relative_position_index", relative_position_index)
        else:
            self.window_size = None
            self.relative_position_bias_table = None
            self.relative_position_index = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rel_pos_bias=None, return_attention=False, return_qkv=False):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple) (B, H, N, C)
        if self.q_norm is not None:
            q = self.q_norm(q).type_as(v)
        if self.k_norm is not None:
            k = self.k_norm(k).type_as(v)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if self.relative_position_bias_table is not None:
            relative_position_bias = \
                self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                    self.window_size[0] * self.window_size[1] + 1,
                    self.window_size[0] * self.window_size[1] + 1, -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            attn = attn + relative_position_bias.unsqueeze(0)

        if rel_pos_bias is not None:
            attn = attn + rel_pos_bias

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        if return_attention:
            return attn
            
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)

        x = self.proj(x)
        x = self.proj_drop(x)

        if return_qkv:
            return x, qkv

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_norm=None, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 window_size=None, attn_head_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_norm=qk_norm, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, window_size=window_size, attn_head_dim=attn_head_dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x, rel_pos_bias=None, return_attention=False, return_qkv=False):
        if return_attention:
            return self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias, return_attention=True)
        if return_qkv:
            y, qkv = self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias, return_qkv=return_qkv)
            x = x + self.drop_path(self.gamma_1 * y)
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
            return x, qkv

        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ EEG to Patch Embedding
    """
    def __init__(self, EEG_size=2000, patch_size=200, in_chans=1, embed_dim=200):
        super().__init__()
        # EEG_size = to_2tuple(EEG_size)
        # patch_size = to_2tuple(patch_size)
        num_patches = 62 * (EEG_size // patch_size)
        self.patch_shape = (1, EEG_size // patch_size)
        self.EEG_size = EEG_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=(1, patch_size), stride=(1, patch_size))

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class TemporalConv(nn.Module):
    """ EEG to Patch Embedding
    """
    def __init__(self, in_chans=1, out_chans=8):
        '''
        in_chans: in_chans of nn.Conv2d()
        out_chans: out_chans of nn.Conv2d(), determing the output dimension
        '''
        super().__init__()
        self.conv1 = nn.Conv2d(in_chans, out_chans, kernel_size=(1, 15), stride=(1, 8), padding=(0, 7))
        self.gelu1 = nn.GELU()
        self.norm1 = nn.GroupNorm(4, out_chans)
        self.conv2 = nn.Conv2d(out_chans, out_chans, kernel_size=(1, 3), padding=(0, 1))
        self.gelu2 = nn.GELU()
        self.norm2 = nn.GroupNorm(4, out_chans)
        self.conv3 = nn.Conv2d(out_chans, out_chans, kernel_size=(1, 3), padding=(0, 1))
        self.norm3 = nn.GroupNorm(4, out_chans)
        self.gelu3 = nn.GELU()

    def forward(self, x, **kwargs):
        x = rearrange(x, 'B N A T -> B (N A) T')
        B, NA, T = x.shape
        x = x.unsqueeze(1)
        x = self.gelu1(self.norm1(self.conv1(x)))
        x = self.gelu2(self.norm2(self.conv2(x)))
        x = self.gelu3(self.norm3(self.conv3(x)))
        x = rearrange(x, 'B C NA T -> B NA (T C)')
        return x


class DynamicNeuralTransformer(nn.Module):
    def __init__(self, EEG_size=1600, patch_size=200, in_chans=1, out_chans=8, num_classes=1000, embed_dim=200, depth=12,
                 num_heads=10, mlp_ratio=4., qkv_bias=False, qk_norm=None, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None,
                 use_abs_pos_emb=True, use_rel_pos_bias=False, use_shared_rel_pos_bias=False,
                 use_mean_pooling=True, init_scale=0.001, corrector_num_heads=8,
                 corrector_dropout=0.1, correction_scale=1, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        # To identify whether it is neural tokenizer or neural decoder. 
        # For the neural decoder, use linear projection (PatchEmbed) to project codebook dimension to hidden dimension.
        # Otherwise, use TemporalConv to extract temporal features from EEG signals.
        self.patch_embed = TemporalConv(out_chans=out_chans) if in_chans == 1 else PatchEmbed(EEG_size=EEG_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.time_window = EEG_size // patch_size
        self.patch_size = patch_size

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if use_abs_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, 128 + 1, embed_dim), requires_grad=True)
        else:
            self.pos_embed = None
        self.time_embed = nn.Parameter(torch.zeros(1, 16, embed_dim), requires_grad=True)
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.completion_scope = "none"
        self.pooling_scope = "low"
        self.target_input_chans_index = None
        self.real_input_chans_index = None

        self.register_buffer(
            "tuev23_channel_prototypes",
            torch.zeros(23, embed_dim),
            persistent=False,
        )
        self.register_buffer(
            "bciiv2a22_channel_prototypes",
            torch.zeros(22, embed_dim),
            persistent=False,
        )
        self.register_buffer(
            "physionet64_channel_prototypes",
            torch.zeros(64, embed_dim),
            persistent=False,
        )
        self.register_buffer(
            "seedv62_channel_prototypes",
            torch.zeros(62, embed_dim),
            persistent=False,
        )
        self.register_buffer(
            "seed62_channel_prototypes",
            torch.zeros(62, embed_dim),
            persistent=False,
        )
        self.register_buffer(
            "tuev23_with_seedv62_extra_channel_prototypes",
            torch.zeros(70, embed_dim),
            persistent=False,
        )
        self.register_buffer(
            "hgd78_channel_prototypes",
            torch.zeros(78, embed_dim),
            persistent=False,
        )
        self.register_buffer(
            "eegmat19_channel_prototypes",
            torch.zeros(19, embed_dim),
            persistent=False,
        )
        self.register_buffer(
            "siena29_channel_prototypes",
            torch.zeros(29, embed_dim),
            persistent=False,
        )
        self.register_buffer(
            "attention26_channel_prototypes",
            torch.zeros(26, embed_dim),
            persistent=False,
        )
        self.register_buffer(
            "erpcore28_channel_prototypes",
            torch.zeros(28, embed_dim),
            persistent=False,
        )

        self.rel_pos_bias = None

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.use_rel_pos_bias = use_rel_pos_bias
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_norm=qk_norm, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values, window_size=None)
            for i in range(depth)])
        self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        def make_corrector_encoder():
            layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=corrector_num_heads,
                dim_feedforward=embed_dim * 4,
                dropout=corrector_dropout,
                batch_first=True,
                norm_first=True,
                activation="gelu",
            )
            return nn.TransformerEncoder(layer, num_layers=1)

        # Dynamic Stage 1 直接作为 DynamicNeuralTransformer 的子模块，不再额外
        # 包装一个 DynamicModel。
        self.corrector = nn.ModuleDict({
            "shared_encoder": make_corrector_encoder(),
            "subject_encoder": make_corrector_encoder(),
            "task_encoder": make_corrector_encoder(),
            "shared_norm": nn.LayerNorm(embed_dim),
            "subject_norm": nn.LayerNorm(embed_dim),
            "task_norm": nn.LayerNorm(embed_dim),
        })
        self.correction_scale = float(correction_scale)

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=.02)
        if self.time_embed is not None:
            trunc_normal_(self.time_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        # trunc_normal_(self.mask_token, std=.02)
        if isinstance(self.head, nn.Linear):
            trunc_normal_(self.head.weight, std=.02)
        self.apply(self._init_weights)
        self.fix_init_weight()

        if isinstance(self.head, nn.Linear):
            self.head.weight.data.mul_(init_scale)
            self.head.bias.data.mul_(init_scale)

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'time_embed'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def freeze_cnn(self):
        for param in self.patch_embed.parameters(): #不计算 patch_embed 参数的梯度，也不更新它的权重。
            param.requires_grad = False 
        self.patch_embed.eval()     #让 patch_embed 里面某些层按推理模式运行。
        #是一个保险写法：如果以后 patch_embed 里加了 Dropout 或 BatchNorm，它们也不会在训练时产生随机行为或更新统计量。

    def freeze_corrector(self):
        for param in self.corrector.parameters():
            param.requires_grad = False
        self.corrector.eval()

    def _patch_tokens(self, x):
        batch_size, channels, num_t, _ = x.shape
        tokens = self.patch_embed(x)
        return tokens.reshape(batch_size, channels, num_t, self.embed_dim)

    def _dynamic_channel_indices(self, device):
        real_positions = [int(value) for value in self.real_input_chans_index[1:]] #只保留导联在 LaBraM position embedding 中的编号
        target_positions = [int(value) for value in self.target_input_chans_index[1:]]
        try:
            obs_indices = [target_positions.index(value) for value in real_positions] #找观测导联在28导联张量中的位置
        except ValueError as error:
            raise ValueError("An observed channel is absent from target layout") from error
        obs_index_set = set(obs_indices)
        miss_indices = [
            index for index in range(len(target_positions)) if index not in obs_index_set
        ]
        return (
            torch.as_tensor(obs_indices, dtype=torch.long, device=device), #[1, 3]
            torch.as_tensor(miss_indices, dtype=torch.long, device=device), #[0, 2, 4]
        )

    def _completion_prototypes(self):
        # 取出对应任务的 prototype
        prototype_by_scope = {
            "tuev13_with_tuev23": self.tuev23_channel_prototypes,
            "bciiv2a13_with_bciiv2a22": self.bciiv2a22_channel_prototypes,
            "physionet23_with_physionet64": self.physionet64_channel_prototypes,
            "physionet32_with_physionet64": self.physionet64_channel_prototypes,
            "seedv23_with_seedv62": self.seedv62_channel_prototypes,
            "seed23_with_seed62": self.seed62_channel_prototypes,
            "tuev23_with_seedv62_extra": self.tuev23_with_seedv62_extra_channel_prototypes,
            "hgd20_with_hgd78": self.hgd78_channel_prototypes,
            "eegmat8_with_eegmat19": self.eegmat19_channel_prototypes,
            "siena13_with_siena29": self.siena29_channel_prototypes,
            "attention10_with_attention26": self.attention26_channel_prototypes,
            "erpcore12_with_erpcore28": self.erpcore28_channel_prototypes,
        }
        try:
            return prototype_by_scope[self.completion_scope]
        except KeyError as error:
            raise ValueError(
                f"Unsupported completion_scope: {self.completion_scope}"
            ) from error

    def _encode_dynamic_tokens(self, h_obs):
        prototypes = self._completion_prototypes().to(
            device=h_obs.device,
            dtype=h_obs.dtype,
        )
        obs_indices, miss_indices = self._dynamic_channel_indices(h_obs.device)

        batch_size, _, num_t, _ = h_obs.shape
        p_all = prototypes.unsqueeze(0).unsqueeze(2).expand(
            batch_size, prototypes.shape[0], num_t, self.embed_dim
        )
        p_obs = p_all.index_select(1, obs_indices)
        p_miss = p_all.index_select(1, miss_indices)

        obs_tokens = h_obs.flatten(1, 2)
        miss_tokens = p_miss.flatten(1, 2)

        num_obs_tokens = obs_tokens.shape[1] #有多少个通道
        tokens = torch.cat((obs_tokens, miss_tokens), dim=1) # 直接拼接 观测导联 token + 缺失导联 prototype token

        shared_tokens = self.corrector["shared_norm"](
            self.corrector["shared_encoder"](tokens)
        )
        subject_tokens = self.corrector["subject_norm"](
            self.corrector["subject_encoder"](shared_tokens)
        )
        task_tokens = self.corrector["task_norm"](
            self.corrector["task_encoder"](shared_tokens)
        )

        subject_missing = subject_tokens[:, num_obs_tokens:, :]
        task_missing = task_tokens[:, num_obs_tokens:, :]
        missing_shape = p_miss.shape
        d_sub = self.correction_scale * torch.tanh(
            subject_missing.reshape(missing_shape)
        )
        d_task = self.correction_scale * torch.tanh(
            task_missing.reshape(missing_shape)
        )
        h_pred_miss = p_miss + d_sub + d_task
        return {
            "h_obs": h_obs,
            "p_all": p_all,
            "p_miss": p_miss,
            "obs_indices": obs_indices,
            "miss_indices": miss_indices,
            "z_sub": subject_missing.mean(dim=1),
            "z_task": task_missing.mean(dim=1),
            "d_sub": d_sub,
            "d_task": d_task,
            "h_pred_miss": h_pred_miss,
        }

    def forward_stage1(self, x_obs, x_full):
        outputs = self._encode_dynamic_tokens(self._patch_tokens(x_obs))
        with torch.no_grad():#不希望梯度经过目标分支
            h_full = self._patch_tokens(x_full)
            # 只取缺失导联
            outputs["h_miss_target"] = h_full.index_select(
                1, outputs["miss_indices"]
            ).detach()
        return outputs

    def forward_features(self, x, input_chans=None, return_patch_tokens=False, return_all_tokens=False, **kwargs):
        # x: [B, N, A, T]
        # B = batch size
        # N = 当前真实输入通道数，例如 TUEV-13 时 N=13
        # A = 每个通道切出来的 patch/time window 数
        # T = 每个 patch 的长度，LaBraM 这里通常是 200
        batch_size, n, a, t = x.shape

        # Low-density layouts such as SEED-23 still use the pretrained global
        # channel-position table. input_chans therefore contains one CLS index
        # plus exactly one position index for each real input channel.
        if input_chans is not None:
            if len(input_chans) != n + 1:
                raise ValueError(
                    "LaBraM input channel-index mismatch: "
                    f"input has {n} real channels, but input_chans has "
                    f"{len(input_chans) - 1} channel positions"
                )
            if int(input_chans[0]) != 0:
                raise ValueError(
                    "LaBraM input_chans must start with CLS position index 0"
                )

        # 当前每个通道有多少个 temporal patch。
        # 常见输入是 [B, N, A, 200]，所以 input_time_window = A。
        input_time_window = a if t == self.patch_size else t

        # 原来这里是：
        #   x = self.patch_embed(x)
        #
        # 原来的 x 只包含真实输入通道 token。
        # 现在先命名为 x_real，因为后面可能还要创建补通道后的 x_full。
        # 第一步：只对真实输入通道做 CNN/TemporalConv patch_embed。
        # 输出 x_real: [B, N * input_time_window, embed_dim]
        x_real = self.patch_embed(x)

        if self.completion_scope == "none":
            # completion_scope=none 时，等价于原来的：
            #   x = self.patch_embed(x)
            # 不补通道，保持原始 LaBraM 行为。
            # 后续 token 只包含真实输入通道。
            x = x_real

            # token_input_chans_index 用来选择 position embedding。
            # 不补通道时，它就是 input_chans。
            # input_chans 由当前真实输入的 ch_names 通过 utils.get_input_chans(ch_names) 算出来。
            token_input_chans_index = input_chans

            pool_token_indices = None
            target_channels_num = n  # 当前 token 对应的通道数
        else:
            # 补通道时，需要先把真实 token reshape 回按通道分组的形式。
            # [B, N * A, C] -> [B, N, A, C]
            x_real = x_real.reshape(batch_size, n, input_time_window, self.embed_dim)

            # Stage 1 与 Stage 2 共用同一个 completion_scope -> prototype 映射。
            prototypes = self._completion_prototypes()

            target_channels_num = prototypes.shape[0]  # 当前 token 对应的通道数

            # 用 prototype 初始化完整 target 通道空间。
            #
            # prototypes 原始形状:
            #   [target_channels_num, embed_dim]
            #
            # expand 后:
            #   [B, target_channels_num, input_time_window, embed_dim]
            #
            # 含义：
            #   每个 target 通道、每个 time patch，先都填这个通道自己的 prototype。
            x_full = prototypes.unsqueeze(0).unsqueeze(2).expand(
                batch_size,
                target_channels_num,
                input_time_window,
                self.embed_dim,
            ).clone()

            if self.real_input_chans_index is None or self.target_input_chans_index is None:
                raise ValueError(
                    "real_input_chans_index and target_input_chans_index must be set "
                    "when completion_scope is not none"
                )

            # real_input_chans_index 和 target_input_chans_index 都是 LaBraM position embedding 索引。
            # 它们都包含 cls token，所以第 0 个元素是 cls，真正通道从 [1:] 开始。
            #
            # 例如：
            #   target_input_chans_index = [0] + TUEV-23 的 LaBraM pos_embed index
            #   real_input_chans_index   = [0] + TUEV-13 的 LaBraM pos_embed index
            real_channel_pos = list(self.real_input_chans_index[1:]) #real_input_chans_index 从当前真实输入的 ch_names 算出来
            target_channel_pos = list(self.target_input_chans_index[1:])

            # 记录真实输入通道在 target tensor 里的通道维下标。
            # 注意：这里保存的是 x_full 的第 1 维下标，例如 TUEV-23 里的 0 到 22。
            # 它不是 LaBraM 128 position embedding 里的 index。
            # pooling_scope=low 时会用它只 pool 真实通道。
            real_channel_indices_in_target_tensor = []
            for real_i, real_pos in enumerate(real_channel_pos):
                # target_i 是该真实通道在 target 空间里的第几个通道。
                # 注意：这不是 LaBraM 128 里的 index，而是 target tensor x_full 的通道维 index。
                target_i = target_channel_pos.index(real_pos)
                real_channel_indices_in_target_tensor.append(target_i)

                # 用真实 patch_embed feature 覆盖 prototype。
                #
                # x_real[:, real_i, :, :]：
                #   第 real_i 个真实输入通道的所有 time patch feature。
                #
                # x_full[:, target_i, :, :]：
                #   target 空间里对应通道的位置。
                x_full[:, target_i, :, :] = x_real[:, real_i, :, :]

            # Dynamic Stage 2 不让缺失导联保持静态 prototype，而是使用
            # Stage 1 训练得到的 corrector 预测缺失通道 token。
            dynamic_completion_scopes = {
                "bciiv2a13_with_bciiv2a22",
                "erpcore12_with_erpcore28",
            }
            if self.completion_scope in dynamic_completion_scopes:
                dynamic_outputs = self._encode_dynamic_tokens(x_real)
                expected_obs_indices = torch.as_tensor(
                    real_channel_indices_in_target_tensor,
                    dtype=torch.long,
                    device=x_real.device,
                )
                if not torch.equal(dynamic_outputs["obs_indices"], expected_obs_indices):
                    raise ValueError("Dynamic observed-channel indices are inconsistent")
                x_full[:, dynamic_outputs["miss_indices"], :, :] = dynamic_outputs[
                    "h_pred_miss"
                ]

            # 补完后，把 [B, target_channels_num, A, C] 展平成 transformer token：
            # [B, target_channels_num * A, C]
            x = x_full.flatten(1, 2)

            # 后面 position embedding 应该使用 target_input_chans_index。
            # 因为此时 token 已经是 target 通道空间，不再是原始真实输入空间。
            token_input_chans_index = self.target_input_chans_index

            # pooling_scope=low：最后只 pool 真实输入通道对应的 token。
            # pooling_scope=high：最后 pool 所有补完后的 target token。
            pool_token_indices = (
                real_channel_indices_in_target_tensor
                if self.pooling_scope == "low"
                else None
            )

        # 原来这里开始就已经进入公共逻辑：
        #   cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        #   x = torch.cat((cls_tokens, x), dim=1)
        #
        # 这两行本身不用变。
        # 变化只在于：这里的 x 可能是原始真实 token，也可能是补通道后的 target token。
        # 加 cls token。
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks

        x = torch.cat((cls_tokens, x), dim=1)

        if self.pos_embed is not None:
            # 原来这里是：
            #   pos_embed_used = self.pos_embed[:, input_chans] if input_chans is not None else self.pos_embed
            #
            # 现在改成使用 token_input_chans_index：
            #   不补通道时 token_input_chans_index = input_chans
            #     input_chans 由当前真实输入的 ch_names 通过 utils.get_input_chans(ch_names) 算出来。
            #     也就是 token_input_chans_index = input_chans。
            #     所以它对应真实输入通道的 LaBraM position embedding 索引。
            #
            #   补通道时 token_input_chans_index = self.target_input_chans_index
            #     因为此时 x 已经从真实输入通道扩展成 target 通道空间，
            #     position embedding 也必须使用 target 通道空间的索引。
            #
            # 原因是补通道后，x 已经是 target 通道空间，position embedding 也要用 target 通道空间。
            # 加 channel position embedding。
            # token_input_chans_index 包含 cls token，所以可以直接索引 pos_embed。
            pos_embed_used = self.pos_embed[:, token_input_chans_index] if token_input_chans_index is not None else self.pos_embed
            pos_embed = pos_embed_used[:, 1:, :].unsqueeze(2).expand(batch_size, -1, input_time_window, -1).flatten(1, 2)
            pos_embed = torch.cat((pos_embed_used[:,0:1,:].expand(batch_size, -1, -1), pos_embed), dim=1)
            x = x + pos_embed
        if self.time_embed is not None:
            # 原来这里是：
            #   nc = n if t == self.patch_size else a   # 这里 nc 是通道数。x.shape = [B, N, A, T]
            #   time_embed = self.time_embed[:, 0:input_time_window, :].unsqueeze(1).expand(
            #       batch_size, nc, -1, -1
            #   ).flatten(1, 2)
            #
            # 现在直接用 target_channels_num：
            #   不补通道时 target_channels_num = n
            #   补通道时 target_channels_num = prototype 的目标通道数
            #
            # 原因是补通道后，token 数量已经变成 target_channels_num * input_time_window。
            # 加 time embedding。
            # 每个通道共享同一套 time embedding。
            time_embed = self.time_embed[:, 0:input_time_window, :].unsqueeze(1).expand(batch_size, target_channels_num, -1, -1).flatten(1, 2)
            x[:, 1:, :] += time_embed

        # 下面 transformer blocks 和原来一样：
        #   x = self.pos_drop(x)
        #   for blk in self.blocks:
        #       x = blk(x, rel_pos_bias=None)
        #   x = self.norm(x)
        x = self.pos_drop(x)
        
        for blk in self.blocks:
            x = blk(x, rel_pos_bias=None)
        
        x = self.norm(x)
        if self.fc_norm is not None:
            if return_all_tokens:
                return self.fc_norm(x)
            patch_tokens = x[:, 1:, :]
            if return_patch_tokens:
                return self.fc_norm(patch_tokens)

            # 原来这里是：
            #   return self.fc_norm(t.mean(1))
            #
            # 原来直接对所有真实输入 patch token 做 mean pooling。
            # 现在如果 pooling_scope=low，需要先从 target token 里取回真实输入通道对应的 token。
            if pool_token_indices is not None:
                # pooling_scope=low:
                # patch_tokens 当前是 [B, target_channels_num * A, C]。
                # 先 reshape 回 [B, target_channels_num, A, C]，
                # 再只取真实输入通道对应的 target index。
                patch_tokens = patch_tokens.reshape(
                    batch_size,
                    target_channels_num,
                    input_time_window,
                    self.embed_dim,
                )
                patch_tokens = patch_tokens[:, pool_token_indices, :, :]
                return self.fc_norm(patch_tokens.flatten(1, 2).mean(1))

            # completion_scope=none 或 pooling_scope=high：
            # 直接对当前全部 patch token 做平均。
            return self.fc_norm(patch_tokens.mean(1))
        else:
            if return_all_tokens:
                return x
            elif return_patch_tokens:
                return x[:, 1:]
            else:
                return x[:, 0]

    def forward(self, x, input_chans=None, return_patch_tokens=False, return_all_tokens=False, **kwargs):
        '''
        x: [batch size, number of electrodes, number of patches, patch size]
        For example, for an EEG sample of 4 seconds with 64 electrodes, x will be [batch size, 64, 4, 200]
        '''
        x = self.forward_features(x, input_chans=input_chans, return_patch_tokens=return_patch_tokens, return_all_tokens=return_all_tokens, **kwargs)
        x = self.head(x)
        return x

    def forward_intermediate(self, x, layer_id=12, norm_output=False):
        x = self.patch_embed(x)
        batch_size, seq_len, _ = x.size()

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            pos_embed = self.pos_embed[:, 1:, :].unsqueeze(2).expand(batch_size, -1, self.time_window, -1).flatten(1, 2)
            pos_embed = torch.cat((self.pos_embed[:,0:1,:].expand(batch_size, -1, -1), pos_embed), dim=1)
            x = x + pos_embed
        if self.time_embed is not None:
            time_embed = self.time_embed.unsqueeze(1).expand(batch_size, 62, -1, -1).flatten(1, 2)
            x[:, 1:, :] += time_embed
        x = self.pos_drop(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        if isinstance(layer_id, list):
            output_list = []
            for l, blk in enumerate(self.blocks):
                x = blk(x, rel_pos_bias=rel_pos_bias)
                # use last norm for all intermediate layers
                if l in layer_id:
                    if norm_output:
                        x_norm = self.fc_norm(self.norm(x[:, 1:]))
                        output_list.append(x_norm)
                    else:
                        output_list.append(x[:, 1:])
            return output_list
        elif isinstance(layer_id, int):
            for l, blk in enumerate(self.blocks):
                if l < layer_id:
                    x = blk(x, rel_pos_bias=rel_pos_bias)
                elif l == layer_id:
                    x = blk.norm1(x)
                else:
                    break
            return x[:, 1:]
        else:
            raise NotImplementedError(f"Not support for layer id is {layer_id} now!")
    
    def get_intermediate_layers(self, x, use_last_norm=False):
        x = self.patch_embed(x)
        batch_size, seq_len, _ = x.size()

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            pos_embed = self.pos_embed[:, 1:, :].unsqueeze(2).expand(batch_size, -1, self.time_window, -1).flatten(1, 2)
            pos_embed = torch.cat((self.pos_embed[:,0:1,:].expand(batch_size, -1, -1), pos_embed), dim=1)
            x = x + pos_embed
        if self.time_embed is not None:
            time_embed = self.time_embed.unsqueeze(1).expand(batch_size, 62, -1, -1).flatten(1, 2)
            x[:, 1:, :] += time_embed
        x = self.pos_drop(x)

        features = []
        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for blk in self.blocks:
            x = blk(x, rel_pos_bias)
            if use_last_norm:
                features.append(self.norm(x))
            else:
                features.append(x)

        return features


@register_model
def labram_dynamic_base_patch200_200(pretrained=False, **kwargs):
    model = DynamicNeuralTransformer(
        patch_size=200, embed_dim=200, depth=12, num_heads=10, mlp_ratio=4, qk_norm=partial(nn.LayerNorm, eps=1e-6), # qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def labram_dynamic_large_patch200_200(pretrained=False, **kwargs):
    model = DynamicNeuralTransformer(
        patch_size=200, embed_dim=400, depth=24, num_heads=16, mlp_ratio=4, out_chans=16, qk_norm=partial(nn.LayerNorm, eps=1e-6), # qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def labram_dynamic_huge_patch200_200(pretrained=False, **kwargs):
    model = DynamicNeuralTransformer(
        patch_size=200, embed_dim=800, depth=48, num_heads=16, mlp_ratio=4, out_chans=32, qk_norm=partial(nn.LayerNorm, eps=1e-6), # qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model
