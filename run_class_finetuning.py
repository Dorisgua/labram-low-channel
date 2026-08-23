# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# By Wei-Bang Jiang
# Based on BEiT-v2, timm, DeiT, and DINO code bases
# https://github.com/microsoft/unilm/tree/master/beitv2
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# ---------------------------------------------------------

import argparse
import datetime
from pyexpat import model
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os

from pathlib import Path
from collections import OrderedDict
from timm.data.mixup import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import ModelEma
from optim_factory import create_optimizer, get_parameter_groups, LayerDecayValueAssigner

from engine_for_finetuning import train_one_epoch, evaluate
from utils import NativeScalerWithGradNormCount as NativeScaler
import utils
from scipy import interpolate
import modeling_finetune
from modeling_adabrain import AdaBrainLaBraMMLPWrapper, AdaBrainLaBraMWrapper
from data_processor.bciiv2a import prepare_BCIIV2A_multisession_dataset
from data_processor.eegmat import prepare_EEGMAT_cross_subject_dataset
from data_processor.physionet import prepare_PhysioNet_motor_imagery_dataset
from data_processor.seed import prepare_SEED_cross_subject_dataset
from data_processor.zuo2025 import prepare_Zuo2025_cross_subject_dataset
from data_processor.hgd import prepare_HGD_official_dataset
from data_processor.siena import prepare_Siena_cross_subject_dataset
from data_processor.fatig import prepare_Fatig_rest_task_dataset
from data_processor.faced import prepare_FACED_cross_subject_dataset
from data_processor.aad import prepare_AAD_cross_subject_dataset
from data_processor.attention import prepare_Attention_cross_subject_dataset
from Channels_definition import (
    ATTENTION_10_CHANNELS,
    ATTENTION_26_CHANNELS,
    BCIIV2A_13_CHANNELS,
    BCIIV2A_22_CHANNELS,
    EEGMAT_8_CHANNELS,
    EEGMAT_19_CHANNELS,
    FATIG_30_CHANNELS,
    FACED_32_CHANNELS,
    HIGH_DENSITY_AAD_84_CHANNELS,
    PHYSIONET_23_CHANNELS,
    PHYSIONET_32_CHANNELS,
    PHYSIONET_64_CHANNELS,
    SEED_23_CHANNELS,
    SEED_62_CHANNELS,
    TUEV_13_CHANNELS,
    TUEV_23_CHANNELS,
    SEEDV_62_CHANNELS,
    TUEV23_SEEDV62_EXTRA_CHANNELS,
    ZUO2025_30_CHANNELS,
    HGD_78_CHANNELS,
    HGD_MOTOR_20_CHANNELS,
    SIENA_13_CHANNELS,
    SIENA_29_CHANNELS,
)

def get_args():
    parser = argparse.ArgumentParser('LaBraM fine-tuning and evaluation script for EEG classification', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--update_freq', default=1, type=int)
    parser.add_argument('--save_ckpt_freq', default=5, type=int)

    # robust evaluation
    parser.add_argument('--robust_test', default=None, type=str,
                        help='robust evaluation dataset')
    
    # Model parameters
    parser.add_argument('--model', default='labram_base_patch200_200', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--qkv_bias', action='store_true')
    parser.add_argument('--disable_qkv_bias', action='store_false', dest='qkv_bias')
    parser.set_defaults(qkv_bias=True)
    parser.add_argument('--rel_pos_bias', action='store_true')
    parser.add_argument('--disable_rel_pos_bias', action='store_false', dest='rel_pos_bias')
    parser.set_defaults(rel_pos_bias=True)
    parser.add_argument('--abs_pos_emb', action='store_true')
    parser.set_defaults(abs_pos_emb=False)
    parser.add_argument('--layer_scale_init_value', default=0.1, type=float, 
                        help="0.1 for base, 1e-5 for large. set 0 to disable layer scale")

    parser.add_argument('--input_size', default=200, type=int,
                        help='EEG input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--attn_drop_rate', type=float, default=0.0, metavar='PCT',
                        help='Attention dropout rate (default: 0.)')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--disable_eval_during_finetuning', action='store_true', default=False)

    parser.add_argument('--model_ema', action='store_true', default=False)
    parser.add_argument('--model_ema_decay', type=float, default=0.9999, help='')
    parser.add_argument('--model_ema_force_cpu', action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")

    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--layer_decay', type=float, default=0.9)

    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='num of steps to warmup LR, will overload warmup_epochs if set > 0')

    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Finetuning params
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--model_key', default='model|module', type=str)
    parser.add_argument('--model_prefix', default='', type=str)
    parser.add_argument('--model_filter_name', default='gzp', type=str)
    parser.add_argument('--init_scale', default=0.001, type=float)
    parser.add_argument('--use_mean_pooling', action='store_true')
    parser.set_defaults(use_mean_pooling=True)
    parser.add_argument('--use_cls', action='store_false', dest='use_mean_pooling')
    parser.add_argument('--disable_weight_decay_on_rel_pos_bias', action='store_true', default=False)

    # Dataset parameters
    parser.add_argument('--nb_classes', default=0, type=int,
                        help='number of the classification types')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)

    parser.add_argument('--save_ckpt', action='store_true')
    parser.add_argument('--no_save_ckpt', action='store_false', dest='save_ckpt')
    parser.set_defaults(save_ckpt=True)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    parser.add_argument('--enable_deepspeed', action='store_true', default=False)
    parser.add_argument('--dataset', default='TUAB', type=str,
                        help='dataset: TUAB | TUEV | bciiv2a | physionet | SEED | EEGMAT | FACED | AAD | Zuo2025 | HGD | Siena | fatig')
    parser.add_argument('--data_path', default='', type=str,
                        help='optional dataset root override')
    parser.add_argument('--sampling_rate', default=200, type=int,
                        help='target sampling rate for datasets that support resampling')
    parser.add_argument('--norm_method', default='z_score', type=str,
                        choices=['z_score', '0.1mv', '95'],
                        help='normalization for bciiv2a (AdaBrain default: z_score)')
    parser.add_argument('--loso_fold', default=0, type=int,
                        help='LOSO fold index for datasets that expose folds, e.g. fatig 0-10')
    
    parser.add_argument('--channel_subset', default='', type=str,
                        help='channel subset name, for example tuev13/tuev23/physionet23/physionet64/seed23/seed62')
    parser.add_argument('--completion_scope', default='none', type=str,
                        choices=['none', 'tuev13_with_tuev23', 'bciiv2a13_with_bciiv2a22',
                                 'physionet23_with_physionet64',
                                 'physionet32_with_physionet64',
                                 'seed23_with_seed62',
                                 'seedv23_with_seedv62', 'tuev23_with_seedv62_extra',
                                 'hgd20_with_hgd78', 'eegmat8_with_eegmat19',
                                 'siena13_with_siena29', 'attention10_with_attention26'],
                        help='target channel completion scope')
    parser.add_argument('--pooling_scope', default='low', type=str,
                        choices=['low', 'high'],
                        help='pool only real input channels or all completed target channels')
    parser.add_argument('--channel_prototype_path', default='', type=str,
                        help='path to channel prototype checkpoint')
    parser.add_argument('--freeze_cnn', action='store_true',
                        help='Freeze patch_embed/TemporalConv and train only transformer/head layers')
    parser.add_argument('--classifier_mode', default='mean_pool', type=str,
                        choices=['mean_pool', 'adabrain_all_token', 'adabrain_mlp_token'],
                        help='classification head: original LaBraM mean pooling, AdaBrain all-token linear head, or all-token MLP head')
    parser.add_argument('--classifier_token_scope', default='all', type=str,
                        choices=['all', 'real'],
                        help='AdaBrain readout tokens: all backbone channels or only real input channels')
    parser.add_argument('--best_metric', default='accuracy', type=str,
                        choices=['accuracy', 'balanced_accuracy', 'f1_weighted', 'cohen_kappa', 'roc_auc', 'pr_auc'],
                        help='validation metric used to select checkpoint-best; recommended: TUAB=roc_auc, TUEV=cohen_kappa, SEEDV=accuracy')

    known_args, _ = parser.parse_known_args()

    if known_args.enable_deepspeed:
        try:
            import deepspeed
            from deepspeed import DeepSpeedConfig
            parser = deepspeed.add_config_arguments(parser)
            ds_init = deepspeed.initialize
        except:
            print("Please 'pip install deepspeed==0.4.0'")
            exit(0)
    else:
        ds_init = None

    return parser.parse_args(), ds_init

def get_models(args):
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        attn_drop_rate=args.attn_drop_rate,
        drop_block_rate=None,
        use_mean_pooling=args.use_mean_pooling,
        init_scale=args.init_scale,
        use_rel_pos_bias=args.rel_pos_bias,
        use_abs_pos_emb=args.abs_pos_emb,
        init_values=args.layer_scale_init_value,
        qkv_bias=args.qkv_bias,
    )

    return model


TUAB_CHANNELS = [
    'FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4',
    'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6',
    'A1', 'A2', 'FZ', 'CZ', 'PZ', 'T1', 'T2',
]


DATASET_CONFIGS = {
    'TUAB': {
        'root': 'path/to/TUAB',
        'prepare_fn': utils.prepare_TUAB_dataset,
        'ch_names': TUAB_CHANNELS,
        'pass_channel_names': False,
        'nb_classes': 1,
        'metrics': ["pr_auc", "roc_auc", "accuracy", "balanced_accuracy"],
    },
    'TUEV': {
        'root': '/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/TUEZ/v2.0.1/processed_labram/processed',
        'prepare_fn': utils.prepare_TUEV_dataset,
        'ch_names': {
            'tuev13': TUEV_13_CHANNELS,
            'tuev23': TUEV_23_CHANNELS,
        },
        'pass_channel_names': True,
        'nb_classes': 6,
        'metrics': ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted"],
    },
    'bciiv2a': {
        'root': '/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/BCI-IV-2A/multi_subject_json',
        'prepare_fn': prepare_BCIIV2A_multisession_dataset,
        'ch_names': {
            'bciiv2a13': BCIIV2A_13_CHANNELS,
            'bciiv2a22': BCIIV2A_22_CHANNELS,
        },
        'pass_channel_names': True,
        'validate_loader_channel_names': True,
        'prepare_kwargs_from_args': {
            'sampling_rate': 'sampling_rate',
            'normalize_method': 'norm_method',
        },
        # AdaBrain normalizes in the loader, so do not apply LaBraM's /100 again.
        'input_scale': 1.0,
        'num_t': 4,
        'nb_classes': 4,
        'metrics': ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted"],
    },
    'physionet': {
        'root': '/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/physionet/physionet.org/files/eegmmidb/processed_eegfmbench/processed/fs_200/motor_mv_img/finetune/1.0.0',
        'prepare_fn': prepare_PhysioNet_motor_imagery_dataset,
        'ch_names': {
            'physionet23': PHYSIONET_23_CHANNELS,
            'physionet32': PHYSIONET_32_CHANNELS,
            'physionet64': PHYSIONET_64_CHANNELS,
        },
        'pass_channel_names': True,
        'validate_loader_channel_names': True,
        # EEG-FM-Bench stores microvolts; match LaBraM's standard /100 input scale.
        'input_scale': 0.01,
        'num_t': 4,
        'nb_classes': 4,
        'metrics': ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted"],
    },
    'SEED': {
        'root': '/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/SEED/processed_data',
        'prepare_fn': prepare_SEED_cross_subject_dataset,
        'ch_names': {
            'seed23': SEED_23_CHANNELS,
            'seed62': SEED_62_CHANNELS,
        },
        'pass_channel_names': True,
        'validate_loader_channel_names': True,
        # Pickles store microvolts; match LaBraM's pretrained 0.1 mV scale.
        'input_scale': 0.01,
        'num_t': 1,
        'nb_classes': 3,
        'metrics': ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted"],
    },
    'EEGMAT': {
        'root': '/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/EEGMAT',
        'prepare_fn': prepare_EEGMAT_cross_subject_dataset,
        'ch_names': {
            'eegmat19': EEGMAT_19_CHANNELS,
            'eegmat8': EEGMAT_8_CHANNELS,
        },
        'pass_channel_names': True,
        'validate_loader_channel_names': True,
        'prepare_kwargs_from_args': {
            'sampling_rate': 'sampling_rate',
            'normalize_method': 'norm_method',
        },
        # The loader applies train-statistics z-score normalization.
        'input_scale': 1.0,
        'num_t': 4,
        'nb_classes': 2,
        'metrics': ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted"],
    },
    'FACED': {
        'root': '/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/FACED/processed_data_10s_200hz',
        'prepare_fn': prepare_FACED_cross_subject_dataset,
        'ch_names': {
            'faced32': FACED_32_CHANNELS,
        },
        'pass_channel_names': True,
        'validate_loader_channel_names': True,
        'prepare_kwargs_from_args': {
            'sampling_rate': 'sampling_rate',
            'normalize_method': 'norm_method',
        },
        # Loader applies train-subject statistics z-score normalization.
        'input_scale': 1.0,
        'num_t': 10,
        'nb_classes': 9,
        'metrics': ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted"],
    },
    'AAD': {
        'root': '/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/AAD/processed_data_4s_200hz',
        'prepare_fn': prepare_AAD_cross_subject_dataset,
        'ch_names': {
            'aad84': HIGH_DENSITY_AAD_84_CHANNELS,
        },
        'pass_channel_names': True,
        'validate_loader_channel_names': True,
        'prepare_kwargs_from_args': {
            'sampling_rate': 'sampling_rate',
            'normalize_method': 'norm_method',
        },
        # Loader applies train-subject statistics z-score normalization.
        'input_scale': 1.0,
        'num_t': 4,
        # Binary-logit path in this repository uses nb_classes=1.
        'nb_classes': 1,
        'metrics': ["pr_auc", "roc_auc", "accuracy", "balanced_accuracy", "cohen_kappa", "f1"],
    },
    'Attention': {
        'root': '/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/Attention/processed_data_4s_200hz',
        'prepare_fn': prepare_Attention_cross_subject_dataset,
        'ch_names': {
            'attention26': ATTENTION_26_CHANNELS,
            'attention10': ATTENTION_10_CHANNELS,
        },
        'pass_channel_names': True,
        'validate_loader_channel_names': True,
        'prepare_kwargs_from_args': {
            'sampling_rate': 'sampling_rate',
            'normalize_method': 'norm_method',
        },
        # Loader applies train-subject statistics z-score normalization.
        'input_scale': 1.0,
        'num_t': 4,
        'nb_classes': 2,
        'metrics': ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted"],
    },
    'Zuo2025': {
        'root': '/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/Zuo2025/processed_data_4s_200hz',
        'prepare_fn': prepare_Zuo2025_cross_subject_dataset,
        'ch_names': {
            'zuo2025_30': ZUO2025_30_CHANNELS,
        },
        'pass_channel_names': True,
        'validate_loader_channel_names': True,
        'prepare_kwargs_from_args': {
            'sampling_rate': 'sampling_rate',
            'normalize_method': 'norm_method',
        },
        # The loader applies channel-wise statistics from training subjects only.
        'input_scale': 1.0,
        'num_t': 4,
        'nb_classes': 2,
        'metrics': ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted"],
    },
    'HGD': {
        'root': '/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/HGD/processed_data_4s_200hz',
        'prepare_fn': prepare_HGD_official_dataset,
        'ch_names': {
            'hgd78': HGD_78_CHANNELS,
            'hgd20': HGD_MOTOR_20_CHANNELS,
        },
        'pass_channel_names': True,
        'validate_loader_channel_names': True,
        'prepare_kwargs_from_args': {
            'sampling_rate': 'sampling_rate',
            'normalize_method': 'norm_method',
        },
        'input_scale': 1.0,
        'num_t': 4,
        'nb_classes': 4,
        'metrics': ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted"],
    },
    'Siena': {
        'root': '/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/Siene/processed_data_10s_200hz',
        'prepare_fn': prepare_Siena_cross_subject_dataset,
        'ch_names': {
            'siena29': SIENA_29_CHANNELS,
            'siena13': SIENA_13_CHANNELS,
        },
        'pass_channel_names': True,
        'validate_loader_channel_names': True,
        'prepare_kwargs_from_args': {
            'sampling_rate': 'sampling_rate',
            'normalize_method': 'norm_method',
        },
        # Loader z-scores using the exact training records only.
        'input_scale': 1.0,
        'num_t': 10,
        # Binary-logit path in this repository uses nb_classes=1.
        'nb_classes': 1,
        'metrics': ["pr_auc", "roc_auc", "accuracy", "balanced_accuracy"],
    },
    'fatig': {
        'root': '/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/7461-chenxinhe/fatig/processed_data_3s_200hz',
        'prepare_fn': prepare_Fatig_rest_task_dataset,
        'ch_names': {
            'fatig30': FATIG_30_CHANNELS,
        },
        'pass_channel_names': True,
        'validate_loader_channel_names': True,
        'prepare_kwargs_from_args': {
            'normalize_method': 'norm_method',
            'loso_fold': 'loso_fold',
        },
        # Loader z-scores using train-subject windows only.
        'input_scale': 1.0,
        'num_t': 3,
        'nb_classes': 2,
        'metrics': ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted"],
    },
}


def get_dataset(args):
    if args.dataset not in DATASET_CONFIGS:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    cfg = DATASET_CONFIGS[args.dataset]
    ch_names = cfg['ch_names']
    if isinstance(ch_names, dict):
        if args.channel_subset not in ch_names:
            raise ValueError(f"Unsupported channel_subset {args.channel_subset} for dataset {args.dataset}")
        ch_names = ch_names[args.channel_subset] #因此传入的ch_names是channel_subset的

    prepare_fn = cfg['prepare_fn']
    prepare_kwargs = {
        kwarg: getattr(args, arg_name)
        for kwarg, arg_name in cfg.get('prepare_kwargs_from_args', {}).items()
    }
    root = args.data_path or cfg['root']
    if cfg.get('pass_channel_names', False):
        train_dataset, test_dataset, val_dataset = prepare_fn(
            root,
            channel_names=ch_names,
            **prepare_kwargs,
        )
    else:
        train_dataset, test_dataset, val_dataset = prepare_fn(
            root,
            **prepare_kwargs,
        )

    if cfg.get('validate_loader_channel_names', False):
        actual_ch_names = train_dataset.get_ch_names()
        if actual_ch_names != ch_names:
            raise ValueError(
                f"{args.dataset} channel order mismatch: manifest={actual_ch_names}, "
                f"configured={ch_names}"
            )

    print(f"{args.dataset} channel_subset={args.channel_subset}")
    print(f"{args.dataset} channels ({len(ch_names)}): {ch_names}")
    args.nb_classes = cfg['nb_classes']
    args.input_scale = cfg.get('input_scale', 0.01)
    args.num_t = cfg.get('num_t')
    metrics = cfg['metrics']
    return train_dataset, test_dataset, val_dataset, ch_names, metrics


def _validate_completion_prototype(args, ch_names, target_ch_names, target_input_chans_index, prototypes):
    for name in ch_names:
        if name not in target_ch_names:
            raise ValueError(f"Channel {name} in real data not in prototype target channels")

    expected_target_ch_names = {
        "tuev13_with_tuev23": TUEV_23_CHANNELS,
        "bciiv2a13_with_bciiv2a22": BCIIV2A_22_CHANNELS,
        "physionet23_with_physionet64": PHYSIONET_64_CHANNELS,
        "physionet32_with_physionet64": PHYSIONET_64_CHANNELS,
        "seed23_with_seed62": SEED_62_CHANNELS,
        "seedv23_with_seedv62": SEEDV_62_CHANNELS,
        "tuev23_with_seedv62_extra": TUEV23_SEEDV62_EXTRA_CHANNELS,
        "hgd20_with_hgd78": HGD_78_CHANNELS,
        "eegmat8_with_eegmat19": EEGMAT_19_CHANNELS,
        "siena13_with_siena29": SIENA_29_CHANNELS,
        "attention10_with_attention26": ATTENTION_26_CHANNELS,
    }.get(args.completion_scope)
    if expected_target_ch_names is None:
        raise ValueError(f"Unsupported completion_scope: {args.completion_scope}")

    if target_ch_names != expected_target_ch_names:
        raise ValueError(
            "Prototype channel order mismatch: "
            f"target_ch_names should equal {args.completion_scope} target channels"
        )

    if prototypes.shape[0] != len(target_ch_names):
        raise ValueError(
            f"Prototype row count mismatch: got {prototypes.shape[0]}, "
            f"expected {len(target_ch_names)}"
        )

    if len(target_input_chans_index) != len(target_ch_names) + 1:
        raise ValueError(
            f"target_input_chans_index length mismatch: got {len(target_input_chans_index)}, "
            f"expected {len(target_ch_names) + 1}"
        )

    if target_input_chans_index[0] != 0:
        raise ValueError("target_input_chans_index[0] should be 0 for cls token")

    for i, name in enumerate(target_ch_names):
        expected_global_idx = utils.standard_1020.index(name) + 1
        actual_global_idx = target_input_chans_index[i + 1]
        if actual_global_idx != expected_global_idx:
            raise ValueError(
                f"Position index mismatch for channel {name}: "
                f"expected {expected_global_idx}, got {actual_global_idx}"
            )


def main(args, ds_init):
    utils.init_distributed_mode(args)

    if ds_init is not None:
        utils.create_ds_config(args)

    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True

    # dataset_train, dataset_test, dataset_val: follows the standard format of torch.utils.data.Dataset.
    # ch_names: list of strings, channel names of the dataset. It should be in capital letters.
    # metrics: list of strings, the metrics you want to use. We utilize PyHealth to implement it.
    dataset_train, dataset_test, dataset_val, ch_names, metrics = get_dataset(args)
    if args.best_metric not in metrics:
        raise ValueError(
            f"best_metric={args.best_metric} is not available for dataset={args.dataset}. "
            f"Available metrics: {metrics}"
        )
    print(f"Best metric name: {args.best_metric}")

    if args.disable_eval_during_finetuning:
        dataset_val = None
        dataset_test = None

    if True:  # args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
            if type(dataset_test) == list:
                sampler_test = [torch.utils.data.DistributedSampler(
                    dataset, num_replicas=num_tasks, rank=global_rank, shuffle=False) for dataset in dataset_test]
            else:
                sampler_test = torch.utils.data.DistributedSampler(
                    dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
            sampler_test = torch.utils.data.SequentialSampler(dataset_test)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    if dataset_val is not None:
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=int(1.5 * args.batch_size),
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )
        if type(dataset_test) == list:
            data_loader_test = [torch.utils.data.DataLoader(
                dataset, sampler=sampler,
                batch_size=int(1.5 * args.batch_size),
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
                drop_last=False
            ) for dataset, sampler in zip(dataset_test, sampler_test)]
        else:
            data_loader_test = torch.utils.data.DataLoader(
                dataset_test, sampler=sampler_test,
                batch_size=int(1.5 * args.batch_size),
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
                drop_last=False
            )
    else:
        data_loader_val = None
        data_loader_test = None

    model = get_models(args)

    patch_size = model.patch_size
    print("Patch size = %s" % str(patch_size))
    args.window_size = (1, args.input_size // patch_size)
    args.patch_size = patch_size

    if args.classifier_mode in {"adabrain_all_token", "adabrain_mlp_token"}:
        if args.num_t is None:
            raise ValueError(
                f"classifier_mode={args.classifier_mode} requires num_t in "
                f"DATASET_CONFIGS[{args.dataset!r}]"
            )

        sample_x = dataset_train[0][0]
        if sample_x.ndim != 2:
            raise ValueError(
                "Expected dataset sample shaped [channels, time], "
                f"got {tuple(sample_x.shape)}"
            )
        if sample_x.shape[0] != len(ch_names):
            raise ValueError(
                f"Dataset/channel-name mismatch: sample has {sample_x.shape[0]} "
                f"channels, but ch_names has {len(ch_names)}"
            )

        expected_length = args.num_t * model.patch_size
        if sample_x.shape[-1] != expected_length:
            raise ValueError(
                f"{args.dataset} sample length mismatch: got {sample_x.shape[-1]}, "
                f"expected num_t({args.num_t}) * patch_size({model.patch_size}) "
                f"= {expected_length}"
            )
        print(
            "Validated AdaBrain temporal layout: "
            f"sample_length={sample_x.shape[-1]}, "
            f"num_t={args.num_t}, patch_size={model.patch_size}"
        )

    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')

        print("Load ckpt from %s" % args.finetune)
        checkpoint_model = None
        for model_key in args.model_key.split('|'):
            if model_key in checkpoint:
                checkpoint_model = checkpoint[model_key]
                print("Load state_dict by model_key = %s" % model_key)
                break
        if checkpoint_model is None:
            checkpoint_model = checkpoint
        if (checkpoint_model is not None) and (args.model_filter_name != ''):
            all_keys = list(checkpoint_model.keys())
            new_dict = OrderedDict()
            for key in all_keys:
                if key.startswith('student.'):
                    new_dict[key[8:]] = checkpoint_model[key]
                else:
                    pass
            checkpoint_model = new_dict

        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        all_keys = list(checkpoint_model.keys())
        for key in all_keys:
            if "relative_position_index" in key:
                checkpoint_model.pop(key)

        utils.load_state_dict(model, checkpoint_model, prefix=args.model_prefix)

    model.completion_scope = args.completion_scope
    model.pooling_scope = args.pooling_scope

    if args.completion_scope != "none":
        if not args.channel_prototype_path:
            raise ValueError("--channel_prototype_path is required when completion_scope is not none")

        prototype_ckpt = torch.load(args.channel_prototype_path, map_location="cpu")
        prototypes = prototype_ckpt["channel_prototypes"]
        target_ch_names = prototype_ckpt["ch_names"]
        target_input_chans_index = [int(v) for v in prototype_ckpt["input_chans_index"]]
        real_input_chans_index = [int(v) for v in utils.get_input_chans(ch_names)]

        print(f"Completion scope: {args.completion_scope}")
        print(f"Pooling scope: {args.pooling_scope}")
        print(f"Prototype path: {args.channel_prototype_path}")
        print(f"Prototype channels ({len(target_ch_names)}): {target_ch_names}")
        print(f"Prototype tensor shape: {tuple(prototypes.shape)}")
        print(f"Target input chans index: {target_input_chans_index}")
        print(f"Real input chans index: {real_input_chans_index}")

        _validate_completion_prototype(
            args=args,
            ch_names=ch_names,
            target_ch_names=target_ch_names,
            target_input_chans_index=target_input_chans_index,
            prototypes=prototypes,
        )

        if args.completion_scope == "tuev13_with_tuev23":
            model.tuev23_channel_prototypes.copy_(prototypes)
        elif args.completion_scope == "bciiv2a13_with_bciiv2a22":
            model.bciiv2a22_channel_prototypes.copy_(prototypes)
        elif args.completion_scope == "physionet23_with_physionet64":
            model.physionet64_channel_prototypes.copy_(prototypes)
        elif args.completion_scope == "physionet32_with_physionet64":
            model.physionet64_channel_prototypes.copy_(prototypes)
        elif args.completion_scope == "seed23_with_seed62":
            model.seed62_channel_prototypes.copy_(prototypes)
        elif args.completion_scope == "seedv23_with_seedv62":
            model.seedv62_channel_prototypes.copy_(prototypes)
        elif args.completion_scope == "tuev23_with_seedv62_extra":
            model.tuev23_with_seedv62_extra_channel_prototypes.copy_(prototypes)
        elif args.completion_scope == "hgd20_with_hgd78":
            model.hgd78_channel_prototypes.copy_(prototypes)
        elif args.completion_scope == "eegmat8_with_eegmat19":
            model.eegmat19_channel_prototypes.copy_(prototypes)
        elif args.completion_scope == "siena13_with_siena29":
            model.siena29_channel_prototypes.copy_(prototypes)
        elif args.completion_scope == "attention10_with_attention26":
            model.attention26_channel_prototypes.copy_(prototypes)
        else:
            raise ValueError(f"Unsupported completion_scope: {args.completion_scope}")

        model.target_input_chans_index = target_input_chans_index
        model.real_input_chans_index = real_input_chans_index

    if args.freeze_cnn:
        model.freeze_cnn()
        frozen_cnn_params = sum(p.numel() for p in model.patch_embed.parameters())  #统计 patch_embed 里一共有多少个参数。
        print(f"Freeze CNN/patch_embed: {frozen_cnn_params} parameters")

    if args.classifier_mode in {"adabrain_all_token", "adabrain_mlp_token"}:
        token_channels = (
            len(target_ch_names)
            if args.completion_scope != "none"
            else len(ch_names)
        )
        readout_channel_indices = None
        if args.classifier_token_scope == "real":
            if args.completion_scope == "none":
                readout_channel_indices = list(range(len(ch_names)))
            else:
                real_channel_pos = list(real_input_chans_index[1:])
                target_channel_pos = list(target_input_chans_index[1:])
                missing_real_positions = [
                    pos for pos in real_channel_pos if pos not in target_channel_pos
                ]
                if missing_real_positions:
                    raise ValueError(
                        "Real input channels are absent from the completed target "
                        f"channel space: {missing_real_positions}"
                    )
                readout_channel_indices = [
                    target_channel_pos.index(pos) for pos in real_channel_pos
                ]

        wrapper_cls = (
            AdaBrainLaBraMMLPWrapper
            if args.classifier_mode == "adabrain_mlp_token"
            else AdaBrainLaBraMWrapper
        )
        wrapper_kwargs = {}
        if args.classifier_mode == "adabrain_mlp_token":
            wrapper_kwargs["dropout"] = args.drop

        model = wrapper_cls(
            backbone=model,
            num_channels=token_channels,
            input_num_channels=len(ch_names),
            num_t=args.num_t,
            num_classes=args.nb_classes,
            readout_channel_indices=readout_channel_indices,
            **wrapper_kwargs,
        )
        print(
            f"Using {args.classifier_mode} classifier: "
            f"input_channels={model.input_num_channels}, "
            f"backbone_channels={token_channels}, num_t={args.num_t}, "
            f"backbone_tokens={model.expected_tokens}, "
            f"token_scope={args.classifier_token_scope}, "
            f"readout_channel_indices={model.readout_channel_indices.tolist()}, "
            f"readout_channels={model.readout_num_channels}, "
            f"readout_tokens={model.expected_readout_tokens}, "
            f"head={model.task_head}"
        )

    model.to(device)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')
        print("Using EMA with decay = %.8f" % args.model_ema_decay)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params:', n_parameters)

    total_batch_size = args.batch_size * args.update_freq * utils.get_world_size()
    num_training_steps_per_epoch = len(dataset_train) // total_batch_size
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Update frequent = %d" % args.update_freq)
    print("Number of training examples = %d" % len(dataset_train))
    print("Number of training training per epoch = %d" % num_training_steps_per_epoch)

    num_layers = model_without_ddp.get_num_layers()
    if args.layer_decay < 1.0:
        assigner = LayerDecayValueAssigner(list(args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)))
    else:
        assigner = None

    if assigner is not None:
        print("Assigned values = %s" % str(assigner.values))

    skip_weight_decay_list = model.no_weight_decay()
    if args.disable_weight_decay_on_rel_pos_bias:
        for i in range(num_layers):
            skip_weight_decay_list.add("blocks.%d.attn.relative_position_bias_table" % i)

    if args.enable_deepspeed:
        loss_scaler = None
        optimizer_params = get_parameter_groups(
            model, args.weight_decay, skip_weight_decay_list,
            assigner.get_layer_id if assigner is not None else None,
            assigner.get_scale if assigner is not None else None)
        model, optimizer, _, _ = ds_init(
            args=args, model=model, model_parameters=optimizer_params, dist_init_required=not args.distributed,
        )

        print("model.gradient_accumulation_steps() = %d" % model.gradient_accumulation_steps())
        assert model.gradient_accumulation_steps() == args.update_freq
    else:
        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
            model_without_ddp = model.module

        optimizer = create_optimizer(
            args, model_without_ddp, skip_list=skip_weight_decay_list,
            get_num_layer=assigner.get_layer_id if assigner is not None else None, 
            get_layer_scale=assigner.get_scale if assigner is not None else None)
        loss_scaler = NativeScaler()

    print("Use step level LR scheduler!")
    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

    if args.nb_classes == 1:
        criterion = torch.nn.BCEWithLogitsLoss()
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

    utils.auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp,
        optimizer=optimizer, loss_scaler=loss_scaler, model_ema=model_ema)
            
    if args.eval:
        balanced_accuracy = []
        accuracy = []
        eval_loaders = data_loader_test if isinstance(data_loader_test, list) else [data_loader_test]
        for data_loader in eval_loaders:
            test_stats = evaluate(
                data_loader, model, device, header='Test:', ch_names=ch_names,
                metrics=metrics, is_binary=(args.nb_classes == 1),
                input_scale=args.input_scale,
            )
            accuracy.append(test_stats['accuracy'])
            balanced_accuracy.append(test_stats['balanced_accuracy'])
        print(f"======Accuracy: {np.mean(accuracy)} {np.std(accuracy)}, balanced accuracy: {np.mean(balanced_accuracy)} {np.std(balanced_accuracy)}")
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    best_score = float('-inf')
    best_epoch = None
    best_val_stats = None
    best_test_stats = None
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch * args.update_freq)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer,
            device, epoch, loss_scaler, args.clip_grad, model_ema,
            log_writer=log_writer, start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values, wd_schedule_values=wd_schedule_values,
            num_training_steps_per_epoch=num_training_steps_per_epoch, update_freq=args.update_freq, 
            ch_names=ch_names, is_binary=args.nb_classes == 1,
            input_scale=args.input_scale,
        )
        
        if args.output_dir and args.save_ckpt:
            utils.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch, model_ema=model_ema, save_ckpt_freq=args.save_ckpt_freq)
            
        if data_loader_val is not None:
            val_stats = evaluate(
                data_loader_val, model, device, header='Val:', ch_names=ch_names,
                metrics=metrics, is_binary=args.nb_classes == 1,
                input_scale=args.input_scale,
            )
            print(f"Accuracy of the network on the {len(dataset_val)} val EEG: {val_stats['accuracy']:.2f}%")
            test_stats = evaluate(
                data_loader_test, model, device, header='Test:', ch_names=ch_names,
                metrics=metrics, is_binary=args.nb_classes == 1,
                input_scale=args.input_scale,
            )
            print(f"Accuracy of the network on the {len(dataset_test)} test EEG: {test_stats['accuracy']:.2f}%")
            print(f"Epoch {epoch} best metric name: {args.best_metric}")
            print(f"Epoch {epoch} val metrics: {val_stats}")
            print(f"Epoch {epoch} test metrics: {test_stats}")
            print(
                f"Epoch {epoch} metric distribution: "
                f"val={json.dumps(val_stats, sort_keys=True, default=float)}, "
                f"test={json.dumps(test_stats, sort_keys=True, default=float)}"
            )
            
            current_score = float(val_stats[args.best_metric])
            if current_score > best_score:
                best_score = current_score
                best_epoch = epoch
                best_val_stats = val_stats
                best_test_stats = test_stats
                if args.output_dir and args.save_ckpt:
                    utils.save_model(
                        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch="best", model_ema=model_ema)
                print(f"Best epoch: {best_epoch}")
                print(f"Best metric name: {args.best_metric}")
                print(f"Best val selected score: {best_score}")
                print(f"Best val metrics distribution: {json.dumps(best_val_stats, sort_keys=True, default=float)}")
                print(f"Test metrics distribution at best val epoch: {json.dumps(best_test_stats, sort_keys=True, default=float)}")

            print(f'Best {args.best_metric} val: {best_score:.4f} at epoch {best_epoch}')
            if log_writer is not None:
                for key, value in val_stats.items():
                    if key == 'accuracy':
                        log_writer.update(accuracy=value, head="val", step=epoch)
                    elif key == 'balanced_accuracy':
                        log_writer.update(balanced_accuracy=value, head="val", step=epoch)
                    elif key == 'f1_weighted':
                        log_writer.update(f1_weighted=value, head="val", step=epoch)
                    elif key == 'pr_auc':
                        log_writer.update(pr_auc=value, head="val", step=epoch)
                    elif key == 'roc_auc':
                        log_writer.update(roc_auc=value, head="val", step=epoch)
                    elif key == 'cohen_kappa':
                        log_writer.update(cohen_kappa=value, head="val", step=epoch)
                    elif key == 'loss':
                        log_writer.update(loss=value, head="val", step=epoch)
                for key, value in test_stats.items():
                    if key == 'accuracy':
                        log_writer.update(accuracy=value, head="test", step=epoch)
                    elif key == 'balanced_accuracy':
                        log_writer.update(balanced_accuracy=value, head="test", step=epoch)
                    elif key == 'f1_weighted':
                        log_writer.update(f1_weighted=value, head="test", step=epoch)
                    elif key == 'pr_auc':
                        log_writer.update(pr_auc=value, head="test", step=epoch)
                    elif key == 'roc_auc':
                        log_writer.update(roc_auc=value, head="test", step=epoch)
                    elif key == 'cohen_kappa':
                        log_writer.update(cohen_kappa=value, head="test", step=epoch)
                    elif key == 'loss':
                        log_writer.update(loss=value, head="test", step=epoch)
                
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'val_{k}': v for k, v in val_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()},
                         'best_metric': args.best_metric,
                         'best_score': best_score,
                         'best_epoch': best_epoch,
                         'epoch': epoch,
                         'n_parameters': n_parameters}
        else:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    if utils.is_main_process():
        if best_epoch is not None:
            print("=" * 80)
            print("Best epoch summary")
            print(f"Best epoch: {best_epoch}")
            print(f"Best metric name: {args.best_metric}")
            print(f"Best val selected score: {best_score}")
            print(f"Best val metrics: {json.dumps(best_val_stats, sort_keys=True, default=float)}")
            print(f"Best test metrics: {json.dumps(best_test_stats, sort_keys=True, default=float)}")
            print("=" * 80)
        else:
            print("No best epoch was selected because validation was not run.")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    opts, ds_init = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts, ds_init)
