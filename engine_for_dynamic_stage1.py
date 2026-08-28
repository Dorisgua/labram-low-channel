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
import sys
from typing import Iterable, Optional
import torch
from timm.utils import ModelEma
import utils
from einops import rearrange
from losses_dynamic import compute_stage1_losses

# 原分类模型调用：
# def train_class_batch(model, samples, target, criterion, ch_names):
#     outputs = model(samples, ch_names)
#     loss = criterion(outputs, target)
#     return loss, outputs


def train_dynamic_stage1_batch(model, x_obs, x_full,
                               missing_weight, reg_weight,
                               subject_summary_contra_weight, task_summary_contra_weight,
                               subject_correction_contra_weight, task_correction_contra_weight,
                               permute_sub_weight, permute_task_weight,
                               sub_pair_inputs=None, task_pair_inputs=None):
    stage1_model = model.module if hasattr(model, "module") else model
    outputs = stage1_model.forward_stage1(x_obs, x_full)
    sub_pair_outputs = _forward_cslpae_pair(stage1_model, sub_pair_inputs)
    task_pair_outputs = _forward_cslpae_pair(stage1_model, task_pair_inputs)
    losses = compute_stage1_losses(
        outputs, missing_weight, reg_weight,
        subject_summary_contra_weight, task_summary_contra_weight,
        subject_correction_contra_weight, task_correction_contra_weight,
        permute_sub_weight, permute_task_weight,
        sub_pair_outputs=sub_pair_outputs,
        task_pair_outputs=task_pair_outputs,
    )
    return losses["total_loss"], losses


def _prepare_cslpae_pair(dataset, property_name, batch_size, device, input_scale):
    left, right, num_groups, samples_per_group = dataset.sample_cslpae_pair_batch(
        property_name,
        batch_size,
    )

    def move_pair_batch(batch):
        x_obs = batch[2].float().to(device, non_blocking=True) * input_scale
        x_full = batch[3].float().to(device, non_blocking=True) * input_scale
        x_obs = rearrange(x_obs, 'B N (A T) -> B N A T', T=200)
        x_full = rearrange(x_full, 'B N (A T) -> B N A T', T=200)
        return x_obs, x_full

    left_x_obs, left_x_full = move_pair_batch(left)
    right_x_obs, right_x_full = move_pair_batch(right)
    return (
        left_x_obs,
        left_x_full,
        right_x_obs,
        right_x_full,
        num_groups,
        samples_per_group,
    )


def _forward_cslpae_pair(stage1_model, pair_inputs):
    if pair_inputs is None:
        return None
    left_x_obs, left_x_full, right_x_obs, right_x_full, groups, samples = pair_inputs
    return (
        stage1_model.forward_stage1(left_x_obs, left_x_full),
        stage1_model.forward_stage1(right_x_obs, right_x_full),
        groups,
        samples,
    )


def _half_cslpae_pair(pair_inputs):
    if pair_inputs is None:
        return None
    left_x_obs, left_x_full, right_x_obs, right_x_full, groups, samples = pair_inputs
    return (
        left_x_obs.half(),
        left_x_full.half(),
        right_x_obs.half(),
        right_x_full.half(),
        groups,
        samples,
    )


def get_loss_scale_for_deepspeed(model):
    optimizer = model.optimizer
    return optimizer.loss_scale if hasattr(optimizer, "loss_scale") else optimizer.cur_scale


# 原分类训练入口：
# def train_one_epoch(model, criterion, data_loader, optimizer, device, epoch, ...):
def train_dynamic_stage1_one_epoch(
                    model: torch.nn.Module, data_loader: Iterable,
                    optimizer: torch.optim.Optimizer, device: torch.device,
                    epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, log_writer=None,
                    start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None,
                    input_scale=0.01, missing_weight=1.0, reg_weight=0.01,
                    subject_summary_contra_weight=0.0, task_summary_contra_weight=0.0,
                    subject_correction_contra_weight=0.0, task_correction_contra_weight=0.0,
                    permute_sub_weight=1.0, permute_task_weight=1.0):
    model.train(True)
    stage1_model = model.module if hasattr(model, "module") else model
    if any(param.requires_grad for param in stage1_model.patch_embed.parameters()):
        raise RuntimeError("Dynamic Stage 1 requires a frozen patch_embed")
    stage1_model.patch_embed.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Dynamic Stage 1: [{}]'.format(epoch)
    print_freq = 10

    if loss_scaler is None:
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()

    # 旧逻辑只能解包二字段 tuple：
    # for data_iter_step, (samples, targets) in enumerate(
    #         metric_logger.log_every(data_loader, print_freq, header)):
    # Stage 1 使用同一批样本的 x_obs 和 x_full。
    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        _, _, x_obs, x_full, _, _ = batch
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group.get("lr_scale", 1.0)
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        # 原分类输入：
        # samples = samples.float().to(device, non_blocking=True) * input_scale
        # samples = rearrange(samples, 'B N (A T) -> B N A T', T=200)
        x_obs = x_obs.float().to(device, non_blocking=True) * input_scale
        x_full = x_full.float().to(device, non_blocking=True) * input_scale
        x_obs = rearrange(x_obs, 'B N (A T) -> B N A T', T=200)
        x_full = rearrange(x_full, 'B N (A T) -> B N A T', T=200)

        pair_batch_size = x_obs.shape[0]
        sub_pair_inputs = None
        if (subject_summary_contra_weight > 0.0
                or subject_correction_contra_weight > 0.0
                or permute_sub_weight > 0.0):
            sub_pair_inputs = _prepare_cslpae_pair(
                data_loader.dataset, "subject", pair_batch_size, device, input_scale
            )
        task_pair_inputs = None
        if (task_summary_contra_weight > 0.0
                or task_correction_contra_weight > 0.0
                or permute_task_weight > 0.0):
            task_pair_inputs = _prepare_cslpae_pair(
                data_loader.dataset, "task", pair_batch_size, device, input_scale
            )

        if loss_scaler is None:
            # 原分类调用：loss, output = train_class_batch(...)
            x_obs = x_obs.half()
            x_full = x_full.half()
            sub_pair_inputs = _half_cslpae_pair(sub_pair_inputs)
            task_pair_inputs = _half_cslpae_pair(task_pair_inputs)
            loss, losses = train_dynamic_stage1_batch(
                model, x_obs, x_full,
                missing_weight, reg_weight,
                subject_summary_contra_weight, task_summary_contra_weight,
                subject_correction_contra_weight, task_correction_contra_weight,
                permute_sub_weight, permute_task_weight,
                sub_pair_inputs, task_pair_inputs)
        else:
            with torch.cuda.amp.autocast():
                loss, losses = train_dynamic_stage1_batch(
                    model, x_obs, x_full,
                    missing_weight, reg_weight,
                    subject_summary_contra_weight, task_summary_contra_weight,
                    subject_correction_contra_weight, task_correction_contra_weight,
                    permute_sub_weight, permute_task_weight,
                    sub_pair_inputs, task_pair_inputs)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        if loss_scaler is None:
            loss /= update_freq
            model.backward(loss)
            model.step()

            if (data_iter_step + 1) % update_freq == 0:
                # model.zero_grad()
                # Deepspeed will call step() & model.zero_grad() automatic
                if model_ema is not None:
                    model_ema.update(model)
            grad_norm = None
            loss_scale_value = get_loss_scale_for_deepspeed(model)
        else:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
            loss_scale_value = loss_scaler.state_dict()["scale"]

        if device.type == "cuda":
            torch.cuda.synchronize()

        # 原分类指标：metric_logger.update(class_acc=class_acc)
            
        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_missing=losses["missing"].item())
        metric_logger.update(loss_reg=losses["reg"].item())
        metric_logger.update(loss_subject_summary_contra=losses["subject_summary_contra"].item())
        metric_logger.update(loss_task_summary_contra=losses["task_summary_contra"].item())
        metric_logger.update(loss_subject_correction_contra=losses["subject_correction_contra"].item())
        metric_logger.update(loss_task_correction_contra=losses["task_correction_contra"].item())
        metric_logger.update(loss_permute_sub=losses["permute_sub"].item())
        metric_logger.update(loss_permute_task=losses["permute_task"].item())
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(loss_missing=losses["missing"].item(), head="loss")
            log_writer.update(loss_reg=losses["reg"].item(), head="loss")
            log_writer.update(loss_subject_summary_contra=losses["subject_summary_contra"].item(), head="loss")
            log_writer.update(loss_task_summary_contra=losses["task_summary_contra"].item(), head="loss")
            log_writer.update(loss_subject_correction_contra=losses["subject_correction_contra"].item(), head="loss")
            log_writer.update(loss_task_correction_contra=losses["task_correction_contra"].item(), head="loss")
            log_writer.update(loss_permute_sub=losses["permute_sub"].item(), head="loss")
            log_writer.update(loss_permute_task=losses["permute_task"].item(), head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")

            log_writer.set_step()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
# 原分类验证入口：
# def evaluate(data_loader, model, device, header='Test:', ch_names=None, ...):
def evaluate_dynamic_stage1(data_loader, model, device, header='Dynamic Stage 1:',
                            input_scale=0.01, missing_weight=1.0, reg_weight=0.01,
                            subject_summary_contra_weight=0.0, task_summary_contra_weight=0.0,
                            subject_correction_contra_weight=0.0, task_correction_contra_weight=0.0,
                            permute_sub_weight=1.0, permute_task_weight=1.0):

    metric_logger = utils.MetricLogger(delimiter="  ")
    #header = 'Test:'

    # switch to evaluation mode
    model.eval()
    for batch in metric_logger.log_every(data_loader, 10, header):
        # 原分类验证使用 EEG 和 target；Stage 1 使用 x_obs 和 x_full。
        _, _, x_obs, x_full, _, _ = batch
        x_obs = x_obs.float().to(device, non_blocking=True) * input_scale
        x_full = x_full.float().to(device, non_blocking=True) * input_scale
        x_obs = rearrange(x_obs, 'B N (A T) -> B N A T', T=200)
        x_full = rearrange(x_full, 'B N (A T) -> B N A T', T=200)

        pair_batch_size = x_obs.shape[0]
        sub_pair_inputs = None
        if (subject_summary_contra_weight > 0.0
                or subject_correction_contra_weight > 0.0
                or permute_sub_weight > 0.0):
            sub_pair_inputs = _prepare_cslpae_pair(
                data_loader.dataset, "subject", pair_batch_size, device, input_scale
            )
        task_pair_inputs = None
        if (task_summary_contra_weight > 0.0
                or task_correction_contra_weight > 0.0
                or permute_task_weight > 0.0):
            task_pair_inputs = _prepare_cslpae_pair(
                data_loader.dataset, "task", pair_batch_size, device, input_scale
            )
        
        # compute output
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            loss, losses = train_dynamic_stage1_batch(
                model, x_obs, x_full,
                missing_weight, reg_weight,
                subject_summary_contra_weight, task_summary_contra_weight,
                subject_correction_contra_weight, task_correction_contra_weight,
                permute_sub_weight, permute_task_weight,
                sub_pair_inputs, task_pair_inputs)

        batch_size = x_obs.shape[0]
        metric_logger.meters['loss'].update(loss.item(), n=batch_size)
        metric_logger.meters['loss_missing'].update(losses['missing'].item(), n=batch_size)
        metric_logger.meters['loss_reg'].update(losses['reg'].item(), n=batch_size)
        metric_logger.meters['loss_subject_summary_contra'].update(losses['subject_summary_contra'].item(), n=batch_size)
        metric_logger.meters['loss_task_summary_contra'].update(losses['task_summary_contra'].item(), n=batch_size)
        metric_logger.meters['loss_subject_correction_contra'].update(losses['subject_correction_contra'].item(), n=batch_size)
        metric_logger.meters['loss_task_correction_contra'].update(losses['task_correction_contra'].item(), n=batch_size)
        metric_logger.meters['loss_permute_sub'].update(losses['permute_sub'].item(), n=batch_size)
        metric_logger.meters['loss_permute_task'].update(losses['permute_task'].item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* loss {losses.global_avg:.3f}'.format(losses=metric_logger.loss))
    return {key: meter.global_avg for key, meter in metric_logger.meters.items()}
