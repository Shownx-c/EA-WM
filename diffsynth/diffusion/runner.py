import math
import os

import torch
from tqdm import tqdm
from accelerate import Accelerator

from .training_module import DiffusionTrainingModule
from .logger import ModelLogger

from torch.utils.tensorboard import SummaryWriter


def _unwrap_distributed_model(model):
    while isinstance(model, (torch.nn.parallel.DistributedDataParallel, torch.nn.DataParallel)):
        model = model.module
    return model


def _build_scheduler(optimizer, scheduler_type: str, warmup_steps: int, total_steps: int):
    scheduler_type = (scheduler_type or "constant").lower()
    warmup_steps = max(0, int(warmup_steps))
    total_steps = max(1, int(total_steps))

    def lr_lambda(current_step: int):
        step = min(current_step, total_steps)

        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))

        if scheduler_type == "linear":
            progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            progress = min(max(progress, 0.0), 1.0)
            return max(0.0, 1.0 - progress)

        if scheduler_type == "cosine":
            progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            progress = min(max(progress, 0.0), 1.0)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def _scalar(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return float(value.detach().float().mean().item())
    return float(value)


def launch_training_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 1,
    save_steps: int = None,
    num_epochs: int = 1,
    args=None,
):
    lr_scheduler_type = "constant"
    lr_warmup_steps = 0
    max_grad_norm = 0.0
    stage1_epochs = 0

    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        num_epochs = args.num_epochs
        lr_scheduler_type = getattr(args, "lr_scheduler_type", "constant")
        lr_warmup_steps = getattr(args, "lr_warmup_steps", 0)
        max_grad_norm = getattr(args, "max_grad_norm", 0.0)
        stage1_epochs = getattr(args, "stage1_epochs", 0)

    stage1_epochs = max(0, min(int(stage1_epochs), int(num_epochs)))
    writer = None
    if accelerator.is_main_process:
        output_path = getattr(args, "output_path", "./models") if args is not None else "./models"
        writer = SummaryWriter(log_dir=os.path.join(output_path, "logs_tensor"))

    if hasattr(model, "set_training_stage"):
        model.set_training_stage("stage1" if stage1_epochs > 0 else "stage2")

    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)

    if hasattr(model, "build_optimizer_param_groups"):
        optimizer_param_groups = model.build_optimizer_param_groups(
            args=args,
            base_learning_rate=learning_rate,
            weight_decay=weight_decay,
        )
        optimizer = torch.optim.AdamW(optimizer_param_groups, lr=learning_rate, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)

    total_steps = max(1, int(num_epochs) * len(dataloader))
    scheduler = _build_scheduler(optimizer, lr_scheduler_type, lr_warmup_steps, total_steps)

    model.to(device=accelerator.device)
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)

    for epoch_id in range(num_epochs):
        if stage1_epochs > 0 and epoch_id == stage1_epochs:
            unwrapped_model = _unwrap_distributed_model(model)
            if hasattr(unwrapped_model, "set_training_stage"):
                unwrapped_model.set_training_stage("stage2")
                accelerator.print(
                    f"[TrainStage] switching to stage2 at epoch={epoch_id}. bidirectional cross-attention is now unfrozen."
                )

        for idx, data in enumerate(tqdm(dataloader)):
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                if dataset.load_from_cache:
                    loss = model({}, inputs=data)
                else:
                    loss = model(data)
                loss_components = None
                unwrapped_model = _unwrap_distributed_model(model)
                pipe = getattr(unwrapped_model, "pipe", None)
                if pipe is not None:
                    loss_components = getattr(pipe, "_last_eawm_loss_components", None)
                accelerator.backward(loss)

                if max_grad_norm is not None and max_grad_norm > 0 and accelerator.sync_gradients:
                    if hasattr(unwrapped_model, "trainable_modules"):
                        params_for_clip = unwrapped_model.trainable_modules()
                    else:
                        params_for_clip = (p for p in model.parameters() if p.requires_grad)
                    accelerator.clip_grad_norm_(params_for_clip, max_grad_norm)

                optimizer.step()
                model_logger.on_step_end(accelerator, model, save_steps, loss=loss)
                scheduler.step()

                global_step = epoch_id * len(dataloader) + idx
                loss_value = _scalar(loss)
                if writer is not None:
                    writer.add_scalar("loss", loss_value, global_step=global_step)
                    writer.add_scalar("epoch", epoch_id, global_step=global_step)
                if loss_components is not None:
                    video_loss = _scalar(loss_components.get("video_loss"))
                    kvaf_loss = _scalar(loss_components.get("kvaf_loss"))
                    event_loss = _scalar(loss_components.get("event_loss"))
                    flow_loss = _scalar(loss_components.get("flow_loss"))
                    weighted_total_loss = _scalar(loss_components.get("total_loss"))
                    if writer is not None:
                        writer.add_scalar("loss/video", video_loss, global_step=global_step)
                        writer.add_scalar("loss/kvaf", kvaf_loss, global_step=global_step)
                        if event_loss is not None:
                            writer.add_scalar("loss/event", event_loss, global_step=global_step)
                        if flow_loss is not None:
                            writer.add_scalar("loss/flow_weighted", flow_loss, global_step=global_step)
                        writer.add_scalar("loss/total_weighted", weighted_total_loss, global_step=global_step)
                    if global_step < 5 or (global_step + 1) % 10 == 0:
                        event_part = "" if event_loss is None else f"event={event_loss:.6f} "
                        accelerator.print(
                            f"[LossComponents] step={global_step} "
                            f"loss={loss_value:.6f} video={video_loss:.6f} kvaf={kvaf_loss:.6f} "
                            f"{event_part}"
                            f"weighted_total={weighted_total_loss:.6f}"
                        )

        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)
    model_logger.on_training_end(accelerator, model, save_steps)

    if writer is not None:
        writer.close()


def launch_data_process_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    num_workers: int = 8,
    args=None,
):
    if args is not None:
        num_workers = args.dataset_num_workers

    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, collate_fn=lambda x: x[0], num_workers=num_workers)
    model.to(device=accelerator.device)
    model, dataloader = accelerator.prepare(model, dataloader)

    for data_id, data in enumerate(tqdm(dataloader)):
        with accelerator.accumulate(model):
            with torch.no_grad():
                folder = os.path.join(model_logger.output_path, str(accelerator.process_index))
                os.makedirs(folder, exist_ok=True)
                save_path = os.path.join(model_logger.output_path, str(accelerator.process_index), f"{data_id}.pth")
                data = model(data)
                torch.save(data, save_path)
