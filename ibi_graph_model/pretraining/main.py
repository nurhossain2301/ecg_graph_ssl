import json
import math
import os
import random
import argparse

import numpy as np
import torch
import torch.distributed as dist
import wandb

from config import Config
from model import IBIGraphBYOLModel
from dataloader import load_data
from train import train_one_epoch, validate_one_epoch
from loss import build_loss_fn


def get_ema_momentum(step, total_steps, base=0.996, final=0.9999):
    progress = step / max(1, total_steps)
    return final - (final - base) * (math.cos(math.pi * progress) + 1) / 2


def curriculum_mask_ratio(epoch, total_epochs, start=0.15, end=0.50):
    """Linearly ramp mask ratio from start to end over training."""
    return start + (end - start) * (epoch / max(1, total_epochs - 1))


def build_scheduler(optimizer, cfg, steps_per_epoch):
    total_steps  = max(1, cfg.epochs * steps_per_epoch)
    warmup_steps = max(1, cfg.warmup_epochs * steps_per_epoch)
    min_ratio    = cfg.min_lr / cfg.lr

    def lr_lambda(step):
        if step < warmup_steps:
            return min_ratio + (1 - min_ratio) * step / warmup_steps
        decay = max(1, total_steps - warmup_steps)
        t     = min(1.0, (step - warmup_steps) / decay)
        return min_ratio + (1 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * t))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def setup_ddp():
    dist.init_process_group(backend="nccl")
    rank       = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return rank, local_rank


def cleanup_ddp():
    dist.destroy_process_group()


def set_seed(seed, rank):
    s = seed + rank
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--run_name",   type=str, default=None)
    args, _ = parser.parse_known_args()

    cfg = Config()
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.run_name:
        cfg.run_name = args.run_name

    loss_fn = build_loss_fn(cfg)

    rank, local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    os.makedirs(cfg.output_dir, exist_ok=True)
    set_seed(cfg.seed, rank)

    if rank == 0:
        wandb.init(
            project="ecg-ibi-graph-ssl",
            name=cfg.run_name,
            config=vars(cfg),
        )

    train_loader, val_loader, train_sampler, collator = load_data(cfg)

    model = IBIGraphBYOLModel(cfg).to(device)
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank], find_unused_parameters=True,
    )

    params = (
        list(model.module.online_encoder.parameters()) +
        list(model.module.online_projector.parameters()) +
        list(model.module.predictor.parameters()) +
        list(model.module.decoder.parameters()) +
        list(model.module.pool.parameters()) +
        list(model.module.hrv_head.parameters()) +
        list(model.module.future_head.parameters()) +
        [model.module.mask_token]
    )
    optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch=len(train_loader))

    best_val    = float("inf")
    global_step = 0
    total_steps = cfg.epochs * len(train_loader)

    for epoch in range(cfg.epochs):
        train_sampler.set_epoch(epoch)

        # Curriculum masking: ramp mask ratio linearly over training
        collator.node_mask_ratio = curriculum_mask_ratio(
            epoch, cfg.epochs, cfg.mask_ratio_start, cfg.mask_ratio_end
        )
        if rank == 0:
            wandb.log({"mask_ratio": collator.node_mask_ratio, "epoch": epoch + 1})

        train_stats = train_one_epoch(
            model, train_loader, optimizer, device, rank, global_step,
            log_interval=50,
            get_momentum=lambda step: get_ema_momentum(
                step, total_steps, cfg.ema_momentum_base, cfg.ema_momentum_final
            ),
            scheduler=scheduler,
            loss_fn=loss_fn,
        )
        global_step = train_stats["global_step"]

        val_stats = validate_one_epoch(model, val_loader, device, loss_fn=loss_fn)

        if rank == 0:
            wandb.log({
                "epoch":              epoch + 1,
                "train/loss":         train_stats["loss"],
                "train/mask_loss":    train_stats["mask_loss"],
                "train/byol_loss":    train_stats["byol_loss"],
                "train/hrv_loss":     train_stats["hrv_loss"],
                "train/future_loss":  train_stats["future_loss"],
                "val/loss":           val_stats["loss"],
                "val/mask_loss":      val_stats["mask_loss"],
                "val/byol_loss":      val_stats["byol_loss"],
                "val/hrv_loss":       val_stats["hrv_loss"],
                "val/future_loss":    val_stats["future_loss"],
                "lr":                 optimizer.param_groups[0]["lr"],
            })
            print(
                f"Epoch {epoch+1:03d} | "
                f"Train: {train_stats['loss']:.4f} "
                f"(mask {train_stats['mask_loss']:.4f} byol {train_stats['byol_loss']:.4f} "
                f"hrv {train_stats['hrv_loss']:.4f} fut {train_stats['future_loss']:.4f}) | "
                f"Val: {val_stats['loss']:.4f} | "
                f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
                f"mask_ratio: {collator.node_mask_ratio:.2f}"
            )

            torch.save(
                {
                    "epoch":       epoch + 1,
                    "model":       model.module.state_dict(),
                    "optimizer":   optimizer.state_dict(),
                    "scheduler":   scheduler.state_dict(),
                    "best_val":    best_val,
                    "global_step": global_step,
                    "cfg":         vars(cfg),
                },
                os.path.join(cfg.output_dir, "last_checkpoint.pt"),
            )

            if val_stats["loss"] < best_val:
                best_val = val_stats["loss"]
                torch.save(
                    model.module.state_dict(),
                    os.path.join(cfg.output_dir, "best_model.pt"),
                )
                wandb.log({"best_val_loss": best_val})
                print(f"  Saved best model (val_loss={best_val:.4f})")

    if rank == 0:
        summary = {
            "run_name":      cfg.run_name,
            "output_dir":    cfg.output_dir,
            "best_val_loss": best_val,
            "epochs":        cfg.epochs,
            "cfg":           vars(cfg),
        }
        with open(os.path.join(cfg.output_dir, "training_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        wandb.finish()

    cleanup_ddp()


if __name__ == "__main__":
    main()
