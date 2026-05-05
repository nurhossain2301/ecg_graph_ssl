import torch
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
import pandas as pd
import numpy as np
import random
import argparse
import wandb
import os
import math

from config import Config
from dataset import ECGGraphBYOLDataset, GraphBYOLCollator
from model import ECGBYOLModel
from dataloader import load_data
from train import train_one_epoch, validate_one_epoch
from loss import build_loss_fn


def get_ema_momentum(step, total_steps, base=0.996, final=0.9999):
    """Cosine-anneal EMA momentum from base→final (BYOL §3.1).

    At step 0  → base  (0.996): target updates quickly enough to bootstrap
    At step K  → final (0.9999): target is nearly frozen once training stabilises
    """
    progress = step / max(1, total_steps)
    return final - (final - base) * (math.cos(math.pi * progress) + 1) / 2


def build_warmup_cosine_scheduler(optimizer, cfg, steps_per_epoch):
    total_steps = max(1, cfg.epochs * steps_per_epoch)
    warmup_steps = max(1, cfg.warmup_epochs * steps_per_epoch)
    min_lr_ratio = cfg.min_lr / cfg.lr

    def lr_lambda(step):
        if step < warmup_steps:
            warmup_progress = step / warmup_steps
            return min_lr_ratio + (1.0 - min_lr_ratio) * warmup_progress

        decay_steps = max(1, total_steps - warmup_steps)
        decay_progress = min(1.0, (step - warmup_steps) / decay_steps)
        cosine = 0.5 * (1.0 + np.cos(np.pi * decay_progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# -------------------------------------------------------
# DDP SETUP
# -------------------------------------------------------
def setup_ddp():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def cleanup_ddp():
    dist.destroy_process_group()

# -------------------------------------------------------
# Reproducibility
# -------------------------------------------------------
def set_seed(seed, rank):
    seed = seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -------------------------------------------------------
# Main Training Loop
# -------------------------------------------------------
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--run_name",   type=str, default=None)
    parser.add_argument("--loss_type",  type=str, default="baseline",
                        choices=["baseline", "huber", "vicreg", "barlow", "spectral"])
    parser.add_argument("--overfit", action="store_true",
                        help="Train on a tiny fixed subset to verify the model can memorize")
    args, _ = parser.parse_known_args()

    cfg = Config()
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.run_name:
        cfg.run_name = args.run_name
    if args.overfit:
        cfg.overfit       = True
        cfg.warmup_epochs = 0     # no warmup — go straight to full LR
        cfg.weight_decay  = 0.0   # no regularisation
        cfg.epochs        = 200   # train long enough to see memorisation

    loss_fn = build_loss_fn(args.loss_type)

    rank, world_size, local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    os.makedirs(cfg.output_dir, exist_ok=True)
    set_seed(cfg.seed, rank)
    # -----------------------------
    # Initialize W&B
    # -----------------------------
    if rank == 0:
        wandb.init(
            project="ecg-graph-ssl",
            name=cfg.run_name,
            config={**vars(cfg), "loss_type": args.loss_type},
        )
    
    # ---------------------------------------------------
    # Datasets
    # ---------------------------------------------------
    train_loader, val_loader, train_sampler = load_data(cfg)

    # ---------------------------------------------------
    # Model
    # ---------------------------------------------------
    model = ECGBYOLModel(cfg).to(device)

    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        find_unused_parameters=True   # 🔥 IMPORTANT for BYOL branches
    )


    params = (
        list(model.module.online_encoder.parameters()) +
        list(model.module.online_projector.parameters()) +
        list(model.module.predictor.parameters()) +
        list(model.module.decoder.parameters()) +
        list(model.module.pool.parameters()) +
        [model.module.mask_token]
    )

    optimizer = torch.optim.AdamW(
        params,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay
    )
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        cfg,
        steps_per_epoch=len(train_loader),
    )
    # scaler = GradScaler(enabled=cfg.amp)

    # ---------------------------------------------------
    # Training
    # ---------------------------------------------------
    best_val = float("inf")
    global_step = 0
    log_interval = 50
    total_steps = cfg.epochs * len(train_loader)

    for epoch in range(cfg.epochs):
        train_sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            rank,
            global_step,
            log_interval,
            get_momentum=lambda step: get_ema_momentum(step, total_steps),
            scheduler=scheduler,
            loss_fn=loss_fn,
        )

        global_step = train_stats["global_step"]

        val_stats = validate_one_epoch(model, val_loader, device, loss_fn=loss_fn)

        # ---------------------------------------------------
        # Logging
        # ---------------------------------------------------
        if rank == 0:
            wandb.log({
                "epoch": epoch + 1,

                "train_loss": train_stats["loss"],
                "train_mask_loss": train_stats["mask_loss"],
                "train_mask_mse": train_stats["mask_mse"],
                "train_mask_cos": train_stats["mask_cos"],
                "train_byol_loss": train_stats["byol_loss"],

                "val_loss": val_stats["loss"],
                "val_mask_loss": val_stats["mask_loss"],
                "val_mask_mse": val_stats["mask_mse"],
                "val_mask_cos": val_stats["mask_cos"],
                "val_byol_loss": val_stats["byol_loss"],
                "lr": optimizer.param_groups[0]["lr"],
            })

            print(
                f"Epoch {epoch+1:03d} | "
                f"Train Loss: {train_stats['loss']:.4f} "
                f"(mask {train_stats['mask_loss']:.4f}, "
                f"mse {train_stats['mask_mse']:.4f}, "
                f"cos {train_stats['mask_cos']:.4f}, "
                f"byol {train_stats['byol_loss']:.4f}) | "
                f"Val Loss: {val_stats['loss']:.4f} "
                f"(mask {val_stats['mask_loss']:.4f}, "
                f"mse {val_stats['mask_mse']:.4f}, "
                f"cos {val_stats['mask_cos']:.4f}, "
                f"byol {val_stats['byol_loss']:.4f}) | "
                f"LR: {optimizer.param_groups[0]['lr']:.2e}"
            )

            # Save last checkpoint every epoch
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model": model.module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_val": best_val,
                    "global_step": global_step,
                    "cfg": vars(cfg),
                },
                os.path.join(cfg.output_dir, "last_checkpoint.pt"),
            )

            # Save best model based on reconstruction loss only; BYOL val is
            # high-variance (no target update during val, augmentation noise)
            # and is not a reliable signal for model quality.
            if val_stats["mask_loss"] < best_val:
                best_val = val_stats["mask_loss"]
                torch.save(model.module.state_dict(),
                           os.path.join(cfg.output_dir, "best_model.pt"))
                wandb.log({"best_val_mask_loss": best_val})
                print(f"✅ Saved Best Model (val_mask_loss={best_val:.4f})")

    if rank == 0:
        import json
        summary = {
            "run_name":      cfg.run_name,
            "loss_type":     args.loss_type,
            "output_dir":    cfg.output_dir,
            "best_val_loss": best_val,
            "epochs":        cfg.epochs,
            "cfg":           vars(cfg),
        }
        with open(os.path.join(cfg.output_dir, "training_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Results saved to {cfg.output_dir}")
        wandb.finish()

    cleanup_ddp()


# -------------------------------------------------------
if __name__ == "__main__":
    main()
