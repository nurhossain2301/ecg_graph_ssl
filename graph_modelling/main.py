import torch
from torch.utils.data import DataLoader
import torch.distributed as dist
import numpy as np
import random
import argparse
import wandb
import os

from config import Config
from dataset import ECGGraphDataset, GraphSSL_Collator
from model import ECGModel
from loss import masked_loss
from dataloader import load_data
from train import train_one_epoch, validate_one_epoch

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

    cfg = Config()
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
            # name=f"sft_{os.path.basename(args.output_dir)}",
            config=vars(cfg),
        )
    
    # ---------------------------------------------------
    # Datasets
    # ---------------------------------------------------
    train_loader, val_loader, train_sampler = load_data(cfg)

    # ---------------------------------------------------
    # Model
    # ---------------------------------------------------
    model = ECGModel(cfg).to(device)
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank]
    )


    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    # scaler = GradScaler(enabled=cfg.amp)

    # ---------------------------------------------------
    # Training
    # ---------------------------------------------------
    best_val = float("inf")
    global_step = 0
    log_interval = 50

    for epoch in range(cfg.epochs):
        train_sampler.set_epoch(epoch)

        # train_loss = train_one_epoch(model, train_loader, optimizer, cfg.device)
        # val_loss = validate_one_epoch(model, val_loader, cfg.device)
        train_loss, global_step = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            rank,
            global_step,
            log_interval
        )

        val_loss = validate_one_epoch(model, val_loader, device)

        # ---------------------------------------------------
        # Logging
        # ---------------------------------------------------
        if rank == 0:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss_epoch": train_loss,
                "val_loss": val_loss
            })

            print(
                f"Epoch {epoch+1:03d} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f}"
            )

            # Save best model
            if val_loss < best_val:
                best_val = val_loss
                torch.save(model.module.state_dict(), "best_model.pt")

                wandb.log({"best_val_loss": best_val})
                print("✅ Saved Best Model")

    if rank == 0:
        wandb.finish()

    cleanup_ddp()


# -------------------------------------------------------
if __name__ == "__main__":
    main()