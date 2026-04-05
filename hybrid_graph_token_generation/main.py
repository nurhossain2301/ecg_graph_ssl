import torch
import torch.distributed as dist
import numpy as np
import random
import wandb
import os

from config import Config
from dataloader import load_data
from model import ECGMotifGraphModel
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
# MAIN
# -------------------------------------------------------
def main():

    cfg = Config()

    rank, world_size, local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")

    os.makedirs(cfg.output_dir, exist_ok=True)
    set_seed(cfg.seed, rank)

    # -----------------------------
    # W&B
    # -----------------------------
    if rank == 0:
        wandb.init(
            project="ecg-token-graph",
            name=cfg.run_name,
            config=vars(cfg),
        )

    # ---------------------------------------------------
    # DATA
    # ---------------------------------------------------
    train_loader, val_loader, train_sampler = load_data(cfg)


    # ---------------------------------------------------
    # MODEL
    # ---------------------------------------------------
    model = ECGMotifGraphModel(
        input_dim=cfg.emb_dim,
        model_dim=cfg.model_dim,
        vocab_size=cfg.n_clusters,
        num_heads=cfg.num_heads,
        num_seq_layers=cfg.num_seq_layers,
        num_graph_layers=cfg.num_graph_layers,
        dropout=cfg.dropout,
        max_len=cfg.max_len,
        fusion_alpha=cfg.fusion_alpha
    ).to(device)

    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank]
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay
    )

    # ---------------------------------------------------
    # TRAIN LOOP
    # ---------------------------------------------------
    best_val = float("inf")
    global_step = 0

    for epoch in range(cfg.epochs):

        train_sampler.set_epoch(epoch)

        train_loss, global_step = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            rank,
            global_step,
            cfg.log_interval,
            cfg
        )

        val_metrics = validate_one_epoch(model, val_loader, device, cfg)

        val_loss = val_metrics["val_mtm"] 

        # -----------------------------
        # LOGGING
        # -----------------------------
        if rank == 0:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss_epoch": train_loss,
                "val_mtm": val_metrics["val_mtm"],
                "val_full": val_metrics["val_full"]
            })

            print(
                f"Epoch {epoch+1:03d} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f}"
            )

            # Save best model
            if val_loss < best_val:
                best_val = val_loss
                torch.save(
                    model.module.state_dict(),
                    os.path.join(cfg.output_dir, "best_model.pt")
                )

                wandb.log({"best_val_loss": best_val})
                print("✅ Saved Best Model")

    if rank == 0:
        wandb.finish()

    cleanup_ddp()


# -------------------------------------------------------
if __name__ == "__main__":
    main()