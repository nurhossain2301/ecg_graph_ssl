import torch
from torch.utils.data import DataLoader
import numpy as np
import random
from loss import masked_loss
from tqdm import tqdm
import wandb
import torch.distributed as dist



# -------------------------------------------------------
# Train One Epoch
# -------------------------------------------------------
def train_one_epoch(model, loader, optimizer, device, rank, global_step, log_interval):

    model.train()

    total_loss = 0
    num_batches = 0
    pbar = tqdm(loader)
    for batch in pbar:

        beats = batch["beats"].to(device)
        rr = batch["rr"].to(device)
        valid_mask = batch["valid_mask"].to(device)
        node_mask = batch["node_mask"].to(device)

        x, recon = model(beats, rr)

        effective_mask = node_mask & valid_mask
        loss = masked_loss(x, recon, effective_mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ---- Logging every N global steps ----
        if rank == 0 and global_step % log_interval == 0:
            wandb.log({
                "train_loss_step": loss.item(),
                "global_step": global_step,
                "lr": optimizer.param_groups[0]["lr"]
            })

        total_loss += loss.item()
        num_batches += 1
        global_step += 1

    return total_loss / num_batches, global_step


# -------------------------------------------------------
# Validation
# -------------------------------------------------------
@torch.no_grad()
def validate_one_epoch(model, loader, device):

    model.eval()

    total_loss = torch.tensor(0.0, device=device)
    count = torch.tensor(0.0, device=device)
    pbar = tqdm(loader)

    for batch in pbar:

        beats = batch["beats"].to(device)
        rr = batch["rr"].to(device)
        valid_mask = batch["valid_mask"].to(device)
        node_mask = batch["node_mask"].to(device)

        x, recon = model(beats, rr)

        effective_mask = node_mask & valid_mask
        loss = masked_loss(x, recon, effective_mask)

        total_loss += loss
        count += 1

    # ---- Average across all GPUs ----
    dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(count, op=dist.ReduceOp.SUM)

    return (total_loss / count).item()




