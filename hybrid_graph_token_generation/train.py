import torch
from torch.utils.data import DataLoader
import numpy as np
import random
from loss import total_loss
from tqdm import tqdm
import wandb
import torch.distributed as dist



# -------------------------------------------------------
# Train One Epoch
# -------------------------------------------------------
def train_one_epoch(model, loader, optimizer, device, rank, global_step, log_interval, cfg):

    model.train()

    total_loss = 0
    num_batches = 0

    pbar = tqdm(loader)

    for batch in pbar:

        tokens = batch["tokens"].to(device)
        features = batch["features"].to(device)
        valid_mask = batch["valid_mask"].to(device)
        node_mask = batch["node_mask"].to(device)
        edge_index = [e.to(device) for e in batch["edge_index"]]

        # ---- Forward ----
        out = model(
            tokens=tokens,
            features=features,
            valid_mask=valid_mask,
            node_mask=node_mask,
            edge_index=edge_index
        )

        # ---- Loss ----
        loss_dict = total_loss(
            outputs=out,
            tokens=tokens,
            node_mask=node_mask,
            valid_mask=valid_mask,
            edge_index=edge_index,
            lambda_smooth=cfg.lambda_smooth,
            lambda_align=cfg.lambda_align,
        )

        loss = loss_dict["loss"]
        loss_mtm = loss_dict["loss_mtm"]
        loss_smooth = loss_dict["loss_smooth"]
        loss_align = loss_dict["loss_align"]

        # ---- Backprop ----
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ---- Logging ----
        if rank == 0 and global_step % log_interval == 0:
            wandb.log({
                "train_loss_step": loss.item(),
                "loss_mtm": loss_mtm.item(),
                "loss_smooth": loss_smooth.item(),
                "loss_align": loss_align.item(),
                "global_step": global_step,
                "lr": optimizer.param_groups[0]["lr"]
            })

        total_loss += loss.item()
        num_batches += 1
        global_step += 1

        pbar.set_description(f"loss: {loss.item():.4f}")

    return total_loss / num_batches, global_step


# -------------------------------------------------------
# Validation
# -------------------------------------------------------
@torch.no_grad()
def validate_one_epoch(model, loader, device, cfg):

    model.eval()

    total_loss = torch.tensor(0.0, device=device)
    count = torch.tensor(0.0, device=device)

    pbar = tqdm(loader)

    for batch in pbar:

        tokens = batch["tokens"].to(device)
        features = batch["features"].to(device)
        valid_mask = batch["valid_mask"].to(device)
        node_mask = batch["node_mask"].to(device)
        edge_index = [e.to(device) for e in batch["edge_index"]]

        out = model(
            tokens=tokens,
            features=features,
            valid_mask=valid_mask,
            node_mask=node_mask,
            edge_index=edge_index
        )

        out = model(
            tokens=tokens,
            features=features,
            valid_mask=valid_mask,
            node_mask=node_mask,
            edge_index=edge_index
        )

        loss_dict = total_loss(
            outputs=out,
            tokens=tokens,
            node_mask=node_mask,
            valid_mask=valid_mask,
            edge_index=edge_index,
            lambda_smooth=cfg.lambda_smooth,
            lambda_align=cfg.lambda_align,
        )

        total_mtm += loss_dict["loss_mtm"]
        total_full += loss_dict["loss"]
        count += 1

    dist.all_reduce(total_mtm)
    dist.all_reduce(total_full)
    dist.all_reduce(count)

    return {
        "val_mtm": (total_mtm / count).item(),
        "val_full": (total_full / count).item()
    }



