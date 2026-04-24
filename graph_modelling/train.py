import torch
from torch.utils.data import DataLoader
import numpy as np
import random
from loss import ssl_total_loss
from tqdm import tqdm
import wandb
import torch.distributed as dist



# -------------------------------------------------------
# Train One Epoch
# -------------------------------------------------------

def train_one_epoch(model, loader, optimizer, device, rank, global_step, log_interval, momentum=0.99):
    model.train()

    total_loss = 0.0
    total_mask = 0.0
    total_byol = 0.0
    num_batches = 0

    for batch in loader:
        beats = batch["beats"].to(device)
        rr = batch["rr"].to(device)
        valid_mask = batch["valid_mask"].to(device)
        node_mask = batch["node_mask"].to(device)

        beats1 = batch["beats_view1"].to(device)
        rr1 = batch["rr_view1"].to(device)
        valid1 = batch["valid_mask_view1"].to(device)

        beats2 = batch["beats_view2"].to(device)
        rr2 = batch["rr_view2"].to(device)
        valid2 = batch["valid_mask_view2"].to(device)

        target, recon, _ = model.module.masked_forward(beats, rr, node_mask, valid_mask)
        p1, z2 = model.module.byol_forward(beats1, rr1, valid1, beats2, rr2, valid2)

        loss, loss_mask, loss_byol = ssl_total_loss(
            recon=recon,
            target=target,
            node_mask=node_mask,
            valid_mask=valid_mask,
            p1=p1,
            z2=z2,
            lambda_byol=0.5,
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.module.parameters(), 1.0)
        optimizer.step()
        model.module.update_target(momentum=momentum)

        if rank == 0 and global_step % log_interval == 0:
            with torch.no_grad():
                _, _, attn1, _ = model.module.byol_forward(
                    beats1, rr1, valid1, beats2, rr2, valid2, return_attn=True
                )

                # entropy-like summary: lower means more focused attention
                attn_mean = attn1.mean().item()
                attn_max = attn1.max().item()

            wandb.log({
                "train_loss_step": loss.item(),
                "train_mask_loss_step": loss_mask.item(),
                "train_byol_loss_step": loss_byol.item(),
                "attn_mean": attn_mean,
                "attn_max": attn_max,
                "global_step": global_step,
                "lr": optimizer.param_groups[0]["lr"],
            })

        total_loss += loss.item()
        total_mask += loss_mask.item()
        total_byol += loss_byol.item()
        num_batches += 1
        global_step += 1

    return {
        "loss": total_loss / num_batches,
        "mask_loss": total_mask / num_batches,
        "byol_loss": total_byol / num_batches,
        "global_step": global_step,
    }


# -------------------------------------------------------
# Validation
# -------------------------------------------------------
@torch.no_grad()
def validate_one_epoch(model, loader, device):

    model.eval()

    total_loss = torch.tensor(0.0, device=device)
    total_mask = torch.tensor(0.0, device=device)
    total_byol = torch.tensor(0.0, device=device)
    count = torch.tensor(0.0, device=device)

    for batch in loader:

        beats = batch["beats"].to(device)
        rr = batch["rr"].to(device)
        valid_mask = batch["valid_mask"].to(device)
        node_mask = batch["node_mask"].to(device)

        beats1 = batch["beats_view1"].to(device)
        rr1 = batch["rr_view1"].to(device)
        valid1 = batch["valid_mask_view1"].to(device)

        beats2 = batch["beats_view2"].to(device)
        rr2 = batch["rr_view2"].to(device)
        valid2 = batch["valid_mask_view2"].to(device)

        # -------------------------
        # Masked SSL
        # -------------------------
        target, recon, _ = model.module.masked_forward(
            beats, rr, node_mask, valid_mask
        )

        # -------------------------
        # BYOL
        # -------------------------
        p1, z2 = model.module.byol_forward(
            beats1, rr1, valid1,
            beats2, rr2, valid2
        )

        # -------------------------
        # Loss
        # -------------------------
        loss, loss_mask, loss_byol = ssl_total_loss(
            recon=recon,
            target=target,
            node_mask=node_mask,
            valid_mask=valid_mask,
            p1=p1,
            z2=z2,
            lambda_byol=0.5
        )

        total_loss += loss
        total_mask += loss_mask
        total_byol += loss_byol
        count += 1

    # -------------------------
    # DDP sync
    # -------------------------
    dist.all_reduce(total_loss)
    dist.all_reduce(total_mask)
    dist.all_reduce(total_byol)
    dist.all_reduce(count)

    return {
        "loss": (total_loss / count).item(),
        "mask_loss": (total_mask / count).item(),
        "byol_loss": (total_byol / count).item(),
    }




