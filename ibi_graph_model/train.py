import torch
from tqdm import tqdm
from loss import masked_loss
import wandb
import torch.distributed as dist


def train_one_epoch(model, loader, optimizer, device, rank, global_step, log_interval, grad_clip=1.0):
    model.train()

    total_loss = 0.0
    num_batches = 0
    pbar = tqdm(loader, disable=(rank != 0))

    for batch in pbar:
        beats = batch["beats"].to(device)
        rr = batch["rr"].to(device)
        valid_mask = batch["valid_mask"].to(device)
        node_mask = batch["node_mask"].to(device)

        target, recon = model(beats, rr, node_mask, valid_mask)
        loss = masked_loss(recon, target, node_mask, valid_mask)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        if rank == 0 and global_step % log_interval == 0:
            wandb.log({
                "train_loss_step": loss.item(),
                "global_step": global_step,
                "lr": optimizer.param_groups[0]["lr"]
            })

        total_loss += loss.item()
        num_batches += 1
        global_step += 1

        if rank == 0:
            pbar.set_description(f"train loss: {loss.item():.4f}")

    return total_loss / max(num_batches, 1), global_step


@torch.no_grad()
def validate_one_epoch(model, loader, device, rank=0):
    model.eval()

    total_loss = torch.tensor(0.0, device=device)
    count = torch.tensor(0.0, device=device)
    pbar = tqdm(loader, disable=(rank != 0))

    for batch in pbar:
        beats = batch["beats"].to(device)
        rr = batch["rr"].to(device)
        valid_mask = batch["valid_mask"].to(device)
        node_mask = batch["node_mask"].to(device)

        target, recon = model(beats, rr, node_mask, valid_mask)
        loss = masked_loss(recon, target, node_mask, valid_mask)

        total_loss += loss.detach()
        count += 1

    dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(count, op=dist.ReduceOp.SUM)

    return (total_loss / torch.clamp(count, min=1.0)).item()