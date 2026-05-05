import torch
from loss import ssl_total_loss
from tqdm import tqdm
import wandb
import torch.distributed as dist


# -------------------------------------------------------
# Train One Epoch
# -------------------------------------------------------

def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    rank,
    global_step,
    log_interval,
    get_momentum=None,
    scheduler=None,
    loss_fn=None,
):
    if loss_fn is None:
        loss_fn = ssl_total_loss

    model.train()

    total_loss = total_mask = total_mse = total_cos = total_aux = 0.0
    num_batches = 0

    for batch in loader:
        beats      = batch["beats"].to(device)
        rr         = batch["rr"].to(device)
        valid_mask = batch["valid_mask"].to(device)
        node_mask  = batch["node_mask"].to(device)
        beats1     = batch["beats_view1"].to(device)
        rr1        = batch["rr_view1"].to(device)
        valid1     = batch["valid_mask_view1"].to(device)
        beats2     = batch["beats_view2"].to(device)
        rr2        = batch["rr_view2"].to(device)
        valid2     = batch["valid_mask_view2"].to(device)

        target, recon, _, p1, z2, p2, z1_target = model(
            beats, rr, valid_mask, node_mask,
            beats1, rr1, valid1,
            beats2, rr2, valid2,
        )

        loss, loss_mask, loss_aux, loss_mse, loss_cos = loss_fn(
            recon=recon, target=target,
            node_mask=node_mask, valid_mask=valid_mask,
            p1=p1, z2=z2, p2=p2, z1=z1_target,
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.module.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        momentum = get_momentum(global_step) if get_momentum is not None else 0.996
        model.module.update_target(momentum=momentum)

        if rank == 0 and global_step % log_interval == 0:
            with torch.no_grad():
                _, _, _, _, attn1, _ = model.module.byol_forward(
                    beats1, rr1, valid1, beats2, rr2, valid2, return_attn=True
                )
            wandb.log({
                "train_loss_step":      loss.item(),
                "train_mask_loss_step": loss_mask.item(),
                "train_mask_mse_step":  loss_mse.item(),
                "train_mask_cos_step":  loss_cos.item(),
                "train_aux_loss_step":  loss_aux.item(),
                "attn_mean":            attn1.mean().item(),
                "attn_max":             attn1.max().item(),
                "global_step":          global_step,
                "lr":                   optimizer.param_groups[0]["lr"],
                "ema_momentum":         momentum,
            })

        total_loss += loss.item()
        total_mask += loss_mask.item()
        total_mse  += loss_mse.item()
        total_cos  += loss_cos.item()
        total_aux  += loss_aux.item()
        num_batches += 1
        global_step += 1

    metrics = torch.tensor(
        [total_loss, total_mask, total_mse, total_cos, total_aux, float(num_batches)],
        device=device,
    )
    dist.all_reduce(metrics)
    total_loss, total_mask, total_mse, total_cos, total_aux, num_batches = metrics.tolist()

    return {
        "loss":       total_loss / num_batches,
        "mask_loss":  total_mask / num_batches,
        "mask_mse":   total_mse  / num_batches,
        "mask_cos":   total_cos  / num_batches,
        "byol_loss":  total_aux  / num_batches,   # kept as "byol_loss" for W&B compat
        "global_step": global_step,
    }


# -------------------------------------------------------
# Validation
# -------------------------------------------------------
@torch.no_grad()
def validate_one_epoch(model, loader, device, loss_fn=None):
    if loss_fn is None:
        loss_fn = ssl_total_loss

    model.eval()

    total_loss = torch.tensor(0.0, device=device)
    total_mask = torch.tensor(0.0, device=device)
    total_mse  = torch.tensor(0.0, device=device)
    total_cos  = torch.tensor(0.0, device=device)
    total_aux  = torch.tensor(0.0, device=device)
    count      = torch.tensor(0.0, device=device)

    for batch in loader:
        beats      = batch["beats"].to(device)
        rr         = batch["rr"].to(device)
        valid_mask = batch["valid_mask"].to(device)
        node_mask  = batch["node_mask"].to(device)
        beats1     = batch["beats_view1"].to(device)
        rr1        = batch["rr_view1"].to(device)
        valid1     = batch["valid_mask_view1"].to(device)
        beats2     = batch["beats_view2"].to(device)
        rr2        = batch["rr_view2"].to(device)
        valid2     = batch["valid_mask_view2"].to(device)

        target, recon, _, p1, z2, p2, z1_target = model(
            beats, rr, valid_mask, node_mask,
            beats1, rr1, valid1,
            beats2, rr2, valid2,
        )

        loss, loss_mask, loss_aux, loss_mse, loss_cos = loss_fn(
            recon=recon, target=target,
            node_mask=node_mask, valid_mask=valid_mask,
            p1=p1, z2=z2, p2=p2, z1=z1_target,
        )

        total_loss += loss
        total_mask += loss_mask
        total_mse  += loss_mse
        total_cos  += loss_cos
        total_aux  += loss_aux
        count      += 1

    for t in [total_loss, total_mask, total_mse, total_cos, total_aux, count]:
        dist.all_reduce(t)

    return {
        "loss":      (total_loss / count).item(),
        "mask_loss": (total_mask / count).item(),
        "mask_mse":  (total_mse  / count).item(),
        "mask_cos":  (total_cos  / count).item(),
        "byol_loss": (total_aux  / count).item(),
    }
