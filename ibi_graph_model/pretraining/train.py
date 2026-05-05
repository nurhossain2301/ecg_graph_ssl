import torch
import torch.distributed as dist
from tqdm import tqdm
import wandb


def train_one_epoch(model, loader, optimizer, device, rank, global_step,
                    log_interval=50, get_momentum=None, scheduler=None,
                    loss_fn=None):
    model.train()

    totals = dict(loss=0., mask=0., byol=0., hrv=0., future=0.)
    num_batches = 0
    pbar = tqdm(loader, disable=(rank != 0))

    for batch in pbar:
        beats      = batch["beats"].to(device)
        valid_mask = batch["valid_mask"].to(device)
        node_mask  = batch["node_mask"].to(device)
        beats1     = batch["beats_view1"].to(device)
        valid1     = batch["valid_mask_view1"].to(device)
        beats2     = batch["beats_view2"].to(device)
        valid2     = batch["valid_mask_view2"].to(device)
        target_hrv = batch["hrv"].to(device)                   # [B, 14]

        (target_node, recon,
         p1, z2, p2, z1t,
         pred_hrv,
         pred_future, tgt_future) = model(
            beats, valid_mask, node_mask,
            beats1, valid1, beats2, valid2,
        )

        loss, l_mask, l_byol, l_hrv, l_future = loss_fn(
            recon=recon, target_node=target_node,
            node_mask=node_mask, valid_mask=valid_mask,
            p1=p1, z2=z2, p2=p2, z1=z1t,
            pred_hrv=pred_hrv, target_hrv=target_hrv,
            pred_future=pred_future, target_future=tgt_future,
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
            wandb.log({
                "train/loss":        loss.item(),
                "train/mask_loss":   l_mask.item(),
                "train/byol_loss":   l_byol.item(),
                "train/hrv_loss":    l_hrv.item(),
                "train/future_loss": l_future.item(),
                "global_step":       global_step,
                "lr":                optimizer.param_groups[0]["lr"],
                "ema_momentum":      momentum,
            })

        totals["loss"]   += loss.item()
        totals["mask"]   += l_mask.item()
        totals["byol"]   += l_byol.item()
        totals["hrv"]    += l_hrv.item()
        totals["future"] += l_future.item()
        num_batches += 1
        global_step += 1

        if rank == 0:
            pbar.set_description(
                f"loss:{loss.item():.4f} mask:{l_mask.item():.4f} "
                f"hrv:{l_hrv.item():.4f} fut:{l_future.item():.4f}"
            )

    metrics = torch.tensor(
        [totals["loss"], totals["mask"], totals["byol"],
         totals["hrv"], totals["future"], float(num_batches)],
        device=device,
    )
    dist.all_reduce(metrics)
    v = metrics.tolist()

    return {
        "loss":        v[0] / v[5],
        "mask_loss":   v[1] / v[5],
        "byol_loss":   v[2] / v[5],
        "hrv_loss":    v[3] / v[5],
        "future_loss": v[4] / v[5],
        "global_step": global_step,
    }


@torch.no_grad()
def validate_one_epoch(model, loader, device, loss_fn=None):
    model.eval()

    totals = {k: torch.tensor(0.0, device=device)
              for k in ["loss", "mask", "byol", "hrv", "future", "count"]}

    for batch in loader:
        beats      = batch["beats"].to(device)
        valid_mask = batch["valid_mask"].to(device)
        node_mask  = batch["node_mask"].to(device)
        beats1     = batch["beats_view1"].to(device)
        valid1     = batch["valid_mask_view1"].to(device)
        beats2     = batch["beats_view2"].to(device)
        valid2     = batch["valid_mask_view2"].to(device)
        target_hrv = batch["hrv"].to(device)

        (target_node, recon,
         p1, z2, p2, z1t,
         pred_hrv,
         pred_future, tgt_future) = model(
            beats, valid_mask, node_mask,
            beats1, valid1, beats2, valid2,
        )

        loss, l_mask, l_byol, l_hrv, l_future = loss_fn(
            recon=recon, target_node=target_node,
            node_mask=node_mask, valid_mask=valid_mask,
            p1=p1, z2=z2, p2=p2, z1=z1t,
            pred_hrv=pred_hrv, target_hrv=target_hrv,
            pred_future=pred_future, target_future=tgt_future,
        )

        totals["loss"]   += loss
        totals["mask"]   += l_mask
        totals["byol"]   += l_byol
        totals["hrv"]    += l_hrv
        totals["future"] += l_future
        totals["count"]  += 1

    for t in totals.values():
        dist.all_reduce(t)

    c = totals["count"]
    return {
        "loss":        (totals["loss"]   / c).item(),
        "mask_loss":   (totals["mask"]   / c).item(),
        "byol_loss":   (totals["byol"]   / c).item(),
        "hrv_loss":    (totals["hrv"]    / c).item(),
        "future_loss": (totals["future"] / c).item(),
    }
