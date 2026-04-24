import torch
import torch.nn.functional as F


def masked_loss(recon, target, node_mask, valid_mask):

    effective_mask = node_mask.bool() & valid_mask.bool()
    if effective_mask.sum() == 0:
        return torch.tensor(0.0, device=recon.device)

    # -------------------------
    # MSE (scale sensitive)
    # -------------------------
    mse = ((recon - target) ** 2).mean(-1)

    # -------------------------
    # Cosine (scale invariant)
    # -------------------------

    target_n = F.normalize(target, dim=-1)
    recon_n = F.normalize(recon, dim=-1, eps=1e-8)  # F.normalize already has eps
    # BUT: explicitly guard against zero vectors
    # zero_mask = (recon.norm(dim=-1) < 1e-8) | (target.norm(dim=-1) < 1e-8)
    # cos[zero_mask] = 0.0  # ignore zero vectors in cosine loss

    cos = 1 - (recon_n * target_n).sum(-1)

    # -------------------------
    # Combine
    # -------------------------
    mse_m = mse[effective_mask]
    cos_m = cos[effective_mask]
    mse_norm = mse_m / (mse_m.detach().mean() + 1e-8)
    return (mse_norm + 0.5 * cos_m).mean()



def byol_loss(p, z):
    z = z.detach() 
    p = F.normalize(p, dim=-1)
    z = F.normalize(z, dim=-1)
    return 2.0 - 2.0 * (p * z).sum(dim=-1).mean()


def ssl_total_loss(
    recon,
    target,
    node_mask,
    valid_mask,
    p1,
    z2,
    lambda_byol=0.5,
):
    loss_mask = masked_loss(recon, target, node_mask, valid_mask)
    loss_byol = byol_loss(p1, z2)
    total = loss_mask + lambda_byol * loss_byol
    return total, loss_mask, loss_byol