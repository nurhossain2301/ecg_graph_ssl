import torch
import torch.nn.functional as F


def masked_loss(recon, target, node_mask, valid_mask):
    effective_mask = node_mask.bool() & valid_mask.bool()
    if effective_mask.sum() == 0:
        return torch.tensor(0.0, device=recon.device)

    mse = ((recon - target) ** 2).mean(-1)

    target_n = F.normalize(target, dim=-1)
    recon_n  = F.normalize(recon,  dim=-1, eps=1e-8)
    cos = 1.0 - (recon_n * target_n).sum(-1)

    mse_loss = mse[effective_mask].mean()
    cos_loss = cos[effective_mask].mean()
    return mse_loss + 0.5 * cos_loss
