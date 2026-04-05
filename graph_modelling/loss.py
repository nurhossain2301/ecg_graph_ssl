import torch.nn.functional as F

def masked_loss(recon, target, node_mask, valid_mask):
    """
    Only compute loss on masked + valid nodes
    """

    effective_mask = node_mask & valid_mask  # [B, N]

    if effective_mask.sum() == 0:
        return torch.tensor(0.0, device=recon.device)

    diff = (recon - target) ** 2  # [B, N, D]

    loss = diff.sum(-1)  # [B, N]

    loss = loss[effective_mask]

    return loss.mean()

# def masked_loss(x, recon, mask):
#     return F.mse_loss(recon[mask], x[mask])