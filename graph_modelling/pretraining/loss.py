import torch
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Masked node-reconstruction losses
# ─────────────────────────────────────────────────────────────────────────────

def _effective(node_mask, valid_mask):
    return node_mask.bool() & valid_mask.bool()


def masked_loss(recon, target, node_mask, valid_mask):
    """Pure MSE on masked nodes."""
    eff = _effective(node_mask, valid_mask)
    if eff.sum() == 0:
        z = torch.tensor(0.0, device=recon.device)
        return z, z, z

    mse_loss = ((recon - target) ** 2).mean(-1)[eff].mean()
    cos_loss = torch.tensor(0.0, device=recon.device)
    return mse_loss, mse_loss, cos_loss


def masked_loss_huber(recon, target, node_mask, valid_mask, delta=1.0):
    """Huber + 0.5 cosine on masked nodes (robust to artifact spikes)."""
    eff = _effective(node_mask, valid_mask)
    if eff.sum() == 0:
        z = torch.tensor(0.0, device=recon.device)
        return z, z, z

    r, t = recon[eff], target[eff]
    huber_loss = F.huber_loss(r, t, delta=delta)

    target_n = F.normalize(target, dim=-1)
    recon_n  = F.normalize(recon,  dim=-1, eps=1e-8)
    cos_loss = (1 - (recon_n * target_n).sum(-1))[eff].mean()

    return huber_loss + 0.5 * cos_loss, huber_loss, cos_loss


def masked_loss_spectral(recon, target, node_mask, valid_mask):
    """MSE + 0.5 cosine + 0.3 spectral-magnitude on masked nodes."""
    eff = _effective(node_mask, valid_mask)
    if eff.sum() == 0:
        z = torch.tensor(0.0, device=recon.device)
        return z, z, z

    r, t = recon[eff], target[eff]

    target_n = F.normalize(target, dim=-1)
    recon_n  = F.normalize(recon,  dim=-1, eps=1e-8)

    mse_loss = ((recon - target) ** 2).mean(-1)[eff].mean()
    cos_loss = (1 - (recon_n * target_n).sum(-1))[eff].mean()

    r_fft = torch.fft.rfft(r, norm="ortho")
    t_fft = torch.fft.rfft(t, norm="ortho")
    spec_loss = F.mse_loss(r_fft.abs(), t_fft.abs())

    return mse_loss + 0.5 * cos_loss + 0.3 * spec_loss, mse_loss, cos_loss


# ─────────────────────────────────────────────────────────────────────────────
# Global (pool-level) SSL losses   — operate on (B, D) embeddings
# ─────────────────────────────────────────────────────────────────────────────

def byol_loss(p1, z2, p2, z1):
    """Symmetric BYOL loss."""
    p1 = F.normalize(p1, dim=-1);  z2 = F.normalize(z2, dim=-1)
    p2 = F.normalize(p2, dim=-1);  z1 = F.normalize(z1, dim=-1)
    l1 = 2.0 - 2.0 * (p1 * z2).sum(-1).mean()
    l2 = 2.0 - 2.0 * (p2 * z1).sum(-1).mean()
    return (l1 + l2) * 0.5


def _off_diagonal(m):
    n = m.size(0)
    return m.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def vicreg_loss(z1, z2, lam=25.0, mu=25.0, nu=1.0):
    """VICReg: variance + invariance + covariance."""
    N, D = z1.shape

    inv = F.mse_loss(z1, z2)

    std1 = torch.sqrt(z1.var(dim=0) + 1e-4)
    std2 = torch.sqrt(z2.var(dim=0) + 1e-4)
    var  = F.relu(1 - std1).mean() + F.relu(1 - std2).mean()

    z1c  = z1 - z1.mean(dim=0)
    z2c  = z2 - z2.mean(dim=0)
    cov1 = (z1c.T @ z1c) / (N - 1)
    cov2 = (z2c.T @ z2c) / (N - 1)
    cov  = (_off_diagonal(cov1).pow(2).sum() +
            _off_diagonal(cov2).pow(2).sum()) / D

    return lam * inv + mu * var + nu * cov


def barlow_twins_loss(z1, z2, lam=0.005):
    """Barlow Twins: cross-correlation matrix → identity."""
    N, D = z1.shape
    z1 = (z1 - z1.mean(0)) / (z1.std(0) + 1e-4)
    z2 = (z2 - z2.mean(0)) / (z2.std(0) + 1e-4)
    c  = z1.T @ z2 / N                             # (D, D)
    on_diag  = (c.diagonal() - 1).pow(2).sum()
    off_diag = _off_diagonal(c).pow(2).sum()
    return on_diag + lam * off_diag


# ─────────────────────────────────────────────────────────────────────────────
# Unified entry points  (same 5-tuple return for every variant)
# Return: (total, mask_loss, aux_loss, mse_loss, cos_loss)
# ─────────────────────────────────────────────────────────────────────────────

def ssl_total_loss(recon, target, node_mask, valid_mask, p1, z2, p2, z1,
                   lambda_byol=0.5):
    loss_mask, mse, cos = masked_loss(recon, target, node_mask, valid_mask)
    loss_aux = byol_loss(p1, z2, p2, z1)
    return loss_mask + lambda_byol * loss_aux, loss_mask, loss_aux, mse, cos


def ssl_huber_loss(recon, target, node_mask, valid_mask, p1, z2, p2, z1,
                   lambda_byol=0.5):
    loss_mask, mse, cos = masked_loss_huber(recon, target, node_mask, valid_mask)
    loss_aux = byol_loss(p1, z2, p2, z1)
    return loss_mask + lambda_byol * loss_aux, loss_mask, loss_aux, mse, cos


def ssl_vicreg_loss(recon, target, node_mask, valid_mask, p1, z2, p2, z1,
                    lambda_aux=0.1):
    loss_mask, mse, cos = masked_loss(recon, target, node_mask, valid_mask)
    loss_aux = vicreg_loss(p1, p2)          # uses online embeddings of both views
    return loss_mask + lambda_aux * loss_aux, loss_mask, loss_aux, mse, cos


def ssl_barlow_loss(recon, target, node_mask, valid_mask, p1, z2, p2, z1,
                    lambda_aux=0.1):
    loss_mask, mse, cos = masked_loss(recon, target, node_mask, valid_mask)
    loss_aux = barlow_twins_loss(p1, p2)    # uses online embeddings of both views
    return loss_mask + lambda_aux * loss_aux, loss_mask, loss_aux, mse, cos


def ssl_spectral_loss(recon, target, node_mask, valid_mask, p1, z2, p2, z1,
                      lambda_byol=0.5):
    loss_mask, mse, cos = masked_loss_spectral(recon, target, node_mask, valid_mask)
    loss_aux = byol_loss(p1, z2, p2, z1)
    return loss_mask + lambda_byol * loss_aux, loss_mask, loss_aux, mse, cos


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

LOSS_REGISTRY = {
    "baseline": ssl_total_loss,
    "huber":    ssl_huber_loss,
    "vicreg":   ssl_vicreg_loss,
    "barlow":   ssl_barlow_loss,
    "spectral": ssl_spectral_loss,
}


def build_loss_fn(loss_type: str):
    if loss_type not in LOSS_REGISTRY:
        raise ValueError(f"Unknown loss_type {loss_type!r}. "
                         f"Choose from {list(LOSS_REGISTRY)}")
    return LOSS_REGISTRY[loss_type]
