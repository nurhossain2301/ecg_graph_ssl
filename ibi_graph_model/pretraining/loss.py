import torch
import torch.nn.functional as F


# ─── Masked node reconstruction ───────────────────────────────────────────────

def masked_mse_normalized(recon, target, node_mask, valid_mask):
    """
    Instance-normalized MSE on masked nodes only (no cosine term).
    Normalization uses target statistics to make loss scale-free.
    """
    eff = node_mask.bool() & valid_mask.bool()
    if eff.sum() == 0:
        return torch.tensor(0.0, device=recon.device)
    mu    = target.mean(dim=-1, keepdim=True)
    std   = target.std(dim=-1, keepdim=True) + 1e-6
    t_n   = (target - mu) / std
    r_n   = (recon  - mu) / std
    return ((r_n - t_n) ** 2).mean(-1)[eff].mean()


# ─── Global (pool-level) losses ───────────────────────────────────────────────

def byol_loss(p1, z2, p2, z1):
    p1 = F.normalize(p1, dim=-1); z2 = F.normalize(z2, dim=-1)
    p2 = F.normalize(p2, dim=-1); z1 = F.normalize(z1, dim=-1)
    l1 = 2.0 - 2.0 * (p1 * z2).sum(-1).mean()
    l2 = 2.0 - 2.0 * (p2 * z1).sum(-1).mean()
    return (l1 + l2) * 0.5


def hrv_loss(pred, target):
    """SmoothL1 on predicted vs. target normalized HRV feature vector."""
    return F.smooth_l1_loss(pred, target)


def future_loss(pred, target):
    """Cosine distance between predicted and EMA target future representation."""
    pred   = F.normalize(pred,   dim=-1)
    target = F.normalize(target, dim=-1)
    return (2.0 - 2.0 * (pred * target).sum(-1)).mean()


# ─── Unified entry point ──────────────────────────────────────────────────────

def ssl_loss(recon, target_node, node_mask, valid_mask,
             p1, z2, p2, z1,
             pred_hrv=None, target_hrv=None,
             pred_future=None, target_future=None,
             lambda_byol=0.5, lambda_hrv=0.1, lambda_future=0.1):
    """
    Combined SSL loss:
      L = L_mask + λ_byol * L_byol + λ_hrv * L_hrv + λ_future * L_future
    """
    loss_mask = masked_mse_normalized(recon, target_node, node_mask, valid_mask)
    loss_byol = byol_loss(p1, z2, p2, z1)
    total     = loss_mask + lambda_byol * loss_byol

    loss_hrv = torch.tensor(0.0, device=recon.device)
    if pred_hrv is not None and target_hrv is not None:
        loss_hrv = hrv_loss(pred_hrv, target_hrv)
        total    = total + lambda_hrv * loss_hrv

    loss_future = torch.tensor(0.0, device=recon.device)
    if pred_future is not None and target_future is not None:
        loss_future = future_loss(pred_future, target_future)
        total       = total + lambda_future * loss_future

    return total, loss_mask, loss_byol, loss_hrv, loss_future


def build_loss_fn(cfg):
    """Return a loss_fn bound to cfg hyperparameters."""
    def _loss_fn(recon, target_node, node_mask, valid_mask,
                 p1, z2, p2, z1,
                 pred_hrv=None, target_hrv=None,
                 pred_future=None, target_future=None):
        return ssl_loss(
            recon, target_node, node_mask, valid_mask,
            p1, z2, p2, z1,
            pred_hrv=pred_hrv, target_hrv=target_hrv,
            pred_future=pred_future, target_future=target_future,
            lambda_byol=cfg.lambda_byol,
            lambda_hrv=cfg.lambda_hrv,
            lambda_future=cfg.lambda_future,
        )
    return _loss_fn
