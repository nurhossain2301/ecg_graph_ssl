import torch.nn.functional as F

def masked_loss(x, recon, mask):
    return F.mse_loss(recon[mask], x[mask])