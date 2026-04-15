import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader

from config import Config
from model import ECGModel



# -----------------------------
# Classifier Wrapper
# -----------------------------
class GraphClassifier(nn.Module):
    def __init__(self, encoder, hidden_dim=128, num_classes=2, freeze_encoder=True):
        super().__init__()

        self.encoder = encoder

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        # You may need to adjust this based on ECGModel output dim
        feature_dim = 128  

        self.head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, batch):
        beats = batch["beats"]
        rr = batch["rr"]
        valid_mask = batch["valid_mask"]

        x = self.encoder(
            beats,
            rr,
            node_mask=None,
            valid_mask=valid_mask,
            return_latent=True,
        )
        
        # x = torch.nn.functional.normalize(x, dim=-1)

        mask = valid_mask.unsqueeze(-1).float()
        x = x * mask
        pooled = x.sum(dim=1) / mask.sum(dim=1).clamp_min(1e-6)

        logits = self.head(pooled)
        return logits


# -----------------------------
# Load pretrained graph model
# -----------------------------
def load_pretrained_encoder(ckpt_path, cfg, device="cuda"):
    model = ECGModel(cfg).to(device)

    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)

    print(f"✅ Loaded pretrained graph model from {ckpt_path}")
    return model
