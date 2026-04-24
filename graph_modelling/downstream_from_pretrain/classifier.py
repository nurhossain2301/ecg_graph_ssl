import torch
import torch.nn as nn
import copy



# ─────────────────────────────────────────────
# Config (must match pre-training config)
# ─────────────────────────────────────────────
from config import Config
cfg = Config()

# ─────────────────────────────────────────────
# Import your model definition
# ─────────────────────────────────────────────
from model import (
    ECGBYOLModel,
    ECGEncoder,
    AttentionPool,
)

# ─────────────────────────────────────────────
# Classifier Head
# ─────────────────────────────────────────────
class ECGClassifier(nn.Module):
    """
    Wraps the pretrained online_encoder + AttentionPool
    and adds a 2-layer MLP classification head.

    Architecture:
        online_encoder  →  AttentionPool  →  Linear+ReLU  →  Linear  →  logits
    """
    def __init__(
        self,
        cfg,
        num_classes: int,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        freeze_encoder: bool = False,
    ):
        super().__init__()

        # ── backbone ──────────────────────────────────────
        self.encoder = ECGEncoder(cfg)          # online encoder weights
        self.pool    = AttentionPool(cfg.d_model)

        # ── 2-layer classifier head ───────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(cfg.d_model, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),  # logits (no softmax)
        )

        if freeze_encoder:
            self._freeze_encoder()

    # ── helpers ───────────────────────────────────────────
    def _freeze_encoder(self):
        """Freeze all backbone parameters (linear probe setting)."""
        for p in self.encoder.parameters():
            p.requires_grad = False
        for p in self.pool.parameters():
            p.requires_grad = False
        print("[ECGClassifier] Encoder frozen – linear probe mode.")

    def unfreeze_encoder(self):
        """Unfreeze backbone for full fine-tuning."""
        for p in self.encoder.parameters():
            p.requires_grad = True
        for p in self.pool.parameters():
            p.requires_grad = True
        print("[ECGClassifier] Encoder unfrozen – fine-tune mode.")

    # ── forward ───────────────────────────────────────────
    def forward(self, beats, rr, valid_mask, return_embedding=False):
        """
        Args
        ----
        beats       : (B, N, L)   beat waveforms
        rr          : (B, N, 2)   RR-interval features
        valid_mask  : (B, N) bool padded-beat mask
        return_embedding : if True also return pooled vector

        Returns
        -------
        logits      : (B, num_classes)
        embedding   : (B, d_model)  [only when return_embedding=True]
        """
        x = self.encoder(beats, rr, valid_mask)          # (B, N, D)
        embedding, _ = self.pool(x, valid_mask)           # (B, D)
        logits = self.classifier(embedding)               # (B, num_classes)

        if return_embedding:
            return logits, embedding
        return logits

# ─────────────────────────────────────────────
# load_and_build_classifier.py
# ─────────────────────────────────────────────

def load_pretrained_byol(ckpt_path: str, cfg, device="cpu") -> ECGBYOLModel:
    """Load the full BYOL pre-trained model from a checkpoint."""
    byol = ECGBYOLModel(cfg)
    
    checkpoint = torch.load(ckpt_path, map_location=device)

    # ── handle common checkpoint formats ──────────────────
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint          # raw state-dict
    else:
        state_dict = checkpoint

    byol.load_state_dict(state_dict, strict=True)
    byol.eval()
    print(f"[load_pretrained_byol] Loaded from {ckpt_path}")
    return byol


