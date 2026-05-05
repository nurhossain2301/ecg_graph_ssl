import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from config import Config
cfg = Config()

from model import (
    ECGBYOLModel,
    ECGEncoder,
    AttentionPool,
)


# ─────────────────────────────────────────────
# Head option 1: plain MLP (baseline)
# ─────────────────────────────────────────────
class MLPHead(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x, valid_mask=None):
        return self.net(x)


# ─────────────────────────────────────────────
# Head option 2 (cosine): projection → L2-norm → cosine sim × temperature
# ─────────────────────────────────────────────
class CosineClassifierHead(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes, dropout=0.3,
                 init_temperature=10.0):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.weight = nn.Parameter(torch.empty(num_classes, hidden_dim))
        nn.init.xavier_uniform_(self.weight)
        self.temperature = nn.Parameter(torch.tensor(float(init_temperature)))

    def forward(self, x, valid_mask=None):
        x = self.proj(x)
        x = F.normalize(x, dim=-1)
        w = F.normalize(self.weight, dim=-1)
        return self.temperature * (x @ w.t())


# ─────────────────────────────────────────────
# Head option 3 (residual_mlp): d_model → 512 → 256 + skip → num_classes
# ─────────────────────────────────────────────
class ResidualMLPHead(nn.Module):
    """
    Two-layer MLP with a skip connection:
        x → Linear(512) → GELU → Linear(256)
                                         ↘ + skip(x→256) → GELU → Linear(num_classes)
    """
    def __init__(self, in_dim, num_classes, dropout=0.3):
        super().__init__()
        self.fc1   = nn.Linear(in_dim, 512)
        self.norm1 = nn.LayerNorm(512)
        self.fc2   = nn.Linear(512, 256)
        self.norm2 = nn.LayerNorm(256)
        self.skip  = nn.Linear(in_dim, 256)   # project input to residual dim
        self.drop  = nn.Dropout(dropout)
        self.out   = nn.Linear(256, num_classes)

    def forward(self, x, valid_mask=None):
        residual = self.skip(x)                         # (B, 256)
        h = self.drop(F.gelu(self.norm1(self.fc1(x)))) # (B, 512)
        h = self.norm2(self.fc2(h))                     # (B, 256)
        h = self.drop(F.gelu(h + residual))             # residual add
        return self.out(h)


# ─────────────────────────────────────────────
# Head option 4 (cls_transformer): prepend [CLS], run 2 transformer layers,
#   take CLS output → Linear(num_classes)
# ─────────────────────────────────────────────
class CLSTransformerHead(nn.Module):
    """
    Prepends a learnable [CLS] token to the encoder token sequence,
    runs 2 additional TransformerEncoder layers, and uses the [CLS]
    output as the classification embedding.

    Receives raw encoder output (B, N, D) + valid_mask (B, N),
    so AttentionPool is bypassed for this head.
    """
    def __init__(self, d_model, num_classes, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.out  = nn.Linear(d_model, num_classes)

    def forward(self, x, valid_mask):
        # x: (B, N, D)   valid_mask: (B, N) bool
        B = x.size(0)
        cls = self.cls_token.expand(B, -1, -1)                    # (B, 1, D)
        x   = torch.cat([cls, x], dim=1)                          # (B, N+1, D)
        cls_valid = torch.ones(B, 1, dtype=torch.bool, device=x.device)
        mask = torch.cat([cls_valid, valid_mask], dim=1)           # (B, N+1)
        x = self.transformer(x, src_key_padding_mask=~mask)
        return self.out(self.norm(x[:, 0]))                        # CLS position


# ─────────────────────────────────────────────
# ECGClassifier
# ─────────────────────────────────────────────
class ECGClassifier(nn.Module):
    """
    head_type options:
        'mlp'             : 2-layer MLP (baseline)
        'cosine'          : projection → L2-norm → cosine sim × temperature
        'residual_mlp'    : d_model → 512 → 256 + skip → num_classes
        'cls_transformer' : [CLS] token + 2 transformer layers → num_classes
    """
    def __init__(self, cfg, num_classes, hidden_dim=256, dropout=0.3,
                 freeze_encoder=False, head_type="mlp"):
        super().__init__()

        self.encoder   = ECGEncoder(cfg)
        self.pool      = AttentionPool(cfg.d_model)
        self.head_type = head_type

        if head_type == "cosine":
            self.classifier = CosineClassifierHead(
                in_dim=cfg.d_model, hidden_dim=hidden_dim,
                num_classes=num_classes, dropout=dropout,
            )
        elif head_type == "residual_mlp":
            self.classifier = ResidualMLPHead(
                in_dim=cfg.d_model, num_classes=num_classes, dropout=dropout,
            )
        elif head_type == "cls_transformer":
            self.classifier = CLSTransformerHead(
                d_model=cfg.d_model, num_classes=num_classes,
                nhead=4, num_layers=2, dropout=0.1,
            )
        else:  # mlp
            self.classifier = MLPHead(
                in_dim=cfg.d_model, hidden_dim=hidden_dim,
                num_classes=num_classes, dropout=dropout,
            )

        if freeze_encoder:
            self._freeze_encoder()

    def _freeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = False
        for p in self.pool.parameters():
            p.requires_grad = False
        print("[ECGClassifier] Encoder frozen – linear probe mode.")

    def unfreeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = True
        for p in self.pool.parameters():
            p.requires_grad = True
        print("[ECGClassifier] Encoder unfrozen – fine-tune mode.")

    def forward(self, beats, rr, valid_mask, return_embedding=False):
        x = self.encoder(beats, rr, valid_mask)   # (B, N, D)

        if self.head_type == "cls_transformer":
            # bypass AttentionPool — CLS head operates on token sequence
            logits = self.classifier(x, valid_mask)
            if return_embedding:
                return logits, x.mean(dim=1)
            return logits

        embedding, _ = self.pool(x, valid_mask)   # (B, D)
        logits = self.classifier(embedding)
        if return_embedding:
            return logits, embedding
        return logits


# ─────────────────────────────────────────────
# Checkpoint loader
# ─────────────────────────────────────────────
def load_pretrained_byol(ckpt_path: str, cfg, device="cpu") -> ECGBYOLModel:
    byol = ECGBYOLModel(cfg)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    byol.load_state_dict(state_dict, strict=True)
    byol.eval()
    print(f"[load_pretrained_byol] Loaded from {ckpt_path}")
    return byol
