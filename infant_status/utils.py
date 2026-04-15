import os
import re
import json
import math
import argparse
import numpy as np
import torch
import torch.nn as nn



# If you have sklearn installed, we use it for nice metrics.
# Otherwise we fall back to simple implementations.
try:
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, cohen_kappa_score
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

# -----------------------------
# Metrics
# -----------------------------
def compute_metrics(y_true, y_pred, num_classes=2):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    if SKLEARN_OK:
        acc = float(accuracy_score(y_true, y_pred))
        f1m = float(f1_score(y_true, y_pred, average="macro"))
        kappa = float(cohen_kappa_score(y_true, y_pred))
        cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes))).astype(int)
        return {"acc": acc, "macro_f1": f1m, "kappa": kappa, "confusion_matrix": cm.tolist()}

    # Fallback: accuracy + macro-F1 (simple)
    acc = float((y_true == y_pred).mean())
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1

    f1s = []
    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        prec = tp / (tp + fp + 1e-12)
        rec = tp / (tp + fn + 1e-12)
        f1 = 2 * prec * rec / (prec + rec + 1e-12)
        f1s.append(f1)
    f1m = float(np.mean(f1s))
    return {"acc": acc, "macro_f1": f1m, "kappa": float("nan"), "confusion_matrix": cm.tolist()}

def load_encoder_from_ecghubert(model, checkpoint_path, device):
    """
    Tries to load your ECGHuBERTModel and return an encoder callable that outputs (B, T', D),
    plus embed_dim.

    Requirements:
      - Your project has: from model import ECGHuBERTModel
      - Your ECGHuBERTModel has one of:
          * .encoder(x) -> (B,T',D)
          * .forward_features(x) -> (B,T',D)
          * .feature_extractor + .transformer (we'll try a few)
      - The checkpoint is either:
          * {"model": state_dict, ...}
          * state_dict directly
    """
    

    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    # Instantiate a model just to reuse its encoder stack.
    # num_clusters doesn't matter for SFT head as long as weights load.

    model.load_state_dict(state, strict=False)
    model.eval()

    # Build an encoder wrapper
    class EncoderWrapper(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base

            # infer embed_dim
            self.embed_dim = getattr(base, "embed_dim", None)

        def forward(self, x):
            if hasattr(self.base, "encoder"):
                out = self.base.encoder(x)
            elif hasattr(self.base, "forward_features"):
                out = self.base.forward_features(x)
            else:
                # Try common pattern: feature_extractor -> transformer
                if not hasattr(self.base, "feature_extractor"):
                    raise RuntimeError("Cannot find encoder/forward_features/feature_extractor in ECGHuBERTModel.")
                z = self.base.feature_extractor(x)  # (B,T',C) hopefully
                if hasattr(self.base, "transformer"):
                    out = self.base.transformer(z)
                elif hasattr(self.base, "context_network"):
                    out = self.base.context_network(z)
                else:
                    out = z

            # Ensure shape is (B,T,D)
            if out.dim() == 2:
                out = out.unsqueeze(1)

            # If embed_dim not set, infer last dim
            if self.embed_dim is None:
                self.embed_dim = int(out.shape[-1])

            return out

    enc = EncoderWrapper(model).to(device)
    # Force one forward to infer embed_dim if needed
    with torch.no_grad():
        dummy = torch.zeros(1, 1, 250 * 30, device=device)  # one 30s epoch
        _ = enc(dummy)
    return enc, int(enc.embed_dim)
