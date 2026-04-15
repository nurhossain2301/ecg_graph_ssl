import torch
import torch.nn as nn


class BRPClassifier(nn.Module):
    def __init__(self, encoder, hidden_dim=256, num_classes=4, freeze_encoder=True):
        super().__init__()

        self.encoder = encoder

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        # feature_dim = encoder.embed_dim  # make sure your model exposes this
        feature_dim = 256

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        """
        x: (B, T)
        """

        feats = self.encoder(x)          # (B, T', D)
        pooled = feats.mean(dim=1)       # (B, D)

        logits = self.classifier(pooled) # (B, 5)

        return logits