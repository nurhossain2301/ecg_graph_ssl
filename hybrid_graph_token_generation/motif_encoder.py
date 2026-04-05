import torch
import torch.nn as nn


class MotifEncoder(nn.Module):
    def __init__(self, input_len=200, emb_dim=128):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        self.fc = nn.Linear(128, emb_dim)

    def forward(self, x):
        # x: [B, L]
        x = x.unsqueeze(1)  # [B,1,L]
        x = self.encoder(x).squeeze(-1)  # [B,128]
        return self.fc(x)

class MotifAutoencoder(nn.Module):
    def __init__(self, input_len=200, emb_dim=128):
        super().__init__()

        self.encoder = MotifEncoder(input_len, emb_dim)

        self.decoder = nn.Sequential(
            nn.Linear(emb_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_len)
        )

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return z, recon