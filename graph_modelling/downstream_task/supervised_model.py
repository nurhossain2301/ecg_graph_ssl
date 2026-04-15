import torch
import torch.nn as nn
import torch.nn.functional as F
from graph_utils import build_graph


# --------------------------------
# Beat Encoder
# --------------------------------
class BeatEncoder(nn.Module):
    def __init__(self, beat_len, d_model=128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv1d(1, 32, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        self.fc = nn.Linear(64 + 2, d_model)  # + RR

    def forward(self, beats, rr):
        B, N, L = beats.shape

        x = beats.view(B * N, 1, L)
        x = self.net(x).squeeze(-1)

        rr = rr.view(B * N, 2)

        x = torch.cat([x, rr], dim=-1)
        x = self.fc(x)

        return x.view(B, N, -1)


# --------------------------------
# Simple GNN
# --------------------------------
class GNNLayer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.self_lin = nn.Linear(d, d)
        self.neigh_lin = nn.Linear(d, d)

    def forward(self, x, A):
        deg = A.sum(-1, keepdim=True).clamp_min(1e-6)
        neigh = torch.bmm(A, x) / deg
        return self.self_lin(x) + self.neigh_lin(neigh)


# --------------------------------
# Full Model
# --------------------------------
class SupervisedECGGraph(nn.Module):
    def __init__(self, d_model=64, num_classes=4, num_layers=2):
        super().__init__()

        self.encoder = BeatEncoder(beat_len=400, d_model=d_model)

        self.gnn = nn.ModuleList([
            GNNLayer(d_model) for _ in range(num_layers)
        ])
        self.attn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 1)
        )

        self.norm = nn.LayerNorm(d_model)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    def attention_pool(self, x, valid_mask):
        mask = valid_mask.unsqueeze(-1)  # [B, N, 1]

        scores = self.attn(x)  # [B, N, 1]
        scores = scores.masked_fill(~mask, -1e9)

        weights = torch.softmax(scores, dim=1)

        pooled = (x * weights).sum(dim=1)  # [B, D]
        return pooled
    def forward(self, batch):
        beats = batch["beats"]
        rr = batch["rr"]
        valid_mask = batch["valid_mask"]

        x = self.encoder(beats, rr)  # [B, N, D]

        A = build_graph(x)

        for layer in self.gnn:
            x = layer(x, A)

        x = self.norm(x)

        # --------------------------
        # Masked mean pooling
        # --------------------------
        # mask = valid_mask.unsqueeze(-1).float()
        # x = x * mask
        # pooled = x.sum(dim=1) / mask.sum(dim=1).clamp_min(1e-6)

        pooled = self.attention_pool(x, valid_mask)

        logits = self.classifier(pooled)

        return logits