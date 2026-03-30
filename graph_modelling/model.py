import torch
import torch.nn as nn
import torch.nn.functional as F
from graph_utils import build_graph

class BeatEncoder(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, 7, padding=3),
            nn.ReLU(),
            nn.Conv1d(32, 64, 5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(64+2, out_dim)

    def forward(self, x, rr):
        B,N,L = x.shape
        x = x.view(B*N,1,L)
        x = self.net(x).squeeze(-1)
        rr = rr.view(B*N,2)
        x = torch.cat([x, rr], dim=-1)
        return self.fc(x).view(B,N,-1)

class TemporalEncoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.rnn = nn.GRU(dim, dim, batch_first=True)

    def forward(self, x):
        out,_ = self.rnn(x)
        return out

class GNNLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.lin = nn.Linear(dim, dim)

    def forward(self, x, A):
        return F.relu(self.lin(A @ x))

class ECGModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = BeatEncoder(cfg.beat_embed_dim)
        self.temporal = TemporalEncoder(cfg.beat_embed_dim)

        self.gnn = nn.ModuleList([
            GNNLayer(cfg.beat_embed_dim) for _ in range(cfg.gnn_layers)
        ])

        self.decoder = nn.Linear(cfg.beat_embed_dim, cfg.beat_embed_dim)

    def forward(self, beats, rr):
        x = self.encoder(beats, rr)
        x = self.temporal(x)

        A = build_graph(x)

        for layer in self.gnn:
            x = layer(x, A)

        recon = self.decoder(x)

        return x, recon