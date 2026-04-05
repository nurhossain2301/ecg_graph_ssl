import torch
import torch.nn as nn
import torch.nn.functional as F
from graph_utils import build_graph

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------------------------------
# Beat Encoder
# -------------------------------------------------------
class BeatEncoder(nn.Module):
    def __init__(self, in_len, d_model, rr_dim=2):
        super().__init__()
        self.rr_dim = rr_dim
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(64 + rr_dim, d_model)  # +1 for RR

    def forward(self, beats, rr):
        # beats: [B, N, L]
        B, N, L = beats.shape

        x = beats.view(B * N, 1, L)
        x = self.net(x).squeeze(-1)  # [B*N, 64]

        rr = rr.view(B * N, self.rr_dim)

        x = torch.cat([x, rr], dim=-1)

        x = self.fc(x)
        x = x.view(B, N, -1)

        return x


# -------------------------------------------------------
# Temporal Encoder (Transformer)
# -------------------------------------------------------
class TemporalEncoder(nn.Module):
    def __init__(self, d_model, nhead=4, num_layers=2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, x):
        return self.encoder(x)


# -------------------------------------------------------
# Simple GNN (message passing)
# -------------------------------------------------------
class SimpleGNNLayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.lin_self = nn.Linear(d_model, d_model)
        self.lin_neigh = nn.Linear(d_model, d_model)

    def forward(self, x, A):
        # x: [B, N, D], A: [B, N, N]
        deg = A.sum(-1, keepdim=True) + 1e-6
        agg = torch.bmm(A, x) / deg
        return self.lin_self(x) + self.lin_neigh(agg)


# -------------------------------------------------------
# Graph Builder
# -------------------------------------------------------
def build_graph(x, k=5):
    # x: [B, N, D]
    B, N, D = x.shape

    sim = torch.matmul(x, x.transpose(1, 2))  # cosine-ish
    _, idx = torch.topk(sim, k=k, dim=-1)

    A = torch.zeros(B, N, N, device=x.device)

    for b in range(B):
        for i in range(N):
            A[b, i, idx[b, i]] = 1.0

    # temporal edges
    for i in range(N - 1):
        A[:, i, i + 1] = 1
        A[:, i + 1, i] = 1

    return A


# -------------------------------------------------------
# FULL MODEL (CORRECT MASKED SSL)
# -------------------------------------------------------
class ECGModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # self.encoder = BeatEncoder(cfg.beat_len, cfg.d_model)
        self.encoder = BeatEncoder(cfg.beat_len, cfg.d_model, rr_dim=2)
        self.temporal = TemporalEncoder(cfg.d_model)
        self.gnn = nn.ModuleList(
            [SimpleGNNLayer(cfg.d_model) for _ in range(cfg.gnn_layers)]
        )

        self.decoder = nn.Linear(cfg.d_model, cfg.d_model)

        # 🔥 learned mask token (critical)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg.d_model))

    def forward(self, beats, rr, node_mask, valid_mask):
        """
        node_mask: True = MASKED (predict)
        valid_mask: True = real node
        """

        # ---------------------------------------------------
        # 1. Encode beats
        # ---------------------------------------------------
        e = self.encoder(beats, rr)  # [B, N, D]

        # ---------------------------------------------------
        # 2. Compute CLEAN TARGET (no masking)
        # ---------------------------------------------------
        with torch.no_grad():
            target = self.temporal(e.clone())
        target = target.detach()

        # ---------------------------------------------------
        # 3. Apply MASK (VERY IMPORTANT)
        # ---------------------------------------------------
        e_masked = e.clone()

        mask_token = self.mask_token.expand(e.size(0), e.size(1), -1)

        # Only replace VALID nodes that are masked
        e_masked = torch.where(
            node_mask.unsqueeze(-1),
            mask_token,
            e_masked
        )

        # Zero-out padded nodes
        e_masked = e_masked * valid_mask.unsqueeze(-1)

        # ---------------------------------------------------
        # 4. Temporal encoding on masked input
        # ---------------------------------------------------
        x = self.temporal(e_masked)

        # ---------------------------------------------------
        # 5. Graph construction + GNN
        # ---------------------------------------------------
        A = build_graph(x)

        for layer in self.gnn:
            x = layer(x, A)

        # ---------------------------------------------------
        # 6. Reconstruction
        # ---------------------------------------------------
        recon = self.decoder(x)

        return target, recon