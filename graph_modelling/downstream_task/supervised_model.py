# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from graph_utils import build_graph


# # --------------------------------
# # Beat Encoder
# # --------------------------------
# class BeatEncoder(nn.Module):
#     def __init__(self, beat_len, d_model=128):
#         super().__init__()

#         self.net = nn.Sequential(
#             nn.Conv1d(1, 32, 5, stride=2, padding=2),
#             nn.ReLU(),
#             nn.Conv1d(32, 64, 5, stride=2, padding=2),
#             nn.ReLU(),
#             nn.AdaptiveAvgPool1d(1),
#         )

#         self.fc = nn.Linear(64 + 2, d_model)  # + RR

#     def forward(self, beats, rr):
#         B, N, L = beats.shape

#         x = beats.view(B * N, 1, L)
#         x = self.net(x).squeeze(-1)

#         rr = rr.view(B * N, 2)

#         x = torch.cat([x, rr], dim=-1)
#         x = self.fc(x)

#         return x.view(B, N, -1)


# # --------------------------------
# # Simple GNN
# # --------------------------------
# class GNNLayer(nn.Module):
#     def __init__(self, d):
#         super().__init__()
#         self.self_lin = nn.Linear(d, d)
#         self.neigh_lin = nn.Linear(d, d)

#     def forward(self, x, A):
#         deg = A.sum(-1, keepdim=True).clamp_min(1e-6)
#         neigh = torch.bmm(A, x) / deg
#         return self.self_lin(x) + self.neigh_lin(neigh)


# # --------------------------------
# # Full Model
# # --------------------------------
# class SupervisedECGGraph(nn.Module):
#     def __init__(self, d_model=128, num_classes=4, num_layers=3, cfg=None):
#         super().__init__()
#         self.cfg = cfg
#         self.encoder = BeatEncoder(beat_len=cfg.beat_len, d_model=d_model)

#         self.gnn = nn.ModuleList([
#             GNNLayer(d_model) for _ in range(num_layers)
#         ])
#         self.attn = nn.Sequential(
#             nn.Linear(d_model, d_model),
#             nn.Tanh(),
#             nn.Linear(d_model, 1)
#         )

#         self.norm = nn.LayerNorm(d_model)

#         self.classifier = nn.Sequential(
#             nn.Linear(d_model, 128),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(128, num_classes)
#         )
#     def attention_pool(self, x, valid_mask):
#         mask = valid_mask.unsqueeze(-1)  # [B, N, 1]

#         scores = self.attn(x)  # [B, N, 1]
#         scores = scores.masked_fill(~mask, -1e9)

#         weights = torch.softmax(scores, dim=1)

#         pooled = (x * weights).sum(dim=1)  # [B, D]
#         return pooled
#     def forward(self, batch):
#         beats = batch["beats"]
#         rr = batch["rr"]
#         valid_mask = batch["valid_mask"]

#         x = self.encoder(beats, rr)  # [B, N, D]

#         A = build_graph(x)

#         for layer in self.gnn:
#             x = layer(x, A)

#         x = self.norm(x)

#         # --------------------------
#         # Masked mean pooling
#         # --------------------------
#         # mask = valid_mask.unsqueeze(-1).float()
#         # x = x * mask
#         # pooled = x.sum(dim=1) / mask.sum(dim=1).clamp_min(1e-6)

#         pooled = self.attention_pool(x, valid_mask)

#         logits = self.classifier(pooled)

#         return logits


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
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32, 64, 5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 128, 3, stride=2, padding=1),   # ← extra depth
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(128 + 2, d_model)   # 128 channels + 2 RR features
        self.norm = nn.LayerNorm(d_model)

    def forward(self, beats, rr, valid_mask=None):
        B, N, L = beats.shape

        x = beats.view(B * N, 1, L)
        x = self.net(x).squeeze(-1)             # [B*N, 128]

        rr_flat = rr.view(B * N, 2)
        x = torch.cat([x, rr_flat], dim=-1)    # [B*N, 130]
        x = self.fc(x)                          # [B*N, d_model]
        x = self.norm(x)
        x = x.view(B, N, -1)                   # [B, N, d_model]

        # zero out padding beat embeddings
        if valid_mask is not None:
            x = x * valid_mask.unsqueeze(-1).float()

        return x


# --------------------------------
# GNN Layer with residual + norm
# --------------------------------
class GNNLayer(nn.Module):
    def __init__(self, d, dropout=0.1):
        super().__init__()
        self.self_lin  = nn.Linear(d, d)
        self.neigh_lin = nn.Linear(d, d)
        self.gate      = nn.Linear(d * 2, d)   # ← gating mechanism
        self.norm      = nn.LayerNorm(d)
        self.dropout   = nn.Dropout(dropout)

    def forward(self, x, A):
        deg   = A.sum(-1, keepdim=True).clamp_min(1e-6)
        neigh = torch.bmm(A, x) / deg          # mean aggregation

        # gating: how much neighbour info to let through
        gate  = torch.sigmoid(self.gate(torch.cat([x, neigh], dim=-1)))

        out   = self.self_lin(x) + gate * self.neigh_lin(neigh)
        out   = F.gelu(out)
        out   = self.dropout(out)
        return self.norm(x + out)               # residual


# --------------------------------
# Multi-Head Attention Pooling
# --------------------------------
class AttentionPool(nn.Module):
    def __init__(self, d_model, num_heads=4):
        super().__init__()
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.Tanh(),
                nn.Linear(d_model // 2, 1)
            )
            for _ in range(num_heads)
        ])
        self.out = nn.Linear(d_model * num_heads, d_model)

    def forward(self, x, valid_mask):
        """
        x:          [B, N, D]
        valid_mask: [B, N] bool
        """
        mask = valid_mask.unsqueeze(-1)         # [B, N, 1]
        pooled_list = []

        for head in self.heads:
            scores  = head(x)                   # [B, N, 1]
            scores  = scores.masked_fill(~mask, -1e9)
            weights = torch.softmax(scores, dim=1)
            pooled  = (x * weights).sum(dim=1) # [B, D]
            pooled_list.append(pooled)

        pooled = torch.cat(pooled_list, dim=-1) # [B, D*num_heads]
        return self.out(pooled)                 # [B, D]


# --------------------------------
# Full Model
# --------------------------------
class SupervisedECGGraph(nn.Module):
    def __init__(self, d_model=128, num_classes=4, num_layers=3,
                 num_heads=4, k_neighbors=6, dropout=0.3, cfg=None):
        super().__init__()
        self.k = k_neighbors

        self.encoder = BeatEncoder(beat_len=cfg.beat_len, d_model=d_model)

        self.gnn = nn.ModuleList([
            GNNLayer(d_model, dropout=0.1) for _ in range(num_layers)
        ])

        self.pool = AttentionPool(d_model, num_heads=num_heads)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, batch):
        beats      = batch["beats"]       # [B, N, L]
        rr         = batch["rr"]          # [B, N, 2]
        valid_mask = batch["valid_mask"]  # [B, N] bool

        # 1. Encode beats → node features
        x = self.encoder(beats, rr, valid_mask=valid_mask)   # [B, N, D]

        # 2. Dynamic graph + message passing
        for layer in self.gnn:
            A = build_graph(x, k=self.k, valid_mask=valid_mask)
            x = layer(x, A)

        # 3. Multi-head attention pooling
        pooled = self.pool(x, valid_mask)   # [B, D]

        # 4. Classify
        return self.classifier(pooled)      # [B, num_classes]