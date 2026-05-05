import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Graph builder (fully vectorized, no Python loops) ────────────────────────

def build_graph(x, valid_mask, k=6):
    """
    Build adjacency from learned embeddings x [B, N, D].
    Edges: top-k cosine similarity (undirected) + sequential i<->i+1 + self-loops.
    Graph is built from x (temporal encoder output, not raw MLP embeddings).
    """
    B, N, D = x.shape
    x_n = F.normalize(x, dim=-1)                              # [B, N, D]
    sim = torch.bmm(x_n, x_n.transpose(1, 2))                # [B, N, N]

    # Mask padding nodes out of similarity
    pad = ~valid_mask                                          # [B, N]
    sim = sim.masked_fill(pad.unsqueeze(1), -1e9)
    sim = sim.masked_fill(pad.unsqueeze(2), -1e9)

    # K-NN (exclude self-loops from selection)
    eye = torch.eye(N, device=x.device, dtype=torch.bool).unsqueeze(0)
    sim_no_self = sim.masked_fill(eye, -1e9)
    kk = min(k, N - 1)
    _, idx = torch.topk(sim_no_self, k=kk, dim=-1)            # [B, N, K]
    A_knn = torch.zeros(B, N, N, device=x.device)
    A_knn.scatter_(2, idx, 1.0)
    A_knn = ((A_knn + A_knn.transpose(1, 2)) > 0).float()    # undirected

    # Temporal edges i <-> i+1 (valid nodes only)
    A_temp = torch.zeros(B, N, N, device=x.device)
    if N > 1:
        diag = torch.ones(N - 1, device=x.device)
        A_temp[:, :N-1, 1:] = torch.diag(diag).unsqueeze(0)
        A_temp = A_temp + A_temp.transpose(1, 2)
    vm = valid_mask.float()
    A_temp = A_temp * vm.unsqueeze(1) * vm.unsqueeze(2)

    # Self-loops for valid nodes
    A_self = torch.diag_embed(vm)

    return (A_knn + A_temp + A_self).clamp(0, 1)


# ─── Building blocks ──────────────────────────────────────────────────────────

class SinusoidalPE(nn.Module):
    def __init__(self, d_model, max_len=256):
        super().__init__()
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:d_model // 2])
        self.register_buffer("pe", pe.unsqueeze(0))            # [1, max_len, D]

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class IBIEncoder(nn.Module):
    def __init__(self, in_dim, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, x):
        return self.net(x)


class TemporalEncoder(nn.Module):
    def __init__(self, d_model, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers)

    def forward(self, x, valid_mask=None):
        pad_mask = ~valid_mask if valid_mask is not None else None
        return self.encoder(x, src_key_padding_mask=pad_mask)


class GNNLayer(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.lin_self  = nn.Linear(d_model, d_model)
        self.lin_neigh = nn.Linear(d_model, d_model)
        self.act  = nn.GELU()
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, A):
        deg = A.sum(-1, keepdim=True) + 1e-6
        agg = torch.bmm(A, x) / deg
        out = self.lin_self(x) + self.lin_neigh(agg)
        return self.norm(self.drop(self.act(out)) + x)


class AttentionPool(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(d_model, d_model), nn.Tanh(), nn.Linear(d_model, 1),
        )

    def forward(self, x, valid_mask):
        scores  = self.attn(x)                                 # [B, N, 1]
        scores  = scores.masked_fill(~valid_mask.unsqueeze(-1), -1e9)
        weights = torch.softmax(scores, dim=1)
        return (x * weights).sum(dim=1), weights               # [B, D], [B, N, 1]


# ─── Shared encoder ───────────────────────────────────────────────────────────

class IBIGraphEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.feat_enc = IBIEncoder(cfg.ibi_feature_dim, cfg.d_model)
        self.pos_enc  = SinusoidalPE(cfg.d_model, max_len=cfg.max_len_beats + 16)
        self.temporal = TemporalEncoder(cfg.d_model, cfg.nhead, cfg.transformer_layers)
        self.gnn      = nn.ModuleList([GNNLayer(cfg.d_model) for _ in range(cfg.gnn_layers)])
        self.norm     = nn.LayerNorm(cfg.d_model)
        self.k        = cfg.knn_k

    def encode_features(self, beats, valid_mask):
        """MLP → sinusoidal PE, before temporal encoder."""
        e = self.feat_enc(beats) * valid_mask.unsqueeze(-1)    # [B, N, D]
        return self.pos_enc(e)

    def forward(self, beats, valid_mask):
        e = self.encode_features(beats, valid_mask)
        x = self.temporal(e, valid_mask)
        # BUG FIX: build graph from temporal output, not raw MLP embeddings
        A = build_graph(x, valid_mask, k=self.k)
        for layer in self.gnn:
            x = layer(x, A)
        return self.norm(x)


# ─── Full SSL model ───────────────────────────────────────────────────────────

class IBIGraphBYOLModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.online_encoder = IBIGraphEncoder(cfg)
        self.target_encoder = copy.deepcopy(self.online_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        self.pool = AttentionPool(cfg.d_model)

        proj_dim = cfg.proj_dim
        self.online_projector = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.BatchNorm1d(cfg.d_model),
            nn.ReLU(),
            nn.Linear(cfg.d_model, proj_dim),
        )
        self.target_projector = copy.deepcopy(self.online_projector)
        for p in self.target_projector.parameters():
            p.requires_grad = False

        self.predictor = nn.Sequential(
            nn.Linear(proj_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
        )

        # Masked reconstruction (2-layer MLP decoder)
        self.decoder = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.LayerNorm(cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )
        self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        nn.init.normal_(self.mask_token, std=0.02)

        # HRV prediction head
        self.hrv_head = nn.Sequential(
            nn.Linear(cfg.d_model, 64),
            nn.GELU(),
            nn.Linear(64, cfg.n_hrv_features),
        )

        # Future beat prediction head
        self.future_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, proj_dim),
        )

    @torch.no_grad()
    def update_target(self, momentum=0.996):
        for po, pt in zip(self.online_encoder.parameters(),
                          self.target_encoder.parameters()):
            pt.data.mul_(momentum).add_(po.data, alpha=1.0 - momentum)
        for po, pt in zip(self.online_projector.parameters(),
                          self.target_projector.parameters()):
            pt.data.mul_(momentum).add_(po.data, alpha=1.0 - momentum)

    def masked_forward(self, beats, node_mask, valid_mask):
        enc = self.online_encoder

        # Clean features + PE (before masking)
        e = enc.encode_features(beats, valid_mask)             # [B, N, D]

        # BUG FIX: teacher = full EMA target encoder (not online temporal only)
        with torch.no_grad():
            target = self.target_encoder(beats, valid_mask)    # [B, N, D]
            target = target * valid_mask.unsqueeze(-1)

        # Clean temporal output → graph topology
        x_clean = enc.temporal(e, valid_mask)
        # BUG FIX: graph from temporal output x_clean, not raw MLP e
        A = build_graph(x_clean, valid_mask, k=enc.k)

        # Apply mask to input features, then run student through temporal + GNN
        mask_tok = self.mask_token.expand(e.size(0), e.size(1), -1)
        e_masked  = torch.where(node_mask.unsqueeze(-1), mask_tok, e)
        e_masked  = e_masked * valid_mask.unsqueeze(-1)

        x_student = enc.temporal(e_masked, valid_mask)
        for layer in enc.gnn:
            x_student = layer(x_student, A)
        x_student = enc.norm(x_student)
        recon = self.decoder(x_student)

        return target, recon

    def byol_forward(self, beats1, valid1, beats2, valid2):
        # Online: view 1
        z1      = self.online_encoder(beats1, valid1)
        z1_pool, _ = self.pool(z1, valid1)
        p1      = self.predictor(self.online_projector(z1_pool))

        # Online: view 2
        z2_on   = self.online_encoder(beats2, valid2)
        z2_pool, _ = self.pool(z2_on, valid2)
        p2      = self.predictor(self.online_projector(z2_pool))

        with torch.no_grad():
            z2t = self.target_encoder(beats2, valid2)
            z2t, _ = self.pool(z2t, valid2)
            z2t = self.target_projector(z2t)

            z1t = self.target_encoder(beats1, valid1)
            z1t, _ = self.pool(z1t, valid1)
            z1t = self.target_projector(z1t)

        return p1, z2t, p2, z1t, z1_pool  # z1_pool reused for HRV head

    def future_forward(self, beats, valid_mask):
        """
        Predict the EMA representation of the full window from the first half only.
        Returns (pred, target) or (None, None) if sequences are too short.
        """
        B, N, _ = beats.shape
        n_valid  = valid_mask.sum(dim=1)                       # [B]
        half     = (n_valid // 2).long()                       # [B]

        valid_samples = (half >= 4)
        if valid_samples.sum() == 0:
            return None, None

        # Build first-half valid mask per sample
        positions   = torch.arange(N, device=beats.device).unsqueeze(0)  # [1, N]
        valid_first = (positions < half.unsqueeze(1)) & valid_mask        # [B, N]

        # Online: encode first half
        z_first       = self.online_encoder(beats, valid_first)
        z_pool, _     = self.pool(z_first, valid_first)
        pred          = self.future_head(z_pool)               # [B, proj_dim]

        # EMA: encode full window
        with torch.no_grad():
            z_full     = self.target_encoder(beats, valid_mask)
            z_target, _ = self.pool(z_full, valid_mask)
            z_target   = self.target_projector(z_target)       # [B, proj_dim]

        return pred[valid_samples], z_target[valid_samples]

    def forward(self, beats, valid_mask, node_mask, beats1, valid1, beats2, valid2):
        target, recon            = self.masked_forward(beats, node_mask, valid_mask)
        p1, z2, p2, z1t, z1_pool = self.byol_forward(beats1, valid1, beats2, valid2)
        pred_hrv                 = self.hrv_head(z1_pool)
        pred_future, tgt_future  = self.future_forward(beats, valid_mask)
        return target, recon, p1, z2, p2, z1t, pred_hrv, pred_future, tgt_future
