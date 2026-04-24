import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from graph_utils import build_graph


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
        self.fc = nn.Linear(64 + rr_dim, d_model)

    def forward(self, beats, rr):
        B, N, L = beats.shape
        x = beats.reshape(B * N, 1, L)
        x = self.net(x).squeeze(-1)
        rr = rr.reshape(B * N, self.rr_dim)
        x = torch.cat([x, rr], dim=-1)
        x = self.fc(x)
        return x.reshape(B, N, -1)


class TemporalEncoder(nn.Module):
    def __init__(self, d_model, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers)

    def forward(self, x):
        return self.encoder(x)


class ResidualGATLayer(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = d_model ** -0.5

    def forward(self, x, valid_mask):
        # x: [B, N, D]
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        attn = torch.matmul(q, k.transpose(1, 2)) * self.scale  # [B,N,N]

        # mask out padded nodes both as query support and key support
        key_mask = ~valid_mask.unsqueeze(1)   # [B,1,N]
        attn = attn.masked_fill(key_mask, -1e9)

        w = torch.softmax(attn, dim=-1)
        w = self.dropout(w)

        out = torch.matmul(w, v)
        out = self.out(out)

        x = self.norm(x + out)
        x = x * valid_mask.unsqueeze(-1)
        return x

class AttentionPool(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 1),
        )

    def forward(self, x, valid_mask):
        """
        x: [B, N, D]
        valid_mask: [B, N] bool
        """
        scores = self.attn(x)  # [B, N, 1]

        # mask padded beats
        scores = scores.masked_fill(~valid_mask.unsqueeze(-1), -1e9)

        weights = torch.softmax(scores, dim=1)  # [B, N, 1]
        pooled = (x * weights).sum(dim=1)       # [B, D]
        return pooled, weights


class ECGEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm_enc = nn.LayerNorm(cfg.d_model)
        self.norm_temp = nn.LayerNorm(cfg.d_model)
        self.norm_out = nn.LayerNorm(cfg.d_model)
        self.beat = BeatEncoder(cfg.beat_len, cfg.d_model, rr_dim=2)
        self.temporal = TemporalEncoder(
            cfg.d_model,
            nhead=4,
            num_layers=2,
            dropout=0.1,
        )
        self.gnn = nn.ModuleList([
            ResidualGATLayer(cfg.d_model, dropout=0.1)
            for _ in range(cfg.gnn_layers)
        ])

    def forward(self, beats, rr, valid_mask):
        x = self.beat(beats, rr)
        x = self.norm_enc(x)
        x = x * valid_mask.unsqueeze(-1)
        x = self.temporal(x)
        x = self.norm_temp(x)
        x = x * valid_mask.unsqueeze(-1)

        for layer in self.gnn:
            x = layer(x, valid_mask)
        x = self.norm_out(x)
        return x


class ECGBYOLModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.online_encoder = ECGEncoder(cfg)
        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.pool = AttentionPool(cfg.d_model)

        for p in self.target_encoder.parameters():
            p.requires_grad = False

        self.online_projector = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.BatchNorm1d(cfg.d_model),
            nn.ReLU(),
            nn.Linear(cfg.d_model, 128),
        )

        self.target_projector = copy.deepcopy(self.online_projector)
        for p in self.target_projector.parameters():
            p.requires_grad = False

        self.predictor = nn.Sequential(
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128),
        )

        self.decoder = nn.Linear(cfg.d_model, cfg.d_model)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        self.mask_token = nn.init.normal_(self.mask_token, std=0.02)

    @torch.no_grad()
    def update_target(self, momentum=0.99):
        for p_o, p_t in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            p_t.data.mul_(momentum).add_(p_o.data, alpha=1.0 - momentum)

        for p_o, p_t in zip(self.online_projector.parameters(), self.target_projector.parameters()):
            p_t.data.mul_(momentum).add_(p_o.data, alpha=1.0 - momentum)

    def masked_forward(self, beats, rr, node_mask, valid_mask):
        # e = self.online_encoder.beat(beats, rr)
        enc = self.online_encoder

        # Step 1: clean beat embeddings (normalized, masked)
        e = enc.beat(beats, rr)
        e = enc.norm_enc(e)
        e = e * valid_mask.unsqueeze(-1)

        # with torch.no_grad():
        #     target = self.online_encoder.temporal(e.clone())
        # target = target.detach()

        # Step 2: target from temporal only (consistent space)
        with torch.no_grad():
            target = enc.temporal(e)
            target = enc.norm_temp(target)
            target = target * valid_mask.unsqueeze(-1)
            target = target.detach()

        # Step 3: mask input embeddings BEFORE temporal
        mask_tok = self.mask_token.expand(e.size(0), e.size(1), -1)
        e_masked = torch.where(node_mask.unsqueeze(-1), mask_tok, e)
        e_masked = e_masked * valid_mask.unsqueeze(-1)
        # e_masked = e.clone()
        # mask_tok = self.mask_token.expand(e.size(0), e.size(1), -1)
        # e_masked = torch.where(node_mask.unsqueeze(-1), mask_tok, e_masked)
        # e_masked = e_masked * valid_mask.unsqueeze(-1)
        # Step 4: reconstruct through same path as target
        x = enc.temporal(e_masked)
        x = enc.norm_temp(x)
        x = x * valid_mask.unsqueeze(-1)

        # x = self.online_encoder.temporal(e_masked)
        # x = x * valid_mask.unsqueeze(-1)

        # Step 5: GNN refinement (optional, separate from target space)
        for layer in enc.gnn:
            x = layer(x, valid_mask)

        recon = self.decoder(x)
        return target, recon, x

    def _pool(self, x, valid_mask, return_attn=False):
        pooled, weights = self.pool(x, valid_mask)
        if return_attn:
            return pooled, weights
        return pooled

    def byol_forward(self, beats1, rr1, valid1, beats2, rr2, valid2, return_attn=False):
        z1 = self.online_encoder(beats1, rr1, valid1)
        if return_attn:
            z1, attn1 = self._pool(z1, valid1, return_attn=True)
        else:
            z1 = self._pool(z1, valid1)

        z1 = self.online_projector(z1)
        p1 = self.predictor(z1)

        with torch.no_grad():
            z2 = self.target_encoder(beats2, rr2, valid2)
            if return_attn:
                z2, attn2 = self._pool(z2, valid2, return_attn=True)
            else:
                z2 = self._pool(z2, valid2)

            z2 = self.target_projector(z2)

        if return_attn:
            return p1, z2, attn1, attn2

        return p1, z2