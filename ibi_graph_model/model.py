import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------------------------------
# Graph Builder
# -------------------------------------------------------
def build_graph(x, valid_mask, k=6):
    """
    x: [B, N, D]
    valid_mask: [B, N]
    """
    B, N, D = x.shape
    x = F.normalize(x, dim=-1)
    sim = torch.matmul(x, x.transpose(1, 2))

    A = torch.zeros(B, N, N, device=x.device)

    for b in range(B):
        valid_idx = torch.where(valid_mask[b])[0]
        if len(valid_idx) == 0:
            continue

        xb = x[b, valid_idx]                          # [Nv, D]
        sim_b = xb @ xb.T                            # [Nv, Nv]
        kk = min(k + 1, len(valid_idx))
        _, idx = torch.topk(sim_b, k=kk, dim=-1)

        for i_local, i_global in enumerate(valid_idx):
            neigh_local = idx[i_local]
            for j_local in neigh_local:
                j_global = valid_idx[j_local]
                if i_global != j_global:
                    A[b, i_global, j_global] = 1.0

        # temporal edges
        for t in range(len(valid_idx) - 1):
            i = valid_idx[t]
            j = valid_idx[t + 1]
            A[b, i, j] = 1.0
            A[b, j, i] = 1.0

        A[b, valid_idx, valid_idx] += torch.eye(len(valid_idx), device=x.device)

    return A


# -------------------------------------------------------
# IBI Encoder
# -------------------------------------------------------
class IBIEncoder(nn.Module):
    def __init__(self, in_dim, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.GELU(),
            nn.Linear(64, d_model),
        )

    def forward(self, x):
        return self.net(x)


# -------------------------------------------------------
# Temporal Encoder
# -------------------------------------------------------
class TemporalEncoder(nn.Module):
    def __init__(self, d_model, nhead=4, num_layers=2):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x, valid_mask):
        # src_key_padding_mask expects True = pad
        pad_mask = ~valid_mask
        return self.encoder(x, src_key_padding_mask=pad_mask)


# -------------------------------------------------------
# Simple GNN
# -------------------------------------------------------
class SimpleGNNLayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.lin_self = nn.Linear(d_model, d_model)
        self.lin_neigh = nn.Linear(d_model, d_model)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, A):
        deg = A.sum(-1, keepdim=True) + 1e-6
        agg = torch.bmm(A, x) / deg
        out = self.lin_self(x) + self.lin_neigh(agg)
        return self.norm(self.act(out) + x)


# -------------------------------------------------------
# IBI Graph SSL Model
# -------------------------------------------------------
class ECGModel(nn.Module):
    """
    Kept the same class name for compatibility.
    Now it is an IBI-only graph SSL model.
    """
    def __init__(self, cfg):
        super().__init__()

        self.encoder = IBIEncoder(cfg.ibi_feature_dim, cfg.d_model)
        self.temporal = TemporalEncoder(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            num_layers=cfg.transformer_layers
        )
        self.gnn = nn.ModuleList(
            [SimpleGNNLayer(cfg.d_model) for _ in range(cfg.gnn_layers)]
        )
        self.decoder = nn.Linear(cfg.d_model, cfg.d_model)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        self.k = cfg.knn_k

    def forward(self, beats, rr, node_mask, valid_mask):
        """
        beats: [B, N, F]   -> IBI node features
        rr:    [B, N]      -> cleaned IBI sequence (kept for compatibility)
        """
        e = self.encoder(beats)   # [B, N, D]

        with torch.no_grad():
            target = self.temporal(e.clone(), valid_mask)
        target = target.detach()

        e_masked = e.clone()
        mask_token = self.mask_token.expand(e.size(0), e.size(1), -1)

        # only replace masked valid nodes
        replace_mask = node_mask & valid_mask
        e_masked = torch.where(replace_mask.unsqueeze(-1), mask_token, e_masked)

        # padded nodes to zero
        e_masked = e_masked * valid_mask.unsqueeze(-1)

        x = self.temporal(e_masked, valid_mask)

        A = build_graph(x, valid_mask, k=self.k)

        for layer in self.gnn:
            x = layer(x, A)

        recon = self.decoder(x)
        return target, recon