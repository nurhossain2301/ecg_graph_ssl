# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# def build_graph(x, valid_mask, k=8, temporal_hops=2):
#     B, N, D = x.shape
#     x = F.normalize(x, dim=-1)
#     A = torch.zeros(B, N, N, device=x.device)

#     for b in range(B):
#         valid_idx = torch.where(valid_mask[b])[0]
#         nv = len(valid_idx)
#         if nv == 0:
#             continue

#         xb = x[b, valid_idx]
#         sim = xb @ xb.T
#         kk = min(k + 1, nv)
#         _, nn_idx = torch.topk(sim, k=kk, dim=-1)

#         for i_local, i_global in enumerate(valid_idx):
#             for j_local in nn_idx[i_local]:
#                 j_global = valid_idx[j_local]
#                 if i_global != j_global:
#                     A[b, i_global, j_global] = 1.0

#         # temporal edges
#         for hop in range(1, temporal_hops + 1):
#             for t in range(nv - hop):
#                 i = valid_idx[t]
#                 j = valid_idx[t + hop]
#                 A[b, i, j] = 1.0
#                 A[b, j, i] = 1.0

#         A[b, valid_idx, valid_idx] = 1.0

#     return A


# class IBIEncoder(nn.Module):
#     def __init__(self, in_dim, hidden_dim, d_model, dropout=0.1):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(in_dim, hidden_dim),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, d_model),
#         )

#     def forward(self, x):
#         return self.net(x)


# class TemporalEncoder(nn.Module):
#     def __init__(self, d_model, nhead=4, num_layers=2, dropout=0.1):
#         super().__init__()
#         layer = nn.TransformerEncoderLayer(
#             d_model=d_model,
#             nhead=nhead,
#             dim_feedforward=4 * d_model,
#             dropout=dropout,
#             batch_first=True,
#             activation="gelu",
#             norm_first=True,
#         )
#         self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

#     def forward(self, x, valid_mask):
#         return self.encoder(x, src_key_padding_mask=~valid_mask)


# class GraphBlock(nn.Module):
#     def __init__(self, d_model, dropout=0.1, use_edge_gate=True):
#         super().__init__()
#         self.use_edge_gate = use_edge_gate
#         self.self_lin = nn.Linear(d_model, d_model)
#         self.msg_lin = nn.Linear(d_model, d_model)
#         self.update = nn.Sequential(
#             nn.Linear(2 * d_model, 2 * d_model),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(2 * d_model, d_model),
#         )
#         self.norm = nn.LayerNorm(d_model)
#         self.dropout = nn.Dropout(dropout)
#         if use_edge_gate:
#             self.edge_gate = nn.Sequential(
#                 nn.Linear(2 * d_model, d_model),
#                 nn.GELU(),
#                 nn.Linear(d_model, 1),
#                 nn.Sigmoid(),
#             )

#     def forward(self, x, A, valid_mask):
#         B, N, D = x.shape
#         deg = A.sum(-1, keepdim=True).clamp(min=1.0)

#         if self.use_edge_gate:
#             xi = x.unsqueeze(2).expand(B, N, N, D)
#             xj = x.unsqueeze(1).expand(B, N, N, D)
#             g = self.edge_gate(torch.cat([xi, xj], dim=-1)).squeeze(-1)
#             Aeff = A * g
#             deg = Aeff.sum(-1, keepdim=True).clamp(min=1.0)
#         else:
#             Aeff = A

#         agg = torch.bmm(Aeff, self.msg_lin(x)) / deg
#         self_feat = self.self_lin(x)
#         out = self.update(torch.cat([self_feat, agg], dim=-1))
#         out = self.norm(x + self.dropout(out))
#         out = out * valid_mask.unsqueeze(-1)
#         return out


# class AttnPool(nn.Module):
#     def __init__(self, d_model):
#         super().__init__()
#         self.score = nn.Sequential(
#             nn.Linear(d_model, d_model),
#             nn.Tanh(),
#             nn.Linear(d_model, 1),
#         )

#     def forward(self, x, valid_mask):
#         scores = self.score(x).squeeze(-1)
#         scores = scores.masked_fill(~valid_mask, -1e9)
#         attn = torch.softmax(scores, dim=-1)
#         pooled = torch.sum(attn.unsqueeze(-1) * x, dim=1)
#         return pooled, attn


# class IBIGraphClassifier(nn.Module):
#     def __init__(self, cfg):
#         super().__init__()
#         self.k = cfg.knn_k
#         self.use_virtual_node = cfg.use_virtual_node

#         self.encoder = IBIEncoder(
#             in_dim=cfg.ibi_feature_dim,
#             hidden_dim=cfg.node_mlp_hidden,
#             d_model=cfg.d_model,
#             dropout=cfg.dropout,
#         )
#         self.temporal = TemporalEncoder(
#             d_model=cfg.d_model,
#             nhead=cfg.nhead,
#             num_layers=cfg.transformer_layers,
#             dropout=cfg.dropout,
#         )
#         self.gnn = nn.ModuleList([
#             GraphBlock(cfg.d_model, dropout=cfg.dropout, use_edge_gate=cfg.use_edge_gate)
#             for _ in range(cfg.gnn_layers)
#         ])
#         self.pool = AttnPool(cfg.d_model)

#         self.cls_head = nn.Sequential(
#             nn.Linear(3 * cfg.d_model, 2 * cfg.d_model),
#             nn.GELU(),
#             nn.Dropout(cfg.dropout),
#             nn.Linear(2 * cfg.d_model, cfg.num_classes),
#         )

#         if self.use_virtual_node:
#             self.virtual_node = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
#             nn.init.normal_(self.virtual_node, std=0.02)

#     def forward(self, beats, rr, valid_mask):
#         x = self.encoder(beats)
#         x = self.temporal(x, valid_mask)

#         if self.use_virtual_node:
#             v = self.virtual_node.expand(x.size(0), 1, x.size(-1))
#             x = x + v

#         A = build_graph(x, valid_mask, k=self.k, temporal_hops=2)

#         for layer in self.gnn:
#             x = layer(x, A, valid_mask)

#         mean_pool = (x * valid_mask.unsqueeze(-1)).sum(1) / valid_mask.sum(1, keepdim=True).clamp(min=1)
#         max_pool = x.masked_fill(~valid_mask.unsqueeze(-1), -1e9).max(dim=1).values
#         attn_pool, attn = self.pool(x, valid_mask)

#         graph_emb = torch.cat([mean_pool, max_pool, attn_pool], dim=-1)
#         logits = self.cls_head(graph_emb)

#         return {
#             "logits": logits,
#             "graph_emb": graph_emb,
#             "node_emb": x,
#             "attn": attn,
#             "adj": A,
#         }

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------------------------------
# Fully batched graph construction
# -------------------------------------------------------
def build_graph(x, valid_mask, k=8, temporal_hops=2):
    """
    x:          [B, N, D]
    valid_mask: [B, N] bool
    returns A:  [B, N, N] float (symmetric, includes self-loops for valid nodes)
    """
    B, N, D = x.shape
    device = x.device

    # semantic kNN
    xn  = F.normalize(x, dim=-1)
    sim = torch.bmm(xn, xn.transpose(1, 2))           # [B, N, N]

    mask2d = valid_mask.unsqueeze(2) & valid_mask.unsqueeze(1)
    sim    = sim.masked_fill(~mask2d, -1e9)
    eye    = torch.eye(N, device=device).bool().unsqueeze(0)
    sim    = sim.masked_fill(eye, -1e9)

    kk        = min(k, N - 1)
    topk_val, _ = torch.topk(sim, kk, dim=-1)
    threshold = topk_val[..., -1:]
    A         = (sim >= threshold).float()

    # symmetrize kNN
    A = torch.maximum(A, A.transpose(1, 2))

    # temporal edges (batched)
    idx = torch.arange(N, device=device)
    for hop in range(1, temporal_hops + 1):
        src       = idx[:N - hop]
        dst       = idx[hop:]
        valid_edge = (valid_mask[:, src] & valid_mask[:, dst]).float()  # [B, N-hop]
        A[:, src, dst] = torch.maximum(A[:, src, dst], valid_edge)
        A[:, dst, src] = torch.maximum(A[:, dst, src], valid_edge)

    # self-loops for valid nodes only
    valid_diag = torch.diag_embed(valid_mask.float())  # [B, N, N]
    A          = torch.maximum(A, valid_diag)

    return A


# -------------------------------------------------------
# Input feature encoder with residual projection
# -------------------------------------------------------
class IBIEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, d_model, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
        )
        # residual projection if dims differ
        self.residual = (
            nn.Linear(in_dim, d_model) if in_dim != d_model else nn.Identity()
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        return self.norm(self.net(x) + self.residual(x))


# -------------------------------------------------------
# Positional encoding for temporal order
# -------------------------------------------------------
class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model, dropout=0.1):
        super().__init__()
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.pos_emb.weight, std=0.02)

    def forward(self, x):
        B, N, D = x.shape
        pos = torch.arange(N, device=x.device).unsqueeze(0)  # [1, N]
        return self.dropout(x + self.pos_emb(pos))


# -------------------------------------------------------
# Transformer temporal encoder
# -------------------------------------------------------
class TemporalEncoder(nn.Module):
    def __init__(self, d_model, nhead=4, num_layers=2, dropout=0.1, max_len=512):
        super().__init__()
        self.pos_enc = LearnedPositionalEncoding(max_len, d_model, dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,            # pre-norm for stability
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x, valid_mask):
        x = self.pos_enc(x)
        return self.encoder(x, src_key_padding_mask=~valid_mask)


# -------------------------------------------------------
# Graph block with memory-efficient gating
# -------------------------------------------------------
class GraphBlock(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()

        # message computation
        self.msg_lin  = nn.Linear(d_model, d_model)
        self.self_lin = nn.Linear(d_model, d_model)

        # decomposed gate — no N² memory expansion
        self.gate_src = nn.Linear(d_model, d_model)
        self.gate_dst = nn.Linear(d_model, d_model)
        self.gate_out = nn.Linear(d_model, 1)

        # update MLP
        self.update = nn.Sequential(
            nn.Linear(2 * d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
        )
        self.norm    = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, A, valid_mask):
        # decomposed edge gate [B, N, 1]
        gate = torch.sigmoid(
            self.gate_out(
                torch.bmm(A, self.gate_dst(x)) + self.gate_src(x)
            )
        )

        # gated aggregation
        Aeff = A * gate.squeeze(-1).unsqueeze(1)    # broadcast over rows
        deg  = Aeff.sum(-1, keepdim=True).clamp(min=1.0)
        agg  = torch.bmm(Aeff, self.msg_lin(x)) / deg

        # update
        out  = self.update(torch.cat([self.self_lin(x), agg], dim=-1))
        out  = self.norm(x + self.dropout(out))
        out  = out * valid_mask.unsqueeze(-1).float()
        return out


# -------------------------------------------------------
# Multi-head attention pooling
# -------------------------------------------------------
class AttnPool(nn.Module):
    def __init__(self, d_model, num_heads=4):
        super().__init__()
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.Tanh(),
                nn.Linear(d_model // 2, 1),
            )
            for _ in range(num_heads)
        ])
        self.out_proj = nn.Linear(d_model * num_heads, d_model)

    def forward(self, x, valid_mask):
        """
        x:          [B, N, D]
        valid_mask: [B, N]
        returns:    pooled [B, D], attn_weights [B, num_heads, N]
        """
        mask = valid_mask.unsqueeze(-1)
        pooled_list, attn_list = [], []

        for head in self.heads:
            scores  = head(x).masked_fill(~mask, -1e9)    # [B, N, 1]
            weights = torch.softmax(scores, dim=1)         # [B, N, 1]
            pooled  = (x * weights).sum(dim=1)            # [B, D]
            pooled_list.append(pooled)
            attn_list.append(weights.squeeze(-1))

        pooled = self.out_proj(
            torch.cat(pooled_list, dim=-1)                 # [B, D*H]
        )
        attn = torch.stack(attn_list, dim=1)               # [B, H, N]
        return pooled, attn


# -------------------------------------------------------
# Main classifier
# -------------------------------------------------------
class IBIGraphClassifier(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.k                = cfg.knn_k
        self.use_virtual_node = cfg.use_virtual_node
        D                     = cfg.d_model

        # --- modules ---
        self.encoder  = IBIEncoder(cfg.ibi_feature_dim, cfg.node_mlp_hidden, D, cfg.dropout)
        self.temporal = TemporalEncoder(D, cfg.nhead, cfg.transformer_layers,
                                        cfg.dropout, max_len=cfg.max_beats)
        self.gnn      = nn.ModuleList([
            GraphBlock(D, dropout=cfg.dropout) for _ in range(cfg.gnn_layers)
        ])
        self.pool     = AttnPool(D, num_heads=cfg.pool_heads)

        # virtual node
        if self.use_virtual_node:
            self.virtual_node    = nn.Parameter(torch.empty(1, 1, D))
            self.vn_update       = nn.Sequential(   # virtual node update MLP
                nn.Linear(2 * D, D),
                nn.GELU(),
                nn.Linear(D, D),
                nn.LayerNorm(D),
            )
            nn.init.normal_(self.virtual_node, std=0.02)

        # classifier — takes mean + max + attn pooling
        self.cls_head = nn.Sequential(
            nn.Linear(3 * D, 2 * D),
            nn.LayerNorm(2 * D),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(2 * D, D),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(D, cfg.num_classes),
        )

    # ------------------------------------------------------------------
    def _pool(self, x, valid_mask):
        mask_f = valid_mask.unsqueeze(-1).float()

        mean_pool = (x * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)
        max_pool  = x.masked_fill(~valid_mask.unsqueeze(-1), -1e9).max(dim=1).values
        attn_pool, attn_weights = self.pool(x, valid_mask)

        return torch.cat([mean_pool, max_pool, attn_pool], dim=-1), attn_weights

    # ------------------------------------------------------------------
    def forward(self, beats, rr, valid_mask):
        B = beats.size(0)

        # 1. node feature encoding
        x = self.encoder(beats)               # [B, N, D]

        # 2. temporal context
        x = self.temporal(x, valid_mask)      # [B, N, D]

        # 3. virtual node — prepend as node 0
        if self.use_virtual_node:
            vn        = self.virtual_node.expand(B, 1, -1)   # [B, 1, D]
            x         = torch.cat([vn, x], dim=1)            # [B, N+1, D]
            vn_mask   = torch.ones(B, 1, dtype=torch.bool, device=x.device)
            valid_mask = torch.cat([vn_mask, valid_mask], dim=1)  # [B, N+1]

        # 4. dynamic graph + GNN
        layer_outputs = []
        for layer in self.gnn:
            A = build_graph(x, valid_mask, k=self.k, temporal_hops=2)
            x = layer(x, A, valid_mask)
            layer_outputs.append(x)

            # update virtual node with global mean after each layer
            if self.use_virtual_node:
                real_x    = x[:, 1:, :]
                real_mask = valid_mask[:, 1:]
                global_mean = (
                    (real_x * real_mask.unsqueeze(-1).float()).sum(1)
                    / real_mask.sum(1, keepdim=True).float().clamp(min=1)
                )                                            # [B, D]
                vn_new  = self.vn_update(
                    torch.cat([x[:, 0, :], global_mean], dim=-1)
                )                                            # [B, D]
                x       = torch.cat([vn_new.unsqueeze(1), x[:, 1:, :]], dim=1)

        # 5. strip virtual node before pooling
        if self.use_virtual_node:
            vn_emb     = x[:, 0, :]     # [B, D] — can be used as extra feature
            x          = x[:, 1:, :]
            valid_mask = valid_mask[:, 1:]

        # 6. readout
        graph_emb, attn_weights = self._pool(x, valid_mask)  # [B, 3D]
        logits                  = self.cls_head(graph_emb)

        return {
            "logits":    logits,
            "graph_emb": graph_emb,
            "node_emb":  x,
            "attn":      attn_weights,
            "adj":       A,
            "vn_emb":    vn_emb if self.use_virtual_node else None,
        }