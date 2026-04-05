import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Utilities
# ============================================================

def masked_mean(x, mask, dim=1, eps=1e-8):
    """
    x:    [B, N, D]
    mask: [B, N] bool
    """
    mask_f = mask.float().unsqueeze(-1)   # [B, N, 1]
    x = x * mask_f
    denom = mask_f.sum(dim=dim).clamp_min(eps)
    return x.sum(dim=dim) / denom


# ============================================================
# Graph Layer
# ============================================================
import torch
import torch.nn as nn
import torch.nn.functional as F


class GATLayer(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.1):
        super().__init__()

        self.heads = heads
        self.dim = dim
        self.head_dim = dim // heads

        assert dim % heads == 0

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

        self.out_proj = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, edge_index_list, valid_mask):
        """
        x: [B, N, D]
        edge_index_list: list of [2, E]
        """

        B, N, D = x.shape
        out = []

        for b in range(B):
            xb = x[b]                  # [N, D]
            vb = valid_mask[b]
            ei = edge_index_list[b]

            n_valid = int(vb.sum().item())
            if n_valid == 0:
                out.append(torch.zeros_like(xb))
                continue

            h = xb.clone()

            Q = self.q_proj(h).view(N, self.heads, self.head_dim)
            K = self.k_proj(h).view(N, self.heads, self.head_dim)
            V = self.v_proj(h).view(N, self.heads, self.head_dim)

            out_h = torch.zeros_like(Q)

            if ei.numel() > 0:
                src = ei[0]
                dst = ei[1]

                valid_edge = (src < n_valid) & (dst < n_valid)
                src = src[valid_edge]
                dst = dst[valid_edge]

                q = Q[dst]   # [E, H, d]
                k = K[src]
                v = V[src]

                attn = (q * k).sum(-1) / (self.head_dim ** 0.5)  # [E, H]
                attn = torch.softmax(attn, dim=0)

                out_h.index_add_(0, dst, attn.unsqueeze(-1) * v)

            out_h = out_h.view(N, D)
            out_h = self.out_proj(out_h)

            h_new = self.norm(h + self.dropout(out_h))
            h_new = h_new * vb.unsqueeze(-1)

            out.append(h_new)

        return torch.stack(out, dim=0)

class SimpleGraphLayer(nn.Module):
    """
    Simple message passing:
        h_i' = W_self h_i + mean_j(W_neigh h_j)
    using dense padded batching with edge_index per sample.
    """

    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.self_proj = nn.Linear(dim, dim)
        self.neigh_proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index_list, valid_mask):
        """
        x: [B, N, D]
        edge_index_list: list of length B, each [2, E]
        valid_mask: [B, N]
        """
        B, N, D = x.shape
        out = []

        for b in range(B):
            xb = x[b]                       # [N, D]
            vb = valid_mask[b]              # [N]
            edge_index = edge_index_list[b] # [2, E]

            n_valid = int(vb.sum().item())
            if n_valid == 0:
                out.append(torch.zeros_like(xb))
                continue

            h = xb.clone()

            # self term
            self_term = self.self_proj(h)

            # neighbor aggregation
            neigh_agg = torch.zeros_like(h)
            deg = torch.zeros(N, device=h.device, dtype=h.dtype)

            if edge_index.numel() > 0:
                src = edge_index[0]
                dst = edge_index[1]

                valid_edge = (src < n_valid) & (dst < n_valid)
                src = src[valid_edge]
                dst = dst[valid_edge]

                if src.numel() > 0:
                    msg = self.neigh_proj(h[src])  # [E, D]
                    neigh_agg.index_add_(0, dst, msg)
                    deg.index_add_(
                        0, dst,
                        torch.ones_like(dst, dtype=h.dtype)
                    )

            deg = deg.clamp_min(1.0).unsqueeze(-1)
            neigh_agg = neigh_agg / deg

            hb = self_term + neigh_agg
            hb = F.gelu(hb)
            hb = self.dropout(hb)
            hb = self.norm(h + hb)

            # zero out padded nodes
            hb = hb * vb.unsqueeze(-1)
            out.append(hb)

        return torch.stack(out, dim=0)  # [B, N, D]


# ============================================================
# Positional Encoding for token order
# ============================================================

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, max_len, dim):
        super().__init__()
        self.pos = nn.Parameter(torch.randn(1, max_len, dim) * 0.02)

    def forward(self, x):
        """
        x: [B, N, D]
        """
        N = x.size(1)
        return x + self.pos[:, :N, :]


# ============================================================
# Sequence Encoder
# ============================================================

class SequenceEncoder(nn.Module):
    def __init__(
        self,
        dim=256,
        num_heads=8,
        num_layers=4,
        ff_mult=4,
        dropout=0.1,
        max_len=2048
    ):
        super().__init__()
        self.pos_enc = LearnablePositionalEncoding(max_len=max_len, dim=dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * ff_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

    def forward(self, x, valid_mask):
        """
        x: [B, N, D]
        valid_mask: [B, N] bool
        """
        x = self.pos_enc(x)
        key_padding_mask = ~valid_mask   # transformer expects True for padding
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        x = x * valid_mask.unsqueeze(-1)
        return x


# ============================================================
# Full Motif + Graph Hybrid Model
# ============================================================

class ECGMotifGraphModel(nn.Module):
    """
    Inputs expected from collator:
        tokens:      [B, N] long
        features:    [B, N, D_in]
        valid_mask:  [B, N] bool
        node_mask:   [B, N] bool
        edge_index:  list[[2, E], ...] length B
    """

    def __init__(
        self,
        input_dim=128,
        model_dim=256,
        vocab_size=512,
        num_heads=8,
        num_seq_layers=4,
        num_graph_layers=3,
        ff_mult=4,
        dropout=0.1,
        max_len=2048,
        mask_token_id=0,
        fusion_alpha=0.5
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.mask_token_id = mask_token_id
        self.fusion_alpha = fusion_alpha

        # token embedding for masked token modeling
        self.token_embed = nn.Embedding(vocab_size + 1, model_dim)

        # project continuous motif features
        self.feature_proj = nn.Sequential(
            nn.Linear(input_dim, model_dim),
            nn.LayerNorm(model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, model_dim),
        )

        self.input_norm = nn.LayerNorm(model_dim)

        # sequence branch
        self.seq_encoder = SequenceEncoder(
            dim=model_dim,
            num_heads=num_heads,
            num_layers=num_seq_layers,
            ff_mult=ff_mult,
            dropout=dropout,
            max_len=max_len
        )

        # graph branch
        # self.graph_layers = nn.ModuleList([
        #     SimpleGraphLayer(model_dim, dropout=dropout)
        #     for _ in range(num_graph_layers)
        # ])

        self.graph_layers = nn.ModuleList([
            GATLayer(model_dim, heads=4, dropout=dropout)
            for _ in range(num_graph_layers)
        ])

        # fusion
        self.fusion_norm = nn.LayerNorm(model_dim)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, model_dim),
        )

        # masked token prediction head
        self.token_head = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, vocab_size)
        )

        # optional graph contrastive projection
        self.proj_head = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim)
        )

        # optional pooled output for downstream tasks
        self.pool_head = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim),
            nn.GELU()
        )

    def apply_masking(self, tokens, features, node_mask):
        """
        Replace masked token embeddings with special mask embedding
        while keeping unmasked nodes as token + feature mixture.
        """
        # clamp padded zeros safely
        tokens_in = tokens.clone()

        token_emb = self.token_embed(tokens_in)     # [B, N, D]
        feat_emb = self.feature_proj(features)      # [B, N, D]

        x = token_emb + feat_emb

        # special learned mask embedding comes from last vocab slot
        mask_embed = self.token_embed.weight[self.vocab_size].view(1, 1, -1)
        x = torch.where(node_mask.unsqueeze(-1), mask_embed, x)

        return self.input_norm(x)

    def forward(
        self,
        tokens,
        features,
        valid_mask,
        node_mask,
        edge_index
    ):
        """
        edge_index: list of per-sample edge_index tensors
        """

        # input embedding with masking
        x = self.apply_masking(tokens, features, node_mask)  # [B, N, D]
        x = x * valid_mask.unsqueeze(-1)

        # sequence branch
        h_seq = self.seq_encoder(x, valid_mask)              # [B, N, D]

        # graph branch
        h_graph = x
        for layer in self.graph_layers:
            h_graph = layer(h_graph, edge_index, valid_mask)

        # fuse
        h = self.fusion_alpha * h_seq + (1.0 - self.fusion_alpha) * h_graph
        h = self.fusion_norm(h + self.fusion_mlp(h))
        h = h * valid_mask.unsqueeze(-1)

        # token logits
        token_logits = self.token_head(h)                    # [B, N, V]

        # projected embeddings for contrastive / graph SSL
        z = self.proj_head(h)                                # [B, N, D]
        z = F.normalize(z, dim=-1)

        # pooled representation
        pooled = masked_mean(h, valid_mask)                  # [B, D]
        pooled = self.pool_head(pooled)

        return {
            "hidden": h,                 # [B, N, D]
            "token_logits": token_logits,# [B, N, vocab_size]
            "proj": z,                   # [B, N, D]
            "pooled": pooled,            # [B, D]
            "seq_hidden": h_seq,
            "graph_hidden": h_graph
        }

    def masked_token_loss(self, token_logits, target_tokens, node_mask, valid_mask):
        """
        Compute CE only on masked + valid nodes.
        """
        mask = node_mask & valid_mask
        if mask.sum() == 0:
            return token_logits.sum() * 0.0

        logits = token_logits[mask]      # [M, V]
        targets = target_tokens[mask]    # [M]
        return F.cross_entropy(logits, targets)

    def smoothness_loss(self, hidden, edge_index, valid_mask):
        """
        Graph smoothness regularizer:
            mean ||h_i - h_j||^2 over edges
        """
        losses = []
        B = hidden.size(0)

        for b in range(B):
            h = hidden[b]
            n_valid = int(valid_mask[b].sum().item())
            ei = edge_index[b]

            if ei.numel() == 0 or n_valid <= 1:
                continue

            src = ei[0]
            dst = ei[1]
            valid_edge = (src < n_valid) & (dst < n_valid)
            src = src[valid_edge]
            dst = dst[valid_edge]

            if src.numel() == 0:
                continue

            diff = h[src] - h[dst]
            losses.append((diff.pow(2).sum(dim=-1)).mean())

        if len(losses) == 0:
            return hidden.sum() * 0.0

        return torch.stack(losses).mean()

    def seq_graph_consistency_loss(self, seq_hidden, graph_hidden, valid_mask):
        """
        Encourage sequence and graph branches to agree.
        """
        diff = (seq_hidden - graph_hidden).pow(2).sum(dim=-1)  # [B, N]
        diff = diff * valid_mask.float()
        denom = valid_mask.float().sum().clamp_min(1.0)
        return diff.sum() / denom