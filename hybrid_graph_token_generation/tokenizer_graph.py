import torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import MiniBatchKMeans


# ============================================================
# 1. SEGMENT ENCODER (Motif feature extractor)
# ============================================================

class SegmentEncoder(nn.Module):
    """
    Lightweight CNN encoder for ECG segments
    Input:  [N, L]
    Output: [N, D]
    """

    def __init__(self, in_len, emb_dim=128):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.GELU(),

            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),

            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.GELU(),

            nn.AdaptiveAvgPool1d(1)
        )

        self.proj = nn.Linear(128, emb_dim)

    def forward(self, x):
        # x: [N, L]
        x = x.unsqueeze(1)        # [N, 1, L]
        h = self.encoder(x)       # [N, 128, 1]
        h = h.squeeze(-1)         # [N, 128]
        z = self.proj(h)          # [N, D]
        return z


# ============================================================
# 2. MOTIF TOKENIZER (k-means)
# ============================================================

class MotifTokenizer:
    """
    Discrete token assignment using k-means
    """

    def __init__(self, n_clusters=512):
        self.n_clusters = n_clusters
        self.kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=10000,
            verbose=0
        )
        self.is_fitted = False

    def fit(self, features):
        """
        features: [N, D] numpy
        """
        self.kmeans.fit(features)
        self.is_fitted = True

    def predict(self, features):
        """
        features: [N, D] numpy
        returns token_ids [N]
        """
        assert self.is_fitted, "Tokenizer must be fitted first"
        return self.kmeans.predict(features)


# ============================================================
# 3. HYBRID GRAPH BUILDER
# ============================================================

class GraphBuilder:
    def __init__(
        self,
        k=3,
        temporal_weight=1.0,
        sim_weight=1.0
    ):
        self.k = k
        self.temporal_weight = temporal_weight
        self.sim_weight = sim_weight

    def build(self, features):
        """
        features: [N, D]
        returns edge_index [2, E]
        """

        N = features.shape[0]
        edges = []

        # ----------------------------------
        # 1. Temporal edges
        # ----------------------------------
        for i in range(N - 1):
            edges.append([i, i + 1])
            edges.append([i + 1, i])

        # ----------------------------------
        # 2. kNN similarity edges
        # ----------------------------------
        with torch.no_grad():
            f = torch.tensor(features, dtype=torch.float32)
            dist = torch.cdist(f, f)  # [N, N]

            knn_idx = dist.topk(self.k + 1, largest=False).indices[:, 1:]

            for i in range(N):
                for j in knn_idx[i]:
                    edges.append([i, j.item()])

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        return edge_index


# ============================================================
# 4. FULL PIPELINE
# ============================================================

class MotifGraphPipeline:
    def __init__(
        self,
        segment_len,
        emb_dim=128,
        n_clusters=512,
        knn_k=3
    ):
        self.encoder = SegmentEncoder(segment_len, emb_dim)
        self.tokenizer = MotifTokenizer(n_clusters)
        self.graph_builder = GraphBuilder(k=knn_k)

    def encode(self, segments):
        """
        segments: torch [N, L]
        """
        with torch.no_grad():
            z = self.encoder(segments)  # [N, D]
        return z

    def fit_tokenizer(self, all_features):
        """
        all_features: numpy [M, D]
        """
        self.tokenizer.fit(all_features)

    def tokenize(self, features):
        """
        features: torch [N, D]
        """
        features_np = features.cpu().numpy()
        token_ids = self.tokenizer.predict(features_np)
        return torch.tensor(token_ids, dtype=torch.long)

    def build_graph(self, features):
        """
        features: torch [N, D]
        """
        return self.graph_builder.build(features)

    def forward(self, segments):
        """
        Full pipeline
        """
        features = self.encode(segments)            # [N, D]
        token_ids = self.tokenize(features)         # [N]
        edge_index = self.build_graph(features)     # [2, E]

        return {
            "features": features,
            "tokens": token_ids,
            "edge_index": edge_index
        }