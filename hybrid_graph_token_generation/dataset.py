import torch
from torch.utils.data import Dataset
import numpy as np
import random
import soundfile as sf

from tokenizer_graph import MotifGraphPipeline


class ECGTokenDataset(Dataset):

    def __init__(
        self,
        file_list,
        sampling_rate=1000,
        window_sec=30,
        dataset_size=None,
        mode="train",
        windows_per_file_val=5,
        normalize=True,
        cfg=None
    ):
        self.file_list = file_list
        self.sr = sampling_rate
        self.window_size = sampling_rate * window_sec
        self.mode = mode
        self.normalize = normalize
        self.cfg = cfg

        # ---- TOKEN CONFIG ----
        self.token_window = int(self.sr * cfg.token_window_ms / 1000)
        self.token_stride = int(self.sr * cfg.token_stride_ms / 1000)

        # ---- PIPELINE ----
        self.pipeline = MotifGraphPipeline(
            segment_len=self.token_window,
            emb_dim=cfg.emb_dim,
            n_clusters=cfg.n_clusters,
            knn_k=cfg.knn_k
        )

        # --- fitted centers ----
        centers = np.load(cfg.codebook_path)
        self.pipeline.tokenizer.kmeans.cluster_centers_ = centers
        self.pipeline.tokenizer.is_fitted = True

        # IMPORTANT: tokenizer must be pre-fitted offline
        assert self.pipeline.tokenizer.is_fitted, \
            "Tokenizer must be fitted before training"

        # ---- FILE META ----
        self.file_info = []
        for f in self.file_list:
            info = sf.info(f)
            self.file_info.append({
                "path": f,
                "frames": info.frames
            })

        # TRAIN
        if self.mode == "train":
            self.dataset_size = dataset_size if dataset_size else len(self.file_list)

        # VAL
        else:
            self.val_indices = []
            for file_idx, meta in enumerate(self.file_info):
                total_frames = meta["frames"]

                for w in range(windows_per_file_val):
                    if total_frames > self.window_size:
                        start = int(
                            (w + 1) * (total_frames - self.window_size) /
                            (windows_per_file_val + 1)
                        )
                    else:
                        start = 0

                    self.val_indices.append((file_idx, start))

    # ------------------------------------------------
    def __len__(self):
        if self.mode == "train":
            return self.dataset_size
        else:
            return len(self.val_indices)

    # ------------------------------------------------
    def _load_window(self, file_path, start):

        x, _ = sf.read(
            file_path,
            start=start,
            frames=self.window_size,
            dtype="float32"
        )

        if len(x) < self.window_size:
            pad = self.window_size - len(x)
            x = np.pad(x, (0, pad), mode="reflect")

        if x.ndim > 1:
            x = x[:, 0]

        if self.normalize:
            x = (x - x.mean()) / (x.std() + 1e-6)

        return x.astype(np.float32)

    # ------------------------------------------------
    def _segment(self, ecg):
        segments = []

        for start in range(0, len(ecg) - self.token_window, self.token_stride):
            seg = ecg[start:start + self.token_window]
            segments.append(seg)

        return np.stack(segments)  # [N, L]

    # ------------------------------------------------
    def __getitem__(self, idx):

        if self.mode == "train":
            meta = random.choice(self.file_info)
            total_frames = meta["frames"]
            file_path = meta["path"]

            if total_frames > self.window_size:
                start = random.randint(0, total_frames - self.window_size)
            else:
                start = 0

        else:
            file_idx, start = self.val_indices[idx]
            meta = self.file_info[file_idx]
            file_path = meta["path"]

        # ---- Load ECG ----
        ecg = self._load_window(file_path, start)

        # ---- Segment ----
        segments = self._segment(ecg)  # [N, L]
        segments = torch.tensor(segments, dtype=torch.float32)

        # ---- Pipeline ----
        out = self.pipeline.forward(segments)

        return {
            "segments": segments,              # [N, L]
            "tokens": out["tokens"],           # [N]
            "features": out["features"],       # [N, D]
            "edge_index": out["edge_index"]    # [2, E]
        }

class GraphSSL_Collator:
    def __init__(self, node_mask_ratio=0.3):
        self.node_mask_ratio = node_mask_ratio

    def __call__(self, batch):
        tokens_list = [b["tokens"] for b in batch]
        feat_list = [b["features"] for b in batch]
        edge_list = [b["edge_index"] for b in batch]

        max_n = max(x.shape[0] for x in tokens_list)
        B = len(batch)
        D = feat_list[0].shape[1]

        tokens = torch.zeros(B, max_n, dtype=torch.long)
        features = torch.zeros(B, max_n, D)
        valid_mask = torch.zeros(B, max_n, dtype=torch.bool)
        edge_index_batch = []

        for i in range(B):
            n = tokens_list[i].shape[0]
            tokens[i, :n] = tokens_list[i]
            features[i, :n] = feat_list[i]
            valid_mask[i, :n] = True
            edge_index_batch.append(edge_list[i])

        node_mask = torch.zeros_like(valid_mask)

        for i in range(B):
            valid_idx = torch.where(valid_mask[i])[0]
            if len(valid_idx) == 0:
                continue

            m = max(1, int(len(valid_idx) * self.node_mask_ratio))
            perm = valid_idx[torch.randperm(len(valid_idx))[:m]]
            node_mask[i, perm] = True

        return {
            "tokens": tokens,
            "features": features,
            "valid_mask": valid_mask,
            "node_mask": node_mask,
            "edge_index": edge_index_batch
        }