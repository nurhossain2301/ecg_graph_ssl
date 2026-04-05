import torch
from torch.utils.data import Dataset
import numpy as np
import random
import pandas as pd

from signal_utils import load_ibi_file, clean_ibi, build_ibi_features, random_window_ibi


class ECGGraphDataset(Dataset):
    """
    Kept the same class name so the rest of your codebase does not need large changes.
    But internally this is now an IBI graph dataset.
    """

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
        self.mode = mode
        self.cfg = cfg

        if self.mode == "train":
            self.dataset_size = dataset_size if dataset_size else len(self.file_list)
        else:
            # fixed deterministic repeats per file
            self.val_indices = []
            for file_idx in range(len(self.file_list)):
                for w in range(windows_per_file_val):
                    self.val_indices.append((file_idx, w))

    def __len__(self):
        if self.mode == "train":
            return self.dataset_size
        return len(self.val_indices)

    def _load_features_from_file(self, file_path, train=True):
        ibi = load_ibi_file(file_path)
        ibi = random_window_ibi(
            ibi,
            window_beats=self.cfg.max_len_beats,
            train=train
        )

        ibi_clean, quality = clean_ibi(
            ibi,
            min_ibi_ms=self.cfg.min_ibi_ms,
            max_ibi_ms=self.cfg.max_ibi_ms
        )

        feats = build_ibi_features(
            ibi_clean,
            quality,
            max_len_beats=self.cfg.max_len_beats
        )

        return feats, ibi_clean

    def __getitem__(self, idx):
        if self.mode == "train":
            file_path = random.choice(self.file_list)
            feats, ibi_clean = self._load_features_from_file(file_path, train=True)
        else:
            file_idx, _ = self.val_indices[idx]
            file_path = self.file_list[file_idx]
            feats, ibi_clean = self._load_features_from_file(file_path, train=False)

        return {
            "beats": torch.tensor(feats, dtype=torch.float32),   # [N, F] kept key name for compatibility
            "rr": torch.tensor(ibi_clean, dtype=torch.float32),  # [N] actual cleaned IBI in sec
        }


class GraphSSL_Collator:
    def __init__(self, node_mask_ratio=0.3):
        self.node_mask_ratio = node_mask_ratio

    def __call__(self, batch):
        feat_list = [b["beats"] for b in batch]   # [N, F]
        ibi_list = [b["rr"] for b in batch]       # [N]

        max_n = max(x.shape[0] for x in feat_list)
        feat_dim = feat_list[0].shape[1]
        B = len(batch)

        feats = torch.zeros(B, max_n, feat_dim)
        ibi = torch.zeros(B, max_n)
        valid_mask = torch.zeros(B, max_n, dtype=torch.bool)

        for i in range(B):
            n = feat_list[i].shape[0]
            feats[i, :n] = feat_list[i]
            ibi[i, :n] = ibi_list[i]
            valid_mask[i, :n] = True

        node_mask = torch.zeros_like(valid_mask)

        for i in range(B):
            valid_idx = torch.where(valid_mask[i])[0]
            if len(valid_idx) == 0:
                continue
            m = max(1, int(len(valid_idx) * self.node_mask_ratio))
            perm = valid_idx[torch.randperm(len(valid_idx))[:m]]
            node_mask[i, perm] = True

        return {
            "beats": feats,            # [B, N, F] now IBI features
            "rr": ibi,                 # [B, N] cleaned IBI sec
            "valid_mask": valid_mask,
            "node_mask": node_mask
        }