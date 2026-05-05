import random
import numpy as np
import torch
from torch.utils.data import Dataset

from signal_utils import (
    load_ibi_with_time, window_ibi_by_time, clean_ibi,
    build_ibi_features, compute_hrv_features, HRV_KEYS, normalize_hrv,
)


# ─── IBI augmentations ────────────────────────────────────────────────────────

def ibi_jitter(ibi: np.ndarray, std_ms: float = 5.0) -> np.ndarray:
    noise = np.random.randn(len(ibi)).astype(np.float32) * (std_ms / 1000.0)
    return np.clip(ibi + noise, 0.1, 3.0)


def ibi_dropout(ibi: np.ndarray, keep_prob: float = 0.9) -> np.ndarray:
    """Random individual-beat dropout (keeps ≥4 beats)."""
    if len(ibi) <= 4:
        return ibi
    mask = np.random.rand(len(ibi)) < keep_prob
    if mask.sum() < 4:
        mask[:4] = True
    return ibi[mask]


def ibi_block_dropout(ibi: np.ndarray, min_gap: int = 3, max_gap: int = 10) -> np.ndarray:
    """Drop a contiguous block of beats (simulates bursty sensor loss)."""
    if len(ibi) <= max_gap + 4:
        return ibi
    gap   = np.random.randint(min_gap, min(max_gap + 1, len(ibi) - 4))
    start = np.random.randint(0, len(ibi) - gap)
    return np.concatenate([ibi[:start], ibi[start + gap:]])


def ibi_scale(ibi: np.ndarray, low: float = 0.92, high: float = 1.08) -> np.ndarray:
    return (ibi * np.random.uniform(low, high)).astype(np.float32)


def augment_ibi(ibi: np.ndarray) -> np.ndarray:
    ibi = ibi_jitter(ibi, std_ms=5.0)
    if len(ibi) > 4 and random.random() < 0.5:
        ibi = ibi_block_dropout(ibi)          # block dropout (primary)
    elif len(ibi) > 4 and random.random() < 0.2:
        ibi = ibi_dropout(ibi, keep_prob=0.9) # random dropout (secondary)
    if random.random() < 0.3:
        ibi = ibi_scale(ibi)
    return ibi


# ─── Span masking ─────────────────────────────────────────────────────────────

def _span_mask_indices(n_valid: int, mask_ratio: float, mean_span: int = 5) -> set:
    """Return a set of local indices (into valid positions) to mask via span masking."""
    n_mask = max(1, int(n_valid * mask_ratio))
    masked = set()
    p = 1.0 / max(mean_span, 1)
    for _ in range(300):
        if len(masked) >= n_mask:
            break
        start = random.randint(0, n_valid - 1)
        span  = int(np.random.geometric(p))
        for k in range(start, min(start + span, n_valid)):
            masked.add(k)
    return masked


# ─── Dataset ─────────────────────────────────────────────────────────────────

class IBIGraphDataset(Dataset):
    def __init__(self, file_list, dataset_size=None, mode="train",
                 windows_per_file_val=5, cfg=None):
        self.file_list    = file_list
        self.mode         = mode
        self.cfg          = cfg

        if mode == "train":
            self.dataset_size = dataset_size or len(file_list)
        else:
            self.val_indices = [
                (fi, w)
                for fi in range(len(file_list))
                for w in range(windows_per_file_val)
            ]

    def __len__(self):
        return self.dataset_size if self.mode == "train" else len(self.val_indices)

    def _load_ibi(self, path, train=True, val_window_idx=0, window_sec=None):
        ibi, time = load_ibi_with_time(path)
        ws = window_sec if window_sec is not None else self.cfg.window_sec
        ibi_win, _ = window_ibi_by_time(
            ibi, time,
            window_sec=ws,
            train=train,
            val_window_idx=val_window_idx,
            val_windows_total=self.cfg.windows_per_file_val,
        )
        return ibi_win

    def _featurise(self, ibi_raw):
        ibi_clean, quality = clean_ibi(
            ibi_raw,
            min_ibi_ms=self.cfg.min_ibi_ms,
            max_ibi_ms=self.cfg.max_ibi_ms,
        )
        feats   = build_ibi_features(ibi_clean, quality,
                                     max_len_beats=self.cfg.max_len_beats)
        hrv_d   = compute_hrv_features(ibi_clean)
        hrv_vec = normalize_hrv(
            np.array([hrv_d[k] for k in HRV_KEYS], dtype=np.float32)
        )
        return feats, ibi_clean, hrv_vec

    def __getitem__(self, idx):
        if self.mode == "train":
            path = random.choice(self.file_list)

            # Sample random window duration (multi-resolution)
            ws = int(random.choice(self.cfg.window_choices))

            ibi_base          = self._load_ibi(path, train=True, window_sec=ws)
            feats, ibi_clean, hrv_vec = self._featurise(ibi_base)

            # Two independent augmented views for BYOL
            ibi_raw1          = self._load_ibi(path, train=True, window_sec=ws)
            feats1, ibi1, _   = self._featurise(augment_ibi(ibi_raw1))

            ibi_raw2          = self._load_ibi(path, train=True, window_sec=ws)
            feats2, ibi2, _   = self._featurise(augment_ibi(ibi_raw2))

        else:
            fi, w    = self.val_indices[idx]
            path     = self.file_list[fi]
            ibi_raw  = self._load_ibi(path, train=False, val_window_idx=w)
            feats, ibi_clean, hrv_vec = self._featurise(ibi_raw)
            feats1, ibi1 = feats.copy(), ibi_clean.copy()
            feats2, ibi2 = feats.copy(), ibi_clean.copy()

        return {
            "beats":       torch.tensor(feats,     dtype=torch.float32),
            "rr":          torch.tensor(ibi_clean, dtype=torch.float32),
            "hrv":         torch.tensor(hrv_vec,   dtype=torch.float32),
            "beats_view1": torch.tensor(feats1,    dtype=torch.float32),
            "rr_view1":    torch.tensor(ibi1,      dtype=torch.float32),
            "beats_view2": torch.tensor(feats2,    dtype=torch.float32),
            "rr_view2":    torch.tensor(ibi2,      dtype=torch.float32),
        }


# ─── Collator ────────────────────────────────────────────────────────────────

class IBIGraphCollator:
    def __init__(self, node_mask_ratio=0.15, span_mean_len=5):
        self.node_mask_ratio = node_mask_ratio  # updated externally for curriculum
        self.span_mean_len   = span_mean_len

    def _pad(self, feat_list, ibi_list):
        max_n    = max(x.shape[0] for x in feat_list)
        feat_dim = feat_list[0].shape[1]
        B        = len(feat_list)

        feats      = torch.zeros(B, max_n, feat_dim)
        ibi        = torch.zeros(B, max_n)
        valid_mask = torch.zeros(B, max_n, dtype=torch.bool)

        for i in range(B):
            n = feat_list[i].shape[0]
            feats[i, :n]      = feat_list[i]
            ibi[i, :n]        = ibi_list[i]
            valid_mask[i, :n] = True

        return feats, ibi, valid_mask

    def __call__(self, batch):
        beats,  rr,  valid_mask = self._pad(
            [b["beats"]       for b in batch], [b["rr"]       for b in batch])
        beats1, rr1, valid1     = self._pad(
            [b["beats_view1"] for b in batch], [b["rr_view1"] for b in batch])
        beats2, rr2, valid2     = self._pad(
            [b["beats_view2"] for b in batch], [b["rr_view2"] for b in batch])

        hrv = torch.stack([b["hrv"] for b in batch])           # [B, 14]

        # Span masking (replaces uniform random masking)
        node_mask = torch.zeros_like(valid_mask)
        for i in range(valid_mask.size(0)):
            valid_idx = torch.where(valid_mask[i])[0]
            if len(valid_idx) == 0:
                continue
            n_valid     = len(valid_idx)
            masked_local = _span_mask_indices(
                n_valid, self.node_mask_ratio, self.span_mean_len
            )
            perm = valid_idx[list(masked_local)]
            node_mask[i, perm] = True

        return {
            "beats":            beats,  "rr":    rr,   "valid_mask":       valid_mask,
            "node_mask":        node_mask,
            "hrv":              hrv,
            "beats_view1":      beats1, "rr_view1": rr1, "valid_mask_view1": valid1,
            "beats_view2":      beats2, "rr_view2": rr2, "valid_mask_view2": valid2,
        }
