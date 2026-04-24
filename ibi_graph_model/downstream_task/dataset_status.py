import os
import torch
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from torch.utils.data import Dataset

from littlebeats.littlebeats.ecg.ibi import detect_ibi_simple
from ibi_graph_model.signal_utils import clean_ibi, build_ibi_features, random_window_ibi


class IBIGraphDataset(Dataset):
    def __init__(
        self,
        csv_file,
        window_sec=30,
        sample_rate=1000,
        cfg=None,
        label2idx=None,
        normalize=True
    ):
        self.meta = pd.read_csv(csv_file)

        self.window_sec = window_sec
        self.sr = sample_rate
        self.window_size = int(window_sec * sample_rate)

        self.cfg = cfg
        self.normalize = normalize
        self.label2idx = label2idx

        # ---- SAME LABEL GROUPING AS BRP ----
        self.LABEL_GROUP_MAP = {
            "active alert": "active",
            "active": "active",
            "drowsy": "sleep",
            "drowsy unsure": "sleep",
            "light sleep": "sleep",
            "deep sleep": "sleep",
            "quiet alert": "quiet",
            "crying": "crying"
        }

        self.samples = []
        self._prepare_index()
        self._build_label_mapping()

    # -------------------------------------------------------
    # Parse annotation (same as BRP)
    # -------------------------------------------------------
    def _parse_annotation(self, ann_path, tone_offset):
        segments = []

        with open(ann_path, "r") as f:
            for line in f:
                if not line.startswith("state"):
                    continue

                parts = line.strip().split("\t")

                start_sec = float(parts[3]) + tone_offset
                end_sec = float(parts[5]) + tone_offset
                label_txt = parts[-1].strip().lower()

                label_group = self.LABEL_GROUP_MAP.get(label_txt, None)

                if label_group is not None:
                    segments.append((start_sec, end_sec, label_group))

        return segments

    # -------------------------------------------------------
    # Build index (window-based)
    # -------------------------------------------------------
    def _prepare_index(self):
        print("Indexing IBI Graph Dataset...")

        for _, row in self.meta.iterrows():
            ecg_path = row["ECG_files"]
            ann_path = row["label_file"]
            tone_offset = float(row["tone_sec"])

            if not os.path.exists(ecg_path) or not os.path.exists(ann_path):
                continue

            # load ECG (txt like save_ibi)
            ecg_txt = ecg_path.replace(".wav", ".txt")
            if not os.path.exists(ecg_txt):
                continue

            ecg_df = pd.read_csv(ecg_txt, names=["time", "ecg"])
            total_time = ecg_df["time"].values[-1]

            segments = self._parse_annotation(ann_path, tone_offset)

            for seg_start, seg_end, label in segments:

                t = seg_start
                while t + self.window_sec <= seg_end:
                    self.samples.append((ecg_txt, t, label))
                    t += self.window_sec

        print(f"Total samples: {len(self.samples)}")

    # -------------------------------------------------------
    # Label mapping
    # -------------------------------------------------------
    def _build_label_mapping(self):
        if self.label2idx is None:
            labels = [label for _, _, label in self.samples]
            unique_labels = sorted(list(set(labels)))
            self.label2idx = {l: i for i, l in enumerate(unique_labels)}

        self.idx2label = {i: l for l, i in self.label2idx.items()}

        print("Label mapping:", self.label2idx)

    # -------------------------------------------------------
    def __len__(self):
        return len(self.samples)

    # -------------------------------------------------------
    # SAME IBI extraction as save_ibi.py
    # -------------------------------------------------------
    def _compute_ibi(self, ecg_txt, start_sec):
        ecg_df = pd.read_csv(ecg_txt, names=["time", "ecg"])

        # slice window
        mask = (ecg_df["time"] >= start_sec) & \
               (ecg_df["time"] < start_sec + self.window_sec)

        segment_df = ecg_df[mask].copy()
        segment_df.attrs['sr'] = 1000

        if len(segment_df) < 10:
            return np.zeros(1)

        ibi_df = detect_ibi_simple(
            segment_df,
            'ecg',
            ['time'],
            debug=False,
            verbose=False
        )

        ibi = ibi_df['ibi'].values.astype(np.float32)

        return ibi

    # -------------------------------------------------------
    def __getitem__(self, idx):
        ecg_txt, start_sec, label = self.samples[idx]

        # ---- Step 1: IBI extraction (same as save_ibi)
        ibi = self._compute_ibi(ecg_txt, start_sec)

        # ---- Step 2: window in beat space
        ibi = random_window_ibi(
            ibi,
            window_beats=self.cfg.max_len_beats,
            train=False
        )

        # ---- Step 3: clean
        ibi_clean, quality = clean_ibi(
            ibi,
            min_ibi_ms=self.cfg.min_ibi_ms,
            max_ibi_ms=self.cfg.max_ibi_ms
        )

        # ---- Step 4: features
        feats = build_ibi_features(
            ibi_clean,
            quality,
            max_len_beats=self.cfg.max_len_beats
        )

        N = feats.shape[0]

        # ---- padding
        if N < self.cfg.max_len_beats:
            pad_n = self.cfg.max_len_beats - N
            feats = np.pad(feats, ((0, pad_n), (0, 0)))
            ibi_clean = np.pad(ibi_clean, (0, pad_n))

            valid_mask = torch.zeros(self.cfg.max_len_beats, dtype=torch.bool)
            valid_mask[:N] = True
        else:
            feats = feats[:self.cfg.max_len_beats]
            ibi_clean = ibi_clean[:self.cfg.max_len_beats]
            valid_mask = torch.ones(self.cfg.max_len_beats, dtype=torch.bool)

        node_mask = torch.zeros(self.cfg.max_len_beats, dtype=torch.bool)

        label_idx = self.label2idx[label]

        return {
            "beats": torch.tensor(feats, dtype=torch.float32),
            "rr": torch.tensor(ibi_clean, dtype=torch.float32),
            "valid_mask": valid_mask,
            "node_mask": node_mask,
            "label": torch.tensor(label_idx, dtype=torch.long)
        }

class IBIGraphCollator:
    def __call__(self, batch: List[Dict]):
        B = len(batch)
        lengths = [x["beats"].shape[0] for x in batch]
        max_n = max(lengths)
        feat_dim = batch[0]["beats"].shape[1]

        beats = torch.zeros(B, max_n, feat_dim, dtype=torch.float32)
        rr = torch.zeros(B, max_n, dtype=torch.float32)
        valid_mask = torch.zeros(B, max_n, dtype=torch.bool)
        labels = torch.zeros(B, dtype=torch.long)
        # files = []

        for i, item in enumerate(batch):
            n = item["beats"].shape[0]
            beats[i, :n] = item["beats"]
            rr[i, :n] = item["rr"][:n]
            valid_mask[i, :n] = True
            labels[i] = item["label"]
            # files.append(item["file"])

        return {
            "beats": beats,
            "rr": rr,
            "valid_mask": valid_mask,
            "label": labels,
        }