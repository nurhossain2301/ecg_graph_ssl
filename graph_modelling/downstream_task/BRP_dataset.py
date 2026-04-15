import os
import torch
import pandas as pd
import numpy as np
from scipy.io import wavfile
from torch.utils.data import Dataset

from signal_utils import detect_r_peaks, extract_beats


class BRPGraphDataset(Dataset):
    def __init__(
        self,
        csv_file,
        window_sec=10,
        sample_rate=1000,
        normalize=True,
        cfg = None,
        label2idx=None,
    ):
        self.meta = pd.read_csv(csv_file)

        self.window_sec = window_sec
        self.sr = sample_rate
        self.window_size = int(window_sec * sample_rate)

        self.max_beats = cfg.max_beats_per_segment
        self.pre_ms = cfg.pre_r_ms
        self.post_ms = cfg.post_r_ms
        self.normalize = normalize
        self.label2idx = label2idx

        # Label grouping (same as your previous)
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

    # -----------------------------
    # Parse annotation
    # -----------------------------
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

    # -----------------------------
    # Build sample index
    # -----------------------------
    def _prepare_index(self):
        print("Indexing BRP Graph Dataset...")

        for _, row in self.meta.iterrows():
            ecg_path = row["ECG_files"]
            ann_path = row["label_file"]
            tone_offset = float(row["tone_sec"])

            if not os.path.exists(ecg_path) or not os.path.exists(ann_path):
                continue

            sr, waveform = wavfile.read(ecg_path)
            total_samples = waveform.shape[0]

            segments = self._parse_annotation(ann_path, tone_offset)

            for seg_start, seg_end, label in segments:
                start_sample = int(seg_start * sr)
                end_sample = int(seg_end * sr)

                if end_sample - start_sample < self.window_size:
                    continue

                for s in range(start_sample, end_sample - self.window_size, self.window_size):
                    self.samples.append((ecg_path, s, label))

        print(f"Total samples: {len(self.samples)}")

    # -----------------------------
    # Label mapping
    # -----------------------------
    def _build_label_mapping(self):
        if self.label2idx is None:
            labels = [label for _, _, label in self.samples]
            unique_labels = sorted(list(set(labels)))
            self.label2idx = {l: i for i, l in enumerate(unique_labels)}
        else:
            self.label2idx = self.label2idx

        self.idx2label = {i: l for l, i in self.label2idx.items()}

        print("Label mapping:", self.label2idx)

    # -----------------------------
    def __len__(self):
        return len(self.samples)

    # -----------------------------
    def __getitem__(self, idx):
        ecg_path, start_sample, label = self.samples[idx]

        sr, waveform = wavfile.read(ecg_path)
        waveform = torch.tensor(waveform, dtype=torch.float32)

        segment = waveform[start_sample:start_sample + self.window_size]

        if segment.shape[0] < self.window_size:
            pad = self.window_size - segment.shape[0]
            segment = torch.nn.functional.pad(segment, (0, pad))

        # Normalize ECG
        if self.normalize:
            segment = (segment - segment.mean()) / (segment.std() + 1e-6)

        # -----------------------------
        # Convert to graph inputs
        # -----------------------------
        ecg_np = segment.numpy()

        r_peaks = detect_r_peaks(ecg_np, self.sr)

        beats, rr = extract_beats(
            ecg_np,
            r_peaks,
            self.sr,
            pre_ms=self.pre_ms,
            post_ms=self.post_ms,
            max_beats=self.max_beats
        )

        beats = torch.tensor(beats, dtype=torch.float32)  # [N, L]
        rr = torch.tensor(rr, dtype=torch.float32)        # [N, 2]

        N = beats.shape[0]

        # -----------------------------
        # Padding to fixed graph size
        # -----------------------------
        if N < self.max_beats:
            pad_n = self.max_beats - N

            beats = torch.nn.functional.pad(beats, (0, 0, 0, pad_n))
            rr = torch.nn.functional.pad(rr, (0, 0, 0, pad_n))

            valid_mask = torch.zeros(self.max_beats, dtype=torch.bool)
            valid_mask[:N] = True
        else:
            beats = beats[:self.max_beats]
            rr = rr[:self.max_beats]
            valid_mask = torch.ones(self.max_beats, dtype=torch.bool)

        # Node mask (no masking during downstream)
        node_mask = torch.zeros(self.max_beats, dtype=torch.bool)

        label_idx = self.label2idx[label]

        return {
            "beats": beats,
            "rr": rr,
            "node_mask": node_mask,
            "valid_mask": valid_mask,
            "label": torch.tensor(label_idx, dtype=torch.long)
        }