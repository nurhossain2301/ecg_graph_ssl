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
            "active alert": "wake",
            "active": "wake",
            "drowsy": "sleep",
            "drowsy unsure": "sleep",
            "light sleep": "sleep",
            "deep sleep": "sleep",
            "quiet alert": "wake",
            "crying": "wake"
        }

        self.samples = []
        self._prepare_index()
        self._build_label_mapping()

    # -----------------------------
    # Parse annotation
    # -----------------------------
    def _parse_annotation(self, ann_path, tone_offset):

        interaction_segments = []
        state_segments = []
        location_segments = []

        with open(ann_path, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 8:
                    continue

                tag = parts[0].strip()
                start = float(parts[3]) + tone_offset
                end = float(parts[5]) + tone_offset
                label = parts[-1].strip().lower()

                if tag == "INTERACTION":
                    interaction_segments.append((start, end, label))
                elif tag == "state":
                    state_segments.append((start, end, label))
                elif tag == "location":
                    location_segments.append((start, end, label))

        # ----------------------------------
        # Helper to get overlapping label
        # ----------------------------------
        def get_label_at(time, segments):
            for s, e, l in segments:
                if s <= time < e:
                    return l
            return None

        # ----------------------------------
        # Build final segments
        # ----------------------------------
        final_segments = []

        for start, end, interaction in interaction_segments:

            mid_time = (start + end) / 2

            state = get_label_at(mid_time, state_segments)
            location = get_label_at(mid_time, location_segments)

            # -------------------------
            # Apply your logic
            # -------------------------
            label_group = None

            # Infant movement (non-spatial)
            if interaction.startswith("i-alone"):
                if state in ["crying", "active alert"] and location == "floor":
                    label_group = "infant-alone"

            # Infant movement in space
            elif interaction.startswith("i-move"):
                if location=="floor":
                    label_group = "infant-move"

            # Caregiver touch (non-spatial)
            elif interaction.startswith("c-active") or interaction.startswith("c-passive"):
                label_group = "caregiver-touch"

            # Caregiver moving infant
            elif interaction.startswith("c-pick"):
                label_group = "caregiver-pickup"

            # Skip others (like O-TOUCH)
            if label_group is not None:
                final_segments.append((start, end, label_group))

        return final_segments

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

if __name__ == "__main__":
    from config import Config
    from collections import Counter
    import matplotlib.pyplot as plt
    cfg = Config()
    TRAIN_CSV="../../BRP_train.csv"
    TEST_CSV="../../BRP_test.csv"
    train_dataset = BRPGraphDataset(TRAIN_CSV, window_sec=10, sample_rate=1000, cfg=cfg)
    label2idx = train_dataset.label2idx
    test_dataset  = BRPGraphDataset(TEST_CSV, window_sec=10, sample_rate=1000, cfg=cfg, label2idx=label2idx)
    labels = [train_dataset[i]["label"].item() for i in range(len(train_dataset))]
    data = Counter(labels)
    fig, ax = plt.subplots()
    idx2label = dict(zip(label2idx.values(), label2idx.keys()))
    # Loop through dictionary to plot each bar separately
    for i, (key, value) in enumerate(data.items()):
        ax.bar(i, value, label=idx2label[key])

    # Clean up the x-axis labels
    ax.set_xticks(range(len(data)))
    ax.set_xticklabels(data.keys())

    # Add the legend
    ax.legend(title="Categories")
    
    plt.savefig("Infant_vs_caregiver_train_4class.png", bbox_inches='tight')

    
