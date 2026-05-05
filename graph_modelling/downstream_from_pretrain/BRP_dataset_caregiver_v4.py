import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from scipy.io import wavfile
from torch.utils.data import Dataset

from signal_utils import detect_r_peaks, extract_beats


class BRPGraphDatasetCaregiverV4(Dataset):
    """
    Caregiver sub-type dataset — 3-class problem.

    Instead of grouping all caregiving into one "caregiver" class, splits by
    interaction type to reveal which sub-type (if any) leaves a detectable
    ECG trace in the infant's cardiac activity:

        0  infant    — i-alone (crying/active alert on floor) + i-move on floor
        1  c_active  — active caregiving touch (c-active*)
        2  c_passive — passive caregiving touch (c-passive*)

    c-pick is excluded: it is a transient spatial movement, not a sustained
    interaction state, and would have too few samples for reliable training.

    Everything else (windowing, beat extraction) is identical to V1.
    """

    def __init__(
        self,
        csv_file,
        window_sec=30,
        sample_rate=1000,
        normalize=True,
        cfg=None,
        label2idx=None,
    ):
        self.meta = pd.read_csv(csv_file)
        self.window_sec  = window_sec
        self.sr          = sample_rate
        self.window_size = int(window_sec * sample_rate)

        self.max_beats = cfg.max_beats_per_segment
        self.pre_ms    = cfg.pre_r_ms
        self.post_ms   = cfg.post_r_ms
        self.normalize = normalize
        self.label2idx = label2idx

        self.samples = []
        self._prepare_index()
        self._build_label_mapping()

    # ------------------------------------------------------------------
    def _parse_annotation(self, ann_path, tone_offset):
        interaction_segments = []
        state_segments       = []
        location_segments    = []

        with open(ann_path, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 8:
                    continue
                tag   = parts[0].strip()
                start = float(parts[3]) + tone_offset
                end   = float(parts[5]) + tone_offset
                label = parts[-1].strip().lower()

                if tag == "INTERACTION":
                    interaction_segments.append((start, end, label))
                elif tag == "state":
                    state_segments.append((start, end, label))
                elif tag == "location":
                    location_segments.append((start, end, label))

        def get_label_at(time, segs):
            for s, e, l in segs:
                if s <= time < e:
                    return l
            return None

        final_segments = []
        for start, end, interaction in interaction_segments:
            mid   = (start + end) / 2
            state = get_label_at(mid, state_segments)
            loc   = get_label_at(mid, location_segments)
            label_group = None

            if interaction.startswith("i-alone"):
                if state in ["crying", "active alert"] and loc == "floor":
                    label_group = "infant"
            elif interaction.startswith("i-move"):
                if loc == "floor":
                    label_group = "infant"
            elif interaction.startswith("c-active"):
                label_group = "c_active"
            elif interaction.startswith("c-passive"):
                label_group = "c_passive"
            # c-pick excluded intentionally

            if label_group is not None:
                final_segments.append((start, end, label_group))

        return final_segments

    # ------------------------------------------------------------------
    def _prepare_index(self):
        print("Indexing BRP Caregiver V4 (sub-type labels: infant / c_active / c_passive)...")

        for _, row in self.meta.iterrows():
            ecg_path    = row["ECG_files"]
            ann_path    = row["label_file"]
            tone_offset = float(row["tone_sec"])

            if not os.path.exists(ecg_path) or not os.path.exists(ann_path):
                continue

            sr, waveform  = wavfile.read(ecg_path)
            total_samples = waveform.shape[0]
            segments      = self._parse_annotation(ann_path, tone_offset)

            for seg_start, seg_end, label in segments:
                start_sample = int(seg_start * sr)
                end_sample   = int(seg_end   * sr)

                if end_sample - start_sample < self.window_size:
                    continue

                for s in range(start_sample, end_sample - self.window_size, self.window_size):
                    self.samples.append((ecg_path, s, label))

        print(f"Total samples: {len(self.samples)}")

    def _build_label_mapping(self):
        if self.label2idx is None:
            labels        = [lbl for _, _, lbl in self.samples]
            unique_labels = sorted(set(labels))
            self.label2idx = {l: i for i, l in enumerate(unique_labels)}
        self.idx2label = {i: l for l, i in self.label2idx.items()}
        print("Label mapping:", self.label2idx)

    def __len__(self):
        return len(self.samples)

    # ------------------------------------------------------------------
    def __getitem__(self, idx):
        ecg_path, start_sample, label = self.samples[idx]

        sr, waveform = wavfile.read(ecg_path)
        waveform     = torch.tensor(waveform, dtype=torch.float32)

        segment = waveform[start_sample : start_sample + self.window_size]
        if segment.shape[0] < self.window_size:
            segment = F.pad(segment, (0, self.window_size - segment.shape[0]))

        if self.normalize:
            segment = (segment - segment.mean()) / (segment.std() + 1e-6)

        ecg_np  = segment.numpy()
        r_peaks = detect_r_peaks(ecg_np, self.sr)
        beats_np, rr_np = extract_beats(
            ecg_np, r_peaks, self.sr,
            pre_ms=self.pre_ms, post_ms=self.post_ms,
            max_beats=self.max_beats,
        )

        beats = torch.tensor(beats_np, dtype=torch.float32)
        rr    = torch.tensor(rr_np,    dtype=torch.float32)
        N     = beats.shape[0]

        if N < self.max_beats:
            pad_n = self.max_beats - N
            beats = F.pad(beats, (0, 0, 0, pad_n))
            rr    = F.pad(rr,    (0, 0, 0, pad_n))
            valid_mask = torch.zeros(self.max_beats, dtype=torch.bool)
            valid_mask[:N] = True
        else:
            beats      = beats[: self.max_beats]
            rr         = rr[: self.max_beats]
            valid_mask = torch.ones(self.max_beats, dtype=torch.bool)

        node_mask = torch.zeros(self.max_beats, dtype=torch.bool)
        label_idx = self.label2idx[label]

        return {
            "beats":      beats,
            "rr":         rr,
            "node_mask":  node_mask,
            "valid_mask": valid_mask,
            "label":      torch.tensor(label_idx, dtype=torch.long),
        }
