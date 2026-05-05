import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from scipy.io import wavfile
from torch.utils.data import Dataset

from signal_utils import detect_r_peaks, extract_beats


class BRPGraphDatasetCaregiverV5(Dataset):
    """
    V2 + V3 combined — all three improvements together:

    1. Transition windows only (V3): only windows within ±TRANSITION_SEC of
       annotation onset or offset are kept, targeting the cardiac response
       window rather than the steady-state plateau.

    2. Session-level normalization (V2): raw segment z-scored with the
       per-recording mean/std, removing between-infant amplitude differences.

    3. Dual-window temporal context (V2): the preceding window is prepended
       to the current window so the model attends across both and can learn
       the delta in cardiac activity. Output sequence length = 2 * max_beats.

    Binary: caregiver vs infant.
    """

    TRANSITION_SEC = 15

    def __init__(
        self,
        csv_file,
        window_sec=30,
        sample_rate=1000,
        normalize=True,
        cfg=None,
        label2idx=None,
    ):
        self.meta        = pd.read_csv(csv_file)
        self.window_sec  = window_sec
        self.sr          = sample_rate
        self.window_size = int(window_sec * sample_rate)
        self.trans_samp  = int(self.TRANSITION_SEC * sample_rate)

        self.max_beats = cfg.max_beats_per_segment
        self.pre_ms    = cfg.pre_r_ms
        self.post_ms   = cfg.post_r_ms
        self.normalize = normalize
        self.label2idx = label2idx

        self.session_stats = {}   # ecg_path -> (mean, std)
        self.samples       = []   # (ecg_path, start, prev_start, label)

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
            elif interaction.startswith("c-active") or interaction.startswith("c-passive"):
                label_group = "caregiver"
            elif interaction.startswith("c-pick"):
                label_group = "caregiver"

            if label_group is not None:
                final_segments.append((start, end, label_group))

        return final_segments

    # ------------------------------------------------------------------
    def _prepare_index(self):
        print(f"Indexing BRP Caregiver V5 (transition ±{self.TRANSITION_SEC}s + session-norm + dual-window)...")

        for _, row in self.meta.iterrows():
            ecg_path    = row["ECG_files"]
            ann_path    = row["label_file"]
            tone_offset = float(row["tone_sec"])

            if not os.path.exists(ecg_path) or not os.path.exists(ann_path):
                continue

            sr, waveform = wavfile.read(ecg_path)

            if ecg_path not in self.session_stats:
                wf = torch.tensor(waveform, dtype=torch.float32)
                self.session_stats[ecg_path] = (
                    wf.mean().item(),
                    (wf.std() + 1e-6).item(),
                )

            segments = self._parse_annotation(ann_path, tone_offset)

            for seg_start, seg_end, label in segments:
                start_sample = int(seg_start * sr)
                end_sample   = int(seg_end   * sr)

                if end_sample - start_sample < self.window_size:
                    continue

                for s in range(start_sample, end_sample - self.window_size, self.window_size):
                    dist_start = s - start_sample
                    dist_end   = end_sample - (s + self.window_size)

                    if dist_start <= self.trans_samp or dist_end <= self.trans_samp:
                        prev_s = s - self.window_size
                        self.samples.append((ecg_path, s, prev_s, label))

        print(f"Total transition-window samples: {len(self.samples)}")

    def _build_label_mapping(self):
        if self.label2idx is None:
            labels        = [lbl for _, _, _, lbl in self.samples]
            unique_labels = sorted(set(labels))
            self.label2idx = {l: i for i, l in enumerate(unique_labels)}
        self.idx2label = {i: l for l, i in self.label2idx.items()}
        print("Label mapping:", self.label2idx)

    def __len__(self):
        return len(self.samples)

    # ------------------------------------------------------------------
    def _extract_window(self, waveform, start_sample, session_mean, session_std):
        total = waveform.shape[0]
        if start_sample < 0 or start_sample + self.window_size > total:
            beat_len = int((self.pre_ms + self.post_ms) * self.sr / 1000)
            return torch.zeros(self.max_beats, beat_len), torch.zeros(self.max_beats, 2), 0

        segment = waveform[start_sample : start_sample + self.window_size]
        if segment.shape[0] < self.window_size:
            segment = F.pad(segment, (0, self.window_size - segment.shape[0]))

        segment = (segment - session_mean) / session_std

        ecg_np  = segment.numpy()
        beats_np, rr_np = extract_beats(
            ecg_np, detect_r_peaks(ecg_np, self.sr),
            self.sr, pre_ms=self.pre_ms, post_ms=self.post_ms,
            max_beats=self.max_beats,
        )

        beats = torch.tensor(beats_np, dtype=torch.float32)
        rr    = torch.tensor(rr_np,    dtype=torch.float32)
        N     = beats.shape[0]

        if N < self.max_beats:
            pad_n = self.max_beats - N
            beats = F.pad(beats, (0, 0, 0, pad_n))
            rr    = F.pad(rr,    (0, 0, 0, pad_n))
        else:
            beats = beats[: self.max_beats]
            rr    = rr[: self.max_beats]
            N     = self.max_beats

        return beats, rr, N

    # ------------------------------------------------------------------
    def __getitem__(self, idx):
        ecg_path, start_sample, prev_start_sample, label = self.samples[idx]

        sr, waveform_np = wavfile.read(ecg_path)
        waveform = torch.tensor(waveform_np, dtype=torch.float32)

        session_mean, session_std = self.session_stats[ecg_path]

        curr_beats, curr_rr, curr_N = self._extract_window(
            waveform, start_sample, session_mean, session_std
        )
        prev_beats, prev_rr, prev_N = self._extract_window(
            waveform, prev_start_sample, session_mean, session_std
        )

        beats = torch.cat([prev_beats, curr_beats], dim=0)
        rr    = torch.cat([prev_rr,    curr_rr],    dim=0)

        prev_valid = torch.zeros(self.max_beats, dtype=torch.bool)
        prev_valid[:prev_N] = True
        curr_valid = torch.zeros(self.max_beats, dtype=torch.bool)
        curr_valid[:curr_N] = True
        valid_mask = torch.cat([prev_valid, curr_valid], dim=0)

        node_mask = torch.zeros(self.max_beats * 2, dtype=torch.bool)
        label_idx = self.label2idx[label]

        return {
            "beats":      beats,
            "rr":         rr,
            "node_mask":  node_mask,
            "valid_mask": valid_mask,
            "label":      torch.tensor(label_idx, dtype=torch.long),
        }
