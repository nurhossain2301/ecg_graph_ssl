import torch
from torch.utils.data import Dataset
import numpy as np
from signal_utils import detect_r_peaks, extract_beats

import torch
from torch.utils.data import Dataset
import numpy as np
import random
import soundfile as sf

from signal_utils import detect_r_peaks, extract_beats


class ECGGraphDataset(Dataset):

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

        # File metadata
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

        # VALIDATION
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

        # Pad
        if len(x) < self.window_size:
            pad = self.window_size - len(x)
            x = np.pad(x, (0, pad), mode="reflect")

        # Stereo → mono
        if x.ndim > 1:
            x = x[:, 0]

        if self.normalize:
            x = (x - x.mean()) / (x.std() + 1e-6)

        return x.astype(np.float32)

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

        # ---- Load ECG window ----
        ecg = self._load_window(file_path, start)

        # ---- R-peak detection ----
        r_peaks = detect_r_peaks(ecg, self.sr, self.cfg.min_rr_ms)

        # ---- Beat extraction ----
        beats, rr = extract_beats(
            ecg,
            r_peaks,
            self.sr,
            self.cfg.pre_r_ms,
            self.cfg.post_r_ms,
            self.cfg.max_beats_per_segment
        )

        return {
            "beats": torch.tensor(beats, dtype=torch.float32),   # [N, L]
            "rr": torch.tensor(rr, dtype=torch.float32),         # [N, 2]
        }


class GraphSSL_Collator:

    def __init__(self, node_mask_ratio=0.3):
        self.node_mask_ratio = node_mask_ratio

    def __call__(self, batch):

        beats_list = [b["beats"] for b in batch]
        rr_list = [b["rr"] for b in batch]

        max_n = max(x.shape[0] for x in beats_list)
        beat_len = beats_list[0].shape[1]

        B = len(batch)

        beats = torch.zeros(B, max_n, beat_len)
        rr = torch.zeros(B, max_n, 2)
        valid_mask = torch.zeros(B, max_n, dtype=torch.bool)

        for i in range(B):
            n = beats_list[i].shape[0]

            beats[i, :n] = beats_list[i]
            rr[i, :n] = rr_list[i]
            valid_mask[i, :n] = True

        # ---- Node masking ----
        node_mask = torch.zeros_like(valid_mask)

        for i in range(B):
            valid_idx = torch.where(valid_mask[i])[0]
            if len(valid_idx) == 0:
                continue

            m = max(1, int(len(valid_idx) * self.node_mask_ratio))
            perm = valid_idx[torch.randperm(len(valid_idx))[:m]]
            node_mask[i, perm] = True

        return {
            "beats": beats,               # [B, N, L]
            "rr": rr,                     # [B, N, 2]
            "valid_mask": valid_mask,     # [B, N]
            "node_mask": node_mask        # [B, N]
        }