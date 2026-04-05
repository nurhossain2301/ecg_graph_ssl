import torch
from torch.utils.data import Dataset
import numpy as np
import random
import soundfile as sf


class ECGCodebookDataset(Dataset):
    """
    Codebook dataset aligned with ECGTokenDataset
    BUT without tokenizer / graph
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
        self.sr = sampling_rate
        self.window_size = sampling_rate * window_sec
        self.mode = mode
        self.normalize = normalize
        self.cfg = cfg

        # ---- TOKEN SEGMENT CONFIG ----
        self.token_window = int(self.sr * cfg.token_window_ms / 1000)
        self.token_stride = int(self.sr * cfg.token_stride_ms / 1000)

        # ---- FILE META (same as your dataset) ----
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

        # VAL (not really needed for codebook but kept consistent)
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

        if len(segments) == 0:
            segments.append(ecg[:self.token_window])

        return np.stack(segments)  # [N, L]

    # ------------------------------------------------
    def __getitem__(self, idx):

        # SAME sampling logic as training dataset
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

        return torch.tensor(segments, dtype=torch.float32)

def codebook_collate_fn(batch):
    """
    batch: list of [Ni, L]
    return: [total_segments, L]
    """
    return torch.cat(batch, dim=0)