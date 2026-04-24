import numpy as np
import torch
from torch.utils.data import Dataset
import random
import soundfile as sf
from signal_utils import detect_r_peaks, extract_beats


def add_gaussian_noise(x, std=0.01):
    return x + torch.randn_like(x) * std


def amplitude_scale(x, low=0.9, high=1.1):
    scale = torch.empty(1).uniform_(low, high).item()
    return x * scale


def baseline_wander(x, sr, amp=0.05, freq_low=0.05, freq_high=0.5):
    t = torch.arange(x.numel(), device=x.device, dtype=x.dtype) / sr
    freq = torch.empty(1).uniform_(freq_low, freq_high).item()
    phase = torch.empty(1).uniform_(0, 2 * np.pi).item()
    wander = amp * torch.sin(2 * np.pi * freq * t + phase)
    return x + wander


def random_time_mask_beats(beats, mask_ratio=0.1):
    # beats: [N, L]
    out = beats.clone()
    N, L = out.shape
    span = max(1, int(L * mask_ratio))
    for i in range(N):
        if random.random() < 0.5:
            start = random.randint(0, max(0, L - span))
            out[i, start:start + span] = 0.0
    return out


def random_beat_dropout(beats, rr, valid_keep_prob=0.9):
    keep = torch.rand(beats.size(0)) < valid_keep_prob
    if keep.sum() == 0:
        keep[random.randint(0, beats.size(0) - 1)] = True
    return beats[keep], rr[keep]


def rr_jitter(rr, std=0.01):
    return rr + torch.randn_like(rr) * std


class ECGGraphBYOLDataset(Dataset):
    def __init__(
        self,
        file_list,
        sampling_rate=1000,
        window_sec=30,
        dataset_size=None,
        mode="train",
        windows_per_file_val=5,
        normalize=True,
        cfg=None,
    ):
        self.file_list = file_list
        self.sr = sampling_rate
        self.window_size = sampling_rate * window_sec
        self.mode = mode
        self.normalize = normalize
        self.cfg = cfg

        self.file_info = []
        for f in self.file_list:
            info = sf.info(f)
            self.file_info.append({"path": f, "frames": info.frames})

        if self.mode == "train":
            self.dataset_size = dataset_size if dataset_size else len(self.file_list)
        else:
            self.val_indices = []
            for file_idx, meta in enumerate(self.file_info):
                total_frames = meta["frames"]
                for w in range(windows_per_file_val):
                    if total_frames > self.window_size:
                        start = int((w + 1) * (total_frames - self.window_size) / (windows_per_file_val + 1))
                    else:
                        start = 0
                    self.val_indices.append((file_idx, start))

    def __len__(self):
        return self.dataset_size if self.mode == "train" else len(self.val_indices)

    def _load_window(self, file_path, start):
        x, _ = sf.read(file_path, start=start, frames=self.window_size, dtype="float32")
        if len(x) < self.window_size:
            x = np.pad(x, (0, self.window_size - len(x)), mode="reflect")
        if x.ndim > 1:
            x = x[:, 0]
        if self.normalize:
            x = (x - x.mean()) / (x.std() + 1e-6)
        return torch.tensor(x.astype(np.float32))

    def _build_graph_inputs(self, ecg):
        r_peaks = detect_r_peaks(ecg.numpy(), self.sr, self.cfg.min_rr_ms)
        beats, rr = extract_beats(
            ecg.numpy(),
            r_peaks,
            self.sr,
            self.cfg.pre_r_ms,
            self.cfg.post_r_ms,
            self.cfg.max_beats_per_segment,
        )
        beats = torch.tensor(beats, dtype=torch.float32)
        rr = torch.tensor(rr, dtype=torch.float32)
        return beats, rr

    def _augment_graph_view(self, beats, rr):
        beats = amplitude_scale(beats)
        beats = add_gaussian_noise(beats, std=0.01)
        beats = random_time_mask_beats(beats, mask_ratio=0.08)
        rr = rr_jitter(rr, std=0.01)

        if beats.size(0) > 2 and random.random() < 0.5:
            beats, rr = random_beat_dropout(beats, rr, valid_keep_prob=0.9)

        return beats, rr

    def __getitem__(self, idx):
        if self.mode == "train":
            meta = random.choice(self.file_info)
            total_frames = meta["frames"]
            file_path = meta["path"]
            start = random.randint(0, total_frames - self.window_size) if total_frames > self.window_size else 0
        else:
            file_idx, start = self.val_indices[idx]
            meta = self.file_info[file_idx]
            file_path = meta["path"]

        ecg = self._load_window(file_path, start)

        beats, rr = self._build_graph_inputs(ecg)

        if self.mode == "train":
            beats1, rr1 = self._augment_graph_view(beats.clone(), rr.clone())
            beats2, rr2 = self._augment_graph_view(beats.clone(), rr.clone())
        else:
            beats1, rr1 = beats.clone(), rr.clone()
            beats2, rr2 = beats.clone(), rr.clone()

        return {
            "beats": beats,
            "rr": rr,
            "beats_view1": beats1,
            "rr_view1": rr1,
            "beats_view2": beats2,
            "rr_view2": rr2,
        }

class GraphBYOLCollator:
    def __init__(self, node_mask_ratio=0.3):
        self.node_mask_ratio = node_mask_ratio

    def _pad_pack(self, beats_list, rr_list):
        max_n = max(x.shape[0] for x in beats_list)
        beat_len = beats_list[0].shape[1]
        B = len(beats_list)

        beats = torch.zeros(B, max_n, beat_len)
        rr = torch.zeros(B, max_n, 2)
        valid_mask = torch.zeros(B, max_n, dtype=torch.bool)

        for i in range(B):
            n = beats_list[i].shape[0]
            beats[i, :n] = beats_list[i]
            rr[i, :n] = rr_list[i]
            valid_mask[i, :n] = True

        return beats, rr, valid_mask

    def __call__(self, batch):
        beats, rr, valid_mask = self._pad_pack(
            [b["beats"] for b in batch],
            [b["rr"] for b in batch],
        )

        beats1, rr1, valid1 = self._pad_pack(
            [b["beats_view1"] for b in batch],
            [b["rr_view1"] for b in batch],
        )

        beats2, rr2, valid2 = self._pad_pack(
            [b["beats_view2"] for b in batch],
            [b["rr_view2"] for b in batch],
        )

        node_mask = torch.zeros_like(valid_mask)
        for i in range(node_mask.size(0)):
            idx = torch.where(valid_mask[i])[0]
            if len(idx) == 0:
                continue
            m = max(1, int(len(idx) * self.node_mask_ratio))
            perm = idx[torch.randperm(len(idx))[:m]]
            node_mask[i, perm] = True

        return {
            "beats": beats,
            "rr": rr,
            "valid_mask": valid_mask,
            "node_mask": node_mask,
            "beats_view1": beats1,
            "rr_view1": rr1,
            "valid_mask_view1": valid1,
            "beats_view2": beats2,
            "rr_view2": rr2,
            "valid_mask_view2": valid2,
        }