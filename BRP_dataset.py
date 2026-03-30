import os
import pandas as pd
import numpy as np
import torch
from scipy.io import wavfile
from torch.utils.data import Dataset, DataLoader



class BRPSleepDataset(Dataset):
    def __init__(self, csv_file, window_sec=30, sample_rate=1000, normalize=True):
        """
        csv_file: train.csv or test.csv
        window_sec: length of each ECG segment
        sample_rate: ECG sampling rate
        """
        self.meta = pd.read_csv(csv_file)
        self.window_sec = window_sec
        self.sr = sample_rate
        self.window_size = int(window_sec * sample_rate)
        self.normalize = normalize
        
        self.samples = []  # list of (ecg_path, start_sample, label)
        self._prepare_index()

    def _parse_annotation(self, ann_path, tone_offset):
        """
        Returns list of (start_sec, end_sec, label)
        """
        segments = []
        with open(ann_path, 'r') as f:
            for line in f:
                if not line.startswith("state"):
                    continue
                
                parts = line.strip().split('\t')
                start_sec = float(parts[3]) + tone_offset
                end_sec   = float(parts[5]) + tone_offset
                label_txt = parts[-1].lower()

                label = 1 if "sleep" in label_txt else 0
                segments.append((start_sec, end_sec, label))
        return segments

    def _prepare_index(self):
        print("Indexing dataset...")
        
        for _, row in self.meta.iterrows():
            ecg_path = row["ECG_files"]
            ann_path = row["label_file"]
            tone_offset = float(row["tone_sec"])
            
            if not os.path.exists(ecg_path) or not os.path.exists(ann_path):
                continue

            # Load ECG to get duration
            sr, waveform = wavfile.read(ecg_path)
            waveform = torch.tensor(waveform, dtype=torch.float32)
            total_samples = waveform.shape[0]
            total_sec = total_samples / sr

            segments = self._parse_annotation(ann_path, tone_offset)

            # Create windows ONLY inside annotated segments

            window_samples = int(self.window_sec * sr)

            for seg_start, seg_end, seg_label in segments:
                
                seg_start_sample = int(seg_start * sr)
                seg_end_sample   = int(seg_end * sr)
                seg_len = seg_end_sample - seg_start_sample

                # Skip very short segments
                if seg_len < window_samples:
                    continue

                # Chunk this annotated segment into 30s windows
                for start_sample in range(seg_start_sample,
                                        seg_end_sample - window_samples,
                                        window_samples):

                    self.samples.append((ecg_path, start_sample, seg_label))

        print(f"Total windows: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ecg_path, start_sample, label = self.samples[idx]
        
        sr, waveform = wavfile.read(ecg_path)
        waveform = torch.tensor(waveform, dtype=torch.float32)
        segment = waveform[start_sample:start_sample+self.window_size]
        if segment.shape[0] < self.window_size:
            pad = self.window_size - segment.shape[0]
            segment = torch.nn.functional.pad(segment, (0, pad))

        
        if self.normalize:
            segment = (segment - segment.mean()) / (segment.std() + 1e-6)

        return segment, torch.tensor(label, dtype=torch.long)


def sanity_check():
    train_dataset = BRPSleepDataset("BRP_train.csv", window_sec=30, sample_rate=1000)
    test_dataset  = BRPSleepDataset("BRP_test.csv", window_sec=30, sample_rate=1000)
    print(len(train_dataset), train_dataset[0][1])
    print(len(test_dataset))

if __name__=="__main__":
    sanity_check()
    