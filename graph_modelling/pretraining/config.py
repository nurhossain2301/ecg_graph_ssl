from dataclasses import dataclass
import torch

@dataclass
class Config:
    sampling_rate: int = 1000
    window_sec: int = 30
    dataset_size: int = 200000
    windows_per_file_val: int = 100

    # beat-window ratios (fraction of median RR used before/after R-peak)
    # 0.38 + 0.52 = 0.90 × median_RR → ~450ms at the 500ms infant median IBI
    # min values enforce ≥350ms floor across the full infant IBI range (350–650ms)
    pre_ratio:   float = 0.38
    post_ratio:  float = 0.52
    min_pre_ms:  int   = 130
    min_post_ms: int   = 220
    max_pre_ms:  int   = 400
    max_post_ms: int   = 600

    min_rr_ms: int = 250   # used only by the fallback detect_r_peaks

    max_beats_per_segment: int = 32

    beat_embed_dim: int = 128
    graph_dim: int = 128
    gnn_layers: int = 3
    d_model: int = 128

    knn_k: int = 6
    node_mask_ratio: float = 0.3
    edge_mask_ratio: float = 0.2

    batch_size: int = 128
    epochs: int = 100
    lr: float = 1e-3
    min_lr: float = 1e-5
    warmup_epochs: int = 5
    weight_decay: float = 1e-4


    dropout: float = 0.0

    num_workers: int = 4
    grad_clip: float = 1.0

    # Overfit mode: use a tiny fixed subset so train loss should → 0
    overfit: bool = False
    overfit_files: int = 5       # number of source files to draw from
    overfit_samples: int = 256   # virtual dataset_size while overfitting

    run_name: str = "graph_bcp_byol_v1"
    seed: int = 42

    test_csv: str = "/work/hdd/bebr/Projects/ecg_foundational_model/ECG_val_files.csv"
    train_csv: str = "/work/hdd/bebr/Projects/ecg_foundational_model/ECG_train_files.csv"
    output_dir: str = "/work/nvme/bebr/mkhan14/ecg_foundation_model/graph_modelling/experiments"

    @property
    def beat_len(self):
        return int((self.max_pre_ms + self.max_post_ms) * self.sampling_rate / 1000)


    # device: str = "cuda" if torch.cuda.is_available() else "cpu"
