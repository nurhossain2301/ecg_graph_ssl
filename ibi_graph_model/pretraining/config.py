from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class Config:
    sampling_rate: int = 1000
    dataset_size: int = 200000
    windows_per_file_val: int = 100

    # IBI windowing
    window_choices: Tuple[int, ...] = (10, 30, 60)  # multi-resolution (seconds)
    window_sec: int = 30                             # used for val only
    min_ibi_ms: int = 300
    max_ibi_ms: int = 2000
    max_len_beats: int = 200                         # 60s @ 150 BPM ~ 150 beats

    # Node features
    ibi_feature_dim: int = 10
    n_hrv_features: int = 14

    # Model
    graph_dim: int = 128
    gnn_layers: int = 3
    transformer_layers: int = 2
    nhead: int = 4
    knn_k: int = 6
    proj_dim: int = 128

    # Masking (curriculum: starts low, ramps up each epoch)
    mask_ratio_start: float = 0.15
    mask_ratio_end: float = 0.50
    span_mean_len: int = 5                           # mean span length for span masking

    # Loss weights
    lambda_byol: float = 0.5
    lambda_hrv: float = 0.1
    lambda_future: float = 0.1

    # BYOL EMA
    ema_momentum_base: float = 0.996
    ema_momentum_final: float = 0.9999

    # Training
    batch_size: int = 128
    epochs: int = 100
    lr: float = 1e-3
    min_lr: float = 1e-5
    warmup_epochs: int = 5
    weight_decay: float = 1e-4
    num_workers: int = 4
    grad_clip: float = 1.0

    run_name: str = "ibi_graph_ssl_v2"
    seed: int = 42

    # Precomputed IBI CSVs (output of run_precompute_ibi.sh)
    train_csv: str = "/work/hdd/bebr/Projects/ecg_foundational_model/ibi_precomputed/IBI_train_files.csv"
    test_csv: str = "/work/hdd/bebr/Projects/ecg_foundational_model/ibi_precomputed/IBI_val_files.csv"
    output_dir: str = "/work/nvme/bebr/mkhan14/ecg_foundation_model/ibi_graph_model/experiments/ibi_graph_v2"

    @property
    def d_model(self):
        return self.graph_dim
