from dataclasses import dataclass
import torch

@dataclass
class Config:
    sampling_rate: int = 1000
    window_sec: int = 30
    dataset_size: int = 200000
    windows_per_file_val: int = 100

    # ---- IBI settings ----
    min_ibi_ms: int = 300
    max_ibi_ms: int = 2000
    max_len_beats: int = 128
    ibi_feature_dim: int = 4   # [ibi_norm, dibi_norm, local_var, quality]

    # ---- model ----
    graph_dim: int = 128
    gnn_layers: int = 3
    transformer_layers: int = 2
    nhead: int = 4
    knn_k: int = 6

    # ---- masking ----
    node_mask_ratio: float = 0.3
    edge_mask_ratio: float = 0.2

    # ---- train ----
    batch_size: int = 128
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 4
    grad_clip: float = 1.0

    run_name: str = "ibi_graph_ssl_v1"
    seed: int = 42

    train_csv: str = "/work/hdd/bebr/Projects/ecg_foundational_model/IBI_train_files.csv"
    test_csv: str = "/work/hdd/bebr/Projects/ecg_foundational_model/IBI_val_files.csv"
    output_dir: str = "/work/nvme/bebr/mkhan14/ecg_foundation_model/graph_modelling/experiments"

    @property
    def d_model(self):
        return self.graph_dim