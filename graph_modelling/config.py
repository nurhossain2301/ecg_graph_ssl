from dataclasses import dataclass
import torch

@dataclass
class Config:
    sampling_rate: int = 1000
    window_sec: int = 30
    dataset_size: int = 200000
    windows_per_file_val: int = 100

    pre_r_ms: int = 250
    post_r_ms: int = 400
    min_rr_ms: int = 250

    max_beats_per_segment: int = 128

    beat_embed_dim: int = 128
    graph_dim: int = 128
    gnn_layers: int = 3

    knn_k: int = 6
    node_mask_ratio: float = 0.3
    edge_mask_ratio: float = 0.2

    batch_size: int = 128
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-4


    num_workers: int = 4
    grad_clip: float = 1.0

    run_name: str = "graph_bcp_v1.0"
    seed: int = 42

    test_csv: str = "/work/hdd/bebr/Projects/ecg_foundational_model/ECG_val_files.csv"
    train_csv: str = "/work/hdd/bebr/Projects/ecg_foundational_model/ECG_train_files.csv"
    output_dir: str = "/work/nvme/bebr/mkhan14/ecg_foundation_model/graph_modelling/experiments"


    # device: str = "cuda" if torch.cuda.is_available() else "cpu"