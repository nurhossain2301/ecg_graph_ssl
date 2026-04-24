from dataclasses import dataclass


@dataclass
class Config:
    # -----------------------------
    # data
    # -----------------------------
    train_csv: str = "/work/nvme/bebr/mkhan14/ecg_foundation_model/BRP_train.csv"
    val_csv: str = "/work/nvme/bebr/mkhan14/ecg_foundation_model/BRP_test.csv"
    output_dir: str = "ibi_graph_supervised_sleep"
    output_dir_eval: str = "ibi_graph_supervised_sleep_eval"
    ckpt: str = "/work/nvme/bebr/mkhan14/ecg_foundation_model/ibi_graph_model/downstream_task/ibi_graph_supervised_status"


    task_type: str = "auto"    # auto | binary | multiclass
    num_classes: int = 2       # used when labels are already numeric and task_type != auto

    # -----------------------------
    # IBI preprocessing
    # -----------------------------
    min_ibi_ms: int = 350
    max_ibi_ms: int = 650
    max_len_beats: int = 80
    ibi_feature_dim: int = 4

    # -----------------------------
    # model
    # -----------------------------
    graph_dim: int = 128
    node_mlp_hidden: int = 128
    gnn_layers: int = 4
    transformer_layers: int = 2
    nhead: int = 4
    knn_k: int = 8
    dropout: float = 0.2
    use_edge_gate: bool = True
    use_virtual_node: bool = True
    max_beats: int = 80
    pool_heads: int = 4

    # -----------------------------
    # training
    # -----------------------------
    batch_size: int = 128
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 4
    grad_clip: float = 1.0
    label_smoothing: float = 0.05
    use_class_weights: bool = True
    early_stop_patience: int = 10

    # -----------------------------
    # runtime
    # -----------------------------
    seed: int = 42
    run_name: str = "ibi_graph_supervised_v1_sleep"
    device: str = "cuda"

    @property
    def d_model(self):
        return self.graph_dim
