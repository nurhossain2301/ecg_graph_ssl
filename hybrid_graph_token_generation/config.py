import os


class Config:

    def __init__(self):

        # -------------------------
        # PATHS
        # -------------------------
        self.test_csv: str = "/work/hdd/bebr/Projects/ecg_foundational_model/ECG_val_files.csv"
        self.train_csv: str = "/work/hdd/bebr/Projects/ecg_foundational_model/ECG_train_files.csv"

        self.output_dir = "./outputs"
        self.codebook_path = "./codebook.npy"

        self.run_name = "ecg_token_graph_v2"

        # -------------------------
        # DATA
        # -------------------------
        self.sampling_rate = 1000
        self.window_sec = 30

        # TOKENIZATION
        self.token_window_ms = 200
        self.token_stride_ms = 100

        # derived
        self.token_window = int(self.sampling_rate * self.token_window_ms / 1000)

        # DATA SIZE
        self.dataset_size = 200000
        self.windows_per_file_val = 100

        # -------------------------
        # MODEL
        # -------------------------
        self.emb_dim = 128
        self.model_dim = 256

        self.n_clusters = 512  # vocab size

        self.num_heads = 8
        self.num_seq_layers = 4
        self.num_graph_layers = 3

        self.dropout = 0.1
        self.max_len = 2048

        self.fusion_alpha = 0.5

        # -------------------------
        # Loss
        # -------------------------
        self.lambda_smooth = 0.05
        self.lambda_align = 0.05

        # GRAPH
        self.knn_k = 3

        # -------------------------
        # TRAINING
        # -------------------------
        self.batch_size = 8
        self.num_workers = 8

        self.epochs = 50

        self.lr = 1e-4
        self.weight_decay = 1e-2

        self.node_mask_ratio = 0.3
        self.log_interval = 50

        # -------------------------
        # SYSTEM
        # -------------------------
        self.seed = 42

        # # -------------------------
        # # SANITY CHECK
        # # -------------------------
        # assert os.path.exists(self.codebook_path), \
        #     f"Codebook not found at {self.codebook_path}"