import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

from config import Config
from motif_encoder import MotifEncoder
from dataset_codebook import ECGCodebookDataset, codebook_collate_fn


# -------------------------------------------------------
# FEATURE EXTRACTOR (VERY SIMPLE VERSION FIRST)
# -------------------------------------------------------

def extract_features(x):
    """
    x: [B, win]

    return: [B, D]
    """
    # 🔥 Start simple: raw → mean/std normalization
    x = (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True) + 1e-6)

    return x  # [B, win] → treat win as feature dim


# -------------------------------------------------------
# BUILD CODEBOOK
# -------------------------------------------------------
def build_codebook_pipeline(cfg):

    print("🔵 Building codebook...")
    train_files = pd.read_csv(cfg.train_csv)
    train_files = train_files["filename"].tolist()

    val_files = pd.read_csv(cfg.test_csv)
    val_files = val_files["filename"].tolist()

    dataset = ECGCodebookDataset(file_list=train_files,
        sampling_rate=cfg.sampling_rate,
        window_sec=cfg.window_sec,
        dataset_size=cfg.dataset_size,
        mode="train",
        cfg = cfg)
    device = "cuda"

    encoder = MotifEncoder(
        input_len=cfg.token_window,
        emb_dim=cfg.emb_dim
    ).to(device)

    encoder.load_state_dict(torch.load("motif_encoder.pt"))
    encoder.eval()

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=codebook_collate_fn,
        drop_last=True
    )

    kmeans = MiniBatchKMeans(
        n_clusters=cfg.n_clusters,
        batch_size=4096,
        verbose=1
    )

    for i, batch in enumerate(tqdm(loader)):

        # batch: [M, win]
        batch = batch.float()

        # features = extract_features(batch)  # [M, win]
        with torch.no_grad():
            batch = batch.to(device)
            features = encoder(batch).cpu().numpy()

        kmeans.partial_fit(features)

        if i % 100 == 0:
            print(f"Processed batch {i}")

    codebook = kmeans.cluster_centers_

    print("✅ Codebook shape:", codebook.shape)

    np.save(cfg.codebook_path, codebook)
    print(f"💾 Saved codebook to {cfg.codebook_path}")


# -------------------------------------------------------
# MAIN
# -------------------------------------------------------
def main():

    cfg = Config()

    build_codebook_pipeline(cfg)


if __name__ == "__main__":
    print("here")
    main()