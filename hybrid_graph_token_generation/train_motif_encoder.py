import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd 
from config import Config
from dataset_codebook import ECGCodebookDataset, codebook_collate_fn
from motif_encoder import MotifAutoencoder


def train_encoder(cfg):

    device = "cuda"
    train_files = pd.read_csv(cfg.train_csv)
    train_files = train_files["filename"].tolist()

    val_files = pd.read_csv(cfg.test_csv)
    val_files = val_files["filename"].tolist()

    dataset = ECGCodebookDataset(
        file_list=train_files,
        sampling_rate=cfg.sampling_rate,
        window_sec=cfg.window_sec,
        dataset_size=10,
        mode="train",
        cfg=cfg
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        collate_fn=codebook_collate_fn
    )

    model = MotifAutoencoder(
        input_len=cfg.token_window,
        emb_dim=cfg.emb_dim
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1):
        model.train()

        total_loss = 0

        for batch in tqdm(loader):
            batch = batch.to(device)

            z, recon = model(batch)

            loss = F.mse_loss(recon, batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch} Loss: {total_loss:.4f}")

    torch.save(model.encoder.state_dict(), "motif_encoder.pt")
    print("✅ Saved encoder")


if __name__ == "__main__":
    cfg = Config()
    train_encoder(cfg)