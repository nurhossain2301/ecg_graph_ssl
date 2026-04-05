from torch.utils.data import DataLoader, DistributedSampler
import pandas as pd
from dataset import ECGTokenDataset, GraphSSL_Collator
from config import Config


def load_data(cfg):
    train_files = pd.read_csv(cfg.train_csv)
    train_files = train_files["filename"].tolist()

    val_files = pd.read_csv(cfg.test_csv)
    val_files = val_files["filename"].tolist()
    

    train_dataset = ECGTokenDataset(
        file_list=train_files,
        sampling_rate=cfg.sampling_rate,
        window_sec=cfg.window_sec,
        dataset_size=cfg.dataset_size,
        mode="train",
        cfg = cfg
    )
    

    val_dataset = ECGTokenDataset(
        file_list=val_files,
        sampling_rate=cfg.sampling_rate,
        window_sec=cfg.window_sec,
        mode="val",
        windows_per_file_val=cfg.windows_per_file_val,
        cfg=cfg
    )

    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)

    collator = GraphSSL_Collator(node_mask_ratio=cfg.node_mask_ratio)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        # shuffle=True,
        sampler=train_sampler,
        collate_fn=collator,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        # shuffle=False,
        sampler=val_sampler,
        collate_fn=collator,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False
    )
    

    return train_loader, val_loader, train_sampler

def test_dataloader(cfg):
    
    train_loader, val_loader = load_data(cfg)
    for i, batch in enumerate(train_loader):
        print(f"\nBatch {i+1}")
        print("Input shape:", batch["beats"].shape)
        print("Mask shape:", batch["node_mask"].shape)
        # print("Mask ratio:",
        #       batch["mask_indices"][0].float().mean().item())

        if i + 1 >= 4:
            break

if __name__=="__main__":
    cfg = Config()
    test_dataloader(cfg)