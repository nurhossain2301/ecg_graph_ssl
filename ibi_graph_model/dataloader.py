from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
import pandas as pd
from dataset import ECGGraphDataset, GraphSSL_Collator


def load_data(cfg):
    train_files = pd.read_csv(cfg.train_csv)["filename"].tolist()
    val_files   = pd.read_csv(cfg.test_csv)["filename"].tolist()

    train_dataset = ECGGraphDataset(
        file_list=train_files,
        dataset_size=cfg.dataset_size,
        mode="train",
        cfg=cfg,
    )
    val_dataset = ECGGraphDataset(
        file_list=val_files,
        mode="val",
        windows_per_file_val=cfg.windows_per_file_val,
        cfg=cfg,
    )

    if dist.is_available() and dist.is_initialized():
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler   = DistributedSampler(val_dataset,   shuffle=False)
    else:
        train_sampler = None
        val_sampler   = None

    collator = GraphSSL_Collator(node_mask_ratio=cfg.node_mask_ratio)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        collate_fn=collator,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        sampler=val_sampler,
        shuffle=False,
        collate_fn=collator,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader, train_sampler
