from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
import pandas as pd
from dataset import ECGGraphBYOLDataset, GraphBYOLCollator
from config import Config


def load_data(cfg):
    train_files = pd.read_csv(cfg.train_csv)["filename"].tolist()
    val_files   = pd.read_csv(cfg.test_csv)["filename"].tolist()

    if cfg.overfit:
        # Pin training to a tiny fixed file subset so the model can memorize it.
        # Val uses the same files so we can confirm train_loss → val_loss → 0.
        train_files = train_files[:cfg.overfit_files]
        val_files   = train_files

    dataset_size = cfg.overfit_samples if cfg.overfit else cfg.dataset_size

    train_dataset = ECGGraphBYOLDataset(
        file_list=train_files,
        sampling_rate=cfg.sampling_rate,
        window_sec=cfg.window_sec,
        dataset_size=dataset_size,
        mode="train",
        cfg=cfg
    )

    val_dataset = ECGGraphBYOLDataset(
        file_list=val_files,
        sampling_rate=cfg.sampling_rate,
        window_sec=cfg.window_sec,
        mode="val",
        windows_per_file_val=cfg.windows_per_file_val if not cfg.overfit else dataset_size,
        cfg=cfg
    )

    if dist.is_available() and dist.is_initialized():
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    collator = GraphBYOLCollator(node_mask_ratio=cfg.node_mask_ratio)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        collate_fn=collator,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        sampler=val_sampler,
        shuffle=False,
        collate_fn=collator,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False
    )
    

    return train_loader, val_loader, train_sampler

def test_dataloader(cfg):
    
    train_loader, val_loader, _ = load_data(cfg)
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
