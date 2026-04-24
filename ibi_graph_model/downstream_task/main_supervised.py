import json
import os
import random
from tqdm import tqdm
from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader

from config_supervised import Config
from dataset_caregiver import IBIGraphCollator, IBIGraphDataset
from model_supervised import IBIGraphClassifier
from train_supervised import EarlyStopper, SupervisedLoss, run_one_epoch, save_checkpoint


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def compute_class_weights(dataset, num_classes):
    labels = [dataset[i]["label"].item() for i in range(len(dataset))]
    counts = Counter(labels)

    print("Class counts:", counts)

    weights = []
    total = sum(counts.values())

    for i in range(num_classes):
        c = counts.get(i, 1)  # avoid division by zero
        weights.append(total / (num_classes * c))

    weights = torch.tensor(weights, dtype=torch.float32)
    print("Class weights:", weights)

    return weights


def main():
    cfg = Config()
    os.makedirs(cfg.output_dir, exist_ok=True)
    set_seed(cfg.seed)

    train_ds = IBIGraphDataset(cfg.train_csv, cfg=cfg)
    val_ds = IBIGraphDataset(cfg.val_csv, cfg=cfg, label2idx=train_ds.label2idx)

    print("Train size:", len(train_ds), "Val size:", len(val_ds))


    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=IBIGraphCollator(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=IBIGraphCollator(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() and cfg.device.startswith("cuda") else "cpu")
    model = IBIGraphClassifier(cfg).to(device)

    class_weights = None
    if cfg.use_class_weights:
        class_weights = compute_class_weights(train_ds, cfg.num_classes).to(device)
        print("Class weights:", class_weights)

    criterion = SupervisedLoss(
        class_weights=class_weights,
        label_smoothing=cfg.label_smoothing,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    stopper = EarlyStopper(patience=cfg.early_stop_patience, mode="max")

    best_score = -1.0

    for epoch in tqdm(range(cfg.epochs)):
        train_metrics = run_one_epoch(model, train_loader, optimizer, criterion, device, train=True, num_classes=cfg.num_classes)
        val_metrics = run_one_epoch(model, val_loader, optimizer, criterion, device, train=False, num_classes=cfg.num_classes)
        scheduler.step()

        score = val_metrics["f1_macro"]
        print(
            f"Epoch {epoch + 1:03d} | "
            f"train loss={train_metrics['loss']:.4f} acc={train_metrics['acc']:.4f} f1={train_metrics['f1_macro']:.4f} | "
            f"val loss={val_metrics['loss']:.4f} acc={val_metrics['acc']:.4f} bal_acc={val_metrics['bal_acc']:.4f} f1={val_metrics['f1_macro']:.4f}"
        )
        if cfg.num_classes == 2:
            print(f"val auroc={val_metrics.get('auroc', float('nan')):.4f}")

        metrics = {"train": train_metrics, "val": val_metrics, "epoch": epoch + 1}
        with open(os.path.join(cfg.output_dir, "last_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        save_checkpoint(
            os.path.join(cfg.output_dir, "last_model.pt"),
            model,
            optimizer,
            epoch + 1,
            metrics,
        )

        if score > best_score:
            best_score = score
            save_checkpoint(
                os.path.join(cfg.output_dir, "best_model.pt"),
                model,
                optimizer,
                epoch + 1,
                metrics,
            )
            with open(os.path.join(cfg.output_dir, "best_metrics.json"), "w") as f:
                json.dump(metrics, f, indent=2)
            print("Saved best model")

        if stopper.step(score):
            print("Early stopping triggered")
            break

    print("Best validation balanced accuracy:", best_score)
    print("Best confusion matrix:", metrics["val"]["cm"])


if __name__ == "__main__":
    main()
