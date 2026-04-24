import json
import os
import random
from tqdm import tqdm
from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader

from config_supervised import Config
from dataset_sleep import IBIGraphCollator, IBIGraphDataset
from model_supervised import IBIGraphClassifier
from train_supervised import EarlyStopper, SupervisedLoss, run_one_epoch, save_checkpoint




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

# -----------------------------
# Confusion Matrix Plot
# -----------------------------
def plot_confusion_matrix(cm, save_path):
    # labels = ["Active", "Crying", "Quiet", "Sleep"]
    labels = ["caregiver", "infant"]

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def main():
    cfg = Config()
    os.makedirs(cfg.output_dir_eval, exist_ok=True)

     # -----------------------------
    # Load checkpoint
    # -----------------------------
    ckpt = torch.load(cfg.ckpt, map_location=device)


    # -----------------------------
    # Dataset
    # -----------------------------
    test_dataset  = IBIGraphDataset(cfg.val_csv, cfg=cfg)

    test_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=IBIGraphCollator(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() and cfg.device.startswith("cuda") else "cpu")
    model = IBIGraphClassifier(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    class_weights = None
    if cfg.use_class_weights:
        class_weights = compute_class_weights(train_ds, cfg.num_classes).to(device)
        print("Class weights:", class_weights)

    criterion = SupervisedLoss(
        class_weights=class_weights,
        label_smoothing=cfg.label_smoothing,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    

    test_metrics = run_one_epoch(model, val_loader, optimizer, criterion, device, train=False, num_classes=cfg.num_classes)
    
    print("\n===== TEST RESULTS =====")
    print(f"Loss  : {loss:.4f}")
    print(f"Acc   : {metrics['acc']:.4f}")
    print(f"F1    : {metrics['macro_f1']:.4f}")
    print(f"Kappa : {metrics['kappa']:.4f}")

    # -----------------------------
    # Save metrics
    # -----------------------------
    metrics_path = os.path.join(cfg.output_dir_eval, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({
            "loss": float(test_metrics['loss']),
            "accuracy": float(test_metrics["acc"]),
            "macro_f1": float(test_metrics["macro_f1"]),
            "kappa": float(test_metrics["kappa"]),
        }, f, indent=4)

    # -----------------------------
    # Confusion Matrix
    # -----------------------------
    if "confusion_matrix" in test_metrics:
        cm = np.array(test_metrics["confusion_matrix"])

        # Save raw matrix
        np.save(os.path.join(cfg.output_dir_eval, "confusion_matrix.npy"), cm)

        # Save readable txt
        with open(os.path.join(cfg.output_dir_eval, "confusion_matrix.txt"), "w") as f:
            f.write("Confusion Matrix (Sleep=0, Wake=1)\n")
            f.write(str(cm))

        # Plot
        plot_confusion_matrix(
            cm,
            os.path.join(cfg.output_dir_eval, "confusion_matrix.png")
        )

    print(f"\nSaved results to: {cfg.output_dir_eval}")

        


if __name__ == "__main__":
    main()
