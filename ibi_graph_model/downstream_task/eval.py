import json
import os
import random
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch.utils.data import DataLoader

from config_supervised import Config
from dataset_sleep import IBIGraphCollator, IBIGraphDataset
from model_supervised import IBIGraphClassifier
from train_supervised import SupervisedLoss, run_one_epoch


def compute_class_weights(dataset, num_classes):
    labels = [dataset.label2idx[label] for _, _, label in dataset.samples]
    counts = Counter(labels)
    print("Class counts:", counts)
    total = sum(counts.values())
    weights = [total / (num_classes * counts.get(i, 1)) for i in range(num_classes)]
    weights = torch.tensor(weights, dtype=torch.float32)
    print("Class weights:", weights)
    return weights


def plot_confusion_matrix(cm, label_names, save_path):
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_names,
        yticklabels=label_names,
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

    device = torch.device(
        "cuda" if torch.cuda.is_available() and cfg.device.startswith("cuda") else "cpu"
    )

    # -----------------------------
    # Dataset
    # -----------------------------
    test_dataset = IBIGraphDataset(cfg.val_csv, cfg=cfg)

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=IBIGraphCollator(),
    )

    # -----------------------------
    # Model
    # -----------------------------
    model = IBIGraphClassifier(cfg).to(device)
    ckpt = torch.load(cfg.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # -----------------------------
    # Loss (class weights from test set since no train set here)
    # -----------------------------
    class_weights = None
    if cfg.use_class_weights:
        class_weights = compute_class_weights(test_dataset, cfg.num_classes).to(device)

    criterion = SupervisedLoss(
        class_weights=class_weights,
        label_smoothing=0.0,  # no smoothing at eval time
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # -----------------------------
    # Evaluate
    # -----------------------------
    test_metrics = run_one_epoch(
        model, test_loader, optimizer, criterion, device,
        train=False, num_classes=cfg.num_classes,
    )

    print("\n===== TEST RESULTS =====")
    print(f"Loss     : {test_metrics['loss']:.4f}")
    print(f"Acc      : {test_metrics['acc']:.4f}")
    print(f"Bal Acc  : {test_metrics['bal_acc']:.4f}")
    print(f"F1 macro : {test_metrics['f1_macro']:.4f}")
    print(f"Kappa    : {test_metrics['kappa']:.4f}")
    if "auroc" in test_metrics:
        print(f"AUROC    : {test_metrics['auroc']:.4f}")

    # -----------------------------
    # Save metrics
    # -----------------------------
    save_dict = {
        "loss":      float(test_metrics["loss"]),
        "accuracy":  float(test_metrics["acc"]),
        "bal_acc":   float(test_metrics["bal_acc"]),
        "f1_macro":  float(test_metrics["f1_macro"]),
        "kappa":     float(test_metrics["kappa"]),
    }
    if "auroc" in test_metrics:
        save_dict["auroc"] = float(test_metrics["auroc"])

    with open(os.path.join(cfg.output_dir_eval, "metrics.json"), "w") as f:
        json.dump(save_dict, f, indent=4)

    # -----------------------------
    # Confusion Matrix
    # -----------------------------
    if "cm" in test_metrics:
        cm = np.array(test_metrics["cm"])
        label_names = [test_dataset.idx2label[i] for i in range(cfg.num_classes)]

        np.save(os.path.join(cfg.output_dir_eval, "confusion_matrix.npy"), cm)

        with open(os.path.join(cfg.output_dir_eval, "confusion_matrix.txt"), "w") as f:
            f.write(f"Labels: {label_names}\n")
            f.write(str(cm))

        plot_confusion_matrix(
            cm,
            label_names,
            os.path.join(cfg.output_dir_eval, "confusion_matrix.png"),
        )

    print(f"\nSaved results to: {cfg.output_dir_eval}")


if __name__ == "__main__":
    main()
