import os
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns

from utils import load_encoder_from_ecghubert
from BRP_dataset import BRPSleepDataset
from classifier import BRPClassifier
from pretrain_model import ECGHuBERTModel
from train import run_one_epoch


# -----------------------------
# Args
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True, help="path to best.pt")
    p.add_argument("--test_csv", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)

    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)

    return p.parse_args()


# -----------------------------
# Confusion Matrix Plot
# -----------------------------
def plot_confusion_matrix(cm, save_path):
    labels = ["Wake", "Sleep"]

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
    plt.title("Confusion Matrix (Sleep vs Wake)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# -----------------------------
# Main
# -----------------------------
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -----------------------------
    # Load checkpoint
    # -----------------------------
    ckpt = torch.load(args.ckpt, map_location=device)
    train_args = ckpt["args"]

    # -----------------------------
    # Dataset
    # -----------------------------
    test_dataset = BRPSleepDataset(
        args.test_csv,
        window_sec=train_args["window_sec"],
        sample_rate=1000
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    # labels = []
    # for batch in test_loader:
    #     x, y = batch
    #     labels.extend(y.numpy())

    # print("UNIQUE LABELS:", np.unique(labels))
    # -----------------------------
    # Model
    # -----------------------------
    pretrained_model = ECGHuBERTModel(num_clusters=100).to(device)
    encoder, embed_dim = load_encoder_from_ecghubert(
        pretrained_model,
        train_args["encoder_ckpt"],
        device
    )

    model = BRPClassifier(
        encoder=encoder,
        num_classes=2,
        hidden_dim=train_args["hidden_dim"],
        freeze_encoder=bool(train_args["freeze_encoder"]),
    ).to(device)

    model.load_state_dict(ckpt["model"])
    model.eval()

    # Dummy optimizer (not used but required by run_one_epoch)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # -----------------------------
    # Evaluation
    # -----------------------------
    loss, metrics = run_one_epoch(
        model,
        test_loader,
        optimizer,
        device,
        train=False
    )

    print("\n===== TEST RESULTS =====")
    print(f"Loss  : {loss:.4f}")
    print(f"Acc   : {metrics['acc']:.4f}")
    print(f"F1    : {metrics['macro_f1']:.4f}")
    print(f"Kappa : {metrics['kappa']:.4f}")

    # -----------------------------
    # Save metrics
    # -----------------------------
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({
            "loss": float(loss),
            "accuracy": float(metrics["acc"]),
            "macro_f1": float(metrics["macro_f1"]),
            "kappa": float(metrics["kappa"]),
        }, f, indent=4)

    # -----------------------------
    # Confusion Matrix
    # -----------------------------
    if "confusion_matrix" in metrics:
        cm = np.array(metrics["confusion_matrix"])

        # Save raw matrix
        np.save(os.path.join(args.output_dir, "confusion_matrix.npy"), cm)

        # Save readable txt
        with open(os.path.join(args.output_dir, "confusion_matrix.txt"), "w") as f:
            f.write("Confusion Matrix (Sleep=0, Wake=1)\n")
            f.write(str(cm))

        # Plot
        plot_confusion_matrix(
            cm,
            os.path.join(args.output_dir, "confusion_matrix.png")
        )

    print(f"\nSaved results to: {args.output_dir}")


if __name__ == "__main__":
    main()