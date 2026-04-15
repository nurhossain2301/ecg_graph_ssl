import os
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns


from BRP_dataset_caregiver import BRPGraphDataset
from classifier import GraphClassifier, load_pretrained_encoder
from train import run_one_epoch
from config import Config
from train import run_one_epoch
from supervised_model import SupervisedECGGraph


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
    p.add_argument("--num_classes", type=int, default=4)
    p.add_argument("--window_sec", type=int, default=10)

    return p.parse_args()


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


# -----------------------------
# Main
# -----------------------------
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    cfg = Config()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -----------------------------
    # Load checkpoint
    # -----------------------------
    ckpt = torch.load(args.ckpt, map_location=device)
    train_args = ckpt["args"]

    # -----------------------------
    # Dataset
    # -----------------------------
    test_dataset  = BRPGraphDataset(args.test_csv, window_sec=args.window_sec, sample_rate=1000, cfg=cfg)

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
    # Load pretrained encoder
    # Load pretrained encoder
    # pretrained_ckpt = "/work/nvme/bebr/mkhan14/ecg_foundation_model/graph_modelling/best_model.pt"
    # encoder = load_pretrained_encoder(pretrained_ckpt, cfg=cfg, device=device)

    # # Build classifier
    # model = GraphClassifier(
    #     encoder=encoder,
    #     num_classes=args.num_classes,
    # ).to(device)
    model = SupervisedECGGraph(num_classes=args.num_classes).to(device)
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
        train=False,
        args=args
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