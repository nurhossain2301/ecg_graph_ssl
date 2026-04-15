import os
import re
import json
import math
import argparse
import numpy as np
import wandb
import torch
import torch.nn as nn
import random
from torch.utils.data import Dataset, DataLoader

from utils import load_encoder_from_ecghubert
from BRP_dataset import BRPSleepDataset
from classifier import BRPClassifier
from train import run_one_epoch
from pretrain_model import ECGHuBERTModel  # your local file

# -----------------------------
# Main
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--train_csv", type=str, required=True)
    p.add_argument("--test_csv", type=str, required=True)

    p.add_argument("--encoder_ckpt", type=str, required=True, help="SSL pretrained ECGHuBERT checkpoint")
    p.add_argument("--freeze_encoder", type=int, default=1, help="1=freeze encoder, 0=fine-tune encoder")

    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--window_sec", type=int, default=10)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--run_name", type=str, default="sft_brp_v1.0")

    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def set_seed(seed: int):
    random_state = int(seed)
    random.seed(random_state)
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    torch.cuda.manual_seed_all(random_state)


def main():
    args = parse_args()
    set_seed(args.seed)

    # -----------------------------
    # Initialize W&B
    # -----------------------------
    wandb.init(
        project="ecg-hubert-brp-sft",
        name=args.run_name,
        # name=f"sft_{os.path.basename(args.output_dir)}",
        config=vars(args),
    )


    os.makedirs(args.output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Records + split
    # records = read_records_list(args.slpdb_dir)
    # train_recs, test_recs = downstream_split(records)

    # Dataset
    train_dataset = BRPSleepDataset(args.train_csv, window_sec=args.window_sec, sample_rate=1000)
    test_dataset  = BRPSleepDataset(args.test_csv, window_sec=args.window_sec, sample_rate=1000)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    val_loader = test_loader  

    # Encoder + classifier
    pretrained_model = ECGHuBERTModel(num_clusters=100).to(device)
    encoder, embed_dim = load_encoder_from_ecghubert(pretrained_model, args.encoder_ckpt, device)
    model = BRPClassifier(
        encoder=encoder,
        num_classes=4,
        hidden_dim=args.hidden_dim,
        freeze_encoder=bool(args.freeze_encoder),
    ).to(device)

    # Optimizer: if encoder frozen, optimizer sees only classifier params
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    # Train loop with best-val checkpointing
    best_val_f1 = -1.0
    best_path = os.path.join(args.output_dir, "best.pt")
    history = []

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_metrics = run_one_epoch(model, train_loader, optimizer, device, train=True, grad_clip=args.grad_clip)
        va_loss, va_metrics = run_one_epoch(model, val_loader, optimizer, device, train=False, grad_clip=args.grad_clip)

        # -----------------------------
        # Log to W&B
        # -----------------------------
        wandb.log({
            "epoch": epoch,
            "train/loss": tr_loss,
            "train/accuracy": tr_metrics["acc"],
            "train/macro_f1": tr_metrics["macro_f1"],
            "train/kappa": tr_metrics["kappa"],
            "val/loss": va_loss,
            "val/accuracy": va_metrics["acc"],
            "val/macro_f1": va_metrics["macro_f1"],
            "val/kappa": va_metrics["kappa"],
        })
        

        print(
            f"[{epoch:03d}/{args.epochs}] "
            f"train loss={tr_loss:.4f} acc={tr_metrics['acc']:.4f} f1={tr_metrics['macro_f1']:.4f} | "
            f"val loss={va_loss:.4f} acc={va_metrics['acc']:.4f} f1={va_metrics['macro_f1']:.4f}"
        )

        # Save best by val macro-F1
        if va_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = va_metrics["macro_f1"]
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "embed_dim": embed_dim,
                    "args": vars(args),
                    "val_metrics": va_metrics,
                },
                best_path,
            )
            wandb.run.summary["best_val_macro_f1"] = best_val_f1
            print(f"  ✅ Saved best checkpoint to {best_path} (val macro-F1={best_val_f1:.4f})")

        

    # -----------------------------
    # Final Test Evaluation
    # -----------------------------
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    te_loss, te_metrics = run_one_epoch(model, test_loader, optimizer, device, train=False)

    print("\n===== TEST RESULTS =====")
    print(f"test loss: {te_loss:.4f}")
    print(f"test acc : {te_metrics['acc']:.4f}")
    print(f"test f1  : {te_metrics['macro_f1']:.4f}")
    print(f"test kappa: {te_metrics['kappa']:.4f}")

    # Log test metrics
    wandb.log({
        "test/loss": te_loss,
        "test/accuracy": te_metrics["acc"],
        "test/macro_f1": te_metrics["macro_f1"],
        "test/kappa": te_metrics["kappa"],
    })

    # Log confusion matrix
    if "confusion_matrix" in te_metrics:
        wandb.log({
            "test/confusion_matrix": wandb.plot.confusion_matrix(
                probs=None,
                y_true=np.array(te_metrics["confusion_matrix"]).argmax(axis=1),
                preds=np.array(te_metrics["confusion_matrix"]).argmax(axis=0),
            )
        })

    wandb.finish()

    print(f"\nDone. Outputs in: {args.output_dir}")


if __name__ == "__main__":
    main()
