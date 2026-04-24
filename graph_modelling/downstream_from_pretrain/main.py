import torch
import numpy as np
import os
from classifier import ECGClassifier, load_pretrained_byol
import argparse
import random
import wandb
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from BRP_dataset_sleep import BRPGraphDataset
from train import run_one_epoch
from config import Config

def set_seed(seed: int):
    random_state = int(seed)
    random.seed(random_state)
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    torch.cuda.manual_seed_all(random_state)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--train_csv", type=str, required=True)
    p.add_argument("--test_csv", type=str, required=True)

    p.add_argument("--encoder_ckpt", type=str, required=True, help="SSL pretrained ECGGraph checkpoint")
    p.add_argument("--freeze_encoder", type=int, default=1, help="1=freeze encoder, 0=fine-tune encoder")

    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--window_sec", type=int, default=10)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--run_name", type=str, default="sft_brp_graph_caregiver")
    p.add_argument("--num_classes", type=int, default=4)

    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()
    
def build_classifier_from_byol(
    ckpt_path: str,
    cfg,
    num_classes: int,
    hidden_dim: int = 256,
    dropout: float = 0.3,
    freeze_encoder: bool = False,
    device: str = "cpu",
) -> ECGClassifier:
    """
    1. Load BYOL checkpoint
    2. Copy online_encoder + pool weights into ECGClassifier
    3. Return ready-to-train classifier
    """
    # ── step 1: load BYOL ─────────────────────────────────
    byol = load_pretrained_byol(ckpt_path, cfg, device)

    # ── step 2: build classifier ──────────────────────────
    model = ECGClassifier(
        cfg,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        dropout=dropout,
        freeze_encoder=freeze_encoder,
    )

    # ── step 3: transfer weights ──────────────────────────
    # online_encoder
    model.encoder.load_state_dict(
        byol.online_encoder.state_dict(), strict=True
    )
    # attention pool
    model.pool.load_state_dict(
        byol.pool.state_dict(), strict=True
    )

    model = model.to(device)
    print(
        f"[build_classifier] Transferred online_encoder + pool weights.\n"
        f"  num_classes={num_classes}, hidden_dim={hidden_dim}, "
        f"freeze={freeze_encoder}"
    )
    return model

def main():
    args = parse_args()
    set_seed(args.seed)
    cfg = Config()

    # -----------------------------
    # Initialize W&B
    # -----------------------------
    wandb.init(
        project="ecg-graph-byol-brp-sft",
        name=args.run_name,
        # name=f"sft_{os.path.basename(args.output_dir)}",
        config=vars(args),
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Phase 1: linear probe (frozen encoder) ────────────
    model = build_classifier_from_byol(
        ckpt_path    = args.encoder_ckpt,
        cfg          = cfg,
        num_classes  = args.num_classes,
        hidden_dim   = 256, #make dynamic
        dropout      = 0.3,
        freeze_encoder = args.freeze_encoder,   # <── only head trains
        device       = device,
    )
    os.makedirs(args.output_dir, exist_ok=True)

    train_dataset = BRPGraphDataset(args.train_csv, window_sec=args.window_sec, sample_rate=1000, cfg=cfg)
    label2idx = train_dataset.label2idx
    test_dataset  = BRPGraphDataset(args.test_csv, window_sec=args.window_sec, sample_rate=1000, cfg=cfg, label2idx=label2idx)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    val_loader = test_loader  


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

    class_weights = compute_class_weights(train_dataset, args.num_classes).to(device)
    class_weights_test = compute_class_weights(test_dataset, args.num_classes).to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    # Train loop with best-val checkpointing
    best_val_f1 = -1.0
    best_path = os.path.join(args.output_dir, "best.pt")
    history = []

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_metrics = run_one_epoch(model, train_loader, optimizer, device, train=True, grad_clip=args.grad_clip, args=args, class_weights=class_weights)
        va_loss, va_metrics = run_one_epoch(model, val_loader, optimizer, device, train=False, grad_clip=args.grad_clip, args=args, class_weights=class_weights)

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
        # Save best by val macro-F1
        if va_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = va_metrics["macro_f1"]
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "args": vars(args),
                    "val_metrics": va_metrics,
                    "label2idx": label2idx
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

    te_loss, te_metrics = run_one_epoch(model, test_loader, optimizer, device, train=False, args=args)

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
# if __name__ == "__main__":

    # optimizer = torch.optim.Adam(
    #     filter(lambda p: p.requires_grad, model.parameters()),
    #     lr=1e-3,
    # )
    # criterion = nn.CrossEntropyLoss()

    # # (replace with your real DataLoader)
    # # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    # # val_loader   = DataLoader(val_dataset,   batch_size=32)

    # print("\n── Phase 1: Linear Probe ──")
    # for epoch in range(10):
    #     # tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
    #     # va_loss, va_acc = evaluate(model, val_loader, criterion, device)
    #     # print(f"Epoch {epoch+1:02d} | train {tr_loss:.4f}/{tr_acc:.3f} | val {va_loss:.4f}/{va_acc:.3f}")
    #     pass

    # # ── Phase 2: full fine-tune (unfreeze backbone) ───────
    # print("\n── Phase 2: Fine-Tuning ──")
    # model.unfreeze_encoder()

    # optimizer = torch.optim.Adam([
    #     {"params": model.encoder.parameters(),    "lr": 1e-4},  # small lr
    #     {"params": model.pool.parameters(),       "lr": 1e-4},
    #     {"params": model.classifier.parameters(), "lr": 5e-4},  # larger lr
    # ])

    # for epoch in range(20):
    #     # tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
    #     # va_loss, va_acc = evaluate(model, val_loader, criterion, device)
    #     # print(f"Epoch {epoch+1:02d} | train {tr_loss:.4f}/{tr_acc:.3f} | val {va_loss:.4f}/{va_acc:.3f}")
    #     pass

    # # ── save fine-tuned model ─────────────────────────────
    # torch.save(
    #     {
    #         "model_state_dict": model.state_dict(),
    #         "cfg": cfg,
    #         "num_classes": 5,
    #     },
    #     "ecg_classifier.pth",
    # )
    # print("Saved ecg_classifier.pth")