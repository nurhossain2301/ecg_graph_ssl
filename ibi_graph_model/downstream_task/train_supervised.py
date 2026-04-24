import os
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from tqdm import tqdm


class EarlyStopper:
    def __init__(self, patience=10, mode="max"):
        self.patience = patience
        self.mode = mode
        self.best = None
        self.bad_epochs = 0

    def step(self, value):
        if self.best is None:
            self.best = value
            return False

        improved = value > self.best if self.mode == "max" else value < self.best
        if improved:
            self.best = value
            self.bad_epochs = 0
            return False

        self.bad_epochs += 1
        return self.bad_epochs >= self.patience


class SupervisedLoss(torch.nn.Module):
    def __init__(self, class_weights=None, label_smoothing=0.0):
        super().__init__()
        self.register_buffer("class_weights", class_weights if class_weights is not None else None)
        self.label_smoothing = label_smoothing

    def forward(self, logits, target):
        return F.cross_entropy(
            logits,
            target,
            weight=self.class_weights,
            label_smoothing=self.label_smoothing,
        )


@torch.no_grad()
def compute_metrics(logits, labels, num_classes: int) -> Dict[str, float]:
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    pred = probs.argmax(axis=1)
    y = labels.cpu().numpy()

    metrics = {
        "acc": float(accuracy_score(y, pred)),
        "bal_acc": float(balanced_accuracy_score(y, pred)),
        "f1_macro": float(f1_score(y, pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y, pred, average="weighted", zero_division=0)),
    }

    if num_classes == 2:
        try:
            metrics["auroc"] = float(roc_auc_score(y, probs[:, 1]))
        except Exception:
            metrics["auroc"] = float("nan")

    metrics["cm"] = confusion_matrix(y, pred, labels=list(range(num_classes))).tolist()
    return metrics


def run_one_epoch(model, loader, optimizer, criterion, device, train: bool, num_classes: int):
    model.train() if train else model.eval()
    total_loss = 0.0
    all_logits = []
    all_labels = []

    pbar = tqdm(loader)
    for batch in pbar:
        beats = batch["beats"].to(device)
        rr = batch["rr"].to(device)
        valid_mask = batch["valid_mask"].to(device)
        labels = batch["label"].to(device)

        with torch.set_grad_enabled(train):
            out = model(beats, rr, valid_mask)
            logits = out["logits"]
            loss = criterion(logits, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        total_loss += loss.item()
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())
        pbar.set_description(("train" if train else "val") + f" loss: {loss.item():.4f}")

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    metrics = compute_metrics(all_logits, all_labels, num_classes)
    metrics["loss"] = total_loss / max(len(loader), 1)
    return metrics


def save_checkpoint(path, model, optimizer, epoch, metrics):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "metrics": metrics,
    }, path)
