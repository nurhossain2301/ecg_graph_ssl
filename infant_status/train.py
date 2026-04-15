import torch
import torch.nn as nn
from utils import compute_metrics
# -----------------------------
# Train / Eval loops
# -----------------------------
def run_one_epoch(model, loader, optimizer, device, train=True, grad_clip=1.0):
    if train:
        model.train()
    else:
        model.eval()

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_n = 0
    all_true, all_pred = [], []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            logits = model(x)
            loss = criterion(logits, y)

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip is not None and grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

        total_loss += float(loss.item()) * x.size(0)
        total_n += x.size(0)

        preds = torch.argmax(logits, dim=-1)
        all_true.extend(y.detach().cpu().numpy().tolist())
        all_pred.extend(preds.detach().cpu().numpy().tolist())

    metrics = compute_metrics(all_true, all_pred, num_classes=4)
    avg_loss = total_loss / max(total_n, 1)
    return avg_loss, metrics