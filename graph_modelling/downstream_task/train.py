import torch
import torch.nn as nn
from utils import compute_metrics
# -----------------------------
# Train / Eval loops
# -----------------------------
def run_one_epoch(model, loader, optimizer, device, train=True, grad_clip=1.0, args=None, class_weights=None):
    if train:
        model.train()
    else:
        model.eval()

    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_n = 0
    all_true, all_pred = [], []

    for batch in loader:
        for k in batch:
            batch[k] = batch[k].to(device)

        labels = batch["label"]
        

        with torch.set_grad_enabled(train):
            logits = model(batch)  # (B, num_classes)
            loss = criterion(logits, labels)

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip is not None and grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size
        total_n += batch_size

        preds = torch.argmax(logits, dim=-1)
        all_true.extend(labels.detach().cpu().numpy().tolist())
        all_pred.extend(preds.detach().cpu().numpy().tolist())

    metrics = compute_metrics(all_true, all_pred, num_classes=args.num_classes)
    avg_loss = total_loss / max(total_n, 1)
    return avg_loss, metrics