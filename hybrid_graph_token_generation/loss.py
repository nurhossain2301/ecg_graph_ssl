import torch
import torch.nn.functional as F


def masked_token_loss(token_logits, target_tokens, node_mask, valid_mask):
    """
    token_logits: [B, N, V]
    target_tokens: [B, N]
    node_mask: [B, N]
    valid_mask: [B, N]
    """
    mask = node_mask & valid_mask

    if mask.sum() == 0:
        return token_logits.sum() * 0.0

    logits = token_logits[mask]      # [M, V]
    targets = target_tokens[mask]    # [M]

    return F.cross_entropy(logits, targets)


def graph_smoothness_loss(hidden, edge_index, valid_mask):
    """
    hidden: [B, N, D]
    edge_index: list of length B, each [2, E]
    valid_mask: [B, N]
    """
    losses = []

    B = hidden.size(0)

    for b in range(B):
        h = hidden[b]
        n_valid = int(valid_mask[b].sum().item())
        ei = edge_index[b]

        if ei.numel() == 0 or n_valid <= 1:
            continue

        src = ei[0]
        dst = ei[1]

        valid_edge = (src < n_valid) & (dst < n_valid)
        src = src[valid_edge]
        dst = dst[valid_edge]

        if src.numel() == 0:
            continue

        diff = h[src] - h[dst]
        loss_b = diff.pow(2).sum(dim=-1).mean()
        losses.append(loss_b)

    if len(losses) == 0:
        return hidden.sum() * 0.0

    return torch.stack(losses).mean()


def seq_graph_consistency_loss(seq_hidden, graph_hidden, valid_mask):
    """
    seq_hidden: [B, N, D]
    graph_hidden: [B, N, D]
    valid_mask: [B, N]
    """
    diff = (seq_hidden - graph_hidden).pow(2).sum(dim=-1)   # [B, N]
    diff = diff * valid_mask.float()

    denom = valid_mask.float().sum().clamp_min(1.0)
    return diff.sum() / denom


def total_loss(
    outputs,
    tokens,
    node_mask,
    valid_mask,
    edge_index,
    lambda_smooth=0.1,
    lambda_align=0.1,
):
    """
    outputs: model forward dict
    """
    loss_mtm = masked_token_loss(
        outputs["token_logits"],
        tokens,
        node_mask,
        valid_mask
    )

    loss_smooth = graph_smoothness_loss(
        outputs["hidden"],
        edge_index,
        valid_mask
    )

    loss_align = seq_graph_consistency_loss(
        outputs["seq_hidden"],
        outputs["graph_hidden"],
        valid_mask
    )

    loss = loss_mtm + lambda_smooth * loss_smooth + lambda_align * loss_align

    return {
        "loss": loss,
        "loss_mtm": loss_mtm,
        "loss_smooth": loss_smooth,
        "loss_align": loss_align
    }