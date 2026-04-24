import torch
import torch.nn.functional as F

# graph_utils.py — add valid_mask support
def build_graph(h, k=6, valid_mask=None):
    B, N, D = h.shape
    x = F.normalize(h, dim=-1)
    sim = torch.bmm(x, x.transpose(1, 2))   # [B, N, N]

    if valid_mask is not None:
        # mask out edges from/to padding nodes
        mask2d = valid_mask.unsqueeze(2) & valid_mask.unsqueeze(1)  # [B, N, N]
        sim = sim.masked_fill(~mask2d, -1e9)

    # zero diagonal
    eye = torch.eye(N, device=h.device).unsqueeze(0).bool()
    sim = sim.masked_fill(eye, -1e9)

    topk_val, _ = torch.topk(sim, k, dim=-1)
    threshold = topk_val[..., -1:]
    A = (sim >= threshold).float()

    if valid_mask is not None:
        A = A * valid_mask.unsqueeze(2).float()
        A = A * valid_mask.unsqueeze(1).float()

    return A

# def build_graph(h, k=6):
#     B, N, D = h.shape
#     A = torch.zeros(B, N, N, device=h.device)

#     for b in range(B):
#         x = F.normalize(h[b], dim=-1)
#         sim = x @ x.T

#         vals, inds = torch.topk(sim, k=min(k+1, N), dim=-1)

#         for i in range(N):
#             for j in inds[i]:
#                 if i != j:
#                     A[b, i, j] = sim[i, j]

#         A[b] += torch.eye(N, device=h.device)

#     return A