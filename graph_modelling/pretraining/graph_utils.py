import torch
import torch.nn.functional as F

def build_graph(h, k=6):
    B, N, D = h.shape
    A = torch.zeros(B, N, N, device=h.device)

    for b in range(B):
        x = F.normalize(h[b], dim=-1)
        sim = x @ x.T

        vals, inds = torch.topk(sim, k=min(k+1, N), dim=-1)

        for i in range(N):
            for j in inds[i]:
                if i != j:
                    A[b, i, j] = sim[i, j]

        A[b] += torch.eye(N, device=h.device)

    return A