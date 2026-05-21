import torch
import torch.nn.functional as F

def construct_anchor_graph_paper(Z, anchor, lambda_reg=0.1):
    Z = F.normalize(Z, p=2, dim=1)
    anchor = F.normalize(anchor, p=2, dim=1)

    device = Z.device
    if anchor.device != device:
        anchor = anchor.to(device)

    n = Z.size(0)
    m = anchor.size(0)
    Z_sq = torch.sum(Z ** 2, dim=1, keepdim=True)
    U_sq = torch.sum(anchor ** 2, dim=1, keepdim=True)
    dist = Z_sq + U_sq.t() - 2 * torch.matmul(Z, anchor.t())
    min_dist, _ = torch.min(dist, dim=1, keepdim=True)
    dist = dist - min_dist

    if lambda_reg <= 1e-8:
        S = torch.zeros(n, m, device=device)
        min_indices = torch.argmin(dist, dim=1)
        S.scatter_(1, min_indices.unsqueeze(1), 1.0)
        return S

    v = -dist / (2 * lambda_reg)
    u, _ = torch.sort(v, descending=True, dim=1)
    cssv = torch.cumsum(u, dim=1)
    ind = torch.arange(1, m + 1, device=device, dtype=Z.dtype).unsqueeze(0)
    cond = u + (1 - cssv) / ind > 0
    rho = torch.sum(cond, dim=1)
    rho = torch.clamp(rho, min=1.0)
    rho_idx = (rho - 1).long().unsqueeze(1)
    cssv_rho = torch.gather(cssv, 1, rho_idx).squeeze(1)
    theta = (1 - cssv_rho) / rho
    S = torch.clamp(v + theta.unsqueeze(1), min=0)
    S = S / (S.sum(dim=1, keepdim=True) + 1e-9)

    return S