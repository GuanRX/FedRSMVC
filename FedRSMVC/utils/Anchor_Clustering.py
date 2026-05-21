import torch
import torch.nn.functional as F


def compute_view_weights_torch(S_list, eps=1e-10):
    avg_entropies = []

    for S in S_list:
        sample_entropies = F.cross_entropy(
            torch.log(S + eps),  # shape=(n, K)
            torch.zeros(S.shape[0], dtype=torch.long, device=S.device),
            reduction='none'
        )
        avg_entropy = torch.mean(sample_entropies)
        avg_entropies.append(avg_entropy)

    avg_entropies = torch.stack(avg_entropies, dim=0)
    initial_weights = 1.0 / (avg_entropies + eps)
    view_weights = initial_weights / torch.sum(initial_weights)

    return view_weights
