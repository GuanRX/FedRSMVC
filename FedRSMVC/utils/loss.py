import torch
import torch.nn as nn
import torch.nn.functional as F

class Wasserstein_Prototype_Contrastive_Loss(nn.Module):
    def __init__(self, temperature_tau=0.1, lambda_sigma=0.5, use_cosine_base=True):
        super(Wasserstein_Prototype_Contrastive_Loss, self).__init__()
        self.tau = temperature_tau
        self.lambda_sigma = lambda_sigma
        self.use_cosine_base = use_cosine_base
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, P_v, P_aligned, sigma_v, sigma_aligned):
        K = P_v.shape[0]
        device = P_v.device
        P_v, P_aligned = nn.functional.normalize(P_v, dim=1), nn.functional.normalize(P_aligned, dim=1)
        sigma_v = sigma_v.view(K, 1)
        sigma_aligned = sigma_aligned.view(K, 1)

        if self.use_cosine_base:
            P_v_norm = F.normalize(P_v, dim=1)
            P_aligned_norm = F.normalize(P_aligned, dim=1)
            cosine_sim = torch.matmul(P_v_norm, P_aligned_norm.t())
            center_dist_sq = 1.0 - cosine_sim
        else:
            P_v_sq = torch.sum(P_v**2, dim=1, keepdim=True)
            P_aligned_sq = torch.sum(P_aligned**2, dim=1, keepdim=True)
            dist_sq = P_v_sq + P_aligned_sq.t() - 2 * torch.matmul(P_v, P_aligned.t())
            center_dist_sq = torch.clamp(dist_sq, min=0.0)

        sigma_diff_sq = (sigma_v - sigma_aligned.t()) ** 2
        wasserstein_sim_logits = - (center_dist_sq + self.lambda_sigma * sigma_diff_sq)
        scaled_logits = wasserstein_sim_logits / self.tau
        labels = torch.arange(K).to(device)
        loss = self.cross_entropy(scaled_logits, labels)

        return loss