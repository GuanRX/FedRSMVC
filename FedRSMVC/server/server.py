# server/server.py
import models
from . import FedAvg_server
from utils.alignment import alignment, euclidean_dist
from utils.loss import Wasserstein_Prototype_Contrastive_Loss
import torch
import torch.nn.functional as F

FCSA_P_SCALE = 10000000
FCSA_C_SCALE = 1e14


class Server:
    def __init__(self, config, device, n_input, n_samples):
        self.config = config
        self.device = device
        self.global_model = models.get_model(
            config['model_name'],
            gae_n_enc_1=self.config["gae_n_enc_1"],
            gae_n_enc_2=self.config["gae_n_enc_2"],
            gae_n_enc_3=self.config["gae_n_enc_3"],
            gae_n_dec_1=self.config["gae_n_dec_1"],
            gae_n_dec_2=self.config["gae_n_dec_2"],
            gae_n_dec_3=self.config["gae_n_dec_3"],
            n_input=n_input, n_samples=n_samples
        ).to(self.device)

        self.global_parameters = self.global_model.state_dict()

    def distribute_model(self):
        Global_parameters = self.global_parameters.copy()
        ignore_keys = ['encoder.gnn_1.weight', 'decoder.gnn_6.weight']
        for key in ignore_keys:
            if key in Global_parameters:
                Global_parameters.pop(key)
        return Global_parameters

    def update_global_model(self, client_updates_list):
        algorithm = self.config['algorithm']
        new_parameters = None

        if algorithm == 'FedAvg':
            new_parameters = FedAvg_server.aggregate(client_updates_list)
        else:
            raise ValueError(f"Algorithm {algorithm} implementation not found for server.")

        if new_parameters:
            self.global_parameters = new_parameters
            self.global_model.load_state_dict(self.global_parameters, strict=False)

    def anchors_alignment(self, P_list, C_list, sigma_list):
        L_fcsa_list = []
        L_wpcl_list = []
        P_Gradients = []
        dataset = self.config['dataset_name']
        wpcl_loss = Wasserstein_Prototype_Contrastive_Loss(temperature_tau=0.1, lambda_sigma=1.0,
                                                           use_cosine_base=True).to(self.device)
        M_vu_list = []
        P_aligned_list = []

        self.global_model.train()
        for i in range(len(P_list)):
            if i == 0:
                continue

            P_v, C_v, sigma_v = P_list[0].detach(), C_list[0].detach(), sigma_list[0].detach()
            P_u, C_u, sigma_u = (
                P_list[i].detach().clone().requires_grad_(True),
                C_list[i].detach().clone().requires_grad_(True),
                sigma_list[i].detach().clone().requires_grad_(True)
            )

            D = euclidean_dist(P_v, P_u)
            M_vu = alignment(D)
            M_vu_list.append(M_vu)

            P_aligned = torch.matmul(M_vu, P_u)
            P_aligned_list.append(P_aligned)

            C_aligned = C_u @ M_vu.to(C_u.device) @ C_u.t()
            sigma_aligned = F.normalize(M_vu.to(sigma_u.device), p=2, dim=1) @ sigma_u

            sigma_aligned = sigma_aligned.to(sigma_u.device)
            P_aligned = P_aligned.to(sigma_u.device)
            P_v = P_v.to(sigma_u.device)

            L_fcsa = F.mse_loss(P_aligned, P_v) / FCSA_P_SCALE +  F.mse_loss(C_aligned, C_v) / FCSA_C_SCALE
            L_fcsa_list.append(L_fcsa)

            L_wpcl = wpcl_loss(P_aligned, P_v, sigma_aligned, sigma_v)
            L_wpcl_list.append(L_wpcl)

            total_loss = self.config[dataset]['lambda_1'] * L_fcsa + self.config[dataset]['lambda_2'] * L_wpcl

            grads = torch.autograd.grad(outputs=total_loss, inputs=P_u)[0]
            P_Gradients.append(grads.detach())

        if len(P_Gradients) > 0:
            Fused_Gradient = torch.mean(torch.stack(P_Gradients), dim=0)
        else:
            Fused_Gradient = None

        return L_fcsa_list, L_wpcl_list, P_aligned_list, M_vu_list, P_Gradients, Fused_Gradient