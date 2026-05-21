# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Parameter


class ProbabilisticEncoder(nn.Module):
    def __init__(self, n_features):
        super(ProbabilisticEncoder, self).__init__()
        self.mu_layer = nn.Linear(n_features, n_features, bias=True)
        self.sigma_layer = nn.Linear(n_features, n_features, bias=True)
        self.prelu_weight = nn.Parameter(torch.Tensor(n_features).fill_(0.25))

    def forward(self, P_hat):
        epsilon = torch.randn_like(P_hat).to(P_hat.device)
        mu = F.prelu(self.mu_layer(P_hat), self.prelu_weight)
        sigma = F.relu(self.sigma_layer(P_hat))
        P = P_hat + (mu + sigma * epsilon)
        P = F.normalize(P, p=2, dim=1)
        return P, mu, sigma, epsilon

class GNNLayer(Module):
    def __init__(self, in_features, out_features):
        super(GNNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.act = nn.Tanh()
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, features, adj, active=False):
        support = torch.mm(features, self.weight)
        output = torch.spmm(adj, support)
        if active:
            output = self.act(output)
        return output

    def __str__(self) -> str:
        return f"GNNLayer(in_features={self.in_features}, out_features={self.out_features})"

    def __repr__(self):
        return self.__str__()


class IGAE_encoder(nn.Module):
    def __init__(self, gae_n_enc_1, gae_n_enc_2, gae_n_enc_3, n_input):
        super(IGAE_encoder, self).__init__()
        self.gnn_1 = GNNLayer(n_input, gae_n_enc_1)
        self.gnn_2 = GNNLayer(gae_n_enc_1, gae_n_enc_2)
        self.gnn_3 = GNNLayer(gae_n_enc_2, gae_n_enc_3)
        self.s = nn.Sigmoid()

    def forward(self, x, adj):
        z = self.gnn_1(x, adj, active=True)
        z = self.gnn_2(z, adj, active=True)
        H = self.gnn_3(z, adj, active=False)
        H_adj = self.s(torch.mm(H, H.t()))
        return H, H_adj


class IGAE_decoder(nn.Module):
    def __init__(self, gae_n_dec_1, gae_n_dec_2, gae_n_dec_3, n_input):
        super(IGAE_decoder, self).__init__()
        self.gnn_4 = GNNLayer(gae_n_dec_1, gae_n_dec_2)
        self.gnn_5 = GNNLayer(gae_n_dec_2, gae_n_dec_3)
        self.gnn_6 = GNNLayer(gae_n_dec_3, n_input)
        self.s = nn.Sigmoid()

    def forward(self, H, adj):
        z = self.gnn_4(H, adj, active=True)
        z = self.gnn_5(z, adj, active=True)
        X_hat = self.gnn_6(z, adj, active=True)
        X_hat_adj = self.s(torch.mm(X_hat, X_hat.t()))
        return X_hat, X_hat_adj


class IGAE(nn.Module):
    def __init__(self, gae_n_enc_1, gae_n_enc_2, gae_n_enc_3, gae_n_dec_1, gae_n_dec_2, gae_n_dec_3, n_input,
                 n_samples):
        super(IGAE, self).__init__()
        self.encoder = IGAE_encoder(
            gae_n_enc_1=gae_n_enc_1,
            gae_n_enc_2=gae_n_enc_2,
            gae_n_enc_3=gae_n_enc_3,
            n_input=n_input
        )

        self.decoder = IGAE_decoder(
            gae_n_dec_1=gae_n_dec_1,
            gae_n_dec_2=gae_n_dec_2,
            gae_n_dec_3=gae_n_dec_3,
            n_input=n_input
        )

    def forward(self, x, adj):
        H, H_adj = self.encoder(x, adj)
        X_hat, X_hat_adj = self.decoder(H, adj)
        adj_hat = H_adj + X_hat_adj
        return H, X_hat, adj_hat

    def __str__(self):
        return f"IGAE(encoder={self.encoder}, decoder={self.decoder})"

    def __repr__(self):
        return self.__str__()


def get_model(model_name, **kwargs):
    if model_name == "IGAE":
        return IGAE(**kwargs)
    elif model_name == "ProbabilisticEncoder":
        return ProbabilisticEncoder(**kwargs)
    else:
        raise ValueError(f"Model {model_name} not implemented")