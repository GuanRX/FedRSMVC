# client/FedAvg_client.py
import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from utils.evaluation import my_clustering_gpu
import torch.nn.functional as F
from utils.Anchor_graph import construct_anchor_graph_paper
from models import ProbabilisticEncoder


def compute_prototype_uncertainty(H, P_hat, predict_labels):
    device = H.device

    if not isinstance(P_hat, torch.Tensor):
        P_hat = torch.tensor(P_hat)
    if not isinstance(predict_labels, torch.Tensor):
        predict_labels = torch.tensor(predict_labels)

    H = F.normalize(H, p=2, dim=1)
    P_hat = F.normalize(P_hat, p=2, dim=1)
    P_hat = P_hat.to(device)
    predict_labels = predict_labels.to(device)
    P_hat_expanded = P_hat[predict_labels]
    distances = torch.norm(H - P_hat_expanded, p=2, dim=1)
    K = P_hat.size(0)
    sigma = torch.zeros(K, device=device)
    ones = torch.ones_like(distances)
    count = torch.zeros(K, device=device)
    predict_labels = predict_labels.long()

    count.scatter_add_(0, predict_labels, ones)
    sigma.scatter_add_(0, predict_labels, distances)
    eps = 1e-8
    sigma = sigma / (count + eps)

    return sigma


def recon_train(model, data_loader, config, client_id, dataset_name, round_idx, best_acc):
    lr = config.get('lr', 1e-3)
    weight_decay = config.get('weight_decay', 1e-5)
    epochs = config.get('local_epochs', 10)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion_mse = nn.MSELoss()
    x = data_loader['x']
    adj = data_loader['adj']
    y = data_loader['y']
    n_classes = data_loader['n_classes']

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        H, X_hat, adj_hat = model(x, adj)
        metrics_dict, predict_labels, P, new_best_acc = model_eval(model=model, x=x, adj=adj, y=y, best_acc=best_acc,
                                                                   n_classes=n_classes, client_id=client_id,
                                                                   round_idx=round_idx,
                                                                   dataset_name=dataset_name)
        L_rec = criterion_mse(X_hat, x)
        dists = torch.cdist(H, P.to(H.device), p=2)
        S = F.softmax(-dists, dim=1)
        C = S.t() @ S
        loss = L_rec
        loss.backward()
        optimizer.step()

        sigma = compute_prototype_uncertainty(H, P, predict_labels)

    Local_parameters = model.state_dict()
    if 'encoder.gnn_1.weight' in Local_parameters:
        Local_parameters.pop('encoder.gnn_1.weight')
    if 'decoder.gnn_6.weight' in Local_parameters:
        Local_parameters.pop('decoder.gnn_6.weight')

    return Local_parameters, metrics_dict, new_best_acc, P, predict_labels, C, sigma


def model_eval(model, x, adj, y, best_acc, n_classes, client_id, round_idx, dataset_name):
    model.eval()
    with torch.no_grad():
        H, _, _ = model(x, adj)
        features_np = H.detach().cpu().numpy()
        labels_np = y.detach().cpu().numpy()
        metrics_tuple = my_clustering_gpu(features_np, labels_np, n_classes)
        OA = metrics_tuple[0]
        predict_labels, P_hat = metrics_tuple[9], metrics_tuple[10]
        n_features = P_hat.shape[1]
        model = ProbabilisticEncoder(n_features=n_features)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        P_hat_gpu = P_hat.to(device)
        P, mu, sigma, epsilon = model(P_hat_gpu)

        metrics_dict = {
            "OA": OA,
            "AA": metrics_tuple[1],
            "Kappa": metrics_tuple[2],
            "NMI": metrics_tuple[3],
            "ARI": metrics_tuple[4],
            "F1": metrics_tuple[5],
            "Precision": metrics_tuple[6],
            "Recall": metrics_tuple[7],
            "Purity": metrics_tuple[8]
        }

        new_best_acc = best_acc
        if OA > best_acc:
            new_best_acc = OA
            save_dir = os.path.join("save", f"client_{client_id}", dataset_name, str(round_idx))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            torch.save(model.state_dict(), os.path.join(save_dir, "model_params.pth"))
            np.save(os.path.join(save_dir, "features.npy"), features_np)
            np.save(os.path.join(save_dir, "labels.npy"), labels_np)

        return metrics_dict, predict_labels, P, new_best_acc


def align_train(model, data_loader, config, gradient, pseudo_labels, fused_P):
    lr = config.get('lr', 1e-3)
    weight_decay = config.get('weight_decay', 1e-5)
    epochs = config.get('local_epochs', 10)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    x = data_loader['x']
    adj = data_loader['adj']

    if not isinstance(pseudo_labels, torch.Tensor):
        pseudo_labels = torch.tensor(pseudo_labels)
    pseudo_labels = pseudo_labels.to(x.device)

    center_grad = gradient.to(x.device)
    unique_labels, counts = torch.unique(pseudo_labels, return_counts=True)
    count_lookup = torch.ones(center_grad.shape[0], device=x.device)
    count_lookup[unique_labels.long()] = counts.float()
    grad_per_sample = torch.nn.functional.embedding(pseudo_labels.long(), center_grad)
    count_per_sample = torch.nn.functional.embedding(pseudo_labels.long(), count_lookup.unsqueeze(1))
    target_sample_grad = grad_per_sample / count_per_sample

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        H, _, _ = model(x, adj)
        if H.shape != target_sample_grad.shape:
            break
        H.backward(target_sample_grad.detach())
        optimizer.step()

    model.eval()
    H, _, _ = model(x, adj)
    S = construct_anchor_graph_paper(H, fused_P)

    Local_parameters = model.state_dict()
    if 'encoder.gnn_1.weight' in Local_parameters:
        Local_parameters.pop('encoder.gnn_1.weight')
    if 'decoder.gnn_6.weight' in Local_parameters:
        Local_parameters.pop('decoder.gnn_6.weight')

    return Local_parameters, S