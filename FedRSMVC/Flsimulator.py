# Flsimulator.py
import os
import time
import numpy as np
import torch
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
from contextlib import redirect_stdout
from client.client import Client
from server.server import Server
from dataset import FedDataset
from utils.Anchor_Clustering import compute_view_weights_torch
from cal_metric import full_metric


def map_labels(y_true, y_pred):
    y_true = np.array(y_true, dtype=np.int64)
    y_pred = np.array(y_pred, dtype=np.int64)

    if y_true.size != y_pred.size:
        return y_pred

    if y_true.ndim > 1: y_true = y_true.flatten()
    if y_pred.ndim > 1: y_pred = y_pred.flatten()

    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    mapping = np.zeros(D, dtype=np.int64)
    for i in range(len(row_ind)):
        mapping[row_ind[i]] = col_ind[i]

    return mapping[y_pred]


class FLSimulator:
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.server = None
        self.clients = []
        self.dataset = None

        with open(os.devnull, 'w') as f, redirect_stdout(f):
            self._setup_environment()

    def _setup_environment(self):
        dataset_name = self.config['dataset_name']
        self.dataset = FedDataset(dataset_name,
                                  self.config[dataset_name]['mm_data_path'],
                                  self.config[dataset_name]['gt_path'],
                                  self.config[dataset_name]['n_pc'],
                                  self.config[dataset_name]['rgb'],
                                  self.config[dataset_name]['n_neighbors'],
                                  self.config[dataset_name]['n_superpixels'],
                                  self.config[dataset_name]['patch_size'])
        if self.config.get('remove_bkg', False):
            self.dataset.remove_background()
        num_clients = len(self.dataset.clients)
        self.config['num_clients'] = num_clients
        n_samples = self.dataset.y.size

        n_input_dim = self.dataset.clients[0]['data'].shape[1]
        self.server = Server(self.config, device=self.device, n_input=n_input_dim, n_samples=n_samples)

        for i in range(num_clients):
            client = Client(
                client_id=i,
                config=self.config,
                raw_data_dict=self.dataset.clients[i],
                device=self.device
            )
            self.clients.append(client)

    def start(self):
        start_time = time.time()
        total_rounds = self.config['n_epoches']
        dataset_name = self.config['dataset_name']

        best_metrics = {
            'OA': 0.0, 'Kappa': 0.0, 'NMI': 0.0,
            'ARI': 0.0, 'F1': 0.0, 'PURITY': 0.0
        }

        for round_idx in tqdm(range(total_rounds), desc=f"Training FedRSMVC on {dataset_name}", leave=True):
            try:
                with open(os.devnull, 'w') as devnull, redirect_stdout(devnull):
                    global_params = self.server.distribute_model()

                    client_updates, P_list, C_list, sigma_list = [], [], [], []
                    for client in self.clients:
                        update, metrics, P, C, sigma_i = client.local_training(global_params, round_idx + 1, 'local_reconstruction')
                        P_list.append(P)
                        C_list.append(C)
                        sigma_list.append(sigma_i)

                    L_fcsa_list, L_wpcl_list, P_aligned_list, M_vu_list, P_Gradients, Fused_Gradient = self.server.anchors_alignment(P_list, C_list, sigma_list)

                    stacked_anchors = torch.stack([P_list[0]] + P_aligned_list, dim=0)
                    fused_P = torch.mean(stacked_anchors, dim=0)

                    S = []
                    for client in self.clients:
                        idx = client.client_id - 1 if client.client_id != 0 else 0
                        c_Grad = P_Gradients[idx] if client.client_id != 0 else Fused_Gradient

                        update, S_mat = client.local_training(None, round_idx + 1, 'prototype', Gradient=c_Grad, fused_P=fused_P)
                        client_updates.append(update)
                        S.append(S_mat)

                    self.server.update_global_model(client_updates)

                    weights = compute_view_weights_torch(S)
                    final_prob = weights[0] * S[0]
                    weights[0] = weights[1] = 0.5
                    for i in range(1, len(S)):
                        final_prob += weights[i] * S[i]

                    y = self.clients[0].data_loader["y"].cpu().numpy()
                    cluster_labels_raw = torch.argmax(final_prob, dim=1).cpu().numpy()

                    OA, AA, KAPPA, NMI, ARI, F1, PRECISION, RECALL, PURITY = full_metric(y, cluster_labels_raw, is_refined=False)
                    if OA > best_metrics['OA']:
                        best_metrics['OA'] = OA
                        best_metrics['Kappa'] = KAPPA
                        best_metrics['NMI'] = NMI
                        best_metrics['ARI'] = ARI
                        best_metrics['F1'] = F1
                        best_metrics['PURITY'] = PURITY

            except Exception as e:
                print(f"\n[Error in Round {round_idx + 1}] {e}")
                continue

        total_time = time.time() - start_time

        print("\n")
        print(f"FedRSMVC Performance on {dataset_name}")
        print("=" * 40)
        print(f"OA     : {best_metrics['OA']:.8f}")
        print(f"Kappa  : {best_metrics['Kappa']:.8f}")
        print(f"NMI    : {best_metrics['NMI']:.8f}")
        print(f"ARI    : {best_metrics['ARI']:.8f}")
        print(f"F1     : {best_metrics['F1']:.8f}")
        print(f"PURITY : {best_metrics['PURITY']:.8f}")
        print(f"Time   : {total_time:.2f}")
        print("=" * 40)

        return best_metrics