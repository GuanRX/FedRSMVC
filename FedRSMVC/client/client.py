# client/client.py
import torch
import numpy as np
import models
from . import FedAvg_client


class Client:
    def __init__(self, client_id, config, raw_data_dict, device):
        self.client_id = client_id
        self.config = config
        self.device = device
        self.data_loader = self._load_data(raw_data_dict)
        self.best_acc = -1.0
        self.model = models.get_model(
            config['model_name'],
            gae_n_enc_1=self.config["gae_n_enc_1"],
            gae_n_enc_2=self.config["gae_n_enc_2"],
            gae_n_enc_3=self.config["gae_n_enc_3"],
            gae_n_dec_1=self.config["gae_n_dec_1"],
            gae_n_dec_2=self.config["gae_n_dec_2"],
            gae_n_dec_3=self.config["gae_n_dec_3"],
            n_input=self.data_loader["x"].shape[1], n_samples=self.data_loader["x"].shape[0]
        ).to(self.device)

    def _load_data(self, raw_data_dict):
        raw = raw_data_dict

        x = torch.from_numpy(raw['data']).float().to(self.device)
        adj = torch.from_numpy(raw['adj']).float().to(self.device)
        y = torch.from_numpy(raw['y']).long().to(self.device)
        association_mat = torch.from_numpy(raw['association_mat']).float().to(self.device)

        sp_labels = raw['sp_labels']
        if isinstance(sp_labels, np.ndarray):
            sp_labels = torch.from_numpy(sp_labels).long().to(self.device)
        elif isinstance(sp_labels, torch.Tensor):
            sp_labels = sp_labels.to(self.device)

        raw_shape = raw['raw_shape']
        n_classes = raw['n_classes']

        Dataset = {
            'x': x,
            'adj': adj,
            'y': y,
            'association_mat': association_mat,
            'sp_labels': sp_labels,
            'raw_shape': raw_shape,
            'n_classes': n_classes
        }

        return Dataset

    def local_training(self, global_parameters, round_idx, mode, Gradient=None, fused_P=None):
        if global_parameters is not None and mode == 'local_reconstruction':
            self.model.load_state_dict(global_parameters, strict=False)

        algorithm = self.config['algorithm']
        local_updates = None
        metrics = {}
        if algorithm == 'FedAvg':
            if mode == 'local_reconstruction':
                local_updates, metrics, new_best_acc, P, predict_labels, C, sigma = FedAvg_client.recon_train(
                    model=self.model,
                    data_loader=self.data_loader,
                    config=self.config,
                    client_id=self.client_id,
                    dataset_name=self.config['dataset_name'],
                    round_idx=round_idx,
                    best_acc=self.best_acc
                )
                self.predict_labels = predict_labels
            else:
                local_updates, S = FedAvg_client.align_train(
                    model=self.model, data_loader=self.data_loader,
                    config=self.config, gradient=Gradient,
                    pseudo_labels=self.predict_labels, fused_P=fused_P
                )
        else:
            raise ValueError(f"Algorithm {algorithm} implementation not found for client.")

        if mode == 'local_reconstruction':
            return local_updates, metrics, P, C, sigma
        else:
            return local_updates, S