import numpy as np
import scipy.io as sio
import copy
import torch
import os
import warnings
from sklearn.preprocessing import StandardScaler, minmax_scale, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph
from collections import Counter

from utils.superpixel_utils import HSI_to_superpixels, create_association_mat, create_spixel_graph, show_superpixel, extract_superpixel_features


def prepare_data(mat_path):
    data = sio.loadmat(mat_path)
    key = [k for k in data if k != '__version__' and k != '__header__' and k != '__globals__']
    pixels = data[key[0]]
    return pixels


def get_sp_labels(seg, gt):
    labels = np.zeros((np.unique(seg).shape[0]), dtype=np.int32)
    for i in range(len(labels)):
        mask = seg == i
        label = gt[mask]
        if len(label) == 0:
            labels[i] = 0
            continue
        c = Counter(label).most_common()
        if c[0][0] == 0 and len(c) > 1:
            labels[i] = c[1][0]
        else:
            labels[i] = c[0][0]
    return labels


class FedDataset():
    def __init__(self, dataset_name, mm_data_path, gt_path, n_pc, rgb, n_neighbors, n_superpixels, patch_size) -> None:
        self.dataset_name = dataset_name
        mm_data_raw = [prepare_data(data_path) for data_path in mm_data_path]
        for i in range(len(mm_data_raw)):
            if len(mm_data_raw[i].shape) == 2:
                mm_data_raw[i] = mm_data_raw[i][:, :, np.newaxis]
        gt = prepare_data(gt_path)

        n_row, n_column, n_band = mm_data_raw[0].shape
        hsi = copy.deepcopy(mm_data_raw[0])

        if n_pc is not None:
            pca = PCA(n_components=n_pc, random_state=42)
            mm_data_raw[0] = minmax_scale(pca.fit_transform(mm_data_raw[0].reshape(n_row * n_column, n_band))).reshape(
                (n_row, n_column, -1))
        else:
            mm_data_raw[0] = minmax_scale(mm_data_raw[0].reshape(n_row * n_column, n_band)).reshape(
                (n_row, n_column, n_band))

        if rgb is None:
            warnings.warn('No matched dataset name. RGB bands set to [0, 1, 2].')
            rgb = [0, 1, 2]

        if self.dataset_name == 'Trento':
            sp_source_img = hsi[:, :, rgb]
        else:
            sp_source_img = mm_data_raw[0][:, :, :3]

        sp_labels = HSI_to_superpixels(sp_source_img, n_superpixels=n_superpixels, save_path=None)
        if not os.path.exists('result'): os.makedirs('result')
        show_superpixel(sp_labels, hsi[:, :, rgb], 'result/' + self.dataset_name + '_sp.pdf')
        self.association_mat = create_association_mat(sp_labels)
        self.association_mat_copy = self.association_mat.copy()
        self.y = get_sp_labels(sp_labels, gt)
        self.gt = gt
        self.sp_labels = sp_labels
        self.n_classes = np.unique(self.y).shape[0] - 1
        self.clients = []
        for i, data in enumerate(mm_data_raw):
            features = extract_superpixel_features(data, sp_labels, mode='center_patch', patch_size=patch_size)
            sp_features = extract_superpixel_features(data, sp_labels, mode='mean_std')
            sp_features = sp_features[:, :data.shape[2]]
            fadj, _, _ = create_spixel_graph(data, sp_labels, n_neighbors)
            client_dict = {
                'client_id': i,
                'modality_name': f'modality_{i}',
                'n_classes': self.n_classes,
                'data': features,
                'adj': fadj,
                'y': self.y,
                'sp_labels': self.sp_labels,
                'association_mat': self.association_mat,
                'raw_shape': (n_row, n_column)
            }
            self.clients.append(client_dict)

    def remove_background(self):
        non_zero_idx = np.nonzero(self.y)[0]
        self.y = self.y[non_zero_idx]
        self.association_mat = self.association_mat[:, non_zero_idx]
        for client in self.clients:
            client['data'] = client['data'][non_zero_idx, :]
            if hasattr(client['adj'], 'toarray'):
                client['adj'] = client['adj'].toarray()
            client['adj'] = client['adj'][non_zero_idx, :][:, non_zero_idx]
            client['y'] = self.y
            client['association_mat'] = self.association_mat

    def recover_background(self, sp_pred):
        if isinstance(sp_pred, torch.Tensor):
            sp_pred = sp_pred.detach().cpu().numpy()
        total_sp_num = self.association_mat_copy.shape[1]
        sp_pred_full = np.zeros(total_sp_num, dtype=sp_pred.dtype)
        try:
            temp_y = get_sp_labels(self.sp_labels, self.gt)
            mask = temp_y != 0
            sp_pred_full[mask] = sp_pred
        except:
            warnings.warn("Recover background failed due to shape mismatch, returning raw pred.")
            return sp_pred
        self.association_mat = self.association_mat_copy
        return sp_pred_full

    def save_clients(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for client in self.clients:
            client_id = client['client_id']
            save_dict = {
                'x': torch.from_numpy(client['data']).float(),
                'adj': torch.from_numpy(client['adj']).float(),
                'y': torch.from_numpy(client['y']).long(),
                'sp_labels': torch.from_numpy(client['sp_labels']).long(),
                'association_mat': torch.from_numpy(client['association_mat']).float(),
                'n_classes': client['n_classes']
            }
            fname = os.path.join(save_dir, f'client_{client_id}.pt')
            torch.save(save_dict, fname)
            print(f'Saved client {client_id} data to {fname}')

    def __len__(self):
        return len(self.clients)

    def __getitem__(self, index):
        return self.clients[index]

    def __str__(self) -> str:
        return f"Dataset: {self.dataset_name} Clients: {len(self.clients)}"

    def __len__(self):
        return len(self.clients)

    def __getitem__(self, index):
        return self.clients[index]