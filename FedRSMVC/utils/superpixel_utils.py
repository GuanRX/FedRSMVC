import numpy as np
from skimage.measure import regionprops
from skimage.segmentation import slic, mark_boundaries, find_boundaries
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale, minmax_scale, normalize, StandardScaler


from matplotlib import cm
import matplotlib as mpl
from sklearn.neighbors import kneighbors_graph
import networkx as nx
import cv2
import typing
from skimage import graph

def HSI_to_superpixels(img, n_superpixels, save_path=None):
    superpixel_label = slic(img, n_segments=n_superpixels, start_label=0)
    if save_path is not None:
        show_superpixel(superpixel_label, img, save_path)
    return superpixel_label


def show_superpixel(label, x=None, save_path='superpixels.pdf'):
    color = (162 / 255, 169 / 255, 175 / 255)
    n_row, n_col, n_band = x.shape
    if x is not None:
        x = minmax_scale(x.reshape(n_row * n_col, n_band)).reshape(n_row, n_col, n_band)
        mask = mark_boundaries(x, label, color=(1, 1, 0), mode='subpixel')
        # mask = x
    else:
        mask_boundary = find_boundaries(label, mode='subpixel')
        mask = np.ones((n_row, n_col, 3))
        mask[mask_boundary] = color
    fig = plt.figure()
    plt.imshow(mask)
    plt.axis('off')
    plt.tight_layout()
    fig.savefig(save_path, format='pdf', bbox_inches='tight', pad_inches=0)
    plt.close()


def create_association_mat(superpixel_labels):
    labels = np.unique(superpixel_labels)
    n_labels = labels.shape[0]
    n_pixels = superpixel_labels.shape[0] * superpixel_labels.shape[1]
    association_mat = np.zeros((n_pixels, n_labels))
    superpixel_labels_ = superpixel_labels.reshape(-1)
    for i, label in enumerate(labels):
        association_mat[np.where(label == superpixel_labels_), i] = 1
    return association_mat

def create_spixel_graph(source_img, superpixel_labels, n_neighbors=50):
    s = source_img.reshape((-1, source_img.shape[-1]))
    a = create_association_mat(superpixel_labels)
    mean_fea = np.matmul(a.T, s)
    regions = regionprops(superpixel_labels + 1)
    n_labels = np.unique(superpixel_labels).shape[0]
    center_indx = np.zeros((n_labels, 2))
    for i, props in enumerate(regions):
        center_indx[i, :] = props.centroid  # centroid coordinates
    ss_fea = np.concatenate((mean_fea, center_indx), axis=1)
    ss_fea = minmax_scale(ss_fea)
    try:
        adj = kneighbors_graph(ss_fea, n_neighbors=n_neighbors, mode='distance', include_self=False).toarray()
    except:
        adj = kneighbors_graph(ss_fea, n_neighbors=np.unique(superpixel_labels).shape[0] // 2, mode='distance', include_self=False).toarray()

    X_var = ss_fea.var()
    gamma = 1.0 / (ss_fea.shape[1] * X_var) if X_var != 0 else 1.0
    adj[np.where(adj != 0)] = np.exp(-np.power(adj[np.where(adj != 0)], 2) * gamma)
    np.fill_diagonal(adj, 0)
    g = graph.RAG(superpixel_labels)
    sadj = np.array(nx.linalg.adjacency_matrix(g).todense())
    sadj = sadj + np.eye(sadj.shape[0])
    return adj, sadj, center_indx

def extract_superpixel_features(img, labels, mode:typing.Literal['mean_std_geo', 'mean_std', 'center_patch']='mean_std_geo', patch_size=None):
    def normalization(data):
        _range = np.max(data) - np.min(data)
        return (data - np.min(data)) / _range

    num_labels = np.unique(labels).shape[0]
    if mode == 'mean_std_geo':
        features = np.zeros((num_labels, img.shape[2] * 2 + 3), dtype='float32')
        for i in range(num_labels):
            mask = np.zeros_like(labels, dtype='uint8')
            mask[labels==i] = 1

            mean = np.mean(img[labels==i], axis=0)
            std = np.std(img[labels==i], axis=0)
            features[i, 0:img.shape[2]] = mean
            features[i, img.shape[2]:img.shape[2] * 2] = std
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            perimeter = cv2.arcLength(contours[0], True)
            area = cv2.contourArea(contours[0])
            aspect_ratio = float(np.max(contours[0][:, 0, 0]) - np.min(contours[0][:, 0, 0])) / \
                        float(np.max(contours[0][:, 0, 1]) - np.min(contours[0][:, 0, 1]) + 1)
            if aspect_ratio > 1:
                aspect_ratio = 1 / aspect_ratio
            features[i, -3] = area
            features[i, -2] = perimeter
            features[i, -1] = aspect_ratio
        features[:, :-3] = normalization(features[:, :-3])
        features[:, -3] = normalization(features[:, -3])
        features[:, -2] = normalization(features[:, -2])

    elif mode == 'mean_std':
        features = np.zeros((num_labels, img.shape[2] * 2), dtype='float32')
        for i in range(num_labels):

            mean = np.mean(img[labels==i], axis=0)
            std = np.std(img[labels==i], axis=0)
            features[i, 0:img.shape[2]] = mean
            features[i, img.shape[2]:img.shape[2] * 2] = std


        regions = regionprops(labels + 1)
        n_labels = np.unique(labels).shape[0]
        center_indx = np.zeros((n_labels, 2))
        for i, props in enumerate(regions):
            center_indx[i, :] = props.centroid
        features = np.concatenate((center_indx, features), axis=1)
        features[:, :2] = normalization(features[:, :2])

    elif mode == 'center_patch':
        assert patch_size % 2 == 1, "Patch size must be odd"
        features = np.zeros((num_labels, patch_size, patch_size, img.shape[2]), dtype='float32')
        regions = regionprops(labels + 1)
        pad_size = patch_size // 2
        pad_img = np.pad(img, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='constant')
        for i, props in enumerate(regions):
            center = props.centroid
            x, y = center
            x = int(x + pad_size)
            y = int(y + pad_size)
            feat = pad_img[x - pad_size:x + pad_size + 1, y - pad_size:y + pad_size + 1, :]
            features[i, :] = feat
        features = np.transpose(features, (0, 3, 1, 2))
        features = minmax_scale(features.reshape(features.shape[0], -1), axis=1)

    return features