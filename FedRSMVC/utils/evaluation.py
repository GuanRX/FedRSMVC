from warnings import simplefilter
import torch
from torch_clustering import PyTorchKMeans
simplefilter(action='ignore', category=FutureWarning)
from cal_metric import full_metric

def my_clustering_gpu(feature, true_labels, cluster_num):
    kmeans = PyTorchKMeans(metric='euclidean', init='k-means++', n_clusters=cluster_num, n_init=10, verbose=False)
    feature = torch.tensor(feature, dtype=torch.float32)
    predict_labels = kmeans.fit_predict(feature)
    center = kmeans.cluster_centers_
    dis=torch.cdist(feature, center, p=2)
    dis=dis.cpu()
    predict_labels = predict_labels.cpu().numpy()
    pixel_pred = predict_labels
    OA, AA, KAPPA, NMI, ARI, F1, PRECISION, RECALL, PURITY = full_metric(true_labels, pixel_pred, is_refined=False)
    return 100*OA, 100*AA, 100*KAPPA, 100*NMI, 100*ARI, 100*F1, 100*PRECISION, 100*RECALL, 100*PURITY, predict_labels, center