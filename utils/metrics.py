import numpy as np
from scipy.optimize import linear_sum_assignment


def hungary_iou(dists, dist_thresh=0.1):
    distances = dists.copy()
    gt_l = distances.shape[0]
    pred_l = distances.shape[1]
    indice = linear_sum_assignment(distances)
    distances = distances[indice[0], indice[1]]
    tp = np.sum(distances < dist_thresh + 0.001)
    return tp / (gt_l + pred_l - tp)


def get_cd(dists):
    cd = np.sum(np.min(dists, axis=1)) / dists.shape[0] + np.sum(np.min(dists.T, axis=1)) / dists.shape[1]
    return cd


