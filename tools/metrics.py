# tools/metrics.py
import numpy as np
from sklearn import metrics


def compute_image_auroc(scores, labels):
    """
    scores: list/np.array, 每张图的 anomaly score（越大越异常）
    labels: list/np.array, 0/1
    """
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    auroc = metrics.roc_auc_score(labels, scores)
    fpr, tpr, th = metrics.roc_curve(labels, scores)
    return {"auroc": auroc, "fpr": fpr, "tpr": tpr, "thresholds": th}


def compute_pixel_auroc(anomaly_maps, masks):
    """
    anomaly_maps: [N, H, W], 浮点 anomaly score
    masks:        [N, H, W], 0/1
    """
    anomaly_maps = np.asarray(anomaly_maps)
    masks = np.asarray(masks)
    flat_scores = anomaly_maps.reshape(-1)
    flat_labels = masks.reshape(-1).astype(int)
    auroc = metrics.roc_auc_score(flat_labels, flat_scores)
    fpr, tpr, th = metrics.roc_curve(flat_labels, flat_scores)
    return {"auroc": auroc, "fpr": fpr, "tpr": tpr, "thresholds": th}
