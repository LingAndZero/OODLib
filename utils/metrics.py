import numpy as np
import sklearn.metrics as sk


def cal_metric(ind_scores, ood_scores):
    
    ind_labels = np.ones(ind_scores.shape[0])
    ood_labels = np.zeros(ood_scores.shape[0])

    labels = np.concatenate([ind_labels, ood_labels])
    scores = np.concatenate([ind_scores, ood_scores])
    
    fpr_list, tpr_list, _ = sk.roc_curve(labels, scores)
    fpr = fpr_list[np.argmax(tpr_list >= 0.95)]

    auroc = sk.auc(fpr_list, tpr_list)

    precision, recall, _ = sk.precision_recall_curve(labels, scores)
    aupr = sk.auc(recall, precision)

    return auroc, aupr, fpr
