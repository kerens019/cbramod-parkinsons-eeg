"""Pooled performance metrics, computed once over all held-out subjects."""

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix, mean_absolute_error,
                             mean_squared_error, r2_score, roc_auc_score)


def classification_metrics(y_true, y_prob, y_pred):
    y_true, y_prob, y_pred = (np.asarray(a) for a in (y_true, y_prob, y_pred))
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "n_subjects": len(y_true),
        "auroc": roc_auc_score(y_true, y_prob),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "sensitivity": tp / (tp + fn),
        "specificity": tn / (tn + fp),
        "accuracy": (tp + tn) / len(y_true),
    }


def regression_metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return {
        "n_subjects": len(y_true),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "pearson_r": float(pearsonr(y_true, y_pred)[0]),
        "spearman_r": float(spearmanr(y_true, y_pred)[0]),
    }
