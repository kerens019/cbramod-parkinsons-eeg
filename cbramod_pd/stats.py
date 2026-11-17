"""Paired permutation test and Benjamini-Hochberg FDR correction (Methods, "Statistical Analysis")."""

import numpy as np
from scipy.stats import false_discovery_control

from . import config as C

N_ITER = 1_000_000


def balanced_accuracy_rows(y_true, preds):
    """Balanced accuracy of each row of a (n_iter, n_subjects) 0/1 prediction matrix."""
    pos, neg = y_true == 1, y_true == 0
    return 0.5 * ((preds[:, pos] == 1).mean(axis=1) + (preds[:, neg] == 0).mean(axis=1))


def rmse_rows(y_true, preds):
    return np.sqrt(np.mean((preds - y_true[None, :]) ** 2, axis=1))


METRICS = {"balanced_accuracy": balanced_accuracy_rows, "rmse": rmse_rows}


def paired_permutation_test(y_true, pred_a, pred_b, metric, n_iter=N_ITER, seed=C.SEED, chunk=50_000):
    """Two-sided paired permutation test of metric(pred_a) - metric(pred_b).

    Null distribution: in every iteration each subject's two predictions are
    swapped between the models with probability 0.5 and the statistic is
    recomputed. p = (1 + b) / (1 + n_iter), where b counts null values with
    |null| >= |observed|.
    Returns (observed difference, p-value).
    """
    y_true, pred_a, pred_b = (np.asarray(a, dtype=float) for a in (y_true, pred_a, pred_b))
    f = METRICS[metric]
    observed = float(f(y_true, pred_a[None])[0] - f(y_true, pred_b[None])[0])

    rng = np.random.default_rng(seed)
    null, done = np.empty(n_iter), 0
    while done < n_iter:
        m = min(chunk, n_iter - done)
        swap = rng.random((m, len(y_true))) < 0.5
        a = np.where(swap, pred_b[None], pred_a[None])
        b = np.where(swap, pred_a[None], pred_b[None])
        null[done:done + m] = f(y_true, a) - f(y_true, b)
        done += m
    p = (1 + np.sum(np.abs(null) >= abs(observed))) / (1 + n_iter)
    return observed, float(p)


def fdr_bh(p_values):
    """Benjamini-Hochberg adjusted p-values (applied within each task)."""
    return false_discovery_control(np.asarray(p_values, dtype=float), method="bh")
