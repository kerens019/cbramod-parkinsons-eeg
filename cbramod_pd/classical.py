"""Classical EEG-feature baseline (Methods, "Classical EEG-Feature Baseline").

Five features per subject (Lempel-Ziv complexity and relative theta, alpha,
beta and gamma power), each averaged across electrodes, used as is (no
scaling or selection). Models: logistic regression + random forest for
classification (balanced class weights, decision threshold chosen in the
same inner folds as CBraMod); ordinary least squares + random forest for
regression. Uses the exact outer and inner folds of the CBraMod run.
"""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import balanced_accuracy_score

from . import config as C
from .cv import loso_splits


def make_models(kind):
    if kind == "classification":
        return {
            "LogisticRegression": LogisticRegression(class_weight="balanced", max_iter=C.LOGREG_MAX_ITER,
                                                     random_state=C.SEED),
            "RandomForest": RandomForestClassifier(n_estimators=C.RF_N_TREES, class_weight="balanced",
                                                   random_state=C.SEED),
        }
    return {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=C.RF_N_TREES, random_state=C.SEED),
    }


def load_subject_features(features_csv):
    """Feature table -> one row per subject.

    Expected columns: subject_id and the five feature columns of
    config.CLASSICAL_FEATURES. If there is one row per electrode (any other
    columns, e.g. "electrode", are ignored), each feature is averaged across
    the subject's electrodes.
    """
    raw = pd.read_csv(features_csv, dtype={"subject_id": str})
    missing = [c for c in ["subject_id"] + C.CLASSICAL_FEATURES if c not in raw.columns]
    if missing:
        raise ValueError(f"Columns missing from {features_csv}: {missing}")
    raw["subject_id"] = raw["subject_id"].str.strip()
    return raw.groupby("subject_id", as_index=False)[C.CLASSICAL_FEATURES].mean()


def align_features(subject_feats, subjects):
    """Attach features to a task's subjects, keeping the cross-validation order.

    `subjects` is the [subject_id, target] table of data.build_task_data;
    row i is outer fold i + 1, exactly as in the CBraMod run.
    """
    df = subjects.rename(columns={"target": "y_true"}).copy()
    df.insert(0, "outer_fold", np.arange(1, len(df) + 1))
    df = df.merge(subject_feats, on="subject_id", how="left")
    no_feats = df.loc[df[C.CLASSICAL_FEATURES].isna().any(axis=1), "subject_id"].tolist()
    if no_feats:
        print(f"  {len(no_feats)} subjects have no classical features and are skipped: {no_feats}")
    return df.reset_index(drop=True)


def run_loso(aligned, kind):
    """Leave-one-subject-out predictions of both classical models.

    `aligned` is in the cross-validation order, so rebuilding the splits
    gives the same outer and inner folds as the CBraMod run.
    Returns {model name: DataFrame}.
    """
    order = aligned["subject_id"].tolist()
    labels = aligned["y_true"].astype(int).tolist() if kind == "classification" else None
    splits = loso_splits(order, labels)

    feats = aligned.dropna(subset=C.CLASSICAL_FEATURES).set_index("subject_id")
    X = lambda ids: feats.loc[ids, C.CLASSICAL_FEATURES].values
    y = lambda ids: feats.loc[ids, "y_true"].values
    have = set(feats.index)

    rows = {name: [] for name in make_models(kind)}
    for sp in splits:
        if sp["test_id"] not in have:
            continue
        train_ids = [s for s in sp["train_ids"] if s in have]
        for name, est in make_models(kind).items():
            row = {"outer_fold": sp["outer_fold"], "subject_id": sp["test_id"],
                   "y_true": feats.loc[sp["test_id"], "y_true"]}
            if kind == "classification":
                # threshold with the best balanced accuracy averaged across inner folds
                scores = []
                for in_tr, in_va in sp["inner"]:
                    in_tr = [s for s in in_tr if s in have]
                    in_va = [s for s in in_va if s in have]
                    p_va = clone(est).fit(X(in_tr), y(in_tr)).predict_proba(X(in_va))[:, 1]
                    scores.append([balanced_accuracy_score(y(in_va), (p_va >= t).astype(int))
                                   for t in C.THRESHOLD_GRID])
                thr = float(C.THRESHOLD_GRID[int(np.argmax(np.mean(scores, axis=0)))])
                prob = float(clone(est).fit(X(train_ids), y(train_ids))
                             .predict_proba(X([sp["test_id"]]))[0, 1])
                row.update(y_true=int(row["y_true"]), selected_threshold=thr,
                           y_prob=prob, y_pred=int(prob >= thr))
            else:
                row["y_pred"] = float(clone(est).fit(X(train_ids), y(train_ids))
                                      .predict(X([sp["test_id"]]))[0])
            rows[name].append(row)
    return {name: pd.DataFrame(r) for name, r in rows.items()}
