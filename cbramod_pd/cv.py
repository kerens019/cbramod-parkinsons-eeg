"""Leave-one-subject-out outer loop with subject-level inner folds."""

from sklearn.model_selection import KFold, LeaveOneOut, StratifiedKFold

from . import config as C


def loso_splits(subject_ids, labels=None, n_inner=C.N_INNER_FOLDS, seed=C.SEED):
    """Build every outer fold of the leave-one-subject-out cross-validation.

    Each outer fold holds out one subject for testing. Its remaining
    subjects are split into `n_inner` inner folds (stratified by class when
    `labels` is given, i.e. for the classification tasks), used only to
    choose the number of epochs and the decision threshold.
    Outer folds are numbered from 1; inner splits are seeded with seed + outer_fold.
    """
    subject_ids = list(subject_ids)
    splits = []
    for fold, (tv_idx, test_idx) in enumerate(LeaveOneOut().split(subject_ids), start=1):
        tv_ids = [subject_ids[i] for i in tv_idx]
        if labels is not None:
            tv_labels = [labels[i] for i in tv_idx]
            inner_cv = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=seed + fold)
            inner_iter = inner_cv.split(tv_ids, tv_labels)
        else:
            inner_cv = KFold(n_splits=n_inner, shuffle=True, random_state=seed + fold)
            inner_iter = inner_cv.split(tv_ids)
        inner = [([tv_ids[i] for i in tr], [tv_ids[i] for i in va]) for tr, va in inner_iter]
        test_id = subject_ids[test_idx[0]]
        for tr, va in inner:                       # no subject in two sets of the same fold
            assert not (set(tr) & set(va)) and test_id not in tr and test_id not in va
        splits.append({"outer_fold": fold, "test_id": test_id, "train_ids": tv_ids, "inner": inner})
    return splits


def job_slice(splits, job_index, num_jobs):
    """Contiguous, size-balanced chunk of outer folds for one parallel job."""
    base, extra = divmod(len(splits), num_jobs)
    start = job_index * base + min(job_index, extra)
    return splits[start:start + base + (1 if job_index < extra else 0)]
