"""Fine-tuning loop for one leave-one-subject-out outer fold.

Training settings (Methods, "CBraMod Fine-Tuning"): AdamW (lr 1e-4, weight
decay 5e-2, batch 64), cosine annealing to 1e-6 stepped every batch,
gradient clipping at norm 1.0, up to 50 epochs. Classification uses BCE
with the positive class weighted by n_negative / n_positive training
subjects; regression uses MSE. Segment outputs are averaged per subject.
"""

import os
import random
from collections import defaultdict
from copy import deepcopy

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score
from torch.nn import BCEWithLogitsLoss, MSELoss
from torch.utils.data import DataLoader, Dataset

from . import config as C
from .model import CBraModFineTune

# DataLoader workers for the once-per-outer-fold loaders (scales with the machine)
DEFAULT_NUM_WORKERS = min(max((os.cpu_count() or 2) - 1, 1), 8)


def set_seed(seed=C.SEED):
    """Seed Python, NumPy and PyTorch, and make cuDNN deterministic."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# -- Data ------------------------------------------------------------------

class SegmentDataset(Dataset):
    """One item per 30-s segment: (patches, subject target, subject_id)."""

    def __init__(self, subject_ids, eeg, targets):
        self.items = [(eeg[sid][i], targets[sid], sid)
                      for sid in subject_ids for i in range(eeg[sid].shape[0])]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        x, y, sid = self.items[idx]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32), sid


def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_loader(subject_ids, eeg, targets, shuffle, num_workers=DEFAULT_NUM_WORKERS):
    """Shuffled loaders get their own generator seeded with SEED, so batch order is reproducible."""
    generator = None
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(C.SEED)
    return DataLoader(SegmentDataset(subject_ids, eeg, targets), batch_size=C.BATCH_SIZE,
                      shuffle=shuffle, num_workers=num_workers, pin_memory=True,
                      generator=generator, worker_init_fn=_seed_worker if num_workers > 0 else None)


# -- Training objects --------------------------------------------------------

def build_training(model, train_loader, n_epochs, kind, train_ids, targets, device):
    """Loss, optimizer and per-batch cosine schedule for one model."""
    if kind == "classification":
        n_pos = sum(1 for s in train_ids if targets[s] == 1)
        n_neg = sum(1 for s in train_ids if targets[s] == 0)
        if n_pos == 0 or n_neg == 0:
            raise ValueError("Both classes must be present in every training set.")
        pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32, device=device)
        criterion = BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs * len(train_loader), eta_min=C.MIN_LR)
    return criterion, optimizer, scheduler


def train_epoch(model, loader, criterion, optimizer, scheduler, device):
    model.train()
    running = torch.zeros((), device=device)
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), C.CLIP_NORM)
        optimizer.step()
        scheduler.step()
        running += loss.detach()
    return (running / len(loader)).item()


@torch.no_grad()
def predict_subjects(model, loader, device, kind, criterion=None):
    """Average segment outputs per subject (sigmoid probabilities for classification).

    Returns (subject_ids, y_true, y_hat, mean batch loss or None).
    """
    model.eval()
    outputs, truth = defaultdict(list), {}
    running = torch.zeros((), device=device) if criterion is not None else None
    n_batches = 0
    for x, y, sids in loader:
        x_dev, y_dev = x.to(device), y.to(device)
        out = model(x_dev)
        if criterion is not None:
            running += criterion(out, y_dev)
            n_batches += 1
        vals = (torch.sigmoid(out) if kind == "classification" else out).cpu().numpy()
        for v, t, s in zip(vals, y.numpy(), sids):
            outputs[s].append(v)
            truth[s] = int(t) if kind == "classification" else float(t)
    sids = list(outputs)
    y_true = np.array([truth[s] for s in sids])
    y_hat = np.array([np.mean(outputs[s]) for s in sids])
    loss = (running / n_batches).item() if criterion is not None else None
    return sids, y_true, y_hat, loss


def train_with_validation(model, train_loader, val_loader, criterion, optimizer, scheduler, device, kind):
    """Train for MAX_EPOCHS, recording the validation loss after every epoch.

    Returns (per-epoch validation losses, state dict of the lowest-loss epoch).
    """
    val_losses, best_loss, best_state = [], np.inf, None
    for _ in range(C.MAX_EPOCHS):
        train_epoch(model, train_loader, criterion, optimizer, scheduler, device)
        *_, val_loss = predict_subjects(model, val_loader, device, kind, criterion)
        val_losses.append(val_loss)
        if val_loss < best_loss:
            best_loss, best_state = val_loss, deepcopy(model.state_dict())
    return val_losses, best_state


def threshold_scores(y_true, y_prob):
    """Balanced accuracy of every threshold in THRESHOLD_GRID."""
    return [balanced_accuracy_score(y_true, (y_prob >= t).astype(int)) for t in C.THRESHOLD_GRID]


# -- One outer fold ------------------------------------------------------------

def run_outer_fold(split, eeg, targets, kind, weights_path, device, inner_num_workers=0):
    """Inner-fold model selection, refit on all training subjects, predict the held-out subject.

    Epochs: the epoch with the lowest validation loss averaged across inner folds.
    Threshold (classification): the grid value with the highest balanced
    accuracy averaged across inner folds.
    """
    inner_losses, inner_thr_scores, inner_rows = [], [], []
    for inner_fold, (tr_ids, va_ids) in enumerate(split["inner"], start=1):
        tr_loader = make_loader(tr_ids, eeg, targets, shuffle=True, num_workers=inner_num_workers)
        va_loader = make_loader(va_ids, eeg, targets, shuffle=False, num_workers=inner_num_workers)
        model = CBraModFineTune(weights_path).to(device)
        criterion, optimizer, scheduler = build_training(
            model, tr_loader, C.MAX_EPOCHS, kind, tr_ids, targets, device)
        losses, best_state = train_with_validation(
            model, tr_loader, va_loader, criterion, optimizer, scheduler, device, kind)
        inner_losses.append(losses)
        if kind == "classification":                 # score thresholds with the best-epoch model
            model.load_state_dict(best_state)
            _, y_va, p_va, _ = predict_subjects(model, va_loader, device, kind)
            inner_thr_scores.append(threshold_scores(y_va, p_va))
        inner_rows.append({"outer_fold": split["outer_fold"], "inner_fold": inner_fold,
                           "best_val_loss": float(np.min(losses)),
                           "best_epoch": int(np.argmin(losses)) + 1})

    n_epochs = int(np.argmin(np.mean(inner_losses, axis=0))) + 1
    threshold = None
    if kind == "classification":
        threshold = float(C.THRESHOLD_GRID[int(np.argmax(np.mean(inner_thr_scores, axis=0)))])

    # Refit on every training subject with the selected number of epochs
    train_ids, test_id = split["train_ids"], split["test_id"]
    train_loader = make_loader(train_ids, eeg, targets, shuffle=True)
    test_loader = make_loader([test_id], eeg, targets, shuffle=False)
    model = CBraModFineTune(weights_path).to(device)
    criterion, optimizer, scheduler = build_training(
        model, train_loader, n_epochs, kind, train_ids, targets, device)
    for _ in range(n_epochs):
        train_epoch(model, train_loader, criterion, optimizer, scheduler, device)

    # Held-out subject: mean of segment outputs and of segment embeddings
    model.eval()
    seg_out, seg_emb = [], []
    with torch.no_grad():
        for x, _, _ in test_loader:
            out, emb = model(x.to(device), return_embedding=True)
            if kind == "classification":
                out = torch.sigmoid(out)
            seg_out.extend(out.cpu().numpy().tolist())
            seg_emb.append(emb.cpu().numpy())
    y_hat = float(np.mean(seg_out))
    embedding = np.concatenate(seg_emb).mean(axis=0)

    row = {"outer_fold": split["outer_fold"], "subject_id": test_id,
           "selected_epochs": n_epochs, "y_true": targets[test_id]}
    if kind == "classification":
        row.update(selected_threshold=threshold, y_prob=y_hat, y_pred=int(y_hat >= threshold))
    else:
        row.update(y_pred=y_hat)
    emb_row = {"outer_fold": split["outer_fold"], "subject_id": test_id,
               **{f"emb_{i}": float(v) for i, v in enumerate(embedding)}}
    return row, emb_row, inner_rows, model
