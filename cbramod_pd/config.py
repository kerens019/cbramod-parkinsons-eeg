"""All analysis settings, as reported in the paper's Methods section."""

from dataclasses import dataclass
from typing import Optional

import numpy as np

# -- Reproducibility ---------------------------------------------------------
SEED = 42

# -- Cross-validation --------------------------------------------------------
N_INNER_FOLDS = 3                                       # inner folds per outer LOSO fold
THRESHOLD_GRID = np.round(np.linspace(0.1, 0.9, 9), 2)  # decision thresholds 0.1, 0.2, ..., 0.9

# -- CBraMod input format (matches CBraMod pretraining) ----------------------
SFREQ = 200          # Hz
N_CHANNELS = 19
N_PATCHES = 30       # 30-s segments = 30 one-second patches
PATCH_LEN = 200      # samples per patch
UV_PER_UNIT = 100.0  # CBraMod units: 1.0 = 100 microvolts

# -- CBraMod fine-tuning -----------------------------------------------------
BATCH_SIZE = 64
MAX_EPOCHS = 50
LR = 1e-4
WEIGHT_DECAY = 5e-2
MIN_LR = 1e-6
CLIP_NORM = 1.0
DROPOUT = 0.1
EMBED_DIM = 200

# -- Classical EEG-feature baseline ------------------------------------------
CLASSICAL_FEATURES = [
    "LZC_mean",
    "theta_relative_bandpower",
    "alpha_relative_bandpower",
    "beta_relative_bandpower",
    "gamma_relative_bandpower",
]
RF_N_TREES = 200
LOGREG_MAX_ITER = 2000

# -- MoCA groups ------------------------------------------------------------
MOCA_MIN = 21        # patients below 21 fall outside both groups and are excluded
MOCA_NORMAL = 26     # 26-30 = normal (label 1), 21-25 = reduced (label 0)


@dataclass(frozen=True)
class Task:
    name: str
    kind: str                         # "classification" or "regression"
    title: str
    score_col: Optional[str] = None   # column of participants.csv (None for PD vs HC)
    class_names: tuple = ()           # (label 0, label 1) for classification
    paper_num_jobs: int = 1           # how the outer folds were split into jobs in the paper's run


TASKS = {
    "pd_hc": Task("pd_hc", "classification", "PD vs HC",
                  class_names=("PD", "HC"), paper_num_jobs=20),
    "moca": Task("moca", "classification", "MoCA (normal vs reduced)", score_col="moca",
                 class_names=("Reduced (MoCA 21-25)", "Normal (MoCA 26-30)"), paper_num_jobs=5),
    "updrs3": Task("updrs3", "regression", "MDS-UPDRS-III", score_col="updrs3", paper_num_jobs=5),
    "gait_speed": Task("gait_speed", "regression", "Gait speed (m/s)", score_col="gait_speed",
                       paper_num_jobs=5),
}
