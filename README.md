# EEG Foundation Models in Parkinson's Disease

Training and statistical-analysis code for the paper *"Beyond Handcrafted Features? Evaluating the Clinical Utility of EEG Foundation Models in Parkinson's Disease"*.

We fine-tune the [CBraMod](https://github.com/wjq-learning/CBraMod) EEG foundation model on resting-state EEG of patients with Parkinson's disease (PD) and healthy controls (HC), and compare it with models trained on five classical EEG features (Lempel-Ziv complexity and relative theta, alpha, beta and gamma power), in four tasks:

| `--task` | Prediction | Type | Classical models |
|---|---|---|---|
| `pd_hc` | PD vs HC | classification | logistic regression, random forest |
| `moca` | normal (MoCA 26–30) vs reduced (21–25) cognition, PD only | classification | logistic regression, random forest |
| `updrs3` | motor severity (MDS-UPDRS-III), PD only | regression | linear regression, random forest |
| `gait_speed` | usual gait speed, PD only | regression | linear regression, random forest |

**Evaluation.** Leave-one-subject-out cross-validation. Within every outer fold, the remaining subjects are split into three inner folds, which select CBraMod's number of training epochs (lowest mean validation loss) and, for classification, every model's decision threshold (highest mean balanced accuracy, grid 0.1–0.9). The classical models use exactly the same folds. All settings are in [`cbramod_pd/config.py`](cbramod_pd/config.py).

This repository contains code only — no participant data.

## Installation

```bash
git clone <this repository> && cd <repository>
pip install -r requirements.txt

# CBraMod model code and pretrained weights
git clone https://github.com/wjq-learning/CBraMod
mkdir pretrained_weights   # put pretrained_weights.pth here (see the CBraMod README)
```

Fine-tuning CBraMod needs an NVIDIA GPU. The classical baseline runs on any computer.

## Input data

```
data/
├── eeg/
│   ├── PD/<subject_id>.npy      # one file per participant
│   └── HC/<subject_id>.npy
├── participants.csv
└── classical_features.csv
```

- **`eeg/PD`, `eeg/HC`** — each participant's preprocessed resting-state EEG: an array of shape `(n_segments, 19, 6000)` (artifact-free 30-s segments, 19 channels, 200 Hz), in CBraMod units (1.0 = 100 µV; use `--npy-units uV` or `V` for other units). The subject ID is the file name up to the first `_` (`S001.npy` and `S001_task-rest_eeg.npy` both give `S001`).
- **`participants.csv`** — one row per patient with PD: `subject_id`, `moca`, `updrs3`, `gait_speed`. Leave a cell empty when a score is missing; that patient is then left out of that task. Patients with MoCA < 21 are left out of the MoCA task.
- **`classical_features.csv`** — `subject_id` and the five feature columns `LZC_mean`, `theta_relative_bandpower`, `alpha_relative_bandpower`, `beta_relative_bandpower`, `gamma_relative_bandpower`. Either one row per subject or one row per electrode (features are then averaged across electrodes; other columns are ignored).

## Running locally

Everything, all four tasks:

```bash
python scripts/run_all.py --data-dir data \
    --cbramod-dir CBraMod/models --weights pretrained_weights/pretrained_weights.pth
```

Or step by step, per task:

```bash
python scripts/train_cbramod.py   --task pd_hc     # CBraMod (GPU)
python scripts/train_classical.py --task pd_hc     # classical baseline, same folds
python scripts/summarize.py                        # pooled metrics of all models
```

(`data`, `CBraMod/models`, `pretrained_weights/pretrained_weights.pth` and `results` are the default paths; change them with `--data-dir`, `--cbramod-dir`, `--weights`, `--results-dir`.)

Fine-tuning is slow: every outer fold trains four CBraMod models (three inner folds + the final model), so a task takes days on a single GPU. An interrupted run keeps every finished fold.

### Statistical analysis

After training, compare CBraMod with each classical model (separate from training; no GPU needed):

```bash
python scripts/permutation_tests.py            # all tasks; --task <name> for one
```

For every task: a paired permutation test of CBraMod vs each classical model — the statistic is CBraMod minus the classical model in balanced accuracy (from the predicted class labels) for classification, or in RMSE for regression; 1,000,000 iterations, each swapping every subject's two predictions with probability 0.5 (seed 42); two-sided p = (1 + b) / (1 + N) — followed by Benjamini-Hochberg FDR correction within the task's two comparisons.

### Outputs (`results/<task>/`)

| File | Content |
|---|---|
| `cbramod_predictions.csv` | per held-out subject: target, prediction, selected epochs (and threshold) |
| `cbramod_embeddings.csv` | per held-out subject: mean 200-d CBraMod embedding |
| `cbramod_inner_folds.csv` | best validation loss and epoch of every inner fold |
| `classical_predictions_<model>.csv` | per held-out subject: target, prediction (and threshold) |
| `classical_features.csv` | the per-subject features used |
| `permutation_tests.csv` | difference, p and FDR-corrected p of CBraMod vs each classical model |
| `performance.csv` | pooled metrics of all models (AUROC, balanced accuracy, sensitivity, specificity, accuracy; or RMSE, MAE, R², Pearson and Spearman r) |

## Running on AWS SageMaker (optional)

The outer folds can be split over several GPUs: `train_cbramod.py --num-jobs N --job-index k` runs only chunk `k` of `N`, and `--merge-only` combines the chunks afterwards. `aws/` uses this to run every chunk as a separate SageMaker training job:

```bash
# 1. upload data/ (plus CBraMod/models/ and pretrained_weights.pth inside it) to s3://<bucket>/data/
# 2. launch (from an environment with AWS credentials and a SageMaker execution role)
pip install sagemaker boto3
python aws/launch_sagemaker.py --bucket <bucket>
# 3. when the jobs have finished: download and merge, then run the classical baseline and summary
python aws/collect_results.py --bucket <bucket>
python scripts/run_all.py --skip-cbramod
python scripts/summarize.py
```

## Reproducibility

- All randomness is seeded (seed 42), and cuDNN runs in deterministic mode.
- The outer folds are processed in chunks, each starting from the same seeded random state. By default the chunks are the ones of the paper's run (20 for `pd_hc`, 5 for the other tasks), so local and AWS runs give the same results, and match the paper's. Changing `--num-jobs` gives statistically equivalent but not identical results.
- GPU model and library versions can cause small numerical differences; random-forest results depend on the scikit-learn version.
- `python tests/smoke_test.py` runs the whole pipeline on tiny synthetic data with a stand-in backbone (CPU, ~2 minutes).

## Citation

*To be added on publication.*

## License

*To be added.*
