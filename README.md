# Cervical Cancer Risk Prediction with Neural Networks

> Comparing three Keras architectures — a simple MLP, a deeper MLP with dropout, and an autoencoder-based classifier — for predicting biopsy-confirmed cervical cancer risk from clinical and lifestyle risk factors.

## Table of Contents

- [Project Description](#project-description)
- [Prerequisites](#prerequisites)
- [Installation & Setup](#installation--setup)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Data Sources](#data-sources)
- [Methodology](#methodology)
- [Results](#results)
- [Reproducibility Notes](#reproducibility-notes)
- [Authors](#authors)
- [License](#license)

## Project Description

Cervical cancer screening data is severely class-imbalanced: in this dataset only
**6.4% of patients (55 of 858)** have a positive biopsy. A model that predicts
"negative" for everyone would still score 93.6% accuracy while catching zero cases —
so accuracy alone is a misleading objective here.

This project asks: **can a neural network trained on routinely-collected risk factors
identify biopsy-positive patients at a useful recall, and does added architectural
complexity help?**

Three architectures are trained and compared on an identical stratified split, with
SMOTE applied to the training fold only. Evaluation focuses on **recall, F1, and
AUC-ROC** rather than raw accuracy, because in a screening context a false negative
(a missed cancer) is far more costly than a false positive (an unnecessary follow-up).

## Prerequisites

| Tool | Version | Purpose |
|---|---|---|
| [Anaconda / Miniconda](https://www.anaconda.com/) | conda 23+ | Environment management |
| Python | 3.11 | Programming language |
| [Git](https://git-scm.com/) | 2.40+ | Version control |

Key libraries (full pins in `environment.yml` / `requirements.txt`): TensorFlow/Keras,
scikit-learn, imbalanced-learn, pandas, NumPy, Matplotlib, seaborn.

## Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/engabdullaalmulla-dev/cervical-cancer-risk-prediction.git
cd cervical-cancer-risk-prediction
```

### 2. Create and activate the environment

```bash
conda env create -f environment.yml
conda activate ds_research
```

<details>
<summary>Alternative: pip instead of conda</summary>

```bash
conda create --name ds_research python=3.11
conda activate ds_research
pip install -r requirements.txt
```

</details>

### 3. Download the dataset

The raw data is **not tracked in Git**. Fetch it from the original UCI source:

```bash
python src/download_data.py
```

This downloads the archive, verifies its MD5 checksum, and writes
`data/raw/kag_risk_factors_cervical_cancer.csv`.

### 4. Register the Jupyter kernel and launch

```bash
python -m ipykernel install --user --name ds_research
jupyter lab
```

Open `notebooks/01_cancer_risk_prediction.ipynb`, select the **ds_research** kernel,
and run all cells from top to bottom.

## Project Structure

```
cervical-cancer-risk-prediction/
├── data/
│   ├── raw/                             # Original CSV (gitignored - see step 3)
│   ├── processed/                       # Cleaned / transformed data
│   └── external/                        # Third-party data
├── notebooks/
│   └── 01_cancer_risk_prediction.ipynb  # Full analysis: EDA -> models -> evaluation
├── src/
│   ├── __init__.py
│   ├── download_data.py                 # Reproducible dataset download + checksum
│   └── data_loader.py                   # Loading, cleaning, and imputation
├── outputs/
│   ├── figures/                         # Exported EDA figures
│   └── reports/                         # Final report files
├── tests/                               # Unit tests for src/
├── environment.yml                      # Conda environment specification
├── requirements.txt                     # pip requirements
├── .gitignore
├── LICENSE
└── README.md
```

## Usage

**Run the full analysis** — open `notebooks/01_cancer_risk_prediction.ipynb` and run all
cells. It covers data cleaning, EDA, the three model architectures, and evaluation.
Total runtime is a few minutes on CPU.

**Reuse the preprocessing from Python:**

```python
from src.data_loader import load_features_target

X, y = load_features_target()   # cleaned, imputed, target separated
print(X.shape, y.value_counts().to_dict())
```

**Run the tests:**

```bash
python -m pytest tests/ -v
```

## Data Sources

| | |
|---|---|
| **Dataset** | Cervical cancer (Risk Factors) Data Set |
| **Source** | [UCI Machine Learning Repository, ID 383](https://archive.ics.uci.edu/dataset/383/cervical+cancer+risk+factors) |
| **Mirror** | Also distributed on Kaggle as `kag_risk_factors_cervical_cancer.csv` (byte-identical) |
| **Collected at** | Hospital Universitario de Caracas, Caracas, Venezuela |
| **Size** | 858 patients × 36 columns |
| **Target** | `Biopsy` (0 = negative, 1 = positive) |
| **Missing data** | 3,622 `?` placeholders across 26 columns |
| **License** | Creative Commons Attribution 4.0 (CC BY 4.0) |

**Citation** — Fernandes, K., Cardoso, J., & Fernandes, J. (2017). *Cervical cancer
(Risk Factors)* [Dataset]. UCI Machine Learning Repository.
https://doi.org/10.24432/C5Z310

The raw CSV is excluded from version control by `.gitignore`; `src/download_data.py`
reproduces it exactly (verified by MD5) so no manual download step is needed.

## Methodology

### Preprocessing

1. **Missing values** — `?` placeholders converted to `NaN`, all columns coerced to numeric.
2. **Column removal** — `STDs: Time since first diagnosis` and `STDs: Time since last diagnosis`
   dropped (both ~92% missing), leaving **33 predictor features**.
3. **Imputation** — median for the 10 continuous features, mode for the binary/categorical ones.
4. **Split** — stratified 70/15/15 → 600 train / 129 validation / 129 test.
5. **Scaling** — `StandardScaler` fit on the training set only, then applied to validation and test.
6. **Class imbalance** — SMOTE applied to the **training fold only** (562 neg / 38 pos → 562 / 562),
   so validation and test sets retain the real-world class distribution.

> Scaling and SMOTE are both fit after the split and on training data only, which avoids
> leaking test-set information into the model.

### Architectures

| Model | Architecture | Notes |
|---|---|---|
| **A — Simple MLP** | 33 → Dense(64, ReLU) → Dense(1, sigmoid) | Baseline |
| **B — Deep MLP** | 33 → Dense(64, ReLU) → Dropout(0.1) → Dense(32, ReLU) → Dense(1, sigmoid) | Tests whether depth + regularisation helps |
| **C — Autoencoder + MLP** | Autoencoder 33 → 20 → **10** → 20 → 33, then Dense(64, ReLU) → Dense(1, sigmoid) on the 10-d latent code | Unsupervised feature learning before classification |

All models: Adam (lr = 0.001), binary cross-entropy, batch size 64, up to 150 epochs
with early stopping on validation loss (patience 20; patience 15 for the autoencoder)
and best-weight restoration.

## Results

Test-set performance (129 held-out patients, real class distribution):

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|---|---|---|---|---|---|
| **Simple MLP** | **0.9535** | **0.6000** | 0.7500 | **0.6667** | 0.8874 |
| Deep MLP | 0.9535 | 0.6000 | 0.7500 | 0.6667 | 0.8781 |
| Autoencoder + MLP | 0.9457 | 0.5455 | 0.7500 | 0.6316 | **0.9101** |

**Findings:**

- **All three models recover 75% of positive cases** — a large improvement over the
  zero-recall baseline that naive accuracy optimisation would produce.
- **Added complexity did not help.** The deeper network matched the simple MLP exactly on
  every threshold metric, and the autoencoder variant traded precision (0.55 vs 0.60) for
  a marginally worse F1. With only 38 positive training examples before SMOTE, the dataset
  is too small for extra capacity to pay off.
- **The Simple MLP is the best overall model** (F1 = 0.667), and is also the cheapest to
  train and the easiest to interpret.
- **The autoencoder achieved the best ranking quality** (AUC-ROC = 0.910) despite the worst
  F1. AUC is threshold-independent, so its latent representation separates the classes well —
  it simply needs a decision threshold other than 0.5 to convert that into better
  precision/recall.
- **Precision near 0.6 means roughly 4 in 10 flagged patients are false positives.** For a
  screening triage tool this may be an acceptable trade for 75% recall, but it is not a
  diagnostic instrument.

EDA figures are in [`outputs/figures/`](outputs/figures/): class imbalance, feature
distributions, the correlation matrix, per-class boxplots, and outlier inspection.

### Limitations

- 55 positive cases in total (38 in training) — results are sensitive to the particular split.
- SMOTE synthesises minority examples from a very small set of real ones, which risks
  over-optimistic decision boundaries.
- Single train/validation/test split rather than cross-validation; a stratified k-fold
  would give more stable estimates.
- Single-centre data from one hospital in Caracas; generalisation to other populations
  is untested.
- Threshold fixed at 0.5 — tuning it per model (especially for Model C) would likely
  improve the precision/recall balance.

## Reproducibility Notes

- Random seeds are set for Python, NumPy, and TensorFlow (`SEED = 123`), TensorFlow
  op determinism is enabled, and splits use `random_state=42`.
- Package versions are pinned in `environment.yml` and `requirements.txt`. Both list
  direct dependencies only, so they solve on macOS, Linux, and Windows; conda and pip
  resolve the transitive dependencies. (A raw `pip freeze` is deliberately not used —
  inside a conda environment it emits local `file:///` build paths that will not
  install on another machine.)
- All paths in the notebook are relative, so it runs from any clone location.
- **Verified reproduction:** the notebook was re-executed from a clean
  `conda env create -f environment.yml` on macOS (Apple Silicon, Python 3.11.14,
  TensorFlow 2.21.0) and reproduced every reported metric to four decimal places,
  despite the original run being on Google Colab with different library versions.
  End-to-end runtime was roughly 30 seconds on CPU.
- Some variation remains possible on other hardware, since GPU kernel non-determinism
  is not fully constrained by `enable_op_determinism()` alone.

## Authors

- **Abdulla Almulla** — eng.abdulla.almulla@gmail.com

Prepared for the MSc Data Science Research Methods module (EM08DS), following the
faculty guide *Research Methodology for Data Science: A Practical Guide to Reproducible
Python Environments & GitHub Workflows*.

## License

Code in this repository is released under the [MIT License](LICENSE).
The underlying dataset is licensed CC BY 4.0 by the UCI Machine Learning Repository.
