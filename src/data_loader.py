"""Loading and preprocessing for the cervical cancer risk factors dataset.

Mirrors the preprocessing performed in
``notebooks/01_cancer_risk_prediction.ipynb`` so that the same pipeline can be
reused from scripts and tests.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = PROJECT_ROOT / "data" / "raw" / "kag_risk_factors_cervical_cancer.csv"

TARGET = "Biopsy"

# ~92% missing; dropped rather than imputed
HIGH_MISSING_COLS = [
    "STDs: Time since first diagnosis",
    "STDs: Time since last diagnosis",
]

CONTINUOUS_COLS = [
    "Age",
    "Number of sexual partners",
    "First sexual intercourse",
    "Num of pregnancies",
    "Smokes (years)",
    "Smokes (packs/year)",
    "Hormonal Contraceptives (years)",
    "IUD (years)",
    "STDs (number)",
    "STDs: Number of diagnosis",
]


def load_raw(csv_path: Path = DEFAULT_CSV) -> pd.DataFrame:
    """Read the raw CSV exactly as published (missing values appear as '?')."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. Run `python src/download_data.py` first."
        )
    return pd.read_csv(csv_path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Replace '?' with NaN, coerce to numeric, and drop high-missing columns."""
    df_clean = df.replace("?", np.nan)
    for col in df_clean.columns:
        df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")
    return df_clean.drop(columns=HIGH_MISSING_COLS)


def impute(X: pd.DataFrame) -> pd.DataFrame:
    """Median-impute continuous features, mode-impute binary features."""
    X = X.copy()
    continuous = [c for c in CONTINUOUS_COLS if c in X.columns]
    binary = [c for c in X.columns if c not in continuous]

    X[continuous] = SimpleImputer(strategy="median").fit_transform(X[continuous])
    X[binary] = SimpleImputer(strategy="most_frequent").fit_transform(X[binary])
    return X


def load_features_target(csv_path: Path = DEFAULT_CSV):
    """Return (X, y) ready for scaling: cleaned, imputed, target separated."""
    df_clean = clean(load_raw(csv_path))
    X = df_clean.drop(columns=[TARGET])
    y = df_clean[TARGET]
    return impute(X), y


if __name__ == "__main__":
    X, y = load_features_target()
    print(f"Features: {X.shape}, Target: {y.shape}")
    print(f"Remaining NaN values: {int(X.isna().sum().sum())}")
    print(f"Class balance: {y.value_counts().to_dict()}")
