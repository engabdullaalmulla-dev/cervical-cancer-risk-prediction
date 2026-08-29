"""Download the Cervical Cancer (Risk Factors) dataset into data/raw/.

The raw data is deliberately excluded from version control (see .gitignore),
so this script reproduces it from the original UCI source.

Usage:
    python src/download_data.py
"""

import hashlib
import io
import urllib.request
import zipfile
from pathlib import Path

UCI_ZIP_URL = (
    "https://archive.ics.uci.edu/static/public/383/"
    "cervical+cancer+risk+factors.zip"
)

# File as published inside the UCI archive
MEMBER_NAME = "risk_factors_cervical_cancer.csv"

# Name used throughout the notebook (the Kaggle mirror's filename)
TARGET_NAME = "kag_risk_factors_cervical_cancer.csv"

# md5 of the UCI release; the Kaggle mirror is byte-identical
EXPECTED_MD5 = "47f4cde87fb017ae0944847cb1c36fc4"

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def download(force: bool = False) -> Path:
    """Fetch the dataset, verify its checksum, and write it to data/raw/."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    target = RAW_DIR / TARGET_NAME

    if target.exists() and not force:
        print(f"Already present: {target}")
        return target

    print(f"Downloading from {UCI_ZIP_URL} ...")
    with urllib.request.urlopen(UCI_ZIP_URL) as response:
        archive_bytes = response.read()

    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        payload = archive.read(MEMBER_NAME)

    digest = hashlib.md5(payload).hexdigest()
    if digest != EXPECTED_MD5:
        raise ValueError(
            f"Checksum mismatch: expected {EXPECTED_MD5}, got {digest}. "
            "The upstream file may have changed."
        )

    target.write_bytes(payload)
    print(f"Saved {len(payload):,} bytes to {target}")
    print(f"md5 verified: {digest}")
    return target


if __name__ == "__main__":
    download()
