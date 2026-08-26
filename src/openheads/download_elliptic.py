"""Download Elliptic Bitcoin Dataset from Kaggle.

Usage:
    python -m openheads.download_elliptic

Requires kagglehub or manual download from:
    https://www.kaggle.com/datasets/ellipticco/elliptic-data-set
"""

import shutil
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)

RAW_DIR = Path("data/raw/elliptic")

EXPECTED_FILES = [
    "elliptic_txs_features.csv",
    "elliptic_txs_edgelist.csv",
    "elliptic_txs_classes.csv",
]


def download() -> Path:
    """Download Elliptic dataset via kagglehub. Returns path to raw directory."""
    try:
        import kagglehub
    except ImportError:
        logger.error(
            "kagglehub_not_installed",
            hint="pip install kagglehub, then run again",
        )
        raise

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    if all((RAW_DIR / f).exists() for f in EXPECTED_FILES):
        logger.info("elliptic_already_downloaded", path=str(RAW_DIR))
        return RAW_DIR

    logger.info("elliptic_downloading")
    cache_path = kagglehub.dataset_download("ellipticco/elliptic-data-set")
    cache_dir = Path(cache_path)

    # Copy CSV files to our data/raw/elliptic/
    for fname in EXPECTED_FILES:
        # kagglehub may nest files in subdirectories
        found = list(cache_dir.rglob(fname))
        if found:
            shutil.copy2(found[0], RAW_DIR / fname)
            logger.info("elliptic_file_copied", file=fname)
        else:
            logger.error("elliptic_file_missing", file=fname, search_dir=str(cache_dir))
            raise FileNotFoundError(f"Expected file {fname} not found in {cache_dir}")

    logger.info("elliptic_download_complete", path=str(RAW_DIR))
    return RAW_DIR


def verify() -> bool:
    """Check that all Elliptic CSV files exist in data/raw/elliptic/."""
    for fname in EXPECTED_FILES:
        if not (RAW_DIR / fname).exists():
            logger.warning("elliptic_missing_file", file=fname)
            return False
    return True


if __name__ == "__main__":
    download()
