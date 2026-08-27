"""Download Elliptic Bitcoin Dataset from Kaggle.

Usage:
    python -m openheads.download_elliptic

The destination is a parameter, not a constant: OPENHEADS_ELLIPTIC_DIR (or
the `raw_dir` argument) overrides the CWD-relative default — a fixed
CWD-relative write path in library code decides where files land based on
where the CALLER happens to stand.

Requires kagglehub or manual download from:
    https://www.kaggle.com/datasets/ellipticco/elliptic-data-set
"""

import os
import shutil
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)

DEFAULT_RAW_DIR = "data/raw/elliptic"

EXPECTED_FILES = [
    "elliptic_txs_features.csv",
    "elliptic_txs_edgelist.csv",
    "elliptic_txs_classes.csv",
]


def _resolve_raw_dir(raw_dir: Path | str | None) -> Path:
    """Explicit argument > OPENHEADS_ELLIPTIC_DIR > CWD-relative default.
    Resolved at call time, not import time, so env changes are honoured."""
    return Path(raw_dir or os.environ.get("OPENHEADS_ELLIPTIC_DIR") or DEFAULT_RAW_DIR)


def download(raw_dir: Path | str | None = None) -> Path:
    """Download Elliptic dataset via kagglehub. Returns path to raw directory."""
    try:
        import kagglehub
    except ImportError:
        logger.error(
            "kagglehub_not_installed",
            hint="pip install kagglehub, then run again",
        )
        raise

    dest = _resolve_raw_dir(raw_dir)
    dest.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    if all((dest / f).exists() for f in EXPECTED_FILES):
        logger.info("elliptic_already_downloaded", path=str(dest))
        return dest

    logger.info("elliptic_downloading")
    cache_path = kagglehub.dataset_download("ellipticco/elliptic-data-set")
    cache_dir = Path(cache_path)

    # Copy CSV files into the destination directory
    for fname in EXPECTED_FILES:
        # kagglehub may nest files in subdirectories
        found = list(cache_dir.rglob(fname))
        if found:
            shutil.copy2(found[0], dest / fname)
            logger.info("elliptic_file_copied", file=fname)
        else:
            logger.error("elliptic_file_missing", file=fname, search_dir=str(cache_dir))
            raise FileNotFoundError(f"Expected file {fname} not found in {cache_dir}")

    logger.info("elliptic_download_complete", path=str(dest))
    return dest


def verify(raw_dir: Path | str | None = None) -> bool:
    """Check that all Elliptic CSV files exist in the destination directory."""
    dest = _resolve_raw_dir(raw_dir)
    for fname in EXPECTED_FILES:
        if not (dest / fname).exists():
            logger.warning("elliptic_missing_file", file=fname)
            return False
    return True


if __name__ == "__main__":
    download()
