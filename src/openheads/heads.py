"""Scoring heads and the training primitives the backbone scripts share.

Two heads sit on one shared embedding, each with its own loss:

- a narrow head (128 -> 64 -> 1) for the rarer behavioural class,
- a wider head (128 -> 256 -> 128 -> 1) for the broad financial-crime class.

They are trained jointly on a weighted sum of independent BCE losses rather
than as one multi-class head, because the two labels are not mutually
exclusive: an address can be both, and a softmax over them would force the
model to choose. The weights (ALPHA/BETA) are the ratio at which the two
losses enter that sum.

Two things here are easy to get wrong and expensive to discover late:

- ``recall_at_fpr`` is the metric that matters for screening, not accuracy or
  AUROC. At a fixed false-positive budget it answers "of the illicit entities
  we know about, what share would this model surface", which is what an alert
  queue is sized against. AUROC can look excellent while recall at a 0.1%
  FPR budget is near zero.
- Reporting one head's score over a pool that contains the other head's
  positives understates recall, because you are asking a head about entities
  it was designed to reject. Score each positive with its own head and
  calibrate the joint false-positive rate empirically.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from typing import Any

# Embedding width of the shared backbone. The heads' first Linear must match
# it, so it lives next to them rather than in a caller's argv.
EMBED_DIM = 128

# Training pools and split.
RANDOM_SEED = 42
TEST_FRACTION = 0.10  # frozen held-out test, stratified, same seed

# Loss weights: the two heads' independent BCE losses enter the total at this
# ratio. Not a tuning knob to sweep blindly — it encodes which class the run
# is willing to lose recall on.
ALPHA_AIAGENT = 0.3
BETA_FINCRIME = 0.7

# Dropout of the JOINT two-head module (build_model). The standalone
# per-chain training scripts train with dropout 0.2 — that value is their
# recipe and is passed explicitly at the call sites.
DROPOUT = 0.3


def setup_logging(log_dir: str = "./runs", name: str = "openheads") -> logging.Logger:
    """Log to stdout and to a file, timestamps in UTC.

    UTC is not cosmetic: a run spanning a daylight-saving change produces
    timestamps that appear to go backwards in local time, and that is the log
    someone reads to work out whether a job was still alive.
    """
    from pathlib import Path

    path = Path(log_dir)
    path.mkdir(parents=True, exist_ok=True)
    log_path = path / f"{name}.log"
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)sZ {name} %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
        force=True,
    )
    logging.Formatter.converter = time.gmtime
    log = logging.getLogger(name)
    log.info("log_path=%s", log_path)
    return log


def log_json(log: logging.Logger, event: str, **kw: Any) -> None:
    """One structured line per event, for post-hoc audit of a finished run."""
    payload = {"event": event, "ts": time.time(), **kw}
    log.info("JSON %s", json.dumps(payload, default=str))


def build_fincrime_head(embed_dim: int = EMBED_DIM, dropout: float = 0.2):  # type: ignore[no-untyped-def]
    """THE fincrime head architecture (embed_dim -> 256 -> 128 -> 1).

    Single source: the training scripts, the tau scan and the joint module
    all build the head here, so the architecture cannot drift between the
    file that trains a checkpoint and the file that loads it strict=True.
    Returned as a bare ``nn.Sequential`` — its state_dict keys (``0.weight``,
    ``3.weight``, ``6.weight``) are the on-disk contract of
    ``fincrime_<chain>.pt``.
    """
    import torch.nn as nn

    return nn.Sequential(
        nn.Linear(embed_dim, 256), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(128, 1),
    )


def build_aiagent_head(embed_dim: int = EMBED_DIM, dropout: float = 0.0):  # type: ignore[no-untyped-def]
    """THE narrow behavioural head (embed_dim -> 64 -> 1).

    The Dropout layer is kept even at p=0.0: it occupies an index, so the
    state_dict keys are 0/3, not 0/2 — dropping it makes strict=True loading
    of an existing checkpoint fail with missing/unexpected keys.
    """
    import torch.nn as nn

    return nn.Sequential(
        nn.Linear(embed_dim, 64), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(64, 1),
    )


def build_model(dropout: float = DROPOUT, embed_dim: int = EMBED_DIM):  # type: ignore[no-untyped-def]
    """Two independent heads over one shared embedding.

    Returns a module whose ``forward`` yields ``(narrow_logits, broad_logits)``
    — logits, not probabilities: the losses below are the
    ``*WithLogits`` variants, which are numerically stable where an explicit
    sigmoid followed by a log is not.
    """
    import torch.nn as nn

    class TwoHeadDetector(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.aiagent = build_aiagent_head(embed_dim, dropout)
            self.fincrime = build_fincrime_head(embed_dim, dropout)

        def forward(self, x):  # type: ignore[no-untyped-def]
            return self.aiagent(x).squeeze(-1), self.fincrime(x).squeeze(-1)

    return TwoHeadDetector()


def recall_at_fpr(y_true: Any, scores: Any, fpr_target: float) -> float:
    """Share of positives recovered at a false-positive rate no higher than
    ``fpr_target``.

    Returns 0.0 for a degenerate pool (all-positive or all-negative): there is
    no false-positive rate to hold, and a metric of 1.0 there would read as
    success.
    """
    import numpy as np
    from sklearn.metrics import roc_curve

    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return 0.0
    fpr, tpr, _ = roc_curve(y_true, scores)
    # searchsorted(..., "right") - 1 lands on the last operating point whose
    # FPR does NOT exceed the budget. Taking the next point up would report a
    # recall the budget does not pay for.
    i = int(np.searchsorted(fpr, fpr_target, side="right")) - 1
    return float(tpr[i]) if i >= 0 else 0.0
